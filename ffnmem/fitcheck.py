"""Memory-fit gate: measure peak GPU memory of the real training step.

Every wave-1 config must have a measured peak on its target GPU type before any
training run starts.

Measure (cluster, 4 GPUs; one isolated torchrun per combination):

    python -m ffnmem.fitcheck --config configs/w1_mem_full.yaml --nproc 4 \
        --micro_batch 1 2 --seq_len 2048 1024 --zero1 off on --out runs/fitcheck/2080ti.jsonl

Predict (local small GPU): measure 1- and 2-layer slices of the config's layer
types at full width, then extrapolate linearly to the full depth:

    python -m ffnmem.fitcheck --config configs/w1_mem_full.yaml --extrapolate --out runs/fitcheck/local.jsonl

A combination that runs out of memory is recorded as ``status: oom``; the grid
continues.  The child step matches ``train.py``: DDP with
``gradient_as_bucket_view``, activation checkpointing, fp16 autocast +
GradScaler, clip, fused AdamW (or ZeRO-1), two micro-batches per step
(exercising ``no_sync``).
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

import torch

from .config import load_config
from .distlaunch import run as dist_run
from .losses import select_positions
from .model import count_params

GB = 2**30


# ----------------------------------------------------------------- child
def run_child(args) -> None:
    from torch.nn.parallel import DistributedDataParallel as DDP

    from .train import Dist, amp_context, build_model, build_optimizer

    overrides = list(args.set) + [f"model.seq_len={args.seq_len[0]}", f"train.micro_batch={args.micro_batch[0]}"]
    cfg = load_config(args.config, overrides)
    if args.layers_json:
        sub = json.loads(args.layers_json)
        cfg["model"]["layers"] = sub["layers"]
        cfg["model"]["memory_layers"] = sub["memory_layers"]
        if not sub["memory_layers"]:
            cfg["model"]["memory"] = None
            cfg["model"]["draft"] = None
            cfg["model"]["ssm"] = None
        elif cfg["model"].get("draft"):
            cfg["model"]["draft"] = dict(cfg["model"]["draft"], layer=sub["memory_layers"][0])
    tcfg = cfg["train"]
    zero1 = args.zero1[0] == "on"
    d = Dist("cuda", force_pg=True)
    row = {
        "config": cfg["name"], "layers": cfg["model"]["layers"], "memory_layers": cfg["model"]["memory_layers"],
        "seq_len": args.seq_len[0], "micro_batch": args.micro_batch[0], "zero1": zero1,
        "world_size": d.world_size, "gpu": torch.cuda.get_device_name(d.device),
        "gpu_total_gb": torch.cuda.get_device_properties(d.device).total_memory / GB,
        "precision": tcfg["precision"], "act_ckpt": tcfg["act_ckpt"],
        "execution_mode": (cfg["model"]["memory"] or {}).get("execution_mode"),
    }
    try:
        model = build_model(cfg, d.device)
        row["params"] = count_params(model)
        ddp = DDP(model, device_ids=[d.local_rank], gradient_as_bucket_view=True)
        opt = build_optimizer(model, tcfg, d.device, zero1)
        scaler = torch.amp.GradScaler("cuda", enabled=tcfg["precision"] == "fp16")
        vocab, mb, seq = cfg["model"]["vocab"], args.micro_batch[0], args.seq_len[0]
        gen = torch.Generator(device="cpu").manual_seed(0)
        torch.cuda.reset_peak_memory_stats(d.device)
        times, peaks = [], []
        for _ in range(args.steps):
            torch.cuda.synchronize(d.device)
            t0 = time.time()
            for i in range(2):
                x = torch.randint(0, vocab, (mb, seq + 1), generator=gen).to(d.device)
                sync = ddp.no_sync() if i == 0 else contextlib.nullcontext()
                with sync:
                    with amp_context(d.device, tcfg["precision"]):
                        if model.draft is not None:  # Wave 3: include the draft loss, as train.py does
                            frac = float(model.draft_cfg.get("loss_position_frac", 0.25))
                            sel = select_positions(list(range(i * mb, (i + 1) * mb)), seq, frac, d.device)
                            loss = ddp(x[:, :-1], x[:, 1:], draft_sel=sel)[1]
                        else:
                            loss = ddp(x[:, :-1], x[:, 1:])[1]
                    scaler.scale(loss / 2).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(d.device)
            times.append(time.time() - t0)
            peaks.append(torch.cuda.max_memory_allocated(d.device) / GB)
        row.update(
            status="ok",
            peak_alloc_gb=max(peaks),
            peak_reserved_gb=torch.cuda.max_memory_reserved(d.device) / GB,
            peaks_per_step_gb=peaks,
            step_time_s=times[-1],
            tok_per_s=d.world_size * 2 * mb * seq / times[-1],
            loss=float(loss),
        )
    except torch.cuda.OutOfMemoryError as e:
        row.update(status="oom", peak_alloc_gb=torch.cuda.max_memory_allocated(d.device) / GB,
                   peak_reserved_gb=torch.cuda.max_memory_reserved(d.device) / GB, error=str(e).splitlines()[0])
    if d.is_main:
        with open(args.result, "w", encoding="utf-8") as f:
            json.dump(row, f)
    d.close()


# ---------------------------------------------------------------- parent
def launch(config: str, seq: int, mb: int, zero1: str, nproc: int, steps: int, sets: list[str],
           layers_json: str | None, timeout: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        result = Path(tmp) / "row.json"
        child = ["-m", "ffnmem.fitcheck", "--child", "--config", config, "--seq_len", str(seq),
                 "--micro_batch", str(mb), "--zero1", zero1, "--steps", str(steps), "--result", str(result)]
        for s in sets:
            child += ["--set", s]
        if layers_json:
            child += ["--layers_json", layers_json]
        t0 = time.time()
        try:
            proc = dist_run(nproc, child, timeout=timeout)
            rc, err = proc.returncode, proc.stderr
        except subprocess.TimeoutExpired:
            rc, err = -1, "timeout"
        if result.exists():
            row = json.loads(result.read_text(encoding="utf-8"))
        else:
            row = {"config": config, "seq_len": seq, "micro_batch": mb, "zero1": zero1 == "on",
                   "status": "oom" if "OutOfMemoryError" in err else "fail", "returncode": rc,
                   "stderr_tail": err[-2000:]}
        row["wall_s"] = time.time() - t0
        return row


def layer_types(model_cfg: dict) -> list[tuple]:
    mem = set(model_cfg["memory_layers"])
    return [(model_cfg["d_ff"][lid], pos in mem) for pos, lid in enumerate(model_cfg["layers"])]


def slice_spec(model_cfg: dict, positions: list[int]) -> str:
    mem = set(model_cfg["memory_layers"])
    layers = [model_cfg["layers"][p] for p in positions]
    memory_layers = [i for i, p in enumerate(positions) if p in mem]
    return json.dumps({"layers": layers, "memory_layers": memory_layers})


def extrapolate(args, seq: int, mb: int, rows_out: list) -> dict:
    """peak(full) ~= P(base) + sum_l D(type_l) - D(base), with D(t) = P(base + t) - P(base).

    Conservative: transient (recompute) activations of the largest layer type
    are counted once per layer of that type beyond the base.
    """
    cfg = load_config(args.config, args.set)
    m = cfg["model"]
    types = layer_types(m)
    counts = Counter(types)
    base_t = max((t for t in counts if not t[1]), key=lambda t: counts[t], default=types[0])
    base_pos = types.index(base_t)
    measured: dict = {}

    def measure(positions: list[int], label: str) -> dict:
        row = launch(args.config, seq, mb, "off", 1, args.steps, args.set, slice_spec(m, positions), args.timeout)
        row.update(kind="slice", slice=label)
        rows_out.append(row)
        print(json.dumps({k: row.get(k) for k in ("slice", "seq_len", "micro_batch", "status", "peak_alloc_gb", "step_time_s")}), flush=True)
        return row

    p_base = measure([base_pos], "base")
    if p_base["status"] != "ok":
        return {"status": "slice_oom", "seq_len": seq, "micro_batch": mb}
    for t in counts:
        others = [i for i, tt in enumerate(types) if tt == t and i != base_pos]
        pos = sorted([base_pos, others[0] if others else base_pos])
        r = measure(pos, f"base+{t}")
        if r["status"] != "ok":
            return {"status": "slice_oom", "seq_len": seq, "micro_batch": mb}
        measured[t] = r["peak_alloc_gb"] - p_base["peak_alloc_gb"]
    predicted = p_base["peak_alloc_gb"] + sum(measured[t] for t in types) - measured[base_t]
    return {
        "kind": "prediction", "config": cfg["name"], "seq_len": seq, "micro_batch": mb, "status": "ok",
        "predicted_peak_alloc_gb_ddp_world1": predicted,
        "per_layer_gb": {f"d_ff={t[0]},mem={t[1]}": v for t, v in measured.items()},
        "base_slice_gb": p_base["peak_alloc_gb"],
        "note": "world_size=1 measurement; ZeRO-1 on W GPUs saves ~8*(1-1/W) bytes/param of Adam state",
    }


def analytic(cfg: dict, world_size: int = 4) -> dict:
    """Persistent bytes per GPU (no activations): fp32 weights + grads + Adam."""
    from .params import model_counts

    p = model_counts(cfg["model"])["total"]
    return {
        "params": p,
        "weights_gb": 4 * p / GB,
        "grads_gb": 4 * p / GB,
        "adam_gb": 8 * p / GB,
        "adam_zero1_gb": 8 * p / GB / world_size,
        "state_total_gb": 16 * p / GB,
        "state_total_zero1_gb": (8 * p + 8 * p / world_size) / GB,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--micro_batch", type=int, nargs="+", default=[1])
    p.add_argument("--seq_len", type=int, nargs="+", default=[2048])
    p.add_argument("--zero1", nargs="+", default=["off"], choices=["off", "on"])
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--out", default="runs/fitcheck/fitcheck.jsonl")
    p.add_argument("--extrapolate", action="store_true")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--result", help=argparse.SUPPRESS)
    p.add_argument("--layers_json", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.child:
        run_child(args)
        return
    args.config = str(Path(args.config).resolve())  # children run from the repo root

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    cfg = load_config(args.config, args.set)
    rows.append({"kind": "analytic", "config": cfg["name"], **analytic(cfg, max(args.nproc, 4))})
    for seq, mb in itertools.product(args.seq_len, args.micro_batch):
        if args.extrapolate:
            rows.append(extrapolate(args, seq, mb, rows))
            print(json.dumps(rows[-1]), flush=True)
            continue
        for z in args.zero1:
            row = launch(args.config, seq, mb, z, args.nproc, args.steps, args.set, None, args.timeout)
            row["kind"] = "measured"
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ("config", "seq_len", "micro_batch", "zero1", "world_size",
                                                      "status", "peak_alloc_gb", "peak_reserved_gb",
                                                      "gpu_total_gb", "tok_per_s")}), flush=True)
    with open(out, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
