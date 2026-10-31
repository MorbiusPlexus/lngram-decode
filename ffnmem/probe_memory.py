"""How much does the Lngram memory contribute? Zero-out probe on a checkpoint.

    python -m ffnmem.probe_memory --ckpt model_step2000.pt --data_dir <climbmix> --zero none --out probe.jsonl
    python -m ffnmem.probe_memory --ckpt ... --zero all        # both memory layers output 0
    python -m ffnmem.probe_memory --ckpt ... --zero 1          # only the memory at kept-layer position 1

Evaluates val loss on the fixed held-out slice with the chosen memory layers'
output replaced by zeros (the rest of the model unchanged), and reports per
memory layer the mean output/residual norm ratio.  With ``--dump N`` it also
saves, for the first N val windows, per-token loss and per-token Sink mass of
each memory layer (float16 .npy) for the failed-lookup analysis.

Standalone on purpose: it does not modify any module used by running jobs.
Accepts a full training checkpoint or a model-only file ({"model", "config"}).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import apply_overrides
from .data.loader import ValWindows
from .memory.wrapper import LngramMemory
from .model import Transformer


@torch.no_grad()
def sink_per_token(mem: LngramMemory, h: torch.Tensor, h_norm: torch.Tensor) -> torch.Tensor:
    """[B, T] Sink mass averaged over readout heads (same path as the module's full mode)."""
    m = mem.module
    codes = m.quantize_route_logits(mem.route_logits(h_norm))
    query = m._project_query(h)
    tokens = m._lookup_route_chunk(codes, None, 0, m.num_routes, None, None)
    key, _ = m._project_memory(tokens)
    b, t = query.shape[:2]
    grouped = query.view(b, t, m.readout_num_kv_heads, m.readout_groups, m.readout_head_dim)
    scores = torch.matmul(grouped, key.transpose(-1, -2)).float() * m.softmax_scale
    real = m._real_route_probabilities(scores)
    return (1.0 - real.sum(-1)).mean(dim=(-1, -2))


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--zero", default="none", help="none | all | comma-separated kept-layer positions, e.g. 1 or 1,11")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--val_tokens", type=int, default=None, help="default: the checkpoint config's val_tokens")
    p.add_argument("--dump", type=int, default=0, help="save per-token loss + sink for the first N val windows")
    p.add_argument("--dump_dir", default=None)
    p.add_argument("--out", default=None, help="append one JSON line here")
    p.add_argument("--label", default="")
    args = p.parse_args(argv)

    torch.set_grad_enabled(False)
    t0 = time.time()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = apply_overrides(state["config"], [f"train.data_dir={args.data_dir}"])
    step = state.get("step")
    device = torch.device("cuda")
    model = Transformer(cfg["model"])
    model.load_state_dict(state["model"])
    del state
    model.set_attention_backend("sdpa")
    model.to(device).eval()

    mem_pos = [i for i, b in enumerate(model.blocks) if b.memory is not None]
    zero = set(mem_pos) if args.zero == "all" else (set() if args.zero == "none" else {int(x) for x in args.zero.split(",")})
    if not zero <= set(mem_pos):
        raise SystemExit(f"--zero {sorted(zero)} not a subset of memory layers {mem_pos}")

    # Instrument memory layers: norm ratio of output vs residual, optional zeroing, optional sink capture.
    stats = {pos: [0.0, 0] for pos in mem_pos}
    capture: dict[int, list] = {pos: [] for pos in mem_pos}
    dumping = {"on": False}
    for pos in mem_pos:
        mem = model.blocks[pos].memory
        orig = mem.forward

        def fwd(h, h_norm, _orig=orig, _pos=pos, _mem=mem):
            y = _orig(h, h_norm)
            ratio = (y.float().norm(dim=-1) / h.float().norm(dim=-1).clamp_min(1e-6)).mean()
            stats[_pos][0] += float(ratio)
            stats[_pos][1] += 1
            if dumping["on"]:
                capture[_pos].append(sink_per_token(_mem, h, h_norm).half().cpu())
            return torch.zeros_like(y) if _pos in zero else y

        mem.forward = fwd

    val_tokens = args.val_tokens or int(cfg["train"]["val_tokens"])
    val = ValWindows(args.data_dir, cfg["model"]["seq_len"], val_tokens)
    loss_sum, n_tok, n_win = 0.0, 0, 0
    tok_losses = []
    for x, y in val.batches(args.batch, args.max_batches):
        x, y = x.to(device), y.to(device)
        dumping["on"] = n_win < args.dump
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        per_tok = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1), reduction="none").view(y.shape)
        if dumping["on"]:
            tok_losses.append(per_tok.half().cpu())
        loss_sum += float(per_tok.sum())
        n_tok += y.numel()
        n_win += x.shape[0]

    result = {
        "label": args.label, "ckpt": str(args.ckpt), "step": step, "config": cfg["name"], "zero": sorted(zero),
        "memory_layers": mem_pos, "val_loss": loss_sum / n_tok, "val_tokens": n_tok,
        "mem_out_to_residual_norm": {str(k): (v[0] / v[1] if v[1] else None) for k, v in stats.items()},
        "seconds": round(time.time() - t0, 1),
    }
    result["val_ppl"] = math.exp(min(result["val_loss"], 50))
    if args.dump and tok_losses:
        d = Path(args.dump_dir or Path(args.ckpt).with_suffix("").as_posix() + "_dump")
        d.mkdir(parents=True, exist_ok=True)
        tag = f"zero-{'-'.join(map(str, sorted(zero))) or 'none'}"
        np.save(d / f"{tag}_token_loss.npy", torch.cat(tok_losses).numpy())
        for pos in mem_pos:
            if capture[pos]:
                np.save(d / f"{tag}_sink_mem{pos}.npy", torch.cat(capture[pos]).numpy())
        result["dump_dir"] = str(d)
    print(json.dumps(result), flush=True)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
    return result


if __name__ == "__main__":
    main()
