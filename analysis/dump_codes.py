"""M2.T1: per-position dumps of a checkpoint on the validation slice.

    python -m analysis.dump_codes --ckpt runs/w3_sb_1b --data /data/climbmix --run sb --max_tokens 5000000
    python -m analysis.dump_codes --ckpt runs/w1_mem_full/checkpoints/step_0001000.pt --data ... --run run1_s1000

Per input position (val windows are contiguous ``seq_len`` chunks of ``val.bin``,
taken in order; the model sees no context across a window start):

- ``tokens`` x[t], ``next`` x[t+1], ``window``, ``pos`` (position in the window),
  ``doc`` (document id: number of EOS tokens strictly before t in ``val.bin``,
  so an EOS belongs to the document it ends);
- per memory block position p: ``codes_L{p}`` uint8 [N, R] (the router's codes),
  ``memnorm_L{p}`` (L2 norm of the memory output added to the residual) and
  ``sink_L{p}`` (Sink mass = 1 - real-route probability, averaged over readout heads);
- ``top1`` (the model's argmax next token), ``topk_ids`` / ``topk_logp`` (top-32
  log-probs), ``logp_next`` (log-prob of the true next token).

Files: ``dumps/<run>/<field>.<shard:05d>.npy`` + ``dumps/<run>/meta.json``.
Codes come from ``capture_route_logits``; memory norms from a forward hook; the
Sink mass from wrapping the memory module's ``_real_route_probabilities`` on the
instance (the memory runs in ``execution_mode: full`` during the dump).
Nothing in ``ffnmem/`` is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ffnmem.data.loader import ValWindows, read_meta
from ffnmem.infer import load_model
from ffnmem.model import Transformer
from ffnmem.train import amp_context

PER_POSITION = ("tokens", "next", "window", "pos", "doc", "top1", "topk_ids", "topk_logp", "logp_next")


# ------------------------------------------------------------------ helpers
def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_precision(device: torch.device) -> str:
    return "fp16" if device.type == "cuda" else "fp32"


def memory_positions(model: Transformer) -> list[int]:
    return [i for i, b in enumerate(model.blocks) if b.memory is not None]


def doc_ids(val_tokens: np.ndarray, n: int, eos_id: int | None) -> np.ndarray:
    """Document id of each of the first ``n`` val positions (EOS closes its document)."""
    toks = np.asarray(val_tokens[:n])
    if eos_id is None:
        return np.zeros(n, dtype=np.int64)
    is_eos = toks == eos_id
    return (np.cumsum(is_eos) - is_eos).astype(np.int64)


def val_windows(data_dir: str | Path, seq_len: int, max_tokens: int, val_tokens: int | None = None) -> ValWindows:
    """The first ceil(max_tokens / seq_len) val windows (ids 0..n-1, in order)."""
    meta = read_meta(data_dir)
    vt = int(val_tokens or meta.get("val_tokens") or 1 << 62)
    val = ValWindows(data_dir, seq_len, vt)
    n = min(val.num_windows, math.ceil(int(max_tokens) / seq_len))
    val.ids = list(range(n))
    val.num_windows = n
    return val


class Capture:
    """Records codes, memory-output norms and Sink mass of every memory layer during ``model(x)``."""

    def __init__(self, model: Transformer, sink: bool = True):
        self.model = model
        self.codes: dict = {}
        self.memnorm: dict = {}
        self.sink: dict = {}
        self._handles = []
        self._patched = []
        self._modes = []
        for pos in memory_positions(model):
            mem = model.blocks[pos].memory
            mem.capture_route_logits = True
            self._handles.append(mem.register_forward_hook(self._out_hook(pos)))
            m = mem.module
            self._modes.append((m, m.execution_mode))
            m.execution_mode = "full"
            if sink:
                orig = m._real_route_probabilities

                def patched(scores, _orig=orig, _pos=pos):
                    p = _orig(scores)
                    self.sink[_pos] = (1.0 - p.sum(-1)).mean(dim=(2, 3))  # [B, T, Hkv, G] -> [B, T]
                    return p

                m._real_route_probabilities = patched
                self._patched.append(m)

    def _out_hook(self, pos: int):
        def hook(mem, args, out):
            self.memnorm[pos] = out.detach().float().norm(dim=-1)
            self.codes[pos] = mem.module.quantize_route_logits(mem.last_route_logits)

        return hook

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        for m in self._patched:
            del m._real_route_probabilities
        for m, mode in self._modes:
            m.execution_mode = mode
        for pos in memory_positions(self.model):
            self.model.blocks[pos].memory.capture_route_logits = False
            self.model.blocks[pos].memory.last_route_logits = None

    def __enter__(self) -> "Capture":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ------------------------------------------------------------------ dump I/O
def write_shard(out_dir: Path, shard: int, arrays: dict) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = None
    for k, v in arrays.items():
        np.save(out_dir / f"{k}.{shard:05d}.npy", v)
        n = len(v) if n is None else n
        if len(v) != n:
            raise ValueError(f"field {k} has {len(v)} rows, expected {n}")
    return int(n or 0)


def write_meta(out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_dump_meta(dump_dir: str | Path) -> dict:
    return json.loads((Path(dump_dir) / "meta.json").read_text(encoding="utf-8"))


def load_dump(dump_dir: str | Path, fields: list[str] | None = None, max_positions: int | None = None) -> tuple[dict, dict]:
    """(meta, {field: array concatenated over shards}); ``max_positions`` truncates."""
    dump_dir = Path(dump_dir)
    meta = read_dump_meta(dump_dir)
    fields = fields or meta["fields"]
    out = {}
    for f in fields:
        parts, have = [], 0
        for s in meta["shards"]:
            if max_positions is not None and have >= max_positions:
                break
            a = np.load(dump_dir / f"{f}.{s['index']:05d}.npy", mmap_mode="r")
            parts.append(a)
            have += len(a)
        arr = np.concatenate(parts) if len(parts) > 1 else np.asarray(parts[0])
        out[f] = arr[:max_positions] if max_positions is not None else arr
    return meta, out


# ------------------------------------------------------------------ main
@torch.no_grad()
def dump(model: Transformer, cfg: dict, data_dir: str | Path, out_dir: str | Path, *, run: str,
         max_tokens: int, batch: int = 8, topk: int = 32, shard_positions: int = 1 << 20,
         device: torch.device | None = None, precision: str | None = None, val_tokens: int | None = None) -> dict:
    device = device or default_device()
    precision = precision or default_precision(device)
    out_dir = Path(out_dir)
    seq_len = int(cfg["model"]["seq_len"])  # training length (the val windows of train.evaluate)
    val = val_windows(data_dir, seq_len, max_tokens, val_tokens)
    meta_data = read_meta(data_dir)
    eos = meta_data.get("eos_id")
    n_total = val.num_windows * seq_len
    docs_all = doc_ids(val.tokens, n_total, eos)
    vocab = int(model.cfg["vocab"])
    id_dtype = np.uint16 if vocab <= 65536 else np.int32
    mpos = memory_positions(model)
    model.to(device).eval()

    shards, buf, buf_n, shard, w0 = [], {}, 0, 0, 0

    def flush():
        nonlocal buf, buf_n, shard
        if not buf_n:
            return
        arrays = {k: np.concatenate(v) for k, v in buf.items()}
        n = write_shard(out_dir, shard, arrays)
        shards.append({"index": shard, "positions": n, "first_window": int(arrays["window"][0])})
        shard += 1
        buf, buf_n = {}, 0

    def add(k, v):
        buf.setdefault(k, []).append(v)

    t0 = time.time()
    with Capture(model) as cap:
        for x, y in val.batches(batch):
            b, t = x.shape
            x, y = x.to(device), y.to(device)
            with amp_context(device, precision):
                logits = model(x)
            win = np.repeat(np.arange(w0, w0 + b), t)
            pos = np.tile(np.arange(t), b)
            add("tokens", x.reshape(-1).cpu().numpy().astype(id_dtype))
            add("next", y.reshape(-1).cpu().numpy().astype(id_dtype))
            add("window", win.astype(np.int32))
            add("pos", pos.astype(np.int32))
            add("doc", docs_all[win * seq_len + pos].astype(np.int32))
            ids, lps, top1, lpn = [], [], [], []
            for i in range(b):
                lp = logits[i].float().log_softmax(-1)
                v, k = lp.topk(topk, dim=-1)
                ids.append(k)
                lps.append(v)
                top1.append(lp.argmax(-1))
                lpn.append(lp.gather(-1, y[i, :, None])[:, 0])
            add("topk_ids", torch.cat(ids).cpu().numpy().astype(id_dtype))
            add("topk_logp", torch.cat(lps).cpu().numpy().astype(np.float16))
            add("top1", torch.cat(top1).cpu().numpy().astype(np.int32))
            add("logp_next", torch.cat(lpn).cpu().numpy().astype(np.float32))
            for p in mpos:
                add(f"codes_L{p}", cap.codes[p].reshape(b * t, -1).to(torch.uint8).cpu().numpy())
                add(f"memnorm_L{p}", cap.memnorm[p].reshape(-1).cpu().numpy().astype(np.float32))
                add(f"sink_L{p}", cap.sink[p].reshape(-1).cpu().numpy().astype(np.float32))
            buf_n += b * t
            w0 += b
            if buf_n >= shard_positions:
                flush()
                print(f"[dump] {w0 * seq_len:,} positions ({time.time() - t0:.0f}s)", flush=True)
    flush()
    fields = list(PER_POSITION) + [f"{k}_L{p}" for p in mpos for k in ("codes", "memnorm", "sink")]
    mem0 = model.blocks[mpos[0]].memory if mpos else None
    meta = {
        "run": run,
        "checkpoint": cfg.get("_checkpoint"),
        "step": cfg.get("_step"),
        "data_dir": str(data_dir),
        "seq_len": seq_len,
        "vocab": vocab,
        "eos_id": eos,
        "memory_layers": mpos,
        "routes": {str(p): model.blocks[p].memory.routes for p in mpos},
        "bits": mem0.bits if mem0 else None,
        "ngrams": list(mem0.module.ngrams) if mem0 else [],
        "router_input": {str(p): model.blocks[p].memory.router_input for p in mpos},
        "topk": topk,
        "precision": precision,
        "n_positions": int(sum(s["positions"] for s in shards)),
        "n_windows": w0,
        "shards": shards,
        "fields": fields,
        "notes": "doc = #EOS strictly before the position in val.bin; pos = position in the window "
                 "(no model context crosses a window start); sink = 1 - sum of real-route probs, mean over heads.",
    }
    write_meta(out_dir, meta)
    return meta


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoint file or run dir (latest checkpoint)")
    ap.add_argument("--data", required=True, help="dataset dir with meta.json and val.bin")
    ap.add_argument("--run", required=True, help="name; output goes to <out_root>/<run>")
    ap.add_argument("--out_root", default="dumps")
    ap.add_argument("--max_tokens", type=int, default=5_000_000)
    ap.add_argument("--val_tokens", type=int, default=20_000_000, help="size of the val slice (first N of val.bin)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--shard_positions", type=int, default=1 << 21)
    ap.add_argument("--device", default=None)
    ap.add_argument("--precision", choices=["fp16", "fp32"], default=None, help="default fp16 on CUDA, fp32 on CPU")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    device = torch.device(args.device) if args.device else default_device()
    model, cfg = load_model(args.ckpt, device=device)
    meta = dump(model, cfg, args.data, Path(args.out_root) / args.run, run=args.run, max_tokens=args.max_tokens,
                batch=args.batch, topk=args.topk, shard_positions=args.shard_positions, device=device,
                precision=args.precision, val_tokens=args.val_tokens)
    print(f"[dump] wrote {meta['n_positions']:,} positions in {len(meta['shards'])} shards to "
          f"{Path(args.out_root) / args.run}", flush=True)
    return meta


if __name__ == "__main__":
    main()
