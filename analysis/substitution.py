"""M2.T3 substitution test: val loss with the router codes replaced at inference.

    python -m analysis.substitution --ckpt runs/w3_sb_1b --data /data/climbmix --run sb \
        --mlp results/rq2_sb_mlp_L1.pt --max_tokens 5000000

Modes (all on the same val windows):

- ``baseline``: the unmodified model;
- ``own``: the patched forward with the model's own codes (must equal ``baseline`` exactly);
- ``mlp``: codes = argmax of the M2.T2 MLP (``CodeMLP`` over embeddings of x[t-2..t]);
- ``random``: iid uniform codes (seeded);
- ``zero``: the memory layer's output replaced by zeros.

The memory wrapper's ``forward`` is replaced on the instance only (restored
afterwards); nothing in ``ffnmem/`` changes.  Every memory layer listed in
``--layers`` (default: all) is substituted at once.  Output
``results/substitution_<run>.json``: mean loss per mode, deltas vs ``baseline``
with paired bootstrap CIs over windows, and (``mlp``) the fraction of route codes
the MLP gets right.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from analysis.code_predictability import CodeMLP, bootstrap_weights, summarize
from analysis.dump_codes import default_device, default_precision, memory_positions, val_windows
from ffnmem.infer import load_model
from ffnmem.model import Transformer
from ffnmem.train import amp_context

MODES = ("baseline", "own", "mlp", "random", "zero")


class Substitution:
    """Context manager that patches the memory wrappers of ``model`` for one mode."""

    def __init__(self, model: Transformer, mode: str, layers: list[int] | None = None,
                 mlps: dict | None = None, seed: int = 0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "mlp" and not mlps:
            raise ValueError("mode mlp needs the T2 MLPs ({layer: CodeMLP})")
        self.model, self.mode, self.seed = model, mode, seed
        self.layers = layers if layers is not None else memory_positions(model)
        self.mlps = mlps or {}
        self.emb_pad = None
        self.idx: Tensor | None = None
        self.match = {p: [0, 0] for p in self.layers}  # mlp: [codes equal, codes total]
        self._gen = None

    def set_input(self, idx: Tensor) -> None:
        self.idx = idx

    def _forward(self, mem, pos: int, h: Tensor, h_norm: Tensor, router_in: Tensor | None = None) -> Tensor:
        if self.mode == "zero":
            return torch.zeros_like(h)
        src = h_norm if mem.router_input == "hidden" else router_in
        codes = mem.module.quantize_route_logits(mem.route_logits(src))
        if self.mode == "mlp":
            with torch.autocast(device_type=h.device.type, enabled=False):
                pred = self.mlps[pos].window_logits(self.emb_pad, self.idx).argmax(-1).to(codes.dtype)
            self.match[pos][0] += int((pred == codes).sum())
            self.match[pos][1] += codes.numel()
            codes = pred
        elif self.mode == "random":
            if self._gen is None:
                self._gen = torch.Generator(device=h.device).manual_seed(self.seed)
            codes = torch.randint(0, 1 << mem.bits, codes.shape, generator=self._gen, device=h.device,
                                  dtype=codes.dtype)
        out = mem.module(h, codes, q_logits=None)
        if mem.rank1_dim:
            out = out + mem._rank1(h_norm, codes)
        return out

    def __enter__(self) -> "Substitution":
        if self.mode == "baseline":
            return self
        if self.mode == "mlp":
            dev = self.model.tok_emb.weight.device
            self.emb_pad = CodeMLP.padded_embeddings(self.model.tok_emb.weight).to(dev)
            for m in self.mlps.values():
                m.to(dev).eval()
        for p in self.layers:
            mem = self.model.blocks[p].memory
            mem.forward = lambda h, h_norm, router_in=None, _m=mem, _p=p: self._forward(_m, _p, h, h_norm, router_in)
        return self

    def __exit__(self, *exc) -> None:
        if self.mode == "baseline":
            return
        for p in self.layers:
            del self.model.blocks[p].memory.forward


@torch.no_grad()
def window_losses(model: Transformer, val, batch: int, sub: Substitution, device: torch.device,
                  precision: str) -> np.ndarray:
    """Mean next-token CE (nats) of every val window under ``sub``."""
    out = []
    with sub:
        for x, y in val.batches(batch):
            x, y = x.to(device), y.to(device)
            sub.set_input(x)
            with amp_context(device, precision):
                logits = model(x)
            ce = F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none")  # [B, T]
            out.append(ce.mean(1).double().cpu().numpy())
    return np.concatenate(out)


def run_substitution(model: Transformer, data_dir: str | Path, *, modes=MODES, layers: list[int] | None = None,
                     mlps: dict | None = None, max_tokens: int = 5_000_000, batch: int = 8, seed: int = 0,
                     n_boot: int = 1000, device: torch.device | None = None, precision: str | None = None,
                     val_tokens: int | None = None) -> dict:
    device = device or default_device()
    precision = precision or default_precision(device)
    model.to(device).eval()
    seq_len = int(model.cfg["seq_len"])
    val = val_windows(data_dir, seq_len, max_tokens, val_tokens)
    losses, extra = {}, {}
    for mode in modes:
        t0 = time.time()
        sub = Substitution(model, mode, layers, mlps, seed)
        losses[mode] = window_losses(model, val, batch, sub, device, precision)
        if mode == "mlp":
            extra["mlp_code_accuracy"] = {str(p): m[0] / max(m[1], 1) for p, m in sub.match.items()}
        print(f"[subst] {mode}: loss {losses[mode].mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
    W = bootstrap_weights(val.num_windows, n_boot, seed)
    res = {"modes": {}, "n_windows": val.num_windows, "n_tokens": val.num_windows * seq_len,
           "layers": layers if layers is not None else memory_positions(model), "precision": precision,
           "ci": "95% paired percentile bootstrap over val windows", **extra}
    for mode, l in losses.items():
        r = {"loss": summarize(W @ l / val.num_windows)}
        if "baseline" in losses and mode != "baseline":
            r["delta_vs_baseline"] = summarize(W @ (l - losses["baseline"]) / val.num_windows)
            r["max_abs_window_diff"] = float(np.abs(l - losses["baseline"]).max())
        res["modes"][mode] = r
    return res


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--mlp", nargs="*", default=[], help="CodeMLP files from code_predictability (one per layer)")
    ap.add_argument("--modes", nargs="*", default=None, help=f"subset of {MODES} (default: all available)")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--max_tokens", type=int, default=5_000_000)
    ap.add_argument("--val_tokens", type=int, default=20_000_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--precision", choices=["fp16", "fp32"], default=None)
    ap.add_argument("--out_dir", default="results")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    device = torch.device(args.device) if args.device else default_device()
    model, cfg = load_model(args.ckpt, device=device)
    mlps = {}
    for path in args.mlp:
        m = CodeMLP.load(path)
        mlps[int(m.config["layer"])] = m
    modes = args.modes or [m for m in MODES if m != "mlp" or mlps]
    layers = args.layers
    if "mlp" in modes and layers is None:
        layers = sorted(mlps)  # substitute exactly the layers that have an MLP
    res = run_substitution(model, args.data, modes=modes, layers=layers, mlps=mlps, max_tokens=args.max_tokens,
                           batch=args.batch, seed=args.seed, n_boot=args.n_boot, device=device,
                           precision=args.precision, val_tokens=args.val_tokens)
    res.update(run=args.run, checkpoint=cfg.get("_checkpoint"), step=cfg.get("_step"), mlp_files=args.mlp)
    out = Path(args.out_dir) / f"substitution_{args.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[subst] wrote {out}", flush=True)
    return res


if __name__ == "__main__":
    main()
