"""GPU passes for the interpretability studies (results feed analysis/interp_studies.py).

    python -m analysis.interp_gpu rows --target CKPT --name sdc --cache_dir ACC_CACHE --drafters pld sdhead sdhead_online=sdhead+online
    python -m analysis.interp_gpu pertoken --ckpt CKPT --name sb --data HELDOUT --zero_memory
    python -m analysis.interp_gpu draft --ckpt CKPT --name sdc --data HELDOUT

rows     : per-row accepted lengths at gamma=4 on cached acceptance records (open + long), plus whether
           prompt lookup found a match; written to <out>/rows_<name>.npz.
pertoken : per-position cross-entropy of the true next token on the held-out windows (same order as
           analysis/dump_codes.py), optionally also with the memory output zeroed; <out>/pertoken_<name>.npz.
draft    : per-position draft-head argmax and draft CE of the true next token (SD models);
           <out>/draft_<name>.npz.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from drafting.common import accepted_lengths, full_rows
from drafting.offline_acceptance import load_records, make_drafter, parse_spec
from ffnmem.data.loader import ValWindows
from ffnmem.infer import load_model

GAMMA = 4


def rows(a) -> None:
    dev = torch.device("cuda")
    model, cfg = load_model(a.target, device=dev, seq_len=2048 + 256)
    amp = torch.float16
    out = {}
    for t in ("open", "long"):
        recs, _ = load_records(Path(a.cache_dir) / f"records_{t}.npz")
        for s in a.drafters:
            spec = parse_spec(s)
            d = make_drafter(spec, model, amp)
            t0 = time.time()
            accs, rowpos, prompt, matched = [], [], [], []
            for i, rec in enumerate(recs):
                dr = d.drafts(rec, GAMMA)
                keep = full_rows(rec, GAMMA)
                accs.append(accepted_lengths(rec, dr)[keep])
                rowpos.append(np.flatnonzero(keep))
                prompt.append(np.full(int(keep.sum()), i))
                if spec["kind"] == "pld":
                    matched.append(dr[keep, 0] >= 0)
            out[f"{t}/{spec['label']}"] = np.concatenate(accs).astype(np.int8)
            out[f"{t}/row"] = np.concatenate(rowpos).astype(np.int16)
            out[f"{t}/prompt"] = np.concatenate(prompt).astype(np.int16)
            if matched:
                out[f"{t}/pld_matched"] = np.concatenate(matched)
            print(f"[rows] {a.name} {t} {spec['label']}: mean {out[f'{t}/{spec['label']}'].mean():.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(a.out) / f"rows_{a.name}.npz", **out)


def windows(data: str, seq: int = 2048):
    v = ValWindows(data, seq, 10**9)
    return v.batches(4)


@torch.no_grad()
def pertoken(a) -> None:
    dev = torch.device("cuda")
    model, cfg = load_model(a.ckpt, device=dev)
    model.eval()

    def run(zero: bool) -> np.ndarray:
        saved = {}
        if zero:
            for b in model.blocks:
                if b.memory is not None:
                    saved[b] = b.memory.forward
                    b.memory.forward = lambda h, h_norm, router_in=None: torch.zeros_like(h)
        ces = []
        try:
            for x, y in windows(a.data):
                x, y = x.to(dev), y.to(dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(x)
                ces.append(F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="none").half().cpu())
        finally:
            for b, f in saved.items():
                b.memory.forward = f
        return torch.cat(ces).numpy()

    out = {"ce": run(False)}
    print(f"[pertoken] {a.name} mean CE {out['ce'].astype(np.float64).mean():.4f}", flush=True)
    if a.zero_memory:
        out["ce_zero"] = run(True)
        print(f"[pertoken] {a.name} zero-memory mean CE {out['ce_zero'].astype(np.float64).mean():.4f}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(a.out) / f"pertoken_{a.name}.npz", **out)


@torch.no_grad()
def draft(a) -> None:
    dev = torch.device("cuda")
    model, cfg = load_model(a.ckpt, device=dev)
    top1, ce, pmax = [], [], []
    for x, y in windows(a.data):
        x, y = x.to(dev), y.to(dev)
        b, t = x.shape
        bi = torch.arange(b, device=dev).repeat_interleave(t)
        ti = torch.arange(t, device=dev).repeat(b)
        for s in range(0, b * t, 2048):
            with torch.autocast("cuda", dtype=torch.float16):
                q = model.draft_logits(x, bi[s:s + 2048], ti[s:s + 2048]).float()
            top1.append(q.argmax(-1).to(torch.int32).cpu())
            pmax.append(q.softmax(-1).amax(-1).half().cpu())
            ce.append(F.cross_entropy(q, y.flatten()[s:s + 2048], reduction="none").half().cpu())
    out = {"draft_top1": torch.cat(top1).numpy(), "draft_ce": torch.cat(ce).numpy(), "draft_pmax": torch.cat(pmax).numpy()}
    print(f"[draft] {a.name} mean draft CE {out['draft_ce'].astype(np.float64).mean():.4f}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(a.out) / f"draft_{a.name}.npz", **out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rows")
    r.add_argument("--target", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--cache_dir", required=True)
    r.add_argument("--drafters", nargs="+", required=True)
    p = sub.add_parser("pertoken")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--zero_memory", action="store_true")
    d = sub.add_parser("draft")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--name", required=True)
    d.add_argument("--data", required=True)
    for s in (r, p, d):
        s.add_argument("--out", default="results/interp")
    a = ap.parse_args(argv)
    {"rows": rows, "pertoken": pertoken, "draft": draft}[a.cmd](a)


if __name__ == "__main__":
    main()
