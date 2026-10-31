"""Is the table worth having? D270 vs Run 2 and vs SB / w2_mem_super1 val loss (M6.T3).

    # final eval records from each run's logs/train.jsonl
    python -m eval.table_worth --runs D270=runs/w3_dense270 Run2=runs/w1_dense_cut_near \\
        SB=runs/w3_sb_1b w2_mem_super1=runs/w2_mem_super1
    # or recompute on one shared val slice, with a paired bootstrap CI over windows
    python -m eval.table_worth --recompute --data_dir /data/climbmix --val_tokens 5000000 --device cuda \\
        --runs D270=runs/w3_dense270 Run2=runs/w1_dense_cut_near/checkpoints/step_0003815.pt
    # -> results/table_worth.json

A comparison ``A:B`` reports loss(A) - loss(B) (negative = A better).  Token
budgets are compared from the eval records' ``tokens`` (or step × tokens/step
for checkpoints); pairs differing by more than ``--budget_tol`` (relative) are
flagged ``comparable: false`` (e.g. SB at 1B vs D270 at 2B), as are pairs whose
logged losses come from different val slices (``val_tokens``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.bootstrap_lmeval import bootstrap_ci  # noqa: E402

DEFAULT_PAIRS = ["D270:Run2", "D270:SB", "D270:w2_mem_super1", "SB:w2_mem_super1"]


def run_dir_of(path: Path) -> Path:
    return path.parent.parent if path.is_file() and path.parent.name == "checkpoints" else path


def last_eval(run_dir: str | Path) -> dict | None:
    """Final ``event: eval`` record of ``<run_dir>/logs/train.jsonl`` (largest step; last one wins on ties)."""
    p = Path(run_dir) / "logs" / "train.jsonl"
    if not p.is_file():
        return None
    best = None
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line from a killed run
            if rec.get("event") == "eval" and "val_loss" in rec:
                if best is None or int(rec["step"]) >= int(best["step"]):
                    best = rec
    return best


def from_logs(name: str, path: Path) -> dict:
    ev = last_eval(run_dir_of(path))
    if ev is None:
        raise FileNotFoundError(f"{name}: no eval record in {run_dir_of(path) / 'logs' / 'train.jsonl'}")
    return {"source": "logs", "val_loss": float(ev["val_loss"]), "val_tokens": ev.get("val_tokens"),
            "tokens_trained": ev.get("tokens"), "step": int(ev["step"])}


def from_checkpoint(name: str, path: Path, data_dir: str | None, val_tokens: int, batch_size: int,
                    device: str, precision: str):
    from ffnmem.data.loader import ValWindows
    from ffnmem.infer import load_model

    from eval.ffn_importance import window_losses

    model, cfg = load_model(path, device=device)
    tcfg = cfg.get("train", {})
    seq_len = int(model.cfg["seq_len"])
    ddir = data_dir or tcfg["data_dir"]
    losses = window_losses(model, ValWindows(ddir, seq_len, val_tokens), batch_size, precision)
    tps = int(tcfg.get("global_batch", 0)) * int(tcfg.get("seq_len", seq_len))
    ev = last_eval(run_dir_of(Path(path)))
    trained = ev.get("tokens") if ev and int(ev["step"]) == cfg["_step"] else cfg["_step"] * tps
    info = {"source": "recomputed", "checkpoint": cfg["_checkpoint"], "step": cfg["_step"],
            "val_loss": float(losses.mean()), "val_tokens": int(len(losses) * seq_len), "seq_len": seq_len,
            "data_dir": str(ddir), "tokens_trained": trained, "tokens_total": tcfg.get("tokens_total")}
    del model
    return info, losses


def compare(a: str, b: str, runs: dict, budget_tol: float, losses: dict | None, n_boot: int, seed: int) -> dict:
    ra, rb = runs[a], runs[b]
    out = {"a": a, "b": b, "loss_a": ra["val_loss"], "loss_b": rb["val_loss"],
           "diff": ra["val_loss"] - rb["val_loss"], "tokens_a": ra.get("tokens_trained"),
           "tokens_b": rb.get("tokens_trained")}
    notes = []
    ta, tb = ra.get("tokens_trained"), rb.get("tokens_trained")
    if ta and tb:
        rel = abs(float(ta) - float(tb)) / max(float(ta), float(tb))
        out["budget_rel_diff"] = rel
        if rel > budget_tol:
            notes.append(f"token budgets differ ({float(ta) / 1e9:.2f}B vs {float(tb) / 1e9:.2f}B): not comparable")
    else:
        notes.append("token budget unknown")
    if ra.get("val_tokens") != rb.get("val_tokens") or ra.get("seq_len", 0) != rb.get("seq_len", 0):
        notes.append(f"different val slices ({ra.get('val_tokens')} vs {rb.get('val_tokens')} tokens)")
    if losses is not None and a in losses and b in losses and len(losses[a]) == len(losses[b]):
        d, lo, hi = bootstrap_ci(losses[a] - losses[b], n_boot, seed)
        out.update({"paired_diff": d, "ci_lo": lo, "ci_hi": hi, "n_windows": int(len(losses[a])),
                    "excludes_zero": bool(lo > 0 or hi < 0)})
    out["comparable"] = not any("not comparable" in n or "different val" in n for n in notes)
    out["notes"] = notes
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=RUN_DIR_OR_CHECKPOINT ...")
    ap.add_argument("--pairs", nargs="*", default=None, help=f"A:B ... (default: those of {DEFAULT_PAIRS} present)")
    ap.add_argument("--recompute", action="store_true", help="evaluate checkpoints on one shared val slice")
    ap.add_argument("--data_dir", default=None, help="recompute: val data (default: first run's train.data_dir)")
    ap.add_argument("--val_tokens", type=int, default=5_000_000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--budget_tol", type=float, default=0.01)
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "table_worth.json"))
    args = ap.parse_args(argv)

    specs = {}
    for s in args.runs:
        name, _, p = s.partition("=")
        specs[name] = Path(p)
    runs, losses = {}, None
    if args.recompute:
        losses = {}
        data_dir = args.data_dir
        for name, p in specs.items():
            runs[name], losses[name] = from_checkpoint(name, p, data_dir, args.val_tokens, args.batch_size,
                                                       args.device, args.precision)
            data_dir = data_dir or runs[name]["data_dir"]  # every run on the first run's slice
            print(f"[{name}] val_loss {runs[name]['val_loss']:.4f} ({runs[name]['val_tokens']} tokens)")
    else:
        for name, p in specs.items():
            runs[name] = from_logs(name, p)
            runs[name]["path"] = str(p)
    pair_specs = args.pairs if args.pairs is not None else DEFAULT_PAIRS
    comps = []
    for spec in pair_specs:
        a, _, b = spec.partition(":")
        if a in runs and b in runs:
            comps.append(compare(a, b, runs, args.budget_tol, losses, args.n_boot, args.seed))
    for c in comps:
        ci = f" [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]" if "ci_lo" in c else ""
        flag = "" if c["comparable"] else "  (NOT COMPARABLE: " + "; ".join(c["notes"]) + ")"
        print(f"{c['a']} - {c['b']}: {c['diff']:+.4f}{ci}{flag}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mode": "recompute" if args.recompute else "logs", "runs": runs,
                               "comparisons": comps, "budget_tol": args.budget_tol,
                               "n_boot": args.n_boot if args.recompute else None}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
