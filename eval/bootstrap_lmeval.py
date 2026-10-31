"""Bootstrap CIs for the lm-eval table from per-sample outputs (M6.T1).

    python -m eval.bootstrap_lmeval --root results/lmeval            # every subdir = one checkpoint
    python -m eval.bootstrap_lmeval --runs SB=results/lmeval/SB SD-B=results/lmeval/SD-B \\
        --pairs SD-B:SB D270:Run2 --n_boot 10000
    # -> results/lmeval_table.json + results/lmeval_table.md

Reads ``samples_<task>.jsonl`` (written by ``eval.lm_eval_adapter``) or the stock
``samples_<task>_<date>.jsonl`` from ``lm_eval --log_samples`` (latest file per
task).  MMLU subtasks (``mmlu_*_continuation``) are pooled into one
``mmlu_continuation`` row (micro average, as lm-eval's size-weighted group
score).  CIs: percentile bootstrap over documents; paired differences resample
the documents both checkpoints share.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK_ORDER = ["hellaswag", "mmlu_continuation", "sciq", "winogrande", "piqa", "arc_easy"]
PRIMARY = {"hellaswag": "acc_norm", "mmlu_continuation": "acc", "sciq": "acc_norm", "winogrande": "acc",
           "piqa": "acc_norm", "arc_easy": "acc_norm"}
METRICS = ("acc", "acc_norm")
DEFAULT_PAIRS = ["D270:Run2", "BD:Run1", "SD-B:SB", "SD-C:SB", "SD-A:SB", "SB:w2_mem_super1"]
_STOCK = re.compile(r"^samples_(.+)_(\d{4}-\d{2}-\d{2}T[\d\-.]+)\.jsonl$")
_OURS = re.compile(r"^samples_(.+)\.jsonl$")


def bootstrap_ci(values, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05,
                 chunk: int = 256) -> tuple[float, float, float]:
    """(mean, lo, hi): percentile bootstrap of the mean of ``values`` (1-D)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s)
        means[s:s + k] = v[rng.integers(0, v.size, size=(k, v.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def task_group(task: str) -> str:
    if task.startswith("mmlu_") and task.endswith("_continuation") and task != "mmlu_continuation":
        return "mmlu_continuation"
    return task


def load_samples(run_dir: str | Path) -> dict[str, dict[str, dict]]:
    """{group task: {doc key: {metric: value}}} for one checkpoint's output dir."""
    files: dict[str, tuple[str, Path]] = {}
    for p in sorted(Path(run_dir).rglob("samples_*.jsonl")):
        m = _STOCK.match(p.name)
        task, stamp = (m.group(1), m.group(2)) if m else (_OURS.match(p.name).group(1), "")
        if task not in files or stamp >= files[task][0]:
            files[task] = (stamp, p)
    out: dict[str, dict[str, dict]] = {}
    for task, (_, p) in sorted(files.items()):
        group = task_group(task)
        rows = out.setdefault(group, {})
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                s = json.loads(line)
                rows[f"{task}/{s['doc_id']}"] = {k: float(s[k]) for k in METRICS if k in s}
    return out


def summarise(runs: dict[str, dict], n_boot: int, seed: int) -> dict:
    table: dict[str, dict] = {}
    for name, tasks in runs.items():
        table[name] = {}
        for task, rows in tasks.items():
            entry = {"n": len(rows)}
            for metric in METRICS:
                vals = [r[metric] for r in rows.values() if metric in r]
                if len(vals) == len(rows) and vals:
                    mean, lo, hi = bootstrap_ci(vals, n_boot, seed)
                    entry[metric] = {"mean": mean, "lo": lo, "hi": hi}
            table[name][task] = entry
    return table


def paired(runs: dict[str, dict], a: str, b: str, n_boot: int, seed: int) -> dict:
    """a - b per task and metric over shared documents."""
    out = {}
    for task in sorted(set(runs[a]) & set(runs[b])):
        ra, rb = runs[a][task], runs[b][task]
        keys = sorted(set(ra) & set(rb))
        entry = {"n": len(keys)}
        for metric in METRICS:
            if keys and all(metric in ra[k] and metric in rb[k] for k in keys):
                d = [ra[k][metric] - rb[k][metric] for k in keys]
                mean, lo, hi = bootstrap_ci(d, n_boot, seed)
                entry[metric] = {"diff": mean, "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)}
        out[task] = entry
    return out


def _ordered(tasks) -> list[str]:
    return [t for t in TASK_ORDER if t in tasks] + sorted(t for t in tasks if t not in TASK_ORDER)


def to_markdown(table: dict, pairs: dict) -> str:
    tasks = _ordered({t for r in table.values() for t in r})
    lines = ["| checkpoint | " + " | ".join(f"{t} ({PRIMARY.get(t, 'acc')})" for t in tasks) + " |",
             "|---|" + "---|" * len(tasks)]
    for name, rows in table.items():
        cells = []
        for t in tasks:
            e = rows.get(t, {}).get(PRIMARY.get(t, "acc")) or rows.get(t, {}).get("acc")
            cells.append("—" if not e else f"{100 * e['mean']:.1f} [{100 * e['lo']:.1f}, {100 * e['hi']:.1f}]")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    if pairs:
        lines += ["", "Paired differences (a − b, percentage points, 95% bootstrap CI; * = excludes 0):", "",
                  "| a − b | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
        for key, rows in pairs.items():
            cells = []
            for t in tasks:
                e = rows.get(t, {}).get(PRIMARY.get(t, "acc"))
                cells.append("—" if not e else f"{100 * e['diff']:+.1f} [{100 * e['lo']:+.1f}, {100 * e['hi']:+.1f}]"
                             + ("*" if e["excludes_zero"] else ""))
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="dir whose subdirs are per-checkpoint outputs")
    ap.add_argument("--runs", nargs="*", default=[], help="NAME=DIR ...")
    ap.add_argument("--pairs", nargs="*", default=None, help=f"A:B ... (default: those of {DEFAULT_PAIRS} present)")
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "lmeval_table.json"))
    args = ap.parse_args(argv)
    dirs: dict[str, Path] = {}
    if args.root:
        dirs.update({p.name: p for p in sorted(Path(args.root).iterdir()) if p.is_dir()})
    for spec in args.runs:
        name, _, d = spec.partition("=")
        dirs[name] = Path(d)
    runs = {n: s for n, s in ((n, load_samples(d)) for n, d in dirs.items()) if s}
    if not runs:
        raise SystemExit("no samples_*.jsonl found")
    pair_specs = args.pairs if args.pairs is not None else DEFAULT_PAIRS
    pairs = {}
    for spec in pair_specs:
        a, _, b = spec.partition(":")
        if a in runs and b in runs:
            pairs[f"{a} - {b}"] = paired(runs, a, b, args.n_boot, args.seed)
        elif args.pairs is not None:
            print(f"skip pair {spec}: missing run")
    table = summarise(runs, args.n_boot, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_boot": args.n_boot, "seed": args.seed, "primary_metric": PRIMARY,
                               "runs": {n: str(d) for n, d in dirs.items() if n in runs},
                               "table": table, "paired": pairs}, indent=2), encoding="utf-8")
    md = to_markdown(table, pairs)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
