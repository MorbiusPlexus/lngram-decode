"""Paper figures from the results JSONs (M6.T4).

    python -m analysis.plot_figures                                  # everything found under results/
    python -m analysis.plot_figures --results results --out results/figures \
        --sd_runs SB=runs/w3_sb_1b SD-B=runs/w3_sdb_1b SD-C=runs/w3_sdc_1b

Each figure is drawn from whatever matching files exist and skipped (with a note)
otherwise; ``figures.json`` lists every figure and the input files it used, so
every plotted number traces back to a results file.

| figure                     | inputs                                   |
|----------------------------|------------------------------------------|
| ladder.png                 | ladder_<run>.json                         |
| code_predictability.png    | rq2_<run>.json                            |
| prefetch_batch.png         | prefetch_<server>_<run>.json              |
| prefetch_table_size.png    | scaled_<server>.json                      |
| acceptance.png             | acceptance_<target>.json (gamma = 4)      |
| live_speedup.png           | live_<run>.json                           |
| sd_tradeoff.png            | --sd_runs train logs + acceptance JSONs   |
| ffn_importance.png         | ffn_importance_<run>.json                 |
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LADDER_ORDER = ["L0_tok_ngram", "L0_random_router", "L0_sketch", "L1_zero_query", "L1_zero_query_conv",
                "L2_draft", "L3_tuned_lens", "L4_model"]
CONTROLS = {"L0_tok_ngram", "L0_random_router", "L1_zero_query_conv"}


def _load(paths) -> list[tuple[Path, dict]]:
    out = []
    for p in sorted(paths):
        try:
            out.append((p, json.loads(p.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[figures] skip {p}: {e}")
    return out


def _err(v: dict) -> tuple[float, float]:
    return max(v["value"] - v["lo"], 0.0), max(v["hi"] - v["value"], 0.0)


def _stem_run(p: Path, prefix: str) -> str:
    return p.stem[len(prefix):]


# ------------------------------------------------------------------ figures
def fig_ladder(files, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for p, d in files:
        run = d.get("run") or _stem_run(p, "ladder_")
        levels = [lv for lv in LADDER_ORDER if lv in d["levels"]]
        main = [lv for lv in levels if lv not in CONTROLS]
        for ax, metric in zip(axes, ("top1_agree", "sum_min", "kl")):
            xs = [LADDER_ORDER.index(lv) for lv in main]
            ys = [d["levels"][lv][metric]["value"] for lv in main]
            errs = list(zip(*[_err(d["levels"][lv][metric]) for lv in main])) if main else [[], []]
            line = ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=run)
            ctrl = [lv for lv in levels if lv in CONTROLS]
            ax.scatter([LADDER_ORDER.index(lv) for lv in ctrl], [d["levels"][lv][metric]["value"] for lv in ctrl],
                       marker="x", color=line[0].get_color(), s=50)
    for ax, title in zip(axes, ("top-1 agreement with the model", "sum min(p, q)", "KL(p || q) [nats]")):
        ax.set_xticks(range(len(LADDER_ORDER)), [s.replace("_", " ", 1) for s in LADDER_ORDER], fontsize=7,
                      rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[2].set_yscale("symlog", linthresh=1e-3)
    axes[2].set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    fig.suptitle("Decodability ladder (x = controls)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_predictability(files, out: Path):
    rows = []
    for p, d in files:
        run = d.get("run") or _stem_run(p, "rq2_")
        for layer, L in d["layers"].items():
            rows.append((f"{run} L{layer}", L["summary"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    width = 0.8 / 3
    for i, pred in enumerate(("uni", "bi", "mlp")):
        vals = [s[f"mean_frac_{pred}"] for _, s in rows]
        axes[0].bar([j + (i - 1) * width for j in range(len(rows))], [v["value"] for v in vals], width,
                    yerr=list(zip(*[_err(v) for v in vals])) if vals else None, capsize=2,
                    label={"uni": "x[t]", "bi": "x[t-1], x[t]", "mlp": "MLP x[t-2..t]"}[pred])
    axes[0].set_xticks(range(len(rows)), [r for r, _ in rows], rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("fraction of code entropy explained")
    axes[0].set_title("Current code from tokens (mean over routes)")
    axes[0].legend(fontsize=8)
    ks = [1, 2, 4, 8]
    for name, s in rows:
        if f"mean_next_recall@1" in s:
            axes[1].plot(ks, [s[f"mean_next_recall@{k}"]["value"] for k in ks], marker="o", label=f"{name} recall")
        if "p_any_miss@1" in s:
            axes[1].plot(ks, [s[f"p_any_miss@{k}"]["value"] for k in ks], marker="x", ls="--",
                         label=f"{name} P(any route misses)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("k (top-k next codes from (c[t-1], c[t]))")
    axes[1].set_title("Next-step code prediction")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_prefetch_batch(files, out: Path):
    fig, axes = plt.subplots(1, len(files), figsize=(5.5 * len(files), 4), squeeze=False)
    for ax, (p, d) in zip(axes[0], files):
        modes = sorted({r["mode"] for r in d["rows"]})
        for m in modes:
            rs = sorted((r for r in d["rows"] if r["mode"] == m and "latency_ms_median" in r), key=lambda r: r["batch"])
            if not rs:
                continue
            line = ax.plot([r["batch"] for r in rs], [r["latency_ms_median"] for r in rs], marker="o", label=m)
            ax.plot([r["batch"] for r in rs], [r["latency_ms_p99"] for r in rs], ls=":",
                    color=line[0].get_color())
        ax.set_xscale("log", base=2)
        ax.set_xlabel("batch")
        ax.set_ylabel("ms / decode step (median; dotted p99)")
        ax.set_title(p.stem.removeprefix("prefetch_"))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_table_size(files, out: Path):
    fig, axes = plt.subplots(1, len(files), figsize=(5.5 * len(files), 4), squeeze=False)
    for ax, (p, d) in zip(axes[0], files):
        keys = sorted({(r["growth"], r["batch"]) for r in d["rows"]})
        for growth, batch in keys:
            rs = sorted((r for r in d["rows"] if r["growth"] == growth and r["batch"] == batch),
                        key=lambda r: r["table_bytes"])
            gb = [r["table_bytes"] / 2**30 for r in rs]
            line = ax.plot(gb, [r["b1"]["median_ms"] for r in rs], marker="o", label=f"B1 {growth} b{batch}")
            if rs and rs[0]["b4"]:
                ax.plot(gb, [r["b4"][0]["exposed"]["median_ms"] for r in rs], marker="s", ls="--",
                        color=line[0].get_color(), label=f"B4 exposed {growth} b{batch}")
        ax.set_xscale("log")
        ax.set_xlabel("table size [GB]")
        ax.set_ylabel("ms / step (median)")
        ax.set_title(p.stem.removeprefix("scaled_"))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_acceptance(files, out: Path, gamma: str = "4"):
    fig, axes = plt.subplots(1, len(files), figsize=(6.5 * len(files), 4.2), squeeze=False)
    for ax, (p, d) in zip(axes[0], files):
        types = [t for t, v in d["text_types"].items() if "skipped" not in v]
        drafters = [k for k, v in d["drafters"].items() if v.get("kind") != "oracle"]
        width = 0.8 / max(len(types), 1)
        for i, t in enumerate(types):
            ys, errs = [], []
            for name in drafters:
                r = d["drafters"][name]["results"].get(t, {}).get(gamma)
                ys.append(r["mean_accepted"] if r else 0.0)
                errs.append((max(r["mean_accepted"] - r["ci95"][0], 0.0), max(r["ci95"][1] - r["mean_accepted"], 0.0))
                            if r else (0, 0))
            ax.bar([j + (i - (len(types) - 1) / 2) * width for j in range(len(drafters))], ys, width,
                   yerr=list(zip(*errs)) if errs else None, capsize=2, label=t)
        ax.set_xticks(range(len(drafters)), drafters, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(f"mean accepted length (gamma = {gamma})")
        ax.set_title(f"target {d.get('target', p.stem)}")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_live(files, out: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for p, d in files:
        gs = sorted(d["by_gamma"], key=int)
        ax.plot([int(g) for g in gs], [d["by_gamma"][g]["speedup"] for g in gs], marker="o",
                label=f"{d.get('run', p.stem)} ({d.get('drafter', '')}, {d.get('dtype', '')})")
    ax.axhline(1.0, color="grey", lw=0.8)
    ax.set_xlabel("gamma")
    ax.set_ylabel("speedup vs plain greedy decoding")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def final_val_loss(run_dir: Path) -> float | None:
    log = run_dir / "logs" / "train.jsonl"
    if not log.exists():
        return None
    last = None
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "eval" and "val_loss" in r:
            last = r["val_loss"]
    return last


def fig_sd_tradeoff(sd_runs: dict, results: Path, out: Path, gamma: str = "4") -> list[str]:
    """x: final val loss (train logs); y: best SD-head / post-hoc accepted length on that target."""
    used, pts = [], []
    for label, run_dir in sd_runs.items():
        run_dir = Path(run_dir)
        loss = final_val_loss(run_dir)
        acc_file = results / f"acceptance_{run_dir.name}.json"
        best = None
        if acc_file.exists():
            d = json.loads(acc_file.read_text(encoding="utf-8"))
            used.append(str(acc_file))
            for name, dr in d["drafters"].items():
                if dr.get("kind") not in ("sdhead", "posthoc"):
                    continue
                for t, res in dr["results"].items():
                    r = res.get(gamma)
                    if r and (best is None or r["mean_accepted"] > best[0]):
                        best = (r["mean_accepted"], f"{name}/{t}")
        if loss is not None:
            used.append(str(run_dir / "logs" / "train.jsonl"))
        if loss is not None and best is not None:
            pts.append((label, loss, best))
    if not pts:
        return []
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, loss, (acc, which) in pts:
        ax.scatter(loss, acc)
        ax.annotate(f"{label}\n{which}", (loss, acc), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("final val loss [nats]")
    ax.set_ylabel(f"best drafter mean accepted length (gamma = {gamma})")
    ax.set_title("Self-drafting: LM loss vs acceptance")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return used


def fig_ffn(files, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    n = len(files)
    for i, (p, d) in enumerate(files):
        L = d["layers"]
        xs = [r["pos"] + (i - (n - 1) / 2) * 0.8 / n for r in L]
        line = ax.errorbar(xs, [r["delta"] for r in L],
                           yerr=[[max(r["delta"] - r["ci_lo"], 0.0) for r in L],
                                 [max(r["ci_hi"] - r["delta"], 0.0) for r in L]],
                           fmt="o", capsize=2, label=d.get("run", p.stem))
        mem = [r for r, x in zip(L, xs) if r.get("has_memory")]
        ax.scatter([x for r, x in zip(L, xs) if r.get("has_memory")], [r["delta"] for r in mem], s=90,
                   facecolors="none", edgecolors=line[0].get_color())
    ax.set_xlabel("layer (circled: memory layer)")
    ax.set_ylabel("val loss increase when the FFN is zeroed [nats]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--gamma", default="4")
    ap.add_argument("--sd_runs", nargs="*", default=[], metavar="LABEL=RUN_DIR")
    a = ap.parse_args(argv)
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    made: dict = {}

    def draw(name: str, pattern: str, fn, *extra):
        files = _load(res.glob(pattern))
        if not files:
            print(f"[figures] {name}: no {pattern}; skipped")
            return
        fn(files, out / name, *extra)
        made[name] = [str(p) for p, _ in files]
        print(f"[figures] {name} <- {len(files)} file(s)")

    draw("ladder.png", "ladder_*.json", fig_ladder)
    draw("code_predictability.png", "rq2_*.json", fig_predictability)
    draw("prefetch_batch.png", "prefetch_*.json", fig_prefetch_batch)
    draw("prefetch_table_size.png", "scaled_*.json", fig_table_size)
    draw("acceptance.png", "acceptance_*.json", fig_acceptance, a.gamma)
    draw("live_speedup.png", "live_*.json", fig_live)
    draw("ffn_importance.png", "ffn_importance_*.json", fig_ffn)
    if a.sd_runs:
        runs = dict(s.split("=", 1) for s in a.sd_runs)
        used = fig_sd_tradeoff(runs, res, out / "sd_tradeoff.png", a.gamma)
        if used:
            made["sd_tradeoff.png"] = used
            print("[figures] sd_tradeoff.png")
        else:
            print("[figures] sd_tradeoff.png: no run with both a val loss and an acceptance file; skipped")
    (out / "figures.json").write_text(json.dumps(made, indent=2), encoding="utf-8")
    return made


if __name__ == "__main__":
    main()
