"""Paper figures from the released results (no checkpoints needed).

    python paper/make_figures.py            # writes paper/figures/*.pdf

Inputs: results/training/*/train.jsonl, results/evaluation/**.json, results/interp/*.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"
TOK_PER_STEP = 256 * 2048

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 200, "pdf.fonttype": 42})
COL = {"SB": "#3b6fb6", "SD-B": "#d9822b", "SD-C": "#2f9e5b", "D270": "#7d7d7d", "PLD": "#9b59b6",
       "n-gram": "#8c6d31", "post-hoc": "#c0392b"}


def evals(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "eval":
            out[r["step"]] = r
    return out


def load(p: str) -> dict:
    return json.loads((RES / p).read_text(encoding="utf-8"))


def fig_loss() -> None:
    runs = [("SB run 1", "training/sb_run1", COL["SB"], "-"), ("SB run 2", "training/sb_run2", COL["SB"], ":"),
            ("SD-B", "training/sdb", COL["SD-B"], "-"), ("SD-C", "training/sdc", COL["SD-C"], "-"),
            ("D270 (2B schedule)", "training/dense270", COL["D270"], "--")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.2))
    curves = {}
    for name, d, c, ls in runs:
        e = evals(RES / d / "train.jsonl")
        s = sorted(e)
        curves[name] = {k: e[k]["val_loss"] for k in s}
        a.plot([k * TOK_PER_STEP / 1e9 for k in s], [e[k]["val_loss"] for k in s], color=c, ls=ls, lw=1.1, label=name)
    a.set_xlabel("training tokens (B)")
    a.set_ylabel("validation loss (nats)")
    a.set_ylim(2.45, 3.4)
    a.grid(alpha=0.3)
    a.legend(frameon=False, ncol=1)
    for name, ref, c, ls in [("SD-B", "SB run 1", COL["SD-B"], "-"), ("SD-C", "SB run 2", COL["SD-C"], "-"),
                             ("SB run 2", "SB run 1", COL["SB"], ":"), ("D270 (2B schedule)", "SB run 1", COL["D270"], "--")]:
        ks = [k for k in sorted(curves[name]) if k in curves[ref] and k >= 600]
        b.plot([k * TOK_PER_STEP / 1e9 for k in ks], [curves[name][k] - curves[ref][k] for k in ks], color=c, ls=ls,
               marker="o", ms=2, lw=1.1, label=f"{name} − {ref}")
    b.axhline(0, color="k", lw=0.6)
    b.set_xlabel("training tokens (B)")
    b.set_ylabel("Δ validation loss (nats)")
    b.grid(alpha=0.3)
    b.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "loss_curves.pdf")
    plt.close(fig)


def fig_acceptance() -> None:
    specs = [("SB", "w3_sb_1b", [("pld", "PLD"), ("ngram", "n-gram"), ("posthoc", "post-hoc"),
                                 ("posthoc_ssm", "post-hoc+SSM"), ("sketch", "sketch"), ("sketch_online", "sketch+online")]),
             ("SD-B", "w3_sdb_1b", [("pld", "PLD"), ("ngram", "n-gram"), ("sdhead", "head"), ("sdhead_online", "head+online")]),
             ("SD-C", "w3_sdc_1b", [("pld", "PLD"), ("ngram", "n-gram"), ("sdhead", "head"), ("sdhead_online", "head+online")])]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.1), sharey=True, gridspec_kw={"width_ratios": [6, 4, 4]})
    for ax, (tname, f, drafters) in zip(axes, specs):
        d = load(f"evaluation/acceptance/acceptance_{f}.json")["drafters"]
        xs = range(len(drafters))
        for i, (t, hatch, off) in enumerate([("open", None, -0.2), ("long", "///", 0.2)]):
            vals = [d[k]["results"][t]["4"]["mean_accepted"] for k, _ in drafters]
            lo = [d[k]["results"][t]["4"]["ci95"][0] for k, _ in drafters]
            hi = [d[k]["results"][t]["4"]["ci95"][1] for k, _ in drafters]
            ax.bar([x + off for x in xs], vals, 0.4, color=COL.get(tname), alpha=0.85 if i == 0 else 0.45,
                   hatch=hatch, edgecolor="k", lw=0.3, label=t,
                   yerr=[[max(v - l, 0) for v, l in zip(vals, lo)], [max(h - v, 0) for v, h in zip(vals, hi)]],
                   error_kw={"lw": 0.6, "capsize": 1.5})
        ax.set_xticks(list(xs), [n for _, n in drafters], rotation=40, ha="right")
        ax.set_title(f"target {tname}")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("accepted tokens / step (γ=4)")
    axes[0].legend(frameon=False, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "acceptance.pdf")
    plt.close(fig)


def fig_ladder() -> None:
    sb, sdb, sdc = (load(f"evaluation/ladder/ladder_{n}.json")["levels"] for n in ("sb", "sdb", "sdc"))
    rows = [("tok. n-gram counts (control)", sb["L0_tok_ngram"], "#bbbbbb"),
            ("random-router code counts (control)", sb["L0_random_router"], "#bbbbbb"),
            ("SB code counts (L0)", sb["L0_sketch"], COL["SB"]),
            ("SB rows, zero query (L1)", sb["L1_zero_query"], COL["SB"]),
            ("SB post-hoc head (L2)", sb["L2_draft"], COL["post-hoc"]),
            ("SB tuned lens, layer 1 (L3)", sb["L3_tuned_lens"], COL["SB"]),
            ("SD-B joint head (L2)", sdb["L2_draft"], COL["SD-B"]),
            ("SD-C joint head (L2)", sdc["L2_draft"], COL["SD-C"])]
    fig, ax = plt.subplots(figsize=(3.1, 2.1))
    for i, (name, r, c) in enumerate(rows):
        v = r["top1_agree"]
        ax.barh(i, v["value"], color=c, edgecolor="k", lw=0.3,
                xerr=[[v["value"] - v["lo"]], [v["hi"] - v["value"]]], error_kw={"lw": 0.6, "capsize": 1.5})
        ax.text(v["value"] + 0.008, i, f"{v['value']:.3f}", va="center", fontsize=6.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.52)
    ax.set_xlabel("top-1 agreement with model")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "ladder.pdf")
    plt.close(fig)


def fig_predictability() -> None:
    models = [("SB", "sb"), ("SD-B", "sdb"), ("SD-C", "sdc")]
    preds = [("uni", "x[t]"), ("bi", "x[t−1], x[t]"), ("mlp", "MLP x[t−2..t]")]
    fig, ax = plt.subplots(figsize=(3.1, 1.9))
    for j, (m, f) in enumerate(models):
        s = load(f"evaluation/predictability/rq2_{f}.json")["layers"]["1"]["summary"]
        for i, (p, lab) in enumerate(preds):
            v = s[f"mean_frac_{p}"]
            ax.bar(j + (i - 1) * 0.26, v["value"], 0.26, color=COL[m], alpha=[0.45, 0.7, 1.0][i], edgecolor="k", lw=0.3,
                   label=lab if j == 0 else None)
    ax.set_xticks(range(3), [m for m, _ in models])
    ax.set_ylabel("fraction of code entropy\nexplained (held-out)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(frameon=False, loc="upper right", title="predictor", title_fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "predictability.pdf")
    plt.close(fig)


def fig_speedup() -> None:
    acc = load("evaluation/acceptance/acceptance_w3_sdc_1b.json")["drafters"]
    ref = load("evaluation/drafter/cpu_bench_sdc_ref.json")
    cpp = load("evaluation/drafter/cpu_bench_sdc.json")
    target_ms = 22.51  # fp32 greedy step on V100 (results/evaluation/drafter/live_sdc_fp32.json: 44.43 tok/s)
    c_ref = min(r["us_per_token_median"] for r in ref["runs"]) / 1000 / target_ms
    c_cpp = min(r["us_per_token_median"] for r in cpp["runs"]) / 1000 / target_ms
    fig, ax = plt.subplots(figsize=(4.6, 2.0))
    gs = [2, 4, 6]
    for name, key, c, col, ls in [("PLD (c≈0)", "pld", 0.0, COL["PLD"], "-"), ("n-gram (c≈0)", "ngram", 0.0, COL["n-gram"], "-"),
                                  ("SD-C head+online, PyTorch drafter", "sdhead_online", c_ref, COL["SD-C"], "-"),
                                  ("SD-C head+online, C++ drafter", "sdhead_online", c_cpp, COL["SD-C"], ":"),
                                  ("SD-C head, PyTorch drafter", "sdhead", c_ref, COL["SD-B"], "-")]:
        ys = [(acc[key]["results"]["open"][str(g)]["mean_accepted"] + 1) / (1 + g * c) for g in gs]
        ax.plot(gs, ys, marker="o", ms=3, color=col, ls=ls, lw=1.1, label=name)
    live = load("evaluation/drafter/live_sdc_fp32.json")["by_gamma"]
    ax.plot([int(g) for g in live], [live[g]["speedup"] for g in live], marker="x", ms=4, color="k", ls="--", lw=0.9,
            label="measured live loop (head, C++)")
    ax.axhline(1, color="k", lw=0.5)
    ax.set_xlabel("draft length γ")
    ax.set_ylabel("speedup vs greedy")
    ax.set_xticks(gs)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=5.4, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "speedup.pdf")
    plt.close(fig)


def fig_draft_training() -> None:
    fig, ax = plt.subplots(figsize=(3.1, 1.9))
    for name, d, c in [("SD-B", "training/sdb", COL["SD-B"]), ("SD-C", "training/sdc", COL["SD-C"])]:
        e = evals(RES / d / "train.jsonl")
        s = [k for k in sorted(e) if "val_draft_top1_agree" in e[k]]
        ax.plot([k * TOK_PER_STEP / 1e9 for k in s], [e[k]["val_draft_top1_agree"] for k in s], color=c, marker="o", ms=2,
                lw=1.1, label=name)
    for name, p, ls in [("post-hoc on SB", "training/posthoc_sb/train.jsonl", "--"),
                        ("post-hoc+SSM on SB", "training/posthoc_sb_ssm/train.jsonl", ":")]:
        last = [json.loads(line) for line in (RES / p).read_text(encoding="utf-8").splitlines() if '"val_top1_agree"' in line][-1]
        ax.axhline(last["val_top1_agree"], color=COL["post-hoc"], ls=ls, lw=1, label=f"{name} (final)")
    ax.set_xlabel("training tokens (B)")
    ax.set_ylabel("draft top-1 agreement (val)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "draft_training.pdf")
    plt.close(fig)




# ------------------------------------------------------------------ interpretability figures
def fig_continuation() -> None:
    acc = load("interp/accept.json")
    rep = load("interp/repetition.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.1), sharey=True)
    mids = [8, 24, 48, 80, 110]
    for ax, t in zip(axes, ("open", "long")):
        bp = acc["SD-C"][t]["by_continuation_position"]
        keys = list(bp)
        for k, lab, col, ls in [("pld", "PLD", COL["PLD"], "-"), ("sdhead", "SD-C head", COL["SD-C"], "--"),
                                ("sdhead_online", "SD-C head+online", COL["SD-C"], "-")]:
            ax.plot(mids, [bp[b][k] for b in keys], marker="o", ms=3, color=col, ls=ls, lw=1.2, label=lab)
        ax2 = ax.twinx()
        rp = rep["SD-C"][t]["by_position"]
        ax2.bar(mids, [rp[b] for b in rp], width=[16, 16, 32, 32, 30], color="#999999", alpha=0.18, zorder=0,
                label="tokens copying an earlier 5-gram")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("copied fraction" if t == "long" else "")
        if t == "open":
            ax2.set_yticklabels([])
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_xlabel("position in greedy continuation")
        ax.set_title(f"{t} prompts, target SD-C")
        ax.grid(alpha=0.3)
        if t == "open":
            ax.set_ylabel("accepted / step (γ=4)")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "continuation.pdf")
    plt.close(fig)


def fig_redundancy() -> None:
    red = load("interp/redundancy.json")
    fig, ax = plt.subplots(figsize=(3.1, 1.8))
    edges = [i / 20 for i in range(21)]
    for m in ("SB", "SD-B", "SD-C"):
        h = red[m]["hist"]
        tot = sum(h)
        ax.step(edges[:-1], [x / tot for x in h], where="post", color=COL[m], lw=1.3,
                label=f"{m} (mean {red[m]['nmi_mean']:.3f})")
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("normalised MI between two routes' codes")
    ax.set_ylabel("fraction of route pairs")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "redundancy.pdf")
    plt.close(fig)


def fig_token_types() -> None:
    ben = load("interp/benefit.json")
    da = load("interp/draftacc.json")
    classes = ["word-start", "word-cont", "punct", "digit", "whitespace/newline"]
    short = ["word\nstart", "word\ncont.", "punct.", "digit", "space/\nnewline"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 1.9))
    x = range(len(classes))
    axes[0].bar([i - 0.2 for i in x], [ben["by_next_token_class"][c]["gain"] for c in classes], 0.4, color=COL["SB"],
                edgecolor="k", lw=0.3, label="memory benefit to SB")
    axes[0].bar([i + 0.2 for i in x], [-ben["sb_minus_dense"]["by_next_token_class"][c] for c in classes], 0.4,
                color=COL["D270"], edgecolor="k", lw=0.3, label="D270 (2B tok.) advantage over SB")
    axes[0].set_xticks(list(x), short)
    axes[0].set_ylabel("nats / token")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].set_title("loss by next-token type (held-out)")
    for m, off in (("SD-B", -0.2), ("SD-C", 0.2)):
        axes[1].bar([i + off for i in x], [da[m]["agree_by_next_class"][c] for c in classes], 0.4, color=COL[m],
                    edgecolor="k", lw=0.3, label=f"{m} head")
    axes[1].set_xticks(list(x), short)
    axes[1].set_ylabel("top-1 agreement")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].set_title("draft agreement by model's next token")
    fig.tight_layout()
    fig.savefig(OUT / "token_types.pdf")
    plt.close(fig)


def fig_confidence() -> None:
    da = load("interp/draftacc.json")
    fig, ax = plt.subplots(figsize=(3.1, 1.9))
    for m in ("SD-B", "SD-C"):
        d = da[m]["agree_by_model_confidence"]
        ks = list(d)
        mids = [(float(k.split("-")[0]) + float(k.split("-")[1])) / 2 for k in ks]
        ax.plot(mids, [d[k]["agree"] for k in ks], marker="o", ms=3, color=COL[m], lw=1.2, label=f"{m}: vs target confidence")
        if "agree_by_draft_confidence" in da[m]:
            g = da[m]["agree_by_draft_confidence"]
            gk = list(g)
            gm = [(float(k.split("-")[0]) + min(float(k.split("-")[1]), 1.0)) / 2 for k in gk]
            ax.plot(gm, [g[k]["agree"] for k in gk], marker="s", ms=3, color=COL[m], ls="--", lw=1.0,
                    label=f"{m}: vs drafter confidence")
    ax.plot([0, 1], [0, 1], color="k", lw=0.5, ls=":")
    ax.set_xlabel("max probability")
    ax.set_ylabel("draft = target argmax")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=5.5)
    fig.tight_layout()
    fig.savefig(OUT / "confidence.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_loss, fig_acceptance, fig_ladder, fig_predictability, fig_speedup, fig_draft_training,
              fig_continuation, fig_redundancy, fig_token_types, fig_confidence):
        f()
        print("wrote", f.__name__)
