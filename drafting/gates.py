"""Decision gates from the offline acceptance results (plan section 4).

    python -m drafting.gates --ssm_file results/acceptance_w3_sb_1b.json \
        --acceptance results/acceptance_*.json --out results/gates.json

- SSM gate: post-hoc drafter + SSM vs without SSM (labels ``--with_ssm`` / ``--without_ssm``,
  same acceptance file, same prompts) on long-document text at gamma = 4: relative gain
  ``mean(+SSM) / mean(-SSM) - 1 >= 5%`` and its paired bootstrap 95% CI excludes 0.
  Pass -> run SD-C, fail -> run SD-A.
- Live gate: any SD head or post-hoc drafter (kinds ``--candidate_kinds``) beats the token
  n-gram drafter (label ``--baseline``) of the same file on any text type at gamma = 4:
  paired bootstrap 95% CI of the difference in mean accepted length excludes 0 (> 0).

Paired bootstrap: 1,000 resamples of prompt indices, applied to both drafters' per-prompt
mean accepted lengths.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def _per_prompt(entry: dict, text_type: str, gamma: int) -> tuple[np.ndarray, list]:
    res = entry["results"][text_type][str(gamma)]
    return np.asarray(res["per_prompt"], dtype=np.float64), entry.get("doc_ids", {}).get(text_type, [])


def paired_bootstrap(a: np.ndarray, b: np.ndarray, stat, n_boot: int = 1000, seed: int = 0) -> tuple[float, list]:
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n_boot, len(a)))
    samples = stat(a[idx].mean(1), b[idx].mean(1))
    return float(stat(a.mean(), b.mean())), [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def _pair(da: dict, db: dict, t: str, gamma: int) -> tuple[np.ndarray, np.ndarray]:
    a, ia = _per_prompt(da, t, gamma)
    b, ib = _per_prompt(db, t, gamma)
    if len(a) != len(b) or (ia and ib and ia != ib):
        raise ValueError(f"unpaired prompts for text type {t}")
    return a, b


def ssm_gate(result: dict, with_ssm: str, without_ssm: str, text_type: str = "long", gamma: int = 4,
             min_gain: float = 0.05, n_boot: int = 1000) -> dict:
    d = result["drafters"]
    if with_ssm not in d or without_ssm not in d:
        return {"status": "missing", "need": [with_ssm, without_ssm], "have": sorted(d)}
    if text_type not in d[with_ssm]["results"] or text_type not in d[without_ssm]["results"]:
        return {"status": "missing", "need_text_type": text_type}
    a, b = _pair(d[with_ssm], d[without_ssm], text_type, gamma)
    gain, ci = paired_bootstrap(a, b, lambda x, y: x / y - 1.0, n_boot)
    passed = gain >= min_gain and ci[0] > 0
    return {"status": "pass" if passed else "fail", "relative_gain": gain, "ci95": ci, "min_gain": min_gain,
            "mean_with_ssm": float(np.nanmean(a)), "mean_without_ssm": float(np.nanmean(b)),
            "text_type": text_type, "gamma": gamma, "target": result.get("target"),
            "decision": "run SD-C" if passed else "run SD-A"}


def live_gate(results: list[dict], baseline: str = "ngram", kinds=("posthoc", "sdhead"), gamma: int = 4,
              n_boot: int = 1000) -> dict:
    comps = []
    for res in results:
        d = res["drafters"]
        if baseline not in d:
            continue
        for label, entry in d.items():
            if entry.get("kind") not in kinds:
                continue
            for t in entry["results"]:
                if t not in d[baseline]["results"]:
                    continue
                a, b = _pair(entry, d[baseline], t, gamma)
                diff, ci = paired_bootstrap(a, b, lambda x, y: x - y, n_boot)
                comps.append({"target": res.get("target"), "drafter": label, "text_type": t, "gamma": gamma,
                              "diff": diff, "ci95": ci, "beats": ci[0] > 0,
                              "mean": float(np.nanmean(a)), "baseline_mean": float(np.nanmean(b))})
    if not comps:
        return {"status": "missing", "comparisons": []}
    winners = [c for c in comps if c["beats"]]
    best = max(winners or comps, key=lambda c: c["diff"])
    return {"status": "pass" if winners else "fail", "comparisons": comps, "best": best,
            "decision": "build the live pipeline for the best drafter" if winners
            else "report offline acceptance + measured timings + speedup model"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acceptance", nargs="+", default=["results/acceptance_*.json"], help="files or globs")
    p.add_argument("--ssm_file", help="acceptance JSON with the post-hoc +/- SSM drafters (SB)")
    p.add_argument("--with_ssm", default="posthoc_ssm")
    p.add_argument("--without_ssm", default="posthoc")
    p.add_argument("--baseline", default="ngram")
    p.add_argument("--candidate_kinds", nargs="+", default=["posthoc", "sdhead"])
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--min_gain", type=float, default=0.05)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--out", default="results/gates.json")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    files = sorted({f for pat in a.acceptance for f in (glob.glob(pat) or ([pat] if Path(pat).exists() else []))})
    results = [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]
    out = {"files": files}
    if a.ssm_file:
        out["ssm_gate"] = ssm_gate(json.loads(Path(a.ssm_file).read_text(encoding="utf-8")), a.with_ssm,
                                   a.without_ssm, "long", a.gamma, a.min_gain, a.n_boot)
    else:
        out["ssm_gate"] = {"status": "missing", "need": "--ssm_file"}
    out["live_gate"] = live_gate(results, a.baseline, tuple(a.candidate_kinds), a.gamma, a.n_boot)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    s, l = out["ssm_gate"], out["live_gate"]
    print(f"[gates] SSM gate: {s['status']}"
          + (f"  gain {s['relative_gain']:+.1%} CI [{s['ci95'][0]:+.1%}, {s['ci95'][1]:+.1%}] -> {s['decision']}"
             if "relative_gain" in s else ""), flush=True)
    print(f"[gates] live gate: {l['status']}"
          + (f"  best {l['best']['drafter']} on {l['best']['text_type']} ({l['best']['target']}): "
             f"{l['best']['diff']:+.3f} CI [{l['best']['ci95'][0]:+.3f}, {l['best']['ci95'][1]:+.3f}]"
             if "best" in l else ""), flush=True)
    return out


if __name__ == "__main__":
    main()
