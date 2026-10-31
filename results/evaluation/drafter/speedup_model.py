"""Implied speculative-decoding speedup for SD-C from offline acceptance + measured drafter/target cost.

speedup(drafter, text, gamma) = E[accepted + 1] / (1 + gamma * c_draft)
  c_draft = drafter time per drafted token / target time per decoding step
          = (C++ us/token, best thread count, cpu_bench_sdc.json) / (1 / greedy_tokens_per_s, live_sdc_<dtype>.json)
  pld / ngram: c_draft assumed 0 (cheap lookups, not measured).
  sdhead_online: same c_draft as sdhead; the online count update per verified token is unmeasured.
CPU only:  python3 jobs/speedup_model.py   (from ~/morbius)
"""
import json
from pathlib import Path

B = Path(__file__).resolve().parents[1]
R = B / "results"
acc = json.loads((B / "inputs/acceptance_w3_sdc_1b.json").read_text())
bench = json.loads((R / "cpu_bench_sdc.json").read_text())
best = min(bench["runs"], key=lambda r: r["us_per_token_median"])
us = best["us_per_token_median"]

live = {}
for dt in ("fp32", "fp16"):
    p = R / f"live_sdc_{dt}.json"
    if p.exists():
        live[dt] = json.loads(p.read_text())

c_draft = {}
for dt, d in live.items():
    g = d["by_gamma"]
    tps = sum(v["greedy_tokens_per_s"] for v in g.values()) / len(g)  # same greedy pass for every gamma
    c_draft[dt] = {"greedy_tokens_per_s": tps, "target_ms_per_step": 1e3 / tps, "c_draft": us * 1e-6 * tps}

rows = []
for name in ("sdhead", "sdhead_online", "pld", "ngram"):
    dr = acc["drafters"].get(name)
    if dr is None:
        continue
    for text, res in dr["results"].items():
        for gk, v in res.items():
            gamma = int(gk)
            ea = v["mean_accepted"] + 1
            lo, hi = (x + 1 for x in v["ci95"])
            row = {"drafter": name, "text": text, "gamma": gamma, "mean_accepted": v["mean_accepted"],
                   "E_accepted_plus_1": ea, "E_ci95": [lo, hi], "implied": {}}
            for dt, c in c_draft.items():
                cd = 0.0 if name in ("pld", "ngram") else c["c_draft"]
                den = 1 + gamma * cd
                row["implied"][dt] = {"c_draft": cd, "speedup": ea / den, "speedup_ci95": [lo / den, hi / den]}
            rows.append(row)

measured = {dt: {g: {k: v[k] for k in ("speedup", "tokens_per_s", "greedy_tokens_per_s", "mean_accepted",
                                         "mean_tokens_per_block", "match_rate")}
                 for g, v in d["by_gamma"].items()} for dt, d in live.items()}
out = {"source": {"acceptance": "inputs/acceptance_w3_sdc_1b.json", "bench": "results/cpu_bench_sdc.json",
                  "live": [f"results/live_sdc_{dt}.json" for dt in live]},
       "drafter_us_per_token": {"impl": best["impl"], "threads": best["threads"], "median": us,
                                "p99": best["us_per_token_p99"]},
       "c_draft": c_draft,
       "assumptions": ["c_draft = 0 for pld and ngram (not measured)",
                       "sdhead_online uses the sdhead c_draft; its per-verified-token hash update is unmeasured",
                       "timings indicative: node shared with another user's job and with lm-eval on the other GPU"],
       "rows": rows, "measured_live": measured}
(R / "speedup_model_sdc.json").write_text(json.dumps(out, indent=2))

dts = list(c_draft)
md = ["# SD-C implied speedup = E[accepted+1] / (1 + γ·c_draft)", "",
      f"Drafter: C++ {best['threads']} thread(s), median {us:.1f} µs/token (results/cpu_bench_sdc.json).", ""]
for dt, c in c_draft.items():
    md.append(f"- {dt}: target {c['target_ms_per_step']:.2f} ms/step ({c['greedy_tokens_per_s']:.1f} tok/s greedy, "
              f"results/live_sdc_{dt}.json) → c_draft = {c['c_draft']:.4f}")
md += ["- pld / ngram: c_draft assumed 0. sdhead_online: online count update unmeasured.", "",
       "| drafter | text | γ | E[acc+1] (95% CI) | " + " | ".join(f"implied speedup ({dt})" for dt in dts) + " |",
       "|---|---|---|---|" + "---|" * len(dts)]
for r in rows:
    cells = [f"{r['implied'][dt]['speedup']:.3f} ({r['implied'][dt]['speedup_ci95'][0]:.3f}–"
             f"{r['implied'][dt]['speedup_ci95'][1]:.3f})" for dt in dts]
    md.append(f"| {r['drafter']} | {r['text']} | {r['gamma']} | {r['E_accepted_plus_1']:.3f} "
              f"({r['E_ci95'][0]:.3f}–{r['E_ci95'][1]:.3f}) | " + " | ".join(cells) + " |")
md += ["", "## Measured live loop (sdhead, no online counts)", "",
       "| dtype | γ | speedup | tokens/s | greedy tokens/s | mean accepted | match rate |", "|---|---|---|---|---|---|---|"]
for dt, m in measured.items():
    for g, v in m.items():
        md.append(f"| {dt} | {g} | {v['speedup']:.3f} | {v['tokens_per_s']:.1f} | {v['greedy_tokens_per_s']:.1f} | "
                  f"{v['mean_accepted']:.3f} | {v['match_rate']:.3f} |")
(R / "speedup_model_sdc.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
