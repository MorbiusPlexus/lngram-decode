"""analysis/plot_figures.py draws every figure from minimal results files of each schema."""

from __future__ import annotations

import json

from analysis.plot_figures import main


def ci(v: float) -> dict:
    return {"value": v, "lo": v - 0.01, "hi": v + 0.01}


def write(p, obj):
    p.write_text(json.dumps(obj), encoding="utf-8")


def test_all_figures(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    write(res / "ladder_r1.json", {"run": "r1", "levels": {
        lv: {m: ci(x) for m in ("top1_agree", "sum_min", "kl", "acc_true")}
        for lv, x in [("L0_sketch", 0.2), ("L0_tok_ngram", 0.1), ("L1_zero_query", 0.3), ("L4_model", 1.0)]}})
    summary = {f"mean_frac_{p}": ci(0.4) for p in ("uni", "bi", "mlp")}
    summary.update({f"mean_next_recall@{k}": ci(0.5) for k in (1, 2, 4, 8)})
    summary.update({f"p_any_miss@{k}": ci(0.3) for k in (1, 2, 4, 8)})
    write(res / "rq2_r1.json", {"run": "r1", "layers": {"1": {"routes": 4, "summary": summary}}})
    write(res / "prefetch_A_r1.json", {"rows": [
        {"mode": m, "batch": b, "latency_ms_median": 1.0 * b, "latency_ms_p99": 2.0 * b}
        for m in ("b0", "b4") for b in (1, 2, 4)]})
    write(res / "scaled_A.json", {"rows": [
        {"growth": "routes", "batch": 1, "table_bytes": gb * 2**30, "b1": {"median_ms": gb},
         "b4": [{"exposed": {"median_ms": gb / 2}}]} for gb in (0.1, 1, 10)]})
    entry = {"mean_accepted": 1.5, "ci95": [1.2, 1.8]}
    write(res / "acceptance_w3_sdb_1b.json", {"target": "w3_sdb_1b", "text_types": {
        "open": {}, "long": {}, "code": {"skipped": "none"}}, "drafters": {
        "ngram": {"kind": "ngram", "results": {"open": {"4": entry}, "long": {"4": entry}}},
        "sd": {"kind": "sdhead", "results": {"open": {"4": entry}, "long": {"4": dict(entry, mean_accepted=2.0)}}},
        "oracle": {"kind": "oracle", "results": {"open": {"4": entry}}}}})
    write(res / "live_r1.json", {"run": "r1", "by_gamma": {"2": {"speedup": 1.3}, "4": {"speedup": 1.6}}})
    write(res / "ffn_importance_r1.json", {"run": "r1", "layers": [
        {"pos": i, "delta": 0.1 * i, "ci_lo": 0.1 * i - 0.01, "ci_hi": 0.1 * i + 0.01, "has_memory": i == 1}
        for i in range(4)]})
    run = tmp_path / "runs" / "w3_sdb_1b" / "logs"
    run.mkdir(parents=True)
    (run / "train.jsonl").write_text(json.dumps({"event": "eval", "val_loss": 3.1}) + "\n", encoding="utf-8")

    made = main(["--results", str(res), "--out", str(tmp_path / "fig"),
                 "--sd_runs", f"SD-B={tmp_path / 'runs' / 'w3_sdb_1b'}"])
    expected = {"ladder.png", "code_predictability.png", "prefetch_batch.png", "prefetch_table_size.png",
                "acceptance.png", "live_speedup.png", "ffn_importance.png", "sd_tradeoff.png"}
    assert set(made) == expected
    for name in expected:
        assert (tmp_path / "fig" / name).stat().st_size > 1000
    assert json.loads((tmp_path / "fig" / "figures.json").read_text())["ladder.png"]


def test_missing_inputs_are_skipped(tmp_path):
    (tmp_path / "results").mkdir()
    assert main(["--results", str(tmp_path / "results"), "--out", str(tmp_path / "fig")]) == {}
