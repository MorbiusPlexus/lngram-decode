"""Bootstrap CIs and the lm-eval table from per-sample files."""

from __future__ import annotations

import json

import numpy as np
import pytest

from eval.bootstrap_lmeval import bootstrap_ci, load_samples, main, paired, summarise


def test_bootstrap_ci_sanity():
    m, lo, hi = bootstrap_ci(np.full(100, 0.25), n_boot=200)
    assert m == lo == hi == pytest.approx(0.25)
    x = np.random.default_rng(0).random(4000) < 0.7
    m, lo, hi = bootstrap_ci(x, n_boot=2000, seed=1)
    se = np.sqrt(m * (1 - m) / x.size)
    assert lo < m < hi and lo < 0.7 < hi
    assert hi - lo == pytest.approx(2 * 1.96 * se, rel=0.15)
    assert bootstrap_ci(x, 500, seed=3) == bootstrap_ci(x, 500, seed=3)


def _write(path, task, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc_id, acc, acc_norm in rows:
            f.write(json.dumps({"task": task, "doc_id": doc_id, "acc": acc, "acc_norm": acc_norm}) + "\n")


def test_load_pairs_and_table(tmp_path):
    rng = np.random.default_rng(0)
    for name, p in [("A", 0.6), ("B", 0.4)]:
        d = tmp_path / "lmeval" / name
        _write(d / "samples_piqa.jsonl", "piqa", [(i, float(rng.random() < p), float(rng.random() < p))
                                                   for i in range(800)])
        for sub in ("mmlu_anatomy_continuation", "mmlu_virology_continuation"):
            _write(d / f"samples_{sub}.jsonl", sub, [(i, float(rng.random() < p), 0.0) for i in range(50)])
    # a stale stock-format file plus a newer one: the newer wins
    stock = tmp_path / "lmeval" / "B" / "x"
    _write(stock / "samples_winogrande_2026-01-01T00-00-00.000001.jsonl", "winogrande", [(0, 0.0, 0.0)])
    _write(stock / "samples_winogrande_2026-02-01T00-00-00.000001.jsonl", "winogrande", [(0, 1.0, 1.0), (1, 1.0, 1.0)])
    runs = {n: load_samples(tmp_path / "lmeval" / n) for n in ("A", "B")}
    assert set(runs["A"]) == {"piqa", "mmlu_continuation"}
    assert len(runs["A"]["mmlu_continuation"]) == 100
    assert len(runs["B"]["winogrande"]) == 2
    table = summarise(runs, n_boot=500, seed=0)
    e = table["A"]["piqa"]["acc"]
    assert e["lo"] < e["mean"] < e["hi"]
    diff = paired(runs, "A", "B", n_boot=500, seed=0)
    assert diff["piqa"]["acc"]["excludes_zero"] and diff["piqa"]["acc"]["diff"] > 0
    assert "winogrande" not in diff
    same = paired({"A": runs["A"], "A2": runs["A"]}, "A", "A2", n_boot=200, seed=0)
    assert same["piqa"]["acc"]["diff"] == 0 and not same["piqa"]["acc"]["excludes_zero"]

    out = tmp_path / "results" / "lmeval_table.json"
    main(["--root", str(tmp_path / "lmeval"), "--pairs", "A:B", "--n_boot", "300", "--out", str(out)])
    res = json.loads(out.read_text())
    assert set(res["table"]) == {"A", "B"} and "A - B" in res["paired"]
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "| A |" in md and "mmlu_continuation" in md
