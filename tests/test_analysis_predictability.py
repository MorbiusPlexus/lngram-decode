"""M2.T2 code predictability on synthetic dumps whose answers are known."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from analysis import code_predictability as cp
from analysis.dump_codes import write_meta, write_shard

from .test_analysis_dump import tiny_checkpoint

T = 64          # window length
SMALL_V = 32    # tokens actually used (vocab of the dump stays 256 = the tiny model's)


def synthetic_dump(out: Path, n_windows: int = 700, seed: int = 0) -> Path:
    """Routes: 0 = x[t] % 16, 1 = iid uniform, 2 = f(x[t-1], x[t]), 3 = c[t-1] + 1 (mod 16)."""
    rng = np.random.default_rng(seed)
    n = n_windows * T
    x = rng.integers(0, SMALL_V, size=n)
    pos = np.tile(np.arange(T), n_windows)
    doc = np.cumsum(rng.random(n) < 1 / 150)          # ~150-token documents, monotone ids
    prev = np.where(pos >= 1, np.roll(x, 1), 256)
    codes = np.zeros((n, 4), np.uint8)
    codes[:, 0] = x % 16
    codes[:, 1] = rng.integers(0, 16, size=n)
    codes[:, 2] = (3 * x + 5 * prev + (prev * x) // 7) % 16
    start = rng.integers(0, 16, size=n_windows)
    codes[:, 3] = (np.repeat(start, T) + pos) % 16
    half = n // 2
    for s, sl in enumerate((slice(0, half), slice(half, n))):
        write_shard(out, s, {"tokens": x[sl].astype(np.uint16), "pos": pos[sl].astype(np.int32),
                             "doc": doc[sl].astype(np.int32), "codes_L1": codes[sl]})
    write_meta(out, {"run": "syn", "checkpoint": None, "vocab": 256, "bits": 4, "memory_layers": [1],
                     "routes": {"1": 4}, "seq_len": T, "n_positions": n,
                     "shards": [{"index": 0, "positions": half}, {"index": 1, "positions": n - half}],
                     "fields": ["tokens", "pos", "doc", "codes_L1"]})
    return out


def test_known_answers_and_bootstrap(tmp_path):
    dump = synthetic_dump(tmp_path / "dump")
    ck = tiny_checkpoint(tmp_path / "run")
    res = cp.main(["--dump", str(dump), "--run", "syn", "--ckpt", str(ck), "--n_boot", "200",
                   "--mlp_steps", "400", "--mlp_batch", "512", "--mlp_hidden", "128", "--device", "cpu",
                   "--out_dir", str(tmp_path / "results")])
    L = res["layers"]["1"]
    pr = L["per_route"]
    # route 0: deterministic in x[t]
    assert pr["H"][0] > 3.9 and pr["frac_uni"][0] > 0.95 and pr["top1_uni"][0] == 1.0
    assert pr["frac_bi"][0] > 0.9 and pr["frac_mlp"][0] > 0.8 and pr["top1_mlp"][0] > 0.95
    # route 1: independent noise -> nothing explained (slightly negative from estimation noise)
    for q in ("uni", "bi", "mlp"):
        assert -0.1 < pr[f"frac_{q}"][1] < 0.02
    assert 0.03 < pr["top1_uni"][1] < 0.1
    # route 2: a function of (x[t-1], x[t]): the bigram table explains it, x[t] alone much less
    assert pr["frac_bi"][2] > 0.8 and pr["frac_uni"][2] < 0.5
    # route 3: next code is the current + 1 -> perfect next-step recall; codes unrelated to tokens
    assert pr["next_recall@1"][3] == 1.0 and abs(pr["frac_uni"][3]) < 0.05
    assert abs(pr["next_recall@1"][0] - 1 / 16) < 0.02 and abs(pr["next_recall@8"][0] - 0.5) < 0.03
    # P(any route misses) is monotone in k and bounded by the per-route misses
    miss = [L["summary"][f"p_any_miss@{k}"]["value"] for k in cp.TOPK]
    assert all(a >= b for a, b in zip(miss, miss[1:])) and miss[0] > 1 - 1 / 16 - 0.03
    # every CI contains its point estimate -- except the plug-in entropy of a near-uniform route,
    # which sits at its maximum, so every resample is below it (known downward bias)
    for s in L["summary"].values():
        assert s["lo"] <= s["value"] + 1e-9 and s["value"] <= s["hi"] + 1e-9
    for stat, cis in L["per_route_ci"].items():
        if stat in ("H", "eff_codes"):
            continue
        for v, (lo, hi) in zip(pr[stat], cis):
            assert lo - 1e-9 <= v <= hi + 1e-9, stat
    out = tmp_path / "results"
    assert json.loads((out / "rq2_syn.json").read_text())["n_eval"] == res["n_eval"]
    assert (out / "rq2_syn_mlp_L1.pt").exists()
    try:
        import matplotlib  # noqa: F401

        assert (out / "figures" / "rq2_syn_L1.png").exists()
    except ImportError:
        pass
    mlp = cp.CodeMLP.load(out / "rq2_syn_mlp_L1.pt")
    assert mlp.routes == 4 and mlp.k == 16


def test_bootstrap_ratio_matches_direct():
    rng = np.random.default_rng(1)
    vals = rng.random(500)
    unit = np.sort(rng.integers(0, 40, size=500))
    _, unit = np.unique(unit, return_inverse=True)
    D = unit.max() + 1
    W = cp.bootstrap_weights(D, 300, seed=2)
    rows = (W @ cp.unit_sums(vals, unit, D)) / (W @ cp.unit_sums(np.ones(500), unit, D))
    s = cp.summarize(rows)
    assert abs(s["value"] - vals.mean()) < 1e-12 and s["lo"] < s["value"] < s["hi"]
    # one explicit resample equals the weighted formula
    pick = rng.integers(0, D, size=D)
    w = np.bincount(pick, minlength=D)
    direct = np.concatenate([vals[unit == u] for u in pick]).mean()
    assert abs((w @ cp.unit_sums(vals, unit, D)) / (w @ cp.unit_sums(np.ones(500), unit, D)) - direct) < 1e-12


def test_window_logits_matches_pointwise():
    torch.manual_seed(0)
    emb = cp.CodeMLP.padded_embeddings(torch.randn(10, 8))
    mlp = cp.CodeMLP(8, routes=3, bits=2, hidden=16).eval()
    idx = torch.randint(0, 10, (2, 5))
    w = mlp.window_logits(emb, idx)
    s = torch.tensor(10)
    for t in range(5):
        x1 = idx[:, t - 1] if t >= 1 else s.expand(2)
        x2 = idx[:, t - 2] if t >= 2 else s.expand(2)
        torch.testing.assert_close(w[:, t], mlp(emb, x2, x1, idx[:, t]))
