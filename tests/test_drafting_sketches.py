"""M3.T1: sketch building against brute-force counts; code sources; recall."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pytest
import torch

from drafting.build_sketches import (TopKSummary, build_sketch, memory_codes, random_router_weight, seq_rows,
                                     sketch_recall, stream_windows)
from drafting.sketches import Sketch
from tests.test_infer import make


def captured_codes(model, x, layer=1):
    mem = model.blocks[layer].memory
    mem.capture_route_logits = True
    try:
        with torch.no_grad():
            model(x)
        return mem.module.quantize_route_logits(mem.last_route_logits)
    finally:
        mem.capture_route_logits = False


def mg_reference(chunks, k):
    """Python Misra-Gries merge of exact per-chunk counts: row -> {item: count}."""
    summ: dict = defaultdict(dict)
    for rows, items in chunks:
        cnt = Counter(zip(rows.tolist(), items.tolist()))
        touched = defaultdict(dict)
        for (r, it), c in cnt.items():
            touched[r][it] = c
        for r, new in touched.items():
            merged = dict(summ[r])
            for it, c in new.items():
                merged[it] = merged.get(it, 0) + c
            order = sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))
            thr = order[k][1] if len(order) > k else 0
            summ[r] = {it: c - thr for it, c in order[:k] if c - thr > 0}
    return summ


def summary_dict(s: TopKSummary):
    ids, counts = s.numpy()
    return {r: {int(i): int(c) for i, c in zip(ids[r], counts[r]) if c > 0} for r in range(len(ids))}


@pytest.mark.parametrize("mode", ["mg", "topk"])
def test_topk_summary_exact_regime(mode):
    g = torch.Generator().manual_seed(0)
    s = TopKSummary(50, 16, 8, mode=mode, max_buffer=1)  # merge after every chunk
    truth: dict = defaultdict(Counter)
    for _ in range(20):
        rows = torch.randint(0, 50, (200,), generator=g)
        items = torch.randint(0, 12, (200,), generator=g)  # <= 12 distinct items per row
        s.add(rows, items)
        for r, it in zip(rows.tolist(), items.tolist()):
            truth[r][it] += 1
    got = summary_dict(s)
    for r in range(50):
        assert got[r] == dict(truth[r])
    ids, counts = s.numpy()
    assert (np.diff(counts.astype(np.int64), axis=1) <= 0).all()  # slots sorted by count


def test_topk_summary_matches_python_misra_gries():
    g = torch.Generator().manual_seed(1)
    chunks = [(torch.randint(0, 6, (300,), generator=g), torch.randint(0, 40, (300,), generator=g) % (
        1 + torch.randint(0, 40, (300,), generator=g))) for _ in range(8)]
    s = TopKSummary(6, 4, 8, mode="mg", max_buffer=1)
    for rows, items in chunks:
        s.add(rows, items)
    ref = mg_reference(chunks, 4)
    got = summary_dict(s)
    for r in range(6):
        assert got[r] == ref[r]


def test_heavy_hitter_survives_merges():
    g = torch.Generator().manual_seed(2)
    s = TopKSummary(1, 4, 10, max_buffer=1)
    for _ in range(30):
        items = torch.randint(0, 1000, (100,), generator=g)
        items[:40] = 7  # 40% heavy hitter
        s.add(torch.zeros(100, dtype=torch.long), items)
    ids, counts = s.numpy()
    assert ids[0, 0] == 7 and counts[0, 0] >= 1200 - 3000 / 5


@pytest.mark.parametrize("variant", ["wave1", "sdb", "sdc"])
def test_memory_codes_equal_full_forward(variant):
    model = make(variant)
    x = torch.randint(0, 256, (2, 32))
    torch.testing.assert_close(memory_codes(model, x, 1), captured_codes(model, x), rtol=0, atol=0)


def test_seq_rows_matches_sketch_convention():
    codes = torch.randint(0, 16, (1, 20, 4))
    sk = Sketch.empty(4, 4, (2, 3), 1, k=2, vocab=8)
    for n in (2, 3):
        assert np.array_equal(seq_rows(codes, n, 4)[0].numpy(), sk.rows_at(codes[0].numpy(), n))


@pytest.mark.parametrize("variant", ["wave1", "sdb"])
def test_build_sketch_equals_brute_force(variant):
    model = make(variant)
    g = torch.Generator().manual_seed(3)
    windows = [torch.randint(4, 16, (3, 33), generator=g) for _ in range(4)]  # 12 distinct tokens
    sk = build_sketch(model, windows, k=16, max_buffer=1)
    truth = {n: defaultdict(Counter) for n in (2, 3)}
    for w in windows:
        codes = captured_codes(model, w[:, :-1])
        for b in range(w.shape[0]):
            for n in (2, 3):
                rows = sk.rows_at(codes[b].numpy(), n)
                for t in range(n - 1, 32):
                    for r in range(4):
                        truth[n][int(rows[t, r])][int(w[b, t + 1])] += 1
    assert sk.tokens_seen == 4 * 3 * 32 and sk.layer == 1 and sk.k == 16
    for n in (2, 3):
        nz = np.flatnonzero(sk.counts[n].sum(1))
        assert set(nz.tolist()) == set(truth[n])
        for row, cnt in truth[n].items():
            got = {int(i): int(c) for i, c in zip(sk.ids[n][row], sk.counts[n][row]) if c > 0}
            assert got == dict(cnt)
    rec = sketch_recall(model, sk, [(w[:, :-1], w[:, 1:]) for w in windows])
    assert rec["recall16_row"] == 1.0 and 0 < rec["top1"] <= rec["recall16"] <= 1.0


def test_random_router_source():
    model = make("wave1")
    w1, w2 = random_router_weight(model, 1, 0), random_router_weight(model, 1, 0)
    assert torch.equal(w1, w2) and w1.shape == model.blocks[1].memory.router.weight.shape
    x = torch.randint(0, 256, (2, 32))
    assert not torch.equal(memory_codes(model, x, 1, w1), memory_codes(model, x, 1))
    sk = build_sketch(model, [x[:, :32]], router_weight=w1)
    assert sk.counts[2].sum() == 2 * 30 * 4  # 31 inputs, bigrams end at t >= 1, 4 routes


def test_stream_windows_and_cli(tiny_data, tmp_path):
    ws = list(stream_windows(tiny_data, 32, 32 * 10, 4))
    assert [w.shape for w in ws] == [(4, 33), (4, 33), (2, 33)]
    from ffnmem import checkpoint as ckpt
    from ffnmem.config import load_config
    from drafting import build_sketches
    from tests.test_infer import CFG

    model = make("sdb")
    cfg = load_config(CFG / "tiny_cpu_sd.yaml", ["model.memory.router_input=embed2", "model.ssm=null"])
    ckpt.save_atomic({"model": model.state_dict(), "config": cfg, "step": 1}, tmp_path / "run" / "checkpoints" / "step_0000001.pt")
    out = tmp_path / "sk.npz"
    info = build_sketches.main(["--target", str(tmp_path / "run"), "--data_dir", str(tiny_data), "--out", str(out),
                                "--tokens", "2048", "--batch", "8", "--val_tokens", "512", "--device", "cpu"])
    sk = Sketch.load(out)
    assert sk.tokens_seen == 2048 and info["val"]["val_positions"] > 0
    assert out.with_suffix(".json").exists()
