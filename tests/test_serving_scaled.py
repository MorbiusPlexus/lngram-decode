"""M4.T3 scaled-table builder: sizes, route replication, and 4-gram addressing (CPU)."""

from __future__ import annotations

import numpy as np
import torch

from ffnmem.memory.lngram_v2 import _aligned_ngram_address
from serving.layout import permute_table
from serving.prefetch import LayerTable
from serving.scaled_tables import (GB, plan_sizes, route_bytes, route_codes, routes_for_bytes, run)


def test_sizes():
    for ngrams in [(2, 3), (2, 3, 4)]:
        for target in [0.1 * GB, 1 * GB, 10 * GB]:
            r = routes_for_bytes(target, ngrams, 128, 2)
            got = (LayerTable.num_rows(r, ngrams, 4) - 1) * 128 * 2
            assert abs(got - target) <= route_bytes(ngrams, 128, 2) / 2 + 1
    assert route_bytes((2, 3, 4), 128, 2) == (256 + 4096 + 65536) * 256
    assert plan_sizes([0.1, 1, 10], 0.5, True, ram=int(8 * GB)) == [0.1 * GB, 1 * GB, 4 * GB]
    assert plan_sizes([0.1, 1, 10], 0.5, False, ram=None) == [0.1 * GB, 1 * GB, 10 * GB]
    lt = LayerTable.random(3, (2, 3), 4, 8, torch.float16, pin=False, block_rows=100)
    assert lt.weight.shape == (3 * (256 + 4096) + 1, 8)
    assert torch.count_nonzero(lt.weight[-1]) == 0 and torch.count_nonzero(lt.weight[:-1]) > 0


def test_route_replication():
    stream = np.random.default_rng(0).integers(0, 16, (100, 3), dtype=np.uint8)
    c = route_codes(stream, 7, 2, 10)
    assert c.shape == (2, 10, 7)
    assert torch.equal(c[0, :, 4], torch.from_numpy(stream[:10, 1].copy()))
    assert torch.equal(c[1, :, 0], torch.from_numpy(stream[50:60, 0].copy()))


def test_four_gram_addressing_matches_v2_order():
    torch.manual_seed(0)
    r, d, ngrams = 2, 4, (2, 3, 4)
    v2 = {o: torch.randn(r * 16 ** o, d) for o in ngrams}
    weight = torch.cat([permute_table(v2[o], r, o, 4) for o in ngrams] + [torch.zeros(1, d)])
    lt = LayerTable(weight, r, ngrams, 4)
    codes = torch.randint(0, 16, (2, 9, r), dtype=torch.uint8)
    rows = lt.gather(lt.exact_index(codes, 9))  # [B, 9, R, 3, D]
    for k, o in enumerate(ngrams):
        addr = _aligned_ngram_address(codes.to(torch.int32), o, 4).long() + (torch.arange(r) * 16 ** o).view(1, 1, r)
        want = torch.cat((torch.zeros(2, o - 1, r, d), v2[o][addr]), 1)
        assert torch.equal(rows[:, :, :, k], want)
    for p in range(9):  # candidates of position p contain its rows
        cand = lt.gather(lt.candidate_index(codes[:, max(0, p - 3):p], p))
        sel = cand.gather(3, codes[:, p].long()[:, :, None, None, None].expand(-1, -1, 3, 1, d))[:, :, :, 0]
        assert torch.equal(sel, rows[:, p])


def test_run_smoke():
    stream = np.random.default_rng(0).integers(0, 16, (500, 4), dtype=np.uint8)
    rows = run(stream, [2e6], ["routes", "order4"], [1, 2], 16, torch.float16, torch.device("cpu"),
               warmup=2, steps=6, gaps=[0.0], pin=False, log=lambda s: None)
    assert len(rows) == 4
    for row in rows:
        lt_rows = LayerTable.num_rows(row["routes"], row["ngrams"], 4)
        assert row["table_bytes"] == lt_rows * 16 * 2
        b4 = row["b4"][0]
        assert b4["h2d_bytes_per_step"] == row["batch"] * row["routes"] * len(row["ngrams"]) * 16 * 16 * 2
        assert row["b1"]["h2d_bytes_per_step"] == row["batch"] * row["routes"] * len(row["ngrams"]) * 16 * 2
