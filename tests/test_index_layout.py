"""M4.T1 / gate G1: v2 index layout facts, and the serving-layout permutation."""

from __future__ import annotations

from pathlib import Path

import torch

from ffnmem.config import load_config
from ffnmem.memory.lngram_v2 import _aligned_ngram_address
from ffnmem.model import Transformer
from serving.layout import (candidate_rows, permute_tables, reverse_digits, select_candidates,
                            serving_route_tokens)

CFG = Path(__file__).resolve().parents[1] / "configs"


def tiny_memory():
    torch.manual_seed(0)
    model = Transformer(load_config(CFG / "tiny_cpu.yaml")["model"]).eval()
    m = model.blocks[1].memory.module
    with torch.no_grad():
        for t in m.tables.values():
            t.weight.normal_()
    return m


def test_v2_suffix_and_digit_order():
    codes = torch.tensor([[[3], [5], [7], [9]]])  # one route, T = 4
    a2 = _aligned_ngram_address(codes, 2, 4)
    a3 = _aligned_ngram_address(codes, 3, 4)
    # window ending at t=1 is (c0=3, c1=5): oldest least significant
    assert a2[0, 0, 0] == 3 + 16 * 5
    assert a3[0, 1, 0] == 5 + 16 * 7 + 256 * 9  # ends at t = 3
    m = tiny_memory()
    tokens = m._lookup_route_chunk(torch.randint(0, 16, (1, 6, m.num_routes)), None, 0, m.num_routes, None, None)
    assert torch.count_nonzero(tokens[:, 0]) == 0  # suffix ending at t: nothing complete at t = 0
    assert torch.count_nonzero(tokens[:, 1, :, m.memory_dim:]) == 0  # no 3-gram at t = 1
    assert torch.count_nonzero(tokens[:, 1, :, :m.memory_dim]) > 0


def test_reverse_digits_is_involution():
    r = reverse_digits(3, 4)
    assert torch.equal(r[r], torch.arange(4096))
    assert int(r[1]) == 256 and int(r[16]) == 16


def test_serving_layout_gives_identical_route_tokens():
    m = tiny_memory()
    tables = permute_tables(m)
    codes = torch.randint(0, 16, (3, 20, m.num_routes), dtype=torch.uint8)
    want = m._lookup_route_chunk(codes, None, 0, m.num_routes, None, None)
    got = serving_route_tokens(tables, codes, m.ngrams, m.bits_per_route)
    assert torch.equal(got, want)


def test_candidates_are_contiguous_and_contain_next_rows():
    m = tiny_memory()
    tables = permute_tables(m)
    codes = torch.randint(0, 16, (2, 12, m.num_routes), dtype=torch.uint8)
    full = m._lookup_route_chunk(codes, None, 0, m.num_routes, None, None)
    for t in range(2, 11):  # codes up to t known; predict rows of t+1
        cands = candidate_rows(tables, codes[:, t - 1:t + 1], m.ngrams, m.bits_per_route)
        for order, block in cands.items():
            assert block.shape == (2, m.num_routes, 16, m.memory_dim)
        rows = select_candidates(cands, codes[:, t + 1], m.ngrams)
        assert torch.equal(rows, full[:, t + 1])
