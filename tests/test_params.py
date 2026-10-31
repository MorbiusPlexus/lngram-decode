from __future__ import annotations

from pathlib import Path

import pytest

from ffnmem.config import load_config
from ffnmem.memory.lngram_v2 import estimate_lngram_v2_parameters_per_layer
from ffnmem.params import check_match, model_counts, solve_dense_dff, solve_routes

CFG = Path(__file__).resolve().parents[1] / "configs"
RUNS = ["w1_mem_full", "w1_mem_cut_near", "w1_mem_cut_far", "w1_dense_full", "w1_dense_cut_near"]


@pytest.fixture(scope="module")
def counts():
    return {r: model_counts(load_config(CFG / f"{r}.yaml")["model"]) for r in RUNS}


def test_hand_computed_components(counts):
    # config S attention: q 1024x1024, k/v 1024x128 each, o 1024x1024 = 2.36M per layer
    assert counts["w1_mem_full"]["attention"] == 24 * (2 * 1024 * 1024 + 2 * 1024 * 128) == 24 * 2_359_296
    assert counts["w1_mem_full"]["ffn"] == 24 * 3 * 1024 * 2816 == 24 * 8_650_752
    # widened: q/o 1024x1536, k/v 1024x256 = 3.67M per layer
    assert counts["w1_mem_cut_near"]["attention"] == 24 * (2 * 1024 * 1536 + 2 * 1024 * 256) == 24 * 3_670_016
    assert counts["w1_dense_full"]["embedding"] == 32768 * 1024
    backbone = counts["w1_mem_full"]["attention"] + counts["w1_mem_full"]["ffn"]
    assert backbone == pytest.approx(264.5e6, rel=1e-3)


def test_cut_equals_widening(counts):
    freed = 8 * 3 * 1024 * 1280
    assert counts["w1_mem_full"]["ffn"] - counts["w1_mem_cut_near"]["ffn"] == freed
    assert counts["w1_mem_cut_near"]["attention"] - counts["w1_mem_full"]["attention"] == freed


def test_all_runs_matched(counts):
    ref = counts["w1_mem_full"]["non_embedding"]
    for r in RUNS:
        check_match(counts[r], ref)  # within 0.5%
    for r in ("w1_mem_cut_near", "w1_mem_cut_far"):
        assert counts[r]["non_embedding"] == ref


def test_memory_count_matches_vendored_estimate(counts):
    per_layer = estimate_lngram_v2_parameters_per_layer(
        hidden_size=1024, bits_per_route=4, num_routes=90, memory_dim=128, ngrams=(2, 3),
        readout_num_heads=16, readout_num_kv_heads=4, readout_head_dim=64, conv_kernel_size=4,
    )
    c = counts["w1_mem_full"]
    assert c["memory_tables"] + c["memory_readout"] == 2 * per_layer
    assert c["memory_tables"] == 2 * 90 * (16**2 + 16**3) * 128


def test_solvers_reproduce_configs(counts):
    ref = counts["w1_mem_full"]["non_embedding"]
    dense = load_config(CFG / "w1_dense_full.yaml")["model"]
    shift, _ = solve_dense_dff(dense, ref)
    assert shift == 0
    mem = load_config(CFG / "w1_mem_cut_near.yaml")["model"]
    routes, _ = solve_routes(mem, ref)
    assert routes == 90
