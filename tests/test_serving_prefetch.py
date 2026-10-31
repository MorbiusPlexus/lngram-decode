"""M4.T2: B1 / B4 host-RAM serving is bitwise equal to B0 (table on the device), CPU."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ffnmem.config import load_config
from ffnmem.infer import DecodeState, decode_step
from ffnmem.model import Transformer
from serving.prefetch import (CandidatePrefetcher, HostGather, HostTables, LayerTable, ZeroCopyGather,
                              offload_tables, restore_tables)

CFG = Path(__file__).resolve().parents[1] / "configs"
VARIANTS = {
    "wave1": ("tiny_cpu.yaml", []),
    "sdb": ("tiny_cpu_sd.yaml", ["model.memory.router_input=embed2", "model.ssm=null"]),
    "sdc": ("tiny_cpu_sd.yaml", []),
    "wave1_2mem": ("tiny_cpu.yaml", ["model.memory_layers=[1,3]"]),
}


def make(variant: str) -> Transformer:
    name, over = VARIANTS[variant]
    torch.manual_seed(0)
    model = Transformer(load_config(CFG / name, over)["model"]).eval()
    torch.manual_seed(1)
    for b in model.blocks:
        if b.memory is not None:
            m = b.memory.module
            torch.nn.init.normal_(m.readout_o_proj.weight, std=0.05)
            torch.nn.init.normal_(m.short_conv.conv.weight, std=0.2)
            for t in m.tables.values():
                torch.nn.init.normal_(t.weight)  # rows must matter
    return model


def run(model, prompts, steps, provider=None):
    st = DecodeState.empty(model, prompts.shape[0], prompts.shape[1] + steps)
    logits = [decode_step(model, st, prompts, row_provider=provider)]
    for _ in range(steps - 1):
        nxt = logits[-1][:, -1].argmax(-1)
        logits.append(decode_step(model, st, nxt[:, None], row_provider=provider))
    return torch.cat(logits, 1)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("threaded", [False, True])
def test_b1_b4_bitwise_equal_b0(variant, threaded):
    model = make(variant)
    prompts = torch.randint(0, 256, (2, 5))
    ref = run(model, prompts, 22)
    host = HostTables.from_model(model)
    offload_tables(model)
    b1 = HostGather(host)
    b4 = CandidatePrefetcher(host, threaded=threaded)
    try:
        assert torch.equal(run(model, prompts, 22, b1), ref)
        assert torch.equal(run(model, prompts, 22, b4), ref)
    finally:
        b4.close()
    s = b4.stats.summary()
    n_mem = len(host)
    # first run of b4: 1 prefill gather + 21 hits per layer; no stale history
    assert s["hits"] == 21 * n_mem and s["misses"] == 0 and s["stale"] == 0
    assert s["tokens"] == 2 * (5 + 21)


@pytest.mark.parametrize("plen", [1, 2])
def test_positions_0_and_1(plen):
    """Prompts shorter than the n-gram orders: zero rows exactly as in the v2 lookup."""
    model = make("sdc")
    prompts = torch.randint(0, 256, (2, plen))
    ref = run(model, prompts, 6)
    host = HostTables.from_model(model)
    offload_tables(model)
    b4 = CandidatePrefetcher(host, threaded=False)
    assert torch.equal(run(model, prompts, 6, b4), ref)
    assert b4.stats.hits == 5


def test_single_token_prompt_from_position_0():
    model = make("wave1")
    prompts = torch.randint(0, 256, (3, 1))
    ref = run(model, prompts, 4)
    host = HostTables.from_model(model)
    b1 = HostGather(host)
    assert torch.equal(run(model, prompts, 4, b1), ref)


def test_rollback_and_new_state_are_misses_not_wrong():
    model = make("sdb")
    x = torch.randint(0, 256, (2, 20))
    st = DecodeState.empty(model, 2, 20)
    want = []
    with torch.no_grad():
        full = model(x)
    host = HostTables.from_model(model)
    offload_tables(model)
    b4 = CandidatePrefetcher(host, threaded=True)
    decode_step(model, st, x[:, :6], row_provider=b4)
    for t in range(6, 12):
        want.append(decode_step(model, st, x[:, t:t + 1], row_provider=b4))
    st.truncate(9)  # roll back; the staged position (12) no longer matches
    for t in range(9, 20):
        want.append(decode_step(model, st, x[:, t:t + 1], row_provider=b4))
    b4.close()
    got = torch.cat(want[:3] + want[6:], 1)
    torch.testing.assert_close(got, full[:, 6:], rtol=1e-4, atol=1e-4)
    assert b4.stats.misses == 1 and b4.stats.stale == 0


def test_candidates_cover_next_rows():
    model = make("wave1")
    m = model.blocks[1].memory.module
    lt = LayerTable.from_module(m)
    codes = torch.randint(0, 16, (2, 10, m.num_routes), dtype=torch.uint8)
    full = m._lookup_route_chunk(codes, None, 0, m.num_routes, None, None)
    for p in range(0, 10):
        hist = codes[:, max(0, p - lt.history):p]
        cand = lt.gather(lt.candidate_index(hist, p))  # [B, R, len, 16, D]
        rows = cand.gather(3, codes[:, p].long()[:, :, None, None, None].expand(-1, -1, len(lt.ngrams), 1, lt.dim))
        assert torch.equal(rows[:, :, :, 0].flatten(2), full[:, p])
        exact = lt.gather(lt.exact_index(codes[:, max(0, p - lt.history):p + 1], 1))
        assert torch.equal(exact.flatten(3)[:, 0], full[:, p])


def test_byte_accounting():
    model = make("wave1")
    m = model.blocks[1].memory.module
    host = HostTables.from_model(model)
    lt = host[1]
    assert lt.nbytes == (m.num_routes * (16 ** 2 + 16 ** 3) + 1) * m.memory_dim * 4
    prompts = torch.randint(0, 256, (2, 4))
    b1, b4 = HostGather(host), CandidatePrefetcher(host, threaded=False)
    run(model, prompts, 5, b1)
    run(model, prompts, 5, b4)
    row = m.memory_dim * 4
    k = len(m.ngrams)
    exact = lambda n: 2 * n * m.num_routes * k * row  # noqa: E731
    assert b1.stats.h2d_bytes == exact(4) + 4 * exact(1)
    cand = 2 * m.num_routes * k * 16 * row
    assert b4.stats.h2d_bytes == exact(4) + 5 * cand  # prefill gather + one candidate set per call
    assert b1.stats.summary()["h2d_bytes_per_token"] == b1.stats.h2d_bytes / 16


def test_offload_and_restore():
    model = make("wave1")
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        ref = model(x)
    host = HostTables.from_model(model)
    assert offload_tables(model) > 0
    assert model.blocks[1].memory.module.tables["ngram_2"].weight.device.type == "meta"
    restore_tables(model, host)
    with torch.no_grad():
        assert torch.equal(model(x), ref)


def test_zero_copy_not_implemented():
    with pytest.raises(NotImplementedError):
        ZeroCopyGather()
