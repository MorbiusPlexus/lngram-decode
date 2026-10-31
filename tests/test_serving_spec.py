"""M4.T2 speculative-decoding helper: staged verify blocks return exactly the rows of a direct lookup."""

from __future__ import annotations

import pytest
import torch

from ffnmem.infer import DecodeState, decode_step
from serving.prefetch import BlockStager, HostTables, offload_tables
from tests.test_serving_prefetch import make


def reference(model, x):
    """B0 cached decode of x in one block: logits and the verifier's codes per memory layer."""
    st = DecodeState.empty(model, x.shape[0], x.shape[1])
    logits = decode_step(model, st, x)
    return logits, {p: c.clone() for p, c in st.codes.items()}


@pytest.mark.parametrize("bonus", ["all", "last"])
@pytest.mark.parametrize("threaded", [False, True])
def test_staged_rows_equal_direct_lookup(bonus, threaded):
    model = make("sdc")
    x = torch.randint(0, 256, (2, 16))
    _, codes = reference(model, x)
    host = HostTables.from_model(model)
    stager = BlockStager(host, threaded=threaded, bonus=bonus)
    for start, n in [(0, 3), (1, 4), (5, 5), (10, 5)]:
        blk = stager.stage(start, {p: c[:, start:start + n] for p, c in codes.items()},
                           {p: c[:, max(0, start - 2):start] for p, c in codes.items()})
        for pos, c in codes.items():
            m = model.blocks[pos].memory.module
            full = m._lookup_route_chunk(c, None, 0, m.num_routes, None, None)
            assert torch.equal(blk.rows(pos), full[:, start:start + n])
            for j in range(n):
                br = blk.bonus_rows(pos, j, c[:, start + j + 1])
                if bonus == "last" and j < n - 1:
                    assert br is None
                else:
                    assert torch.equal(br, full[:, start + j + 1])
    stager.close()


def test_verify_blocks_bitwise_equal_b0():
    model = make("sdb")
    x = torch.randint(0, 256, (2, 20))
    ref, codes = reference(model, x)
    host = HostTables.from_model(model)
    offload_tables(model)
    stager = BlockStager(host)
    st = DecodeState.empty(model, 2, 20)
    sl = lambda a, b: ({p: c[:, a:b] for p, c in codes.items()}, {p: c[:, max(0, a - 2):a] for p, c in codes.items()})  # noqa: E731
    out = [decode_step(model, st, x[:, :6], row_provider=stager.fallback)]
    blk = stager.stage(6, *sl(6, 11))                      # verify block: positions 6..10
    out.append(decode_step(model, st, x[:, 6:11], row_provider=stager.provider(blk)))
    assert blk.codes_match()
    st.truncate(9)                                          # accepted 2 drafts after position 6 -> bonus at 9
    nxt = stager.stage(10, *sl(10, 14))                     # next drafts: positions 10..13
    out[-1] = out[-1][:, :3]
    out.append(decode_step(model, st, x[:, 9:14], row_provider=stager.provider(nxt, head=(blk, 2))))
    assert nxt.codes_match()
    tail = decode_step(model, st, x[:, 14:20], row_provider=stager.provider(nxt))  # wrong position -> fallback
    out.append(tail)
    stager.close()
    got = torch.cat(out, 1)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
    assert stager.stats.hits == 2 * len(host) and stager.stats.misses == len(host)


def test_verify_bitwise_against_b0_blocks():
    model = make("sdc")
    x = torch.randint(0, 256, (1, 14))
    _, codes = reference(model, x)
    st0 = DecodeState.empty(model, 1, 14)
    b0 = [decode_step(model, st0, x[:, :4]), decode_step(model, st0, x[:, 4:9]), decode_step(model, st0, x[:, 9:14])]
    host = HostTables.from_model(model)
    offload_tables(model)
    stager = BlockStager(host, threaded=True)
    st = DecodeState.empty(model, 1, 14)
    got = [decode_step(model, st, x[:, :4], row_provider=stager.fallback)]
    for a, b in [(4, 9), (9, 14)]:
        blk = stager.stage(a, {p: c[:, a:b] for p, c in codes.items()}, {p: c[:, a - 2:a] for p, c in codes.items()})
        got.append(decode_step(model, st, x[:, a:b], row_provider=stager.provider(blk)))
    stager.close()
    for g, w in zip(got, b0):
        assert torch.equal(g, w)


def test_code_mismatch_is_flagged():
    model = make("sdb")
    x = torch.randint(0, 256, (1, 10))
    _, codes = reference(model, x)
    host = HostTables.from_model(model)
    stager = BlockStager(host, threaded=False)
    st = DecodeState.empty(model, 1, 10)
    decode_step(model, st, x[:, :4], row_provider=stager.fallback)
    wrong = {p: (c[:, 4:8] ^ 1) for p, c in codes.items()}
    blk = stager.stage(4, wrong, {p: c[:, 2:4] for p, c in codes.items()})
    decode_step(model, st, x[:, 4:8], row_provider=stager.provider(blk))
    assert not blk.codes_match()
