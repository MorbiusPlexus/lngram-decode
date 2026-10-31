"""M4.T2 benchmark pieces on CPU: the fixed-shape (graphable) decode step and a bench smoke run."""

from __future__ import annotations

import pytest
import torch

from ffnmem.infer import DecodeState, decode_step
from serving.bench_prefetch import StaticDecoder, run_bench
from serving.exactness import run_exactness
from serving.prefetch import CandidatePrefetcher, HostTables, offload_tables
from tests.test_serving_prefetch import make


def eager_ref(model, prompts, steps):
    st = DecodeState.empty(model, prompts.shape[0], prompts.shape[1] + steps)
    lg = decode_step(model, st, prompts)
    out = []
    for _ in range(steps):
        nxt = lg[:, -1].argmax(-1)
        lg = decode_step(model, st, nxt[:, None])
        out.append(lg)
    return torch.cat(out, 1)


def static_run(model, prompts, steps, mode, pf=None):
    b, p = prompts.shape
    st = DecodeState.empty(model, b, p + steps)
    lg = decode_step(model, st, prompts, row_provider=pf)
    dec = StaticDecoder(model, st, mode, pf)
    dec.start(p, lg[:, -1].argmax(-1))
    return torch.cat([dec.step().clone() for _ in range(steps)], 1), st


@pytest.mark.parametrize("variant", ["wave1", "sdc", "wave1_2mem"])
def test_static_step_matches_decode_step_and_b4_is_bitwise_b0(variant):
    model = make(variant)
    prompts = torch.randint(0, 256, (2, 12))
    ref = eager_ref(model, prompts, 20)
    s0, st0 = static_run(model, prompts, 20, "b0")
    torch.testing.assert_close(s0, ref, rtol=1e-4, atol=1e-4)
    assert torch.equal(s0.argmax(-1), ref.argmax(-1))
    host = HostTables.from_model(model)
    offload_tables(model)
    pf = CandidatePrefetcher(host, threaded=True)
    s4, st4 = static_run(model, prompts, 20, "b4", pf)
    pf.close()
    assert torch.equal(s4, s0)
    assert st4.length == 32 and pf.stats.stale == 0 and pf.stats.hits == 20 * len(host)


def test_static_needs_history():
    model = make("wave1")
    st = DecodeState.empty(model, 1, 8)
    dec = StaticDecoder(model, st)
    with pytest.raises(ValueError):
        dec.start(1, torch.zeros(1, dtype=torch.long))


def test_bench_smoke():
    model = make("sdb")
    prompts = torch.randint(0, 256, (2, 12))
    rows = run_bench(model, prompts, [1, 2], ["b0", "b0g", "b1", "b4", "b4g"], warmup=2, tokens=4, repeats=2,
                     log=lambda s: None)
    assert len(rows) == 10
    by = {(r["mode"], r["batch"]): r for r in rows}
    assert by[("b4", 2)]["hits"] == 8 and by[("b4", 2)]["misses"] == 0 and by[("b4g", 2)]["stale"] == 0
    assert by[("b1", 1)]["h2d_bytes_per_token"] > 0
    # restored tables: B0 still works afterwards
    st = DecodeState.empty(model, 1, 4)
    decode_step(model, st, prompts[:1, :4])


def test_exactness_runner():
    model = make("wave1")
    prompts = torch.randint(0, 256, (4, 6))
    res = run_exactness(model, prompts, 12, batch=2)
    for k in ("b1", "b4"):
        assert res[k]["exact"] and res[k]["tokens_identical_frac"] == 1.0 and res[k]["max_abs_logit_diff"] == 0.0
