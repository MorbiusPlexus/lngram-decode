"""M5.T3 chain: fixed-shape verify == decode_step; live speculative decoding == plain greedy (fp32)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drafting.cpu_drafter import DraftOut, RefDrafter
from drafting.export_drafter import export, load_bundle
from drafting.live_spec import (HostExactRows, Verifier, compare_exact, greedy_with_margins,
                                speculative_generate, static_verify)
from ffnmem.infer import DecodeState, decode_step, greedy_generate
from tests.test_live_export import make_model, make_posthoc, random_sketch, round_tables_fp16


@pytest.mark.parametrize("variant", ["wave1", "sdb", "sdc", "sda"])
def test_static_verify_equals_decode_step(variant):
    model = make_model(variant)
    x = torch.randint(0, 256, (1, 30))
    a = DecodeState.empty(model, 1, 40)
    b = DecodeState.empty(model, 1, 40)
    for lo, hi in [(0, 4), (4, 5), (5, 9), (9, 13), (13, 17)]:  # starts at L = 0, then n = 4 / 1 blocks
        ref = decode_step(model, a, x[:, lo:hi])
        out = static_verify(model, b, x[:, lo:hi], torch.tensor(lo))
        b.length = hi
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    for p in a.codes:
        assert torch.equal(a.codes[p][:, :17], b.codes[p][:, :17])
    torch.testing.assert_close(a.k[1][:, :, :17], b.k[1][:, :, :17])
    # rollback then a different block
    a.truncate(10)
    b.length = 10
    ver = Verifier(model, b, 4, "static")
    ref = decode_step(model, a, x[:, 20:24])
    torch.testing.assert_close(ver(x[0, 20:24].tolist()), ref, rtol=1e-5, atol=1e-5)
    assert b.length == 14


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    out = {}
    for v in ("sdb", "sdc", "sda"):
        m = make_model(v)
        round_tables_fp16(m)
        path = tmp_path_factory.mktemp("b") / f"{v}.bin"
        out[v] = (m, load_bundle(export(path, m, sketch=random_sketch(m, seed=2), shortlist_cap=16)))
    return out


@pytest.mark.parametrize("variant", ["sdb", "sdc", "sda"])
@pytest.mark.parametrize("gamma", [1, 3])
def test_live_equals_greedy(models, variant, gamma):
    model, bundle = models[variant]
    d = RefDrafter(bundle)
    for seed, P in [(0, 9), (1, 1)]:
        torch.manual_seed(seed)
        prompt = torch.randint(0, 256, (P,))
        ref, _ = greedy_generate(model, prompt[None], 24)
        for verifier in ("eager", "static"):
            toks, st = speculative_generate(model, d, prompt, 24, gamma, verifier=verifier)
            assert toks == ref[0].tolist(), (variant, gamma, P, verifier)
            assert sum(a + 1 for a in st["accepted"]) >= 24 and st["blocks"] == len(st["accepted"])
            assert sum(st["branch"].values()) == st["blocks"] - 1 or sum(a + 1 for a in st["accepted"]) > 24


class OracleDrafter:
    """Proposes the target's greedy continuation; at positions in ``wrong`` its top-1 is wrong and
    the right token is the runner-up (exercises acceptance, main and repair branches)."""

    name = "oracle"

    def __init__(self, seq: list, wrong=(), vocab: int = 256):
        self.seq, self.wrong, self.V = list(seq), set(wrong), vocab
        self.cfg = {"layer": 1}
        self._t: list = []

    def reset(self):
        self._t = []

    @property
    def length(self):
        return len(self._t)

    @property
    def tokens(self):
        return list(self._t)

    def committed_codes(self):
        return np.zeros((len(self._t), 4), np.uint8)

    def truncate(self, n):
        del self._t[n:]

    def commit(self, tokens, codes=None):
        self._t.extend(int(x) for x in tokens)

    def draft(self, gamma, hyp=(), topk=None, last_codes=False):
        ctx = self._t + list(hyp)
        ids = np.zeros((gamma, 3), np.int64)
        toks = []
        for i in range(gamma):
            p = len(ctx)
            on = ctx == self.seq[:p] and p < len(self.seq)
            true = self.seq[p] if on else 0
            top = (true + 1) % self.V if (on and p in self.wrong) else true
            ids[i] = [top, true if top != true else (true + 2) % self.V, (true + 3) % self.V]
            toks.append(top)
            ctx.append(top)
        return DraftOut(toks, ids, np.zeros((gamma, 3), np.float32), np.zeros((0, 4), np.uint8))


def test_oracle_drafter_hits_branches():
    model = make_model("sdb")
    prompt = torch.randint(0, 256, (6,))
    ref, _ = greedy_generate(model, prompt[None], 60)
    seq = prompt.tolist() + ref[0].tolist()
    toks, st = speculative_generate(model, OracleDrafter(seq), prompt, 40, 3)
    assert toks == ref[0, :40].tolist()
    assert all(a == 3 for a in st["accepted"]) and st["branch"]["main"] == st["blocks"] - 1
    # wrong top-1 at the first draft position of some blocks -> repair branch
    wrong = {10, 15, 20, 25, 30}  # first draft positions: 6, 10 (repair -> realigned by one), 15, 20, ...
    toks, st = speculative_generate(model, OracleDrafter(seq, wrong), prompt, 40, 3)
    assert toks == ref[0, :40].tolist()
    assert st["branch"]["repair"] >= 3 and 0 in st["accepted"]
    toks, st = speculative_generate(model, OracleDrafter(seq, wrong), prompt, 40, 3, branches=False)
    assert toks == ref[0, :40].tolist() and st["branch"]["miss"] == st["blocks"] - 1


def test_host_rows_and_target_codes(models):
    model, bundle = models["sdc"]
    rows = HostExactRows(model, 1)
    prompt = torch.randint(0, 256, (7,))
    ref, _ = greedy_generate(model, prompt[None], 20)
    toks, _ = speculative_generate(model, RefDrafter(bundle), prompt, 20, 3, row_source=rows)
    assert toks == ref[0].tolist()
    assert rows.hits > 0 and rows.misses == 0  # SD-C: drafter codes == target codes
    # post-hoc drafter on an unmodified (hidden-router) target, history from the target's codes
    target, ph = make_posthoc(ssm_dim=8)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        b = load_bundle(export(Path(tmp) / "p.bin", target, posthoc=ph), mmap=False)
    ref, _ = greedy_generate(target, prompt[None], 20)
    rows = HostExactRows(target, 1)
    for tc in (True, False):
        toks, _ = speculative_generate(target, RefDrafter(b), prompt, 20, 2, target_codes=tc, row_source=rows)
        assert toks == ref[0].tolist()
    assert rows.misses > 0  # own codes differ from the hidden router's -> synchronous fallback


def test_greedy_margins_and_compare():
    model = make_model("sdb")
    prompt = torch.randint(0, 256, (5,))
    toks, margins = greedy_with_margins(model, prompt, 12)
    ref, _ = greedy_generate(model, prompt[None], 12)
    assert toks == ref[0].tolist() and (margins >= 0).all()
    assert compare_exact(toks, toks, margins)["match"]
    bad = list(toks)
    bad[4] = (bad[4] + 1) % 256
    e = compare_exact(bad, toks, margins)
    assert not e["match"] and e["first_mismatch"] == 4 and e["near_tie"] == bool(margins[4] < 1e-2)
