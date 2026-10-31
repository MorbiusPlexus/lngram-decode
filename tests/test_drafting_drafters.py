"""Drafters: oracle acceptance, batched = per-row loop, live-information rule, online counts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drafting.baselines import PromptLookupDrafter, TokenNgramDrafter, build_ngram
from drafting.build_sketches import build_sketch
from drafting.common import SeqRecord, accepted_lengths, full_rows
from drafting.drafters import OracleDrafter, PosthocNeuralDrafter, SDHeadDrafter, SketchDrafter
from drafting.offline_acceptance import generate_records, per_prompt_accepted
from drafting.posthoc_model import PosthocDrafter
from drafting.sketches import Sketch
from tests.test_infer import make

PLEN, NEW = 12, 10


def records(model, n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    prompts = torch.randint(4, 256, (n, PLEN), generator=g).numpy()
    return generate_records(model, prompts, NEW, "open", np.arange(n), batch=n)


def posthoc(model, ssm_dim=0):
    torch.manual_seed(5)
    d = PosthocDrafter(model, q_heads=2, head_dim=16, ssm_dim=ssm_dim, ssm_chunk=8).eval()
    torch.nn.init.normal_(d.router.weight, std=0.5)
    return d


def sketch_for(model, seed=1):
    g = torch.Generator().manual_seed(seed)
    return build_sketch(model, [torch.randint(4, 64, (8, 33), generator=g) for _ in range(4)])


@pytest.mark.parametrize("variant", ["wave1", "sdc", "sda"])
def test_record_codes_equal_full_forward(variant):
    model = make(variant)
    for rec in records(model):
        mem = model.blocks[1].memory
        mem.capture_route_logits = True
        with torch.no_grad():
            model(torch.as_tensor(rec.tokens)[None])
        mem.capture_route_logits = False
        full = mem.module.quantize_route_logits(mem.last_route_logits)[0].numpy()
        assert rec.codes[1].shape == (PLEN + NEW, 4) and np.array_equal(rec.codes[1], full)


@pytest.mark.parametrize("variant", ["wave1", "sdc"])
def test_oracle_drafter_accepts_gamma_everywhere(variant):
    model = make(variant)
    recs = records(model)
    oracle = OracleDrafter(model)
    for rec in recs:
        for gamma in (2, 4):
            acc = accepted_lengths(rec, oracle.drafts(rec, gamma))
            assert acc.tolist() == [min(gamma, rec.n_cont - j) for j in range(rec.n_cont)]
            assert (acc[full_rows(rec, gamma)] == gamma).all()
    pp = per_prompt_accepted(oracle, recs, (2, 4, 6))
    assert all((pp[g] == g).all() for g in (2, 4, 6))


# ------------------------------------------------------------------ batched = per-row loop
def ref_head(model, rec, j, gamma, verified):
    L = rec.prompt_len + j
    x, out = list(rec.tokens[:L]), []
    m = model.blocks[model.draft_pos].memory
    with torch.no_grad():
        for _ in range(gamma):
            idx = torch.tensor(x)[None]
            emb = model.tok_emb(idx).detach()
            s = model.ssm(emb) if model.ssm is not None else None
            q = model.draft_route_logits(emb, s)
            codes = m.module.quantize_route_logits(q)
            if verified == "exact" and m.router_input == "hidden":
                codes[0, :L] = torch.as_tensor(rec.codes[model.draft_pos][:L])
            logits = model._draft_chunk(emb, s, q, codes, torch.tensor([0]), torch.tensor([len(x) - 1]))
            x.append(int(logits.argmax()))
            out.append(x[-1])
    return out


def ref_posthoc(d, rec, j, gamma, verified):
    L = rec.prompt_len + j
    x, out = list(rec.tokens[:L]), []
    with torch.no_grad():
        for _ in range(gamma):
            idx = torch.tensor(x)[None]
            codes = None
            if verified == "exact":
                codes = d.codes(idx)
                codes[0, :L] = torch.as_tensor(rec.codes[d.layer][:L])
            x.append(int(d(idx, torch.tensor([0]), torch.tensor([len(x) - 1]), codes=codes).argmax()))
            out.append(x[-1])
    return out


@pytest.mark.parametrize("variant,verified", [("sdb", "exact"), ("sdc", "exact"), ("sda", "exact"),
                                              ("sda", "own")])
def test_sd_head_batched_equals_per_row_loop(variant, verified):
    model = make(variant)
    torch.manual_seed(7)
    for p in model.draft.parameters():
        torch.nn.init.normal_(p, std=0.3)
    rec = records(model, n=1)[0]
    got = SDHeadDrafter(model, verified=verified).drafts(rec, 4)
    for j in range(rec.n_cont):
        assert got[j].tolist() == ref_head(model, rec, j, 4, verified), j


@pytest.mark.parametrize("ssm_dim,verified", [(0, "exact"), (0, "own"), (8, "exact")])
def test_posthoc_batched_equals_per_row_loop(ssm_dim, verified):
    model = make("wave1")
    d = posthoc(model, ssm_dim)
    rec = records(model, n=1)[0]
    got = PosthocNeuralDrafter(d, verified=verified).drafts(rec, 4)
    for j in range(rec.n_cont):
        assert got[j].tolist() == ref_posthoc(d, rec, j, 4, verified), j


# ------------------------------------------------------------------ level 0
def test_sketch_drafter_first_draft_is_level0_argmax():
    model = make("sdb")
    sk = sketch_for(model)
    rec = records(model, n=1)[0]
    d = SketchDrafter(sk, model).drafts(rec, 3)
    codes = rec.codes[1]
    for j in range(rec.n_cont):
        t = rec.prompt_len + j - 1
        rows = {n: sk.rows_at(codes[: t + 1], n)[-1:] for n in sk.ngrams}
        dense = sk.decode_dense(rows, smoothing=0.0)[0]
        if dense.max() > 0:
            assert dense[d[j, 0]] == dense.max()
        else:
            assert d[j, 0] == -1


def test_sketch_drafter_without_token_router_drafts_one_token():
    model = make("wave1")
    sk = sketch_for(model)
    rec = records(model, n=1)[0]
    d = SketchDrafter(sk, model).drafts(rec, 3)
    assert (d[:, 1:] == -1).all()
    d2 = SketchDrafter(sk, model, router=posthoc(model)).drafts(rec, 3)
    assert np.array_equal(d2[:, 0], d[:, 0]) and (d2[:, 1:] >= 0).any()


def test_online_counts_alone_continue_a_repeated_pattern():
    model = make("sdb")
    empty = Sketch.empty(4, 4, (2, 3), 1, k=16, vocab=256)
    toks = np.array([7, 30, 41, 52, 63] * 8)
    rec = SeqRecord(toks, 20)
    plain = SketchDrafter(empty, model).drafts(rec, 4)
    assert (plain == -1).all()
    online = SketchDrafter(empty, model, online=True).drafts(rec, 4)
    acc = accepted_lengths(rec, online)
    assert (acc[full_rows(rec, 4)] == 4).all()


# ------------------------------------------------------------------ live-information rule
def perturb(rec: SeqRecord, L: int, seed: int) -> SeqRecord:
    rng = np.random.default_rng(seed)
    toks = rec.tokens.copy()
    toks[L:] = rng.integers(4, 256, len(toks) - L)
    codes = {p: c.copy() for p, c in rec.codes.items()}
    for c in codes.values():
        c[L:] = rng.integers(0, 16, c[L:].shape)
    return SeqRecord(toks, rec.prompt_len, rec.text_type, codes, rec.doc_id)


def _drafters(variant):
    model = make(variant)
    g = torch.Generator().manual_seed(4)
    ngram = build_ngram([torch.randint(4, 256, (4, 65), generator=g)], 256, buckets={3: 512, 2: 1 << 16, 1: 256})
    out = [PromptLookupDrafter(), TokenNgramDrafter(ngram)]
    sk = sketch_for(model)
    if variant == "wave1":
        ph = posthoc(model, 8)
        out += [SketchDrafter(sk, model, router=ph, online=True), PosthocNeuralDrafter(ph, online=True),
                PosthocNeuralDrafter(ph, verified="own")]
    else:
        out += [SketchDrafter(sk, model, online=True), SDHeadDrafter(model, online=True),
                SDHeadDrafter(model, verified="own")]
    return model, out


@pytest.mark.parametrize("variant", ["wave1", "sdc", "sda"])
def test_live_information_rule(variant):
    model, drafters = _drafters(variant)
    rec = records(model, n=1)[0]
    rec.tokens[3:9] = rec.tokens[PLEN:PLEN + 6]  # give PLD / online counts something to match
    for drafter in drafters:
        base = drafter.drafts(rec, 4)
        for j in (0, 4, 9):
            L = rec.prompt_len + j
            other = drafter.drafts(perturb(rec, L, seed=j), 4)
            assert np.array_equal(other[j], base[j]), (type(drafter).__name__, j)
