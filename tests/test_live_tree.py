"""M5.T3 stretch: tree mask / positions / ancestor n-gram codes; tree verification == chain decoding."""

from __future__ import annotations

import pytest
import torch

from drafting.cpu_drafter import RefDrafter
from drafting.export_drafter import export, load_bundle
from drafting.live_spec import speculative_generate
from drafting.tree import (TreeSpec, chain_tree, compact_path, default_tree, tree_accept, tree_attention_mask,
                           tree_generate, tree_ngram_codes, tree_positions, tree_verify_step)
from ffnmem.infer import DecodeState, decode_step, greedy_generate
from tests.test_live_export import make_model, random_sketch, round_tables_fp16
from tests.test_live_spec import OracleDrafter

PARENT = (-1, 0, 1, 0, 3, 1)  # 0 -> {1, 3}; 1 -> {2, 5}; 3 -> {4}


def test_mask_positions_and_ancestor_codes():
    m = tree_attention_mask(PARENT, 2)
    expect = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 0, 0, 0],
                           [1, 1, 1, 0, 0, 1, 0, 0],
                           [1, 1, 1, 0, 0, 1, 1, 0],
                           [1, 1, 1, 1, 0, 0, 0, 1]], dtype=torch.bool)
    assert torch.equal(m, expect)
    assert tree_positions(PARENT, 5).tolist() == [5, 6, 7, 6, 7, 7]
    codes = torch.tensor([0, 10, 20, 30, 40, 50]).view(1, 6, 1)
    hist = torch.tensor([100, 101]).view(1, 2, 1)
    ctx, ok = tree_ngram_codes(PARENT, codes, hist, 2)
    assert ctx[0, :, :, 0].tolist() == [[100, 101, 0], [101, 0, 10], [0, 10, 20], [101, 0, 30], [0, 30, 40],
                                        [0, 10, 50]]
    assert ok.all()
    ctx, ok = tree_ngram_codes(PARENT, codes, None, 2)
    assert ok.tolist() == [[False, False, True], [False, True, True], [True, True, True], [False, True, True],
                           [True, True, True], [True, True, True]]
    assert ctx[0, 1, :, 0].tolist() == [0, 0, 10] and ctx[0, 4, :, 0].tolist() == [0, 30, 40]
    with pytest.raises(ValueError):
        TreeSpec((-1, 2, 0), (0, 0, 0))


def test_tree_accept():
    toks = [7, 1, 2, 3, 4, 5]
    #           node:  0  1  2  3  4  5   argmax at each node = prediction of its child
    path, new = tree_accept(PARENT, toks, [3, 9, 9, 4, 8, 9])
    assert path == [0, 3, 4] and new == [3, 4, 8]
    path, new = tree_accept(PARENT, toks, [1, 5, 9, 9, 9, 6])
    assert path == [0, 1, 5] and new == [1, 5, 6]
    path, new = tree_accept(PARENT, toks, [9, 9, 9, 9, 9, 9])
    assert path == [0] and new == [9]


@pytest.mark.parametrize("variant", ["wave1", "sdb", "sdc", "sda"])
def test_tree_nodes_equal_chain_decoding_of_their_paths(variant):
    model = make_model(variant)
    prompt = torch.randint(0, 256, (1, 9))
    nodes = torch.randint(0, 256, (1, 6))
    nodes[0, 0] = prompt[0, -1]
    st = DecodeState.empty(model, 1, 30)
    decode_step(model, st, prompt[:, :-1])
    L = st.length
    out = tree_verify_step(model, st, nodes, PARENT)
    for i, path in enumerate([[0], [0, 1], [0, 1, 2], [0, 3], [0, 3, 4], [0, 1, 5]]):
        ch = DecodeState.empty(model, 1, 30)
        decode_step(model, ch, prompt[:, :-1])
        ref = decode_step(model, ch, nodes[:, path])
        torch.testing.assert_close(out[0, i], ref[0, -1], rtol=1e-4, atol=1e-4)
    # accept 0 -> 3 -> 4, compact, continue: equals chain decoding of the accepted tokens
    compact_path(st, L, [0, 3, 4])
    assert st.length == L + 3
    nxt = torch.randint(0, 256, (1, 2))
    got = decode_step(model, st, nxt)
    ch = DecodeState.empty(model, 1, 30)
    ref = decode_step(model, ch, torch.cat((prompt[:, :-1], nodes[:, [0, 3, 4]], nxt), 1))
    torch.testing.assert_close(got, ref[:, -2:], rtol=1e-4, atol=1e-4)
    for p in st.codes:
        assert torch.equal(st.codes[p][:, :st.length], ch.codes[p][:, :st.length])


@pytest.fixture(scope="module")
def drafters(tmp_path_factory):
    out = {}
    for v in ("sdb", "sdc", "sda"):
        m = make_model(v)
        round_tables_fp16(m)
        b = load_bundle(export(tmp_path_factory.mktemp("t") / f"{v}.bin", m, sketch=random_sketch(m, seed=4),
                               shortlist_cap=16))
        out[v] = (m, b)
    return out


@pytest.mark.parametrize("variant", ["sdb", "sdc", "sda"])
def test_degenerate_tree_equals_chain(drafters, variant):
    model, bundle = drafters[variant]
    prompt = torch.randint(0, 256, (7,))
    ref, _ = greedy_generate(model, prompt[None], 20)
    for gamma in (1, 3):
        chain, cst = speculative_generate(model, RefDrafter(bundle), prompt, 20, gamma, branches=False)
        tree, tst = tree_generate(model, RefDrafter(bundle), prompt, 20, chain_tree(gamma + 1))
        assert tree == chain == ref[0].tolist()
        assert tst["accepted"] == cst["accepted"]
    tree, _ = tree_generate(model, RefDrafter(bundle), prompt, 20, default_tree())
    assert tree == ref[0].tolist()


def test_tree_with_oracle_accepts_runner_up_paths():
    model = make_model("sdc")
    prompt = torch.randint(0, 256, (6,))
    ref, _ = greedy_generate(model, prompt[None], 60)
    seq = prompt.tolist() + ref[0].tolist()
    toks, st = tree_generate(model, OracleDrafter(seq), prompt, 40, default_tree())
    assert toks == ref[0, :40].tolist() and all(a == 4 for a in st["accepted"][:-1])
    # wrong top-1 at the first draft position of every block: the runner-up branch (nodes 5 -> 6) is accepted
    wrong = set(range(6, 60))
    toks, st = tree_generate(model, OracleDrafter(seq, wrong), prompt, 40, default_tree())
    assert toks == ref[0, :40].tolist() and max(st["accepted"]) == 1
