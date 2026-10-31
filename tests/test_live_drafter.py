"""M5.T2: stateful commit / draft / rollback of the CPU drafter; C++ extension vs the reference."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drafting.cpu_drafter import CppDrafter, RefDrafter, extension_error, load_extension
from drafting.export_drafter import export, load_bundle
from tests.test_live_export import make_model, make_posthoc, random_sketch, round_tables_fp16


@pytest.fixture(scope="module", params=["sdb", "sdc", "sda"])
def bundle(request, tmp_path_factory):
    model = make_model(request.param)
    round_tables_fp16(model)
    path = tmp_path_factory.mktemp("b") / f"{request.param}.bin"
    return load_bundle(export(path, model, sketch=random_sketch(model, seed=1), shortlist_cap=16))


def chain_by_trace(d: RefDrafter, prefix: list, gamma: int) -> list:
    """Greedy chain recomputed from scratch with teacher-forced traces (no state reuse)."""
    seq = list(prefix)
    out = []
    for _ in range(gamma):
        _, ids, lg = d.trace(seq)
        nxt = int(ids[-1][np.argmax(lg[-1])])
        out.append(nxt)
        seq.append(nxt)
    return out


def test_draft_equals_teacher_forced_chain(bundle):
    d = RefDrafter(bundle)
    prefix = torch.randint(0, 256, (20,)).tolist()
    d.commit(prefix)
    out = d.draft(4)
    assert out.tokens == chain_by_trace(RefDrafter(bundle), prefix, 4)
    assert out.topk_ids.shape == (4, 3) and (out.topk_ids[:, 0] == out.tokens).all()
    assert (np.diff(out.topk_logits, axis=1) <= 0).all()
    assert out.codes.shape == (3, 4)  # last drafted token not advanced
    assert d.draft(4, last_codes=True).codes.shape == (4, 4)


def test_draft_does_not_mutate_and_rollback_is_exact(bundle):
    d = RefDrafter(bundle)
    a = torch.randint(0, 256, (15,)).tolist()
    b = torch.randint(0, 256, (7,)).tolist()
    d.commit(a)
    first = d.draft(3)
    codes_a = d.committed_codes().copy()
    d.draft(5, hyp=b)
    assert d.length == 15 and d.draft(3).tokens == first.tokens
    d.commit(b)
    with_b = d.draft(3)
    d.truncate(15)
    np.testing.assert_array_equal(d.committed_codes(), codes_a)
    assert d.draft(3).tokens == first.tokens
    # hyp == commit
    assert d.draft(3, hyp=b).tokens == with_b.tokens
    np.testing.assert_array_equal(d.draft(3, hyp=b).topk_logits, with_b.topk_logits)
    # commit in pieces == commit at once
    e = RefDrafter(bundle)
    for i in range(0, 15, 4):
        e.commit(a[i:i + 4])
    np.testing.assert_array_equal(e.committed_codes(), codes_a)
    assert e.draft(3).tokens == first.tokens
    d.reset()
    assert d.length == 0
    with pytest.raises(ValueError):
        d.draft(1)
    assert d.draft(2, hyp=a[:1]).tokens == chain_by_trace(RefDrafter(bundle), a[:1], 2)


def test_commit_with_target_codes():
    target, ph = make_posthoc()
    round_tables_fp16(target)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        b = load_bundle(export(Path(tmp) / "p.bin", target, posthoc=ph), mmap=False)
    d = RefDrafter(b)
    toks = torch.randint(0, 256, (12,)).tolist()
    codes = np.random.default_rng(0).integers(0, 16, (12, 4)).astype(np.uint8)
    d.commit(toks, codes=codes)
    np.testing.assert_array_equal(d.committed_codes(), codes)
    with torch.no_grad():
        ref = ph(torch.tensor([toks]), torch.zeros(1, dtype=torch.long), torch.tensor([11]),
                 codes=torch.from_numpy(codes.astype(np.int32))[None])
    out = d.draft(1, topk=1)
    assert out.tokens[0] == int(ref.argmax())
    np.testing.assert_allclose(out.topk_logits[0, 0], float(ref.max()), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------- C++ extension
def _ext():
    ext = load_extension()
    if ext is None:
        pytest.skip(f"C++ drafter extension not buildable here: {extension_error()}")
    return ext


@pytest.mark.parametrize("threads", [1, 3])
def test_cpp_trace_matches_reference(bundle, threads):
    _ext()
    ref, cpp = RefDrafter(bundle), CppDrafter(bundle, threads=threads)
    toks = torch.randint(0, 256, (1500,)).tolist()
    c_ref, i_ref, l_ref = ref.trace(toks)
    c_cpp, _, _ = cpp.trace(toks)
    assert (c_ref != c_cpp).mean() < 1e-3  # router sign flips only at |logit| ~ 1e-7
    _, i_cpp, l_cpp = cpp.trace(toks, codes=c_ref)
    np.testing.assert_array_equal(i_ref, i_cpp)
    fin = np.isfinite(l_ref)
    np.testing.assert_array_equal(fin, np.isfinite(l_cpp))
    np.testing.assert_allclose(l_cpp[fin], l_ref[fin], rtol=0, atol=1e-4)
    _, f_ref, fl_ref = ref.trace(toks[:200], full=True)
    _, f_cpp, fl_cpp = cpp.trace(toks[:200], codes=c_ref[:200], full=True)
    np.testing.assert_array_equal(f_ref, f_cpp)
    np.testing.assert_allclose(fl_cpp, fl_ref, rtol=0, atol=1e-4)


def test_cpp_stateful_api_matches_reference(bundle):
    _ext()
    ref, cpp = RefDrafter(bundle), CppDrafter(bundle, threads=2)
    a = torch.randint(0, 256, (30,)).tolist()
    for d in (ref, cpp):
        d.commit(a)
    np.testing.assert_array_equal(cpp.committed_codes(), ref.committed_codes())
    for hyp in ([], a[:3]):
        r, c = ref.draft(4, hyp=hyp, last_codes=True), cpp.draft(4, hyp=hyp, last_codes=True)
        assert r.tokens == c.tokens
        np.testing.assert_array_equal(r.topk_ids, c.topk_ids)
        np.testing.assert_allclose(r.topk_logits, c.topk_logits, atol=1e-4)
        np.testing.assert_array_equal(r.codes, c.codes)
    cpp.truncate(10)
    ref.truncate(10)
    assert cpp.length == 10 and cpp.tokens == ref.tokens
    assert cpp.draft(3).tokens == ref.draft(3).tokens
    codes = np.random.default_rng(1).integers(0, 16, (5, 4))
    ref.commit(a[10:15], codes=codes)
    cpp.commit(a[10:15], codes=codes)
    np.testing.assert_array_equal(cpp.committed_codes(), ref.committed_codes())
    assert cpp.draft(3).tokens == ref.draft(3).tokens
    cpp.reset()
    assert cpp.length == 0


def test_cpp_thread_count_does_not_change_results(bundle):
    _ext()
    toks = torch.randint(0, 256, (300,)).tolist()
    outs = [CppDrafter(bundle, threads=t).trace(toks) for t in (1, 2, 4)]
    for o in outs[1:]:
        for x, y in zip(outs[0], o):
            np.testing.assert_array_equal(x, y)
