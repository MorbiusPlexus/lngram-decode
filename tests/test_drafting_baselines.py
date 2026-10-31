"""M3.T2: prompt-lookup and token n-gram drafters on hand-made sequences."""

from __future__ import annotations

import numpy as np
import torch

from drafting.baselines import (PromptLookupDrafter, TokenNgramDrafter, TokenNgramModel, build_ngram,
                                context_keys, hashed, previous_occurrence)
from drafting.common import SeqRecord, accepted_lengths


def rec_of(tokens, prompt_len):
    return SeqRecord(np.asarray(tokens, dtype=np.int64), prompt_len)


def test_previous_occurrence():
    s = np.array([1, 2, 3, 1, 2, 3, 1, 2])
    assert previous_occurrence(s, 1).tolist() == [-1, -1, -1, 0, 1, 2, 3, 4]
    assert previous_occurrence(s, 3).tolist() == [-1, -1, -1, -1, -1, 2, 3, 4]


def test_pld_copies_after_longest_most_recent_match():
    #         0  1  2  3  4  5  6  7  8  9 10 11 12
    tokens = [5, 6, 7, 8, 9, 1, 6, 7, 3, 4, 5, 6, 7]
    d = PromptLookupDrafter().drafts(rec_of(tokens, 12), 3)
    # context ends with (4, 5, 6): no earlier 3-gram, 2-gram (5, 6) at 0..1 beats the
    # more recent 1-gram 6 at 6 -> copy 7 8 9
    assert d[0].tolist() == [7, 8, 9]
    rec = rec_of(tokens, 10)  # rows 0..2 have contexts [:10], [:11], [:12]
    d = PromptLookupDrafter().drafts(rec, 4)
    assert d[0].tolist() == [-1, -1, -1, -1]          # ends with 4: never seen before
    assert d[1].tolist() == [6, 7, 8, 9]              # ends with 5: 1-gram match at 0
    assert d[2].tolist() == [7, 8, 9, 1]
    # 1-gram: the most recent earlier occurrence wins (6 at 6, not at 1)
    d = PromptLookupDrafter().drafts(rec_of([6, 1, 6, 2, 9, 6, 0], 6), 1)
    assert d[0].tolist() == [2]


def test_pld_overlapping_copy_and_acceptance():
    tokens = [1, 2, 1, 2, 1, 2, 1, 2]
    rec = rec_of(tokens, 4)
    d = PromptLookupDrafter().drafts(rec, 4)
    assert d[0].tolist() == [1, 2, 1, 2]  # match ends 1 positions back: copy runs into its own drafts
    acc = accepted_lengths(rec, d)
    assert acc.tolist() == [4, 3, 2, 1]


def test_hash_is_identical_in_numpy_and_torch():
    rng = np.random.default_rng(0)
    ctx = rng.integers(0, 32768, size=(1000, 3))
    kn = context_keys(ctx, 15)
    kt = context_keys(torch.as_tensor(ctx), 15)
    assert np.array_equal(kn, kt.numpy())
    rn, fn = hashed(kn, 1 << 22, False)
    rt, ft = hashed(kt, 1 << 22, False)
    assert np.array_equal(rn, rt.numpy()) and np.array_equal(fn, ft.numpy())
    assert rn.max() < (1 << 22) and fn.max() < 65536 and rn.min() >= 0


def test_ngram_counts_backoff_and_drafting(tmp_path):
    # vocab 64: order 1 and 2 exact, order 3 hashed into 128 buckets
    seq = [10, 11, 12, 13] * 5 + [20, 11, 12, 30] * 2 + [40, 41]
    w = torch.as_tensor([seq])
    m = build_ngram([w], 64, orders=(3, 2, 1), buckets={3: 128, 2: 1 << 12, 1: 64}, k=4)
    assert not m.exact(3) and m.exact(2) and m.exact(1)
    assert m.tokens_seen == len(seq) - 1
    # unigram row 11: followed by 12 seven times
    assert m.ids[1][11, 0] == 12 and m.counts[1][11, 0] == 7
    # bigram (11, 12): 13 x5, 30 x2
    row = 11 + (12 << 6)
    assert m.ids[2][row, :2].tolist() == [13, 30] and m.counts[2][row, :2].tolist() == [5, 2]
    # trigram decides: (20, 11, 12) -> 30 although (11, 12) -> 13 is more frequent
    assert m.predict(np.array([[20, 11, 12], [10, 11, 12], [-1, 99 % 64, 41]])).tolist()[:2] == [30, 13]
    # unseen context backs off to unigram, then to the fallback token
    assert m.predict(np.array([[63, 62, 40]]))[0] == 41
    assert m.predict(np.array([[63, 62, 61]]))[0] == m.fallback == 11  # most frequent (ties: lowest id)
    path = tmp_path / "ng.npz"
    m.save(path)
    m2 = TokenNgramModel.load(path)
    rec = rec_of([5, 20, 11, 12, 30, 40, 41, 5], 2)
    d1, d2 = TokenNgramDrafter(m).drafts(rec, 3), TokenNgramDrafter(path).drafts(rec, 3)
    assert np.array_equal(d1, d2)
    # row 0: context [5, 20] -> 11 (unigram 20->11), then (5,20,11)->? bigram (20,11)->12, then (20,11,12)->30
    assert d1[0].tolist() == [11, 12, 30]
