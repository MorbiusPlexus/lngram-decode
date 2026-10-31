from __future__ import annotations

import json

import numpy as np
import pytest

from ffnmem.data.loader import TrainWindows, ValWindows, build_index, index_paths, read_meta


def test_index_reproducible(tiny_data, tmp_path):
    a = TrainWindows(tiny_data, 32, 1234, global_batch=8, micro_batch=2)
    b = TrainWindows(tiny_data, 32, 1234, global_batch=8, micro_batch=2)
    assert np.array_equal(np.asarray(a.perm), np.asarray(b.perm))
    # rebuilding from scratch with the same seed gives the same order
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(tiny_data, copy)
    build_index(copy, 32, 1234, overwrite=True)
    c = TrainWindows(copy, 32, 1234, global_batch=8, micro_batch=2)
    assert np.array_equal(np.asarray(a.perm), np.asarray(c.perm))
    with pytest.raises(FileExistsError):
        build_index(copy, 32, 1234)


def test_ranks_disjoint_covering_and_world_size_invariant(tiny_data):
    for step in (0, 3, 17):
        full = TrainWindows(tiny_data, 32, 1234, 8, 8).step_window_ids(step)
        assert len(set(full.tolist())) == 8
        for world in (1, 2, 4):
            parts = [TrainWindows(tiny_data, 32, 1234, 8, 1, rank=r, world_size=world).rank_window_ids(step)
                     for r in range(world)]
            joined = np.concatenate(parts)
            assert np.array_equal(joined, full)  # same windows, same order, any world size
            flat = [set(p.tolist()) for p in parts]
            for i in range(world):
                for j in range(i + 1, world):
                    assert not flat[i] & flat[j]


def test_steps_use_distinct_windows(tiny_data):
    tw = TrainWindows(tiny_data, 32, 1234, 8, 2)
    seen = np.concatenate([tw.step_window_ids(s) for s in range(tw.max_steps)])
    assert len(np.unique(seen)) == len(seen)
    with pytest.raises(IndexError):
        tw.step_window_ids(tw.max_steps)


def test_windows_inside_shards_and_targets_shifted(tiny_data):
    tw = TrainWindows(tiny_data, 32, 1234, 8, 2)
    meta = read_meta(tiny_data)
    for i, s in enumerate(meta["shards"]):
        assert tw.info["windows_per_shard"][i] * 32 + 1 <= s["tokens"]
    for w in range(tw.num_windows):
        shard, off = tw.locate(w)
        assert off + 33 <= meta["shards"][shard]["tokens"]
    (x, y), *_ = tw.micro_batches(0)
    assert x.shape == (2, 32) and y.shape == (2, 32)
    assert np.array_equal(x[:, 1:].numpy(), y[:, :-1].numpy())
    assert len(tw.micro_batches(0)) == 4


def test_heldout_disjoint_from_training(tiny_data):
    _, info_path = index_paths(tiny_data, 32, 1234)
    info = json.loads(info_path.read_text())
    assert "val.bin" not in info["shards"]
    val = ValWindows(tiny_data, 32, val_tokens=10**9)
    tw = TrainWindows(tiny_data, 32, 1234, 8, 2)
    val_windows = {bytes(np.asarray(val.tokens[i * 32 : i * 32 + 33]).tobytes()) for i in range(val.num_windows)}
    train_windows = {bytes(tw.window(w).astype("<u2").tobytes()) for w in range(tw.num_windows)}
    assert not val_windows & train_windows
    # ranks split the val slice without overlap
    ids = [set(ValWindows(tiny_data, 32, 10**9, rank=r, world_size=3).ids) for r in range(3)]
    assert set().union(*ids) == set(range(val.num_windows)) and sum(map(len, ids)) == val.num_windows


def test_vocab_guard(tiny_data):
    with pytest.raises(ValueError):
        TrainWindows(tiny_data, 32, 1234, 8, 2, model_vocab=100)
