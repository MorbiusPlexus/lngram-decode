"""Loader over the shards written by ``data/download_dataset.py``.

Format (from that script's ``meta.json``): raw little-endian token ids of dtype
``meta["dtype"]`` (uint16 for the 32,768-id tokenizer); every document is
followed by EOS; every shard starts at a document boundary and ends on EOS;
``val.bin`` holds the first ``val_tokens`` of the stream, disjoint from shards.

Windows: ``seq_len + 1`` tokens starting at multiples of ``seq_len`` inside one
shard (inputs = first seq_len, targets = last seq_len).  Documents are simply
concatenated with EOS between them -- no document masking.

Order: a fixed permutation of all train windows (seed ``data_seed``), written
once by ``python -m ffnmem.data.build_index``.  Optimizer step ``s`` consumes
permutation entries ``[s*G, (s+1)*G)`` (G = global batch).  Rank ``r`` takes a
contiguous ``G / world_size`` slice of that block and splits it into
micro-batches.  The sequence of windows per step therefore does not depend on
world size or micro-batch size, and resuming only needs the step number.

The tokenizer is never loaded here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def read_meta(data_dir: str | Path) -> dict:
    with open(Path(data_dir) / "meta.json", encoding="utf-8") as f:
        return json.load(f)


def shard_fingerprint(meta: dict) -> str:
    """Identifies the shard set an index was built from."""
    blob = json.dumps([(s["file"], int(s["tokens"])) for s in meta["shards"]], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def index_paths(data_dir: str | Path, seq_len: int, seed: int) -> tuple[Path, Path]:
    stem = f"index_seq{seq_len}_seed{seed}"
    d = Path(data_dir)
    return d / f"{stem}.npy", d / f"{stem}.json"


def windows_per_shard(meta: dict, seq_len: int) -> np.ndarray:
    # A window starting at j*seq_len needs seq_len+1 tokens inside the shard.
    return np.array([max(0, (int(s["tokens"]) - 1) // seq_len) for s in meta["shards"]], dtype=np.int64)


def build_index(data_dir: str | Path, seq_len: int, seed: int, overwrite: bool = False) -> Path:
    meta = read_meta(data_dir)
    npy_path, json_path = index_paths(data_dir, seq_len, seed)
    if npy_path.exists() and not overwrite:
        raise FileExistsError(f"{npy_path} exists (pass overwrite=True / --overwrite to rebuild)")
    counts = windows_per_shard(meta, seq_len)
    total = int(counts.sum())
    perm = np.random.default_rng(seed).permutation(total).astype(np.int64)
    tmp = npy_path.with_name(npy_path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, perm)
    tmp.replace(npy_path)
    info = {
        "seq_len": seq_len,
        "seed": seed,
        "num_windows": total,
        "windows_per_shard": counts.tolist(),
        "shards": [s["file"] for s in meta["shards"]],
        "shard_fingerprint": shard_fingerprint(meta),
        "dtype": meta["dtype"],
        "vocab_size": meta["vocab_size"],
    }
    tmp = json_path.with_name(json_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    tmp.replace(json_path)
    return npy_path


class ShardSet:
    """Memory-mapped train shards + val file."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.meta = read_meta(self.data_dir)
        if self.meta.get("byte_order", "little") != "little":
            raise ValueError("only little-endian shards are supported")
        self.dtype = np.dtype(self.meta["dtype"]).newbyteorder("<")
        self.vocab_size = int(self.meta["vocab_size"])
        self.eos_id = self.meta.get("eos_id")
        self._maps: dict[str, np.memmap] = {}

    def memmap(self, name: str) -> np.memmap:
        if name not in self._maps:
            self._maps[name] = np.memmap(self.data_dir / name, dtype=self.dtype, mode="r")
        return self._maps[name]


class TrainWindows:
    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int,
        seed: int,
        global_batch: int,
        micro_batch: int,
        rank: int = 0,
        world_size: int = 1,
        model_vocab: int | None = None,
    ):
        if global_batch % (micro_batch * world_size):
            raise ValueError(
                f"global_batch {global_batch} not divisible by micro_batch*world_size {micro_batch}*{world_size}"
            )
        self.shards = ShardSet(data_dir)
        if model_vocab is not None and self.shards.vocab_size > model_vocab:
            raise ValueError(f"dataset vocab {self.shards.vocab_size} > model vocab {model_vocab}")
        npy_path, json_path = index_paths(data_dir, seq_len, seed)
        if not npy_path.exists():
            raise FileNotFoundError(
                f"{npy_path} missing; run `python -m ffnmem.data.build_index --data {data_dir} "
                f"--seq_len {seq_len} --seed {seed}` once before training"
            )
        with open(json_path, encoding="utf-8") as f:
            self.info = json.load(f)
        if self.info["shard_fingerprint"] != shard_fingerprint(self.shards.meta):
            raise ValueError("index was built from a different shard set; rebuild it")
        self.perm = np.load(npy_path, mmap_mode="r")
        self.seq_len = seq_len
        self.global_batch = global_batch
        self.micro_batch = micro_batch
        self.rank, self.world_size = rank, world_size
        self.per_rank = global_batch // world_size
        self.accum_steps = self.per_rank // micro_batch
        counts = np.asarray(self.info["windows_per_shard"], dtype=np.int64)
        self._starts = np.concatenate(([0], np.cumsum(counts)))  # global window id -> shard
        self.num_windows = int(counts.sum())
        self.max_steps = self.num_windows // global_batch

    def locate(self, window_id: int) -> tuple[int, int]:
        """Global window id -> (shard index, token offset)."""
        shard = int(np.searchsorted(self._starts, window_id, side="right") - 1)
        return shard, int(window_id - self._starts[shard]) * self.seq_len

    def window(self, window_id: int) -> np.ndarray:
        shard, offset = self.locate(window_id)
        tokens = self.shards.memmap(self.info["shards"][shard])
        return np.asarray(tokens[offset : offset + self.seq_len + 1], dtype=np.int64)

    def step_window_ids(self, step: int) -> np.ndarray:
        """All global window ids consumed at optimizer step ``step`` (every rank)."""
        if step >= self.max_steps:
            raise IndexError(
                f"step {step} needs windows beyond the dataset ({self.num_windows} windows, "
                f"{self.max_steps} steps at global batch {self.global_batch})"
            )
        lo = step * self.global_batch
        return np.asarray(self.perm[lo : lo + self.global_batch])

    def rank_window_ids(self, step: int) -> np.ndarray:
        ids = self.step_window_ids(step)
        return ids[self.rank * self.per_rank : (self.rank + 1) * self.per_rank]

    def micro_batches(self, step: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        ids = self.rank_window_ids(step)
        out = []
        for i in range(0, len(ids), self.micro_batch):
            batch = np.stack([self.window(int(w)) for w in ids[i : i + self.micro_batch]])
            t = torch.from_numpy(batch)
            out.append((t[:, :-1].contiguous(), t[:, 1:].contiguous()))
        return out


class ValWindows:
    """Fixed held-out slice: the first ``val_tokens`` tokens of ``val.bin``."""

    def __init__(self, data_dir: str | Path, seq_len: int, val_tokens: int, rank: int = 0, world_size: int = 1):
        self.shards = ShardSet(data_dir)
        name = self.shards.meta.get("val_file")
        if not name:
            raise FileNotFoundError("meta.json has no val_file")
        self.tokens = self.shards.memmap(name)
        usable = min(int(val_tokens), len(self.tokens))
        self.num_windows = max(0, (usable - 1) // seq_len)
        self.seq_len = seq_len
        self.ids = list(range(rank, self.num_windows, world_size))

    def batches(self, batch_size: int, max_batches: int | None = None):
        n = 0
        for i in range(0, len(self.ids), batch_size):
            if max_batches is not None and n >= max_batches:
                return
            chunk = self.ids[i : i + batch_size]
            batch = np.stack(
                [np.asarray(self.tokens[w * self.seq_len : w * self.seq_len + self.seq_len + 1], dtype=np.int64) for w in chunk]
            )
            t = torch.from_numpy(batch)
            yield t[:, :-1].contiguous(), t[:, 1:].contiguous()
            n += 1
