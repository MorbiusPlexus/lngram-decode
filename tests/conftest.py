from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("USE_LIBUV", "0")

TINY = ROOT / "configs" / "tiny_cpu.yaml"


def make_fake_dataset(out: Path, vocab: int = 256, eos: int = 2, n_shards: int = 3,
                      shard_tokens: int = 40_000, val_tokens: int = 4_000, seed: int = 0) -> Path:
    """Same on-disk format as data/download_dataset.py: docs + EOS, shards end on EOS."""
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)

    def write(name: str, target: int) -> int:
        parts, n = [], 0
        while n < target:
            doc = rng.integers(4, vocab, size=int(rng.integers(20, 300))).astype(np.uint16)
            doc = np.append(doc, np.uint16(eos))
            parts.append(doc)
            n += len(doc)
        arr = np.concatenate(parts)
        arr.astype("<u2").tofile(out / name)
        return len(arr)

    meta = {"dataset": "fake", "vocab_size": vocab, "eos_id": eos, "dtype": "uint16", "byte_order": "little",
            "val_file": "val.bin", "val_tokens": write("val.bin", val_tokens), "shards": [], "total_tokens": 0,
            "complete": True}
    for i in range(n_shards):
        name = f"shard_{i:05d}.bin"
        n = write(name, shard_tokens)
        meta["shards"].append({"file": name, "tokens": n})
        meta["total_tokens"] += n
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory) -> Path:
    from ffnmem.data.loader import build_index

    d = make_fake_dataset(tmp_path_factory.mktemp("data"))
    build_index(d, seq_len=32, seed=1234)
    return d


@pytest.fixture
def tiny_cfg():
    from ffnmem.config import load_config

    return load_config(TINY)
