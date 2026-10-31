"""Build the fixed window order once, before any training job.

    python -m ffnmem.data.build_index --data <dir> [--seq_len 2048] [--seed 1234]

Writes ``index_seq{L}_seed{S}.npy`` (permutation of global window ids) and a
matching ``.json`` into the data directory.  Training jobs only read it.
"""

from __future__ import annotations

import argparse
import json

from .loader import build_index, index_paths


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    path = build_index(args.data, args.seq_len, args.seed, overwrite=args.overwrite)
    with open(index_paths(args.data, args.seq_len, args.seed)[1], encoding="utf-8") as f:
        info = json.load(f)
    print(f"wrote {path}: {info['num_windows']:,} windows of {args.seq_len}+1 tokens "
          f"from {len(info['shards'])} shards (fingerprint {info['shard_fingerprint']})")


if __name__ == "__main__":
    main()
