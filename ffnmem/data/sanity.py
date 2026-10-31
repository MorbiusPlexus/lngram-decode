"""Data sanity check for the loader (not a replacement for data/verify_dataset.py).

    python -m ffnmem.data.sanity --data <dir> [--seq_len 2048] [--seed 1234]

- decodes 3 random training windows (and 1 val window) back to text;
- checks every indexed window lies inside its shard (no boundary crossing);
- asserts token ids < vocab size on the sampled windows and on the first steps.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from .loader import TrainWindows, ValWindows

DEFAULT_TOKENIZER = Path(__file__).resolve().parents[2] / "data" / "tokenizer" / "tokenizer.json"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--check_steps", type=int, default=4, help="scan all windows of the first N steps (global batch 256)")
    args = p.parse_args(argv)

    from tokenizers import Tokenizer  # only this script needs the tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    tw = TrainWindows(args.data, args.seq_len, args.seed, global_batch=256, micro_batch=1)
    vocab = tw.shards.vocab_size
    print(f"dtype={tw.shards.dtype} vocab={vocab} eos={tw.shards.eos_id} tokenizer_vocab={tok.get_vocab_size()}")
    assert tok.get_vocab_size() == vocab, "tokenizer vocab differs from meta.json"
    print(f"{tw.num_windows:,} train windows -> {tw.max_steps:,} steps at global batch 256")

    # Every window must fit in its shard.
    for i, name in enumerate(tw.info["shards"]):
        n_tok = len(tw.shards.memmap(name))
        n_win = tw.info["windows_per_shard"][i]
        assert n_win * args.seq_len + 1 <= n_tok, f"{name}: windows exceed shard ({n_win} x {args.seq_len}+1 > {n_tok})"
    print("boundary check: every window lies inside its shard")

    for step in range(min(args.check_steps, tw.max_steps)):
        for w in tw.step_window_ids(step):
            arr = tw.window(int(w))
            assert len(arr) == args.seq_len + 1
            assert int(arr.max()) < vocab and int(arr.min()) >= 0, f"window {w}: id out of range"
    print(f"id-range check: first {args.check_steps} steps OK")

    rng = random.Random(args.seed)
    for k in range(args.samples):
        w = rng.randrange(tw.num_windows)
        shard, offset = tw.locate(w)
        arr = tw.window(w)
        text = tok.decode(arr[:200].tolist(), skip_special_tokens=False)
        print(f"\n--- train window {w} ({tw.info['shards'][shard]} @ {offset:,}) ---")
        print(text.replace("\n", "\n")[:800])
    val = ValWindows(args.data, args.seq_len, val_tokens=10**12)
    x, _ = next(val.batches(1))
    print(f"\n--- val window 0 ({val.num_windows:,} val windows available) ---")
    print(tok.decode(x[0, :200].tolist(), skip_special_tokens=False).replace("\n", "\n")[:800])
    eos_rate = np.mean([tw.window(int(w)) == tw.shards.eos_id for w in tw.step_window_ids(0)[:32]])
    print(f"\nEOS rate in first 32 windows: {eos_rate:.4f}")
    print("SANITY OK")


if __name__ == "__main__":
    main()
