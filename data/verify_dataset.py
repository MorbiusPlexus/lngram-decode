"""
verify_dataset.py
-----------------
Sanity-checks the shards produced by download_dataset.py:

  1. Prints meta.json and checks it is complete.
  2. Checks every file exists, has a whole number of tokens, and matches the
     token count recorded in meta.json.
  3. One pass per file (chunked, low memory): id range, out-of-vocab ids,
     EOS rate, and counts of special tokens that should NOT appear in
     pretraining data (<unk>, <s>, <pad>).
  4. Decodes random windows, starting at a document boundary, so you can
     eyeball quality.

Usage
─────
  python verify_dataset.py --data_dir data/climbmix
  python verify_dataset.py --data_dir data/climbmix --max_shards 3 --n_samples 10
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

CHUNK = 50_000_000  # tokens per chunk when scanning a file


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir",   type=str, default="data/climbmix")
    p.add_argument("--tokenizer",  type=str, default=None, help="Override tokenizer path from meta.json.")
    p.add_argument("--max_shards", type=int, default=None, help="Only check the first N train shards.")
    p.add_argument("--n_samples",  type=int, default=5)
    p.add_argument("--window",     type=int, default=256)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def human(n: float) -> str:
    for unit, thresh in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(n) >= thresh:
            return f"{n / thresh:.2f}{unit}"
    return str(int(n))


def sep(char="─", width=72):
    print(char * width)


def scan(path: Path, dtype, vocab_size: int):
    """Return (n_tokens, counts per id) in one chunked pass, or raise."""
    itemsize = np.dtype(dtype).itemsize
    size = path.stat().st_size
    if size % itemsize:
        raise ValueError(f"file size {size} is not a multiple of {itemsize} bytes (truncated write?)")
    tokens = np.memmap(path, dtype=dtype, mode="r")
    # uint16 can hold ids up to 65535, so count that whole range to catch ids >= vocab_size
    n_bins = max(vocab_size, 65536) if dtype == np.uint16 else vocab_size
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(0, len(tokens), CHUNK):
        chunk = np.asarray(tokens[i:i + CHUNK])
        if dtype != np.uint16 and (chunk.min() < 0 or chunk.max() >= n_bins):
            raise ValueError(f"token ids out of range [{chunk.min()}, {chunk.max()}]")
        counts += np.bincount(chunk, minlength=n_bins)
    return len(tokens), counts


def main():
    args = parse_args()
    random.seed(args.seed)
    data_dir = Path(args.data_dir)
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        sys.exit(f"ERROR: {meta_path} not found")
    meta = json.load(open(meta_path))

    dtype = np.dtype(meta.get("dtype", "uint16")).type
    vocab_size = meta["vocab_size"]
    eos_id = meta.get("eos_id")

    sep("=")
    print("  DATASET METADATA")
    sep("=")
    for k in ["dataset", "split", "tokenizer", "vocab_size", "eos_id", "dtype", "seed", "docs_seen"]:
        print(f"  {k:16s}: {meta.get(k)}")
    print(f"  {'train tokens':16s}: {human(meta['total_tokens'])} ({meta['total_tokens']:,}) "
          f"in {len(meta['shards'])} shards")
    print(f"  {'val tokens':16s}: {human(meta.get('val_tokens', 0))} ({meta.get('val_file')})")
    print(f"  {'complete':16s}: {meta.get('complete')}")
    print(f"  {'elapsed':16s}: {meta.get('elapsed_seconds', 0) / 60:.1f} min")
    sep("=")

    warnings = 0
    if not meta.get("complete"):
        print("  ⚠  Run is not marked complete (interrupted?). Checking what exists.")
        warnings += 1

    # ── Tokenizer and special ids ──────────────────────────────────────────
    tok_path = args.tokenizer or meta["tokenizer"]
    print(f"\nLoading tokenizer: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if len(tokenizer) != vocab_size:
        print(f"  ⚠  tokenizer has {len(tokenizer)} tokens but meta says {vocab_size}. Wrong tokenizer?")
        warnings += 1
    forbidden = {name: tid for name, tid in [("unk", tokenizer.unk_token_id),
                                              ("bos", tokenizer.bos_token_id),
                                              ("pad", tokenizer.pad_token_id)]
                 if tid is not None and tid != eos_id}

    # ── Files to check ─────────────────────────────────────────────────────
    files = []
    if meta.get("val_file"):
        files.append((meta["val_file"], meta["val_tokens"]))
    shards = meta["shards"][:args.max_shards] if args.max_shards else meta["shards"]
    files += [(s["file"], s["tokens"]) for s in shards]

    print()
    sep()
    print(f"  FILE CHECK  ({len(files)} files)")
    sep()
    total_counts = None
    ok_files = []
    for name, expected in files:
        path = data_dir / name  # paths are relative to data_dir, so the folder can be moved
        if not path.exists():
            print(f"  ✗  MISSING     {name}")
            warnings += 1
            continue
        try:
            n, counts = scan(path, dtype, vocab_size)
        except Exception as e:
            print(f"  ✗  BAD         {name}: {e}")
            warnings += 1
            continue
        n_oov = int(counts[vocab_size:].sum())
        n_eos = int(counts[eos_id]) if eos_id is not None else 0
        bad = {k: int(counts[v]) for k, v in forbidden.items() if counts[v]}
        size_ok = n == expected
        status = "✓" if size_ok and not n_oov and not bad else "⚠"
        if status == "⚠":
            warnings += 1
        nz = np.nonzero(counts)[0]
        print(f"  {status}  {name:18s} {human(n):>8s} tok"
              f" | EOS {n_eos / max(n, 1) * 100:.2f}% (~{human(n / max(n_eos, 1))} tok/doc)"
              f" | ids [{nz.min()}, {nz.max()}]"
              + ("" if size_ok else f" | EXPECTED {expected:,}")
              + (f" | OOV {n_oov:,}" if n_oov else "")
              + "".join(f" | {k.upper()} {v:,}" for k, v in bad.items()))
        total_counts = counts if total_counts is None else total_counts + counts
        ok_files.append((path, n))

    if total_counts is not None:
        sep()
        n_total = int(total_counts.sum())
        used = int((total_counts[:vocab_size] > 0).sum())
        print(f"  Tokens scanned : {human(n_total)} ({n_total:,})")
        print(f"  Vocab used     : {used:,} / {vocab_size:,} ids appear at least once")
        unk_id = tokenizer.unk_token_id
        if unk_id is not None and total_counts[unk_id]:
            rate = total_counts[unk_id] / n_total
            print(f"  <unk> tokens   : {int(total_counts[unk_id]):,} ({rate * 1e6:.1f} per million)"
                  " — bytes missing from the tokenizer's base alphabet")
    sep()

    # ── Detokenization spot-check ──────────────────────────────────────────
    print(f"\n  DETOKENIZATION SPOT-CHECK ({args.n_samples} windows of {args.window} tokens)")
    candidates = [(p, n) for p, n in ok_files if n > args.window * 4]
    for i in range(args.n_samples if candidates else 0):
        path, n = random.choice(candidates)
        tokens = np.memmap(path, dtype=dtype, mode="r")
        start = random.randint(0, n - args.window * 4)
        # jump to the next document start so the sample begins cleanly
        if eos_id is not None:
            hits = np.nonzero(np.asarray(tokens[start:start + args.window * 3]) == eos_id)[0]
            if len(hits):
                start += int(hits[0]) + 1
        window = np.asarray(tokens[start:start + args.window]).tolist()
        text = tokenizer.decode(window, skip_special_tokens=False)
        print(f"\n  Sample {i + 1} | {path.name} | offset {start:,}")
        sep("·")
        print("  " + text.replace("\n", "↵ ")[:600] + (" ..." if len(text) > 600 else ""))
    print()
    sep("=")
    print("  ALL CHECKS PASSED ✓" if warnings == 0 else f"  COMPLETED WITH {warnings} WARNING(S) ⚠")
    sep("=")


if __name__ == "__main__":
    main()
