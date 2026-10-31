"""
download_dataset.py
-------------------
Stream ClimbMix from HuggingFace, tokenize with the project tokenizer, and
write token IDs to flat binary shard files on disk.

Output layout
─────────────
  <output_dir>/
    val.bin            # held-out validation tokens (first --val_tokens of the stream)
    shard_00000.bin    # training tokens, raw little-endian uint16 (or int32 if vocab > 65536)
    shard_00001.bin
    ...
    meta.json          # updated after every shard, so an interrupted run can resume

Each shard starts at a document boundary and ends right after an EOS, so shards
are ~shard_size tokens (slightly more), never split a document, and --resume
can restart cleanly from the last completed shard.

Load a shard with:  np.memmap(path, dtype=meta["dtype"], mode="r")

Usage
─────
  python download_dataset.py                               # 10B train tokens + 50M val
  python download_dataset.py --target_tokens 2_000_000_000 # smaller run
  python download_dataset.py --resume                      # continue an interrupted run
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import transformers as _transformers
from tqdm.auto import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description="Download & tokenize ClimbMix into binary shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",       type=str, default="OptimalScale/ClimbMix")
    p.add_argument("--data_files",    type=str, default=None,
                   help="Optional data_files for load_dataset (e.g. local json/parquet for testing).")
    p.add_argument("--split",         type=str, default="train")
    p.add_argument("--text_column",   type=str, default="text")
    p.add_argument("--tokenizer",     type=str, default="climbmix-tokenizer",
                   help="Local path or HF hub id of the tokenizer.")
    p.add_argument("--output_dir",    type=str, default="data/climbmix")
    p.add_argument("--target_tokens", type=int, default=10_000_000_000,
                   help="Stop after writing this many TRAIN tokens (val is extra).")
    p.add_argument("--val_tokens",    type=int, default=50_000_000,
                   help="Tokens reserved for val.bin before any training shard. 0 disables.")
    p.add_argument("--shard_size",    type=int, default=500_000_000,
                   help="Approx tokens per shard (250M uint16 ≈ 500 MB).")
    p.add_argument("--shuffle_buffer", type=int, default=10_000,
                   help="Streaming shuffle buffer (also shuffles source file order). 0 disables.")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--batch_docs",    type=int, default=32_000,
                   help="Documents per tokenizer call; batching uses all CPU cores.")
    p.add_argument("--no_eos_between_docs", dest="eos_between_docs", action="store_false",
                   help="Do not append EOS after each document.")
    p.add_argument("--resume",        action="store_true",
                   help="Resume from meta.json in output_dir (restarts after last full shard).")
    p.add_argument("--log_every",     type=int, default=50_000, help="Log every N documents.")
    p.add_argument("--hf_token",      type=str, default=None,
                   help="HuggingFace access token for faster downloads and gated datasets. "
                        "Defaults to the HF_TOKEN environment variable if set.")
    return p.parse_args()


def human(n: float) -> str:
    for unit, thresh in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(n) >= thresh:
            return f"{n / thresh:.2f}{unit}"
    return str(int(n))


class BinWriter:
    """Appends token arrays to one file, flushing every few MB (never buffers a whole shard)."""

    FLUSH_TOKENS = 4_000_000

    def __init__(self, path: Path, dtype):
        self.path, self.dtype = path, dtype
        self.fh = open(path, "wb")
        self.pending: list[np.ndarray] = []
        self.pending_len = 0
        self.n_tokens = 0

    def write(self, arr: np.ndarray):
        self.pending.append(arr)
        self.pending_len += len(arr)
        self.n_tokens += len(arr)
        if self.pending_len >= self.FLUSH_TOKENS:
            self.flush()

    def flush(self):
        if self.pending:
            self.fh.write(np.concatenate(self.pending).astype(self.dtype, copy=False).tobytes())
            self.pending, self.pending_len = [], 0

    def close(self):
        self.flush()
        self.fh.close()


def save_meta(output_dir: Path, meta: dict):
    tmp = output_dir / "meta.json.tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    tmp.replace(output_dir / "meta.json")  # atomic, so a crash never leaves half-written meta


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── HuggingFace authentication ─────────────────────────────────────────
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("HuggingFace: logged in with provided token (faster downloads enabled).")
    else:
        print("HuggingFace: no token provided (set --hf_token or HF_TOKEN env var for faster downloads).")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.tokenizer}")
    # AutoTokenizer can't resolve the custom 'TokenizersBackend' class in tokenizer_config.json,
    # so we load directly with PreTrainedTokenizerFast (which reads tokenizer.json natively).
    # We briefly suppress the harmless "class mismatch" warning that fires as a result.
    _prev_verbosity = _transformers.logging.get_verbosity()
    _transformers.logging.set_verbosity_error()
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer, token=hf_token)
    finally:
        _transformers.logging.set_verbosity(_prev_verbosity)
    eos_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)  # includes added/special tokens
    dtype = np.uint16 if vocab_size <= 65536 else np.int32
    if args.eos_between_docs and eos_id is None:
        raise SystemExit("Tokenizer has no EOS token; pass --no_eos_between_docs or fix the tokenizer.")
    print(f"  vocab_size : {vocab_size:,}  |  eos_id : {eos_id}  |  dtype : {np.dtype(dtype).name}")

    # ── Dataset ────────────────────────────────────────────────────────────
    print(f"\nStreaming: {args.dataset}  split={args.split}")
    ds = load_dataset(
        args.dataset,
        data_files=args.data_files,
        split=args.split,
        streaming=True,
        token=hf_token,
    )
    if args.shuffle_buffer > 0:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    # ── Resume state ───────────────────────────────────────────────────────
    meta = {
        "dataset": args.dataset, "split": args.split, "tokenizer": args.tokenizer,
        "vocab_size": vocab_size, "eos_id": eos_id, "dtype": np.dtype(dtype).name,
        "byte_order": "little", "seed": args.seed, "shuffle_buffer": args.shuffle_buffer,
        "target_tokens": args.target_tokens, "shard_size_tokens": args.shard_size,
        "val_file": None, "val_tokens": 0, "shards": [], "total_tokens": 0,
        "docs_seen": 0, "elapsed_seconds": 0.0, "complete": False, "stream_state": None,
    }
    if args.resume and (output_dir / "meta.json").exists():
        with open(output_dir / "meta.json") as f:
            meta = json.load(f)
        if meta.get("complete"):
            print("meta.json says this run is already complete. Nothing to do.")
            return
        if meta.get("stream_state") is None:
            raise SystemExit("meta.json has no stream_state to resume from; start a fresh run.")
        ds.load_state_dict(meta["stream_state"])
        print(f"Resuming after {len(meta['shards'])} shard(s), {human(meta['total_tokens'])} train tokens.")
        print("  (note: resuming a shuffled stream drops the old shuffle buffer, ~a few thousand docs)")
    elif (output_dir / "meta.json").exists():
        raise SystemExit(f"{output_dir}/meta.json exists. Use --resume, or pick a new --output_dir.")

    need_val = args.val_tokens > 0 and meta["val_file"] is None
    shard_idx = len(meta["shards"])

    def new_writer():
        if need_val:
            return BinWriter(output_dir / "val.bin", dtype)
        return BinWriter(output_dir / f"shard_{shard_idx:05d}.bin", dtype)

    writer = new_writer()
    print(f"\nTarget : {human(args.target_tokens)} train tokens"
          f"{f' + {human(args.val_tokens)} val' if need_val else ''}  →  {output_dir.resolve()}\n")

    t0 = time.time() - meta["elapsed_seconds"]
    batch: list[str] = []
    done = False

    # n_fmt/total_fmt: tqdm sets total=None once n overshoots it (last shard), which broke "{total:.3g}"
    _bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} tok [{elapsed}<{remaining}, {rate_fmt}]"
    val_pbar = tqdm(
        total=args.val_tokens if need_val else 0,
        unit="tok",
        unit_scale=True,
        unit_divisor=1000,
        dynamic_ncols=True,
        desc=" val tokens",
        bar_format=_bar_fmt,
        disable=not need_val,
    )
    pbar = tqdm(
        total=args.target_tokens,
        initial=meta["total_tokens"],
        unit="tok",
        unit_scale=True,
        unit_divisor=1000,
        dynamic_ncols=True,
        desc="train tokens",
        bar_format=_bar_fmt,
    )

    def finish_file():
        """Close the current file at a document boundary and record it in meta."""
        nonlocal writer, need_val, shard_idx
        writer.close()
        if need_val:
            meta["val_file"], meta["val_tokens"] = writer.path.name, writer.n_tokens
            need_val = False
            val_pbar.close()  # val done — train bar takes over
        else:
            meta["shards"].append({"file": writer.path.name, "tokens": writer.n_tokens})
            meta["total_tokens"] += writer.n_tokens
            shard_idx += 1
            pbar.set_postfix_str(f"shard {shard_idx:05d}  {human(meta['total_tokens'])} tok total", refresh=True)
        meta["stream_state"] = ds.state_dict()
        meta["elapsed_seconds"] = round(time.time() - t0, 1)
        save_meta(output_dir, meta)

    def process(texts: list[str]):
        nonlocal writer, done
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        batch_new_tokens = 0
        for ids in encoded:
            if not ids:
                continue
            if args.eos_between_docs:
                ids.append(eos_id)
            writer.write(np.asarray(ids, dtype=dtype))
            meta["docs_seen"] += 1
            batch_new_tokens += len(ids)

            limit = args.val_tokens if need_val else args.shard_size
            if writer.n_tokens >= limit:
                finish_file()
                if meta["total_tokens"] >= args.target_tokens:
                    done = True
                    pbar.update(batch_new_tokens)
                    return
                writer = new_writer()

        # advance whichever bar is active
        if need_val:
            val_pbar.update(batch_new_tokens)
            val_pbar.set_postfix(docs=f"{meta['docs_seen']:,}", refresh=False)
        else:
            pbar.update(batch_new_tokens)
            pbar.set_postfix(docs=f"{meta['docs_seen']:,}", shard=f"{shard_idx:05d}", refresh=False)

    try:
        for item in ds:
            text = item.get(args.text_column) or ""
            if text:
                batch.append(text)
            if len(batch) >= args.batch_docs:
                process(batch)
                batch = []
                if done:
                    break
        if batch and not done:
            process(batch)
    except KeyboardInterrupt:
        val_pbar.close()
        pbar.close()
        print("\n⚠  Interrupted. Completed shards are saved; rerun with --resume.")
        writer.close()
        writer.path.unlink(missing_ok=True)  # partial shard is discarded; resume redoes it
        return

    val_pbar.close()
    pbar.close()

    if not done:  # stream ran out before the target
        print(f"\n⚠  Source ran out before the target: {human(meta['total_tokens'] + writer.n_tokens)}"
              f" of {human(args.target_tokens)} train tokens.")
        if writer.n_tokens > 0:
            finish_file()
        else:
            writer.close()
            writer.path.unlink(missing_ok=True)

    meta["complete"] = True
    meta["elapsed_seconds"] = round(time.time() - t0, 1)
    save_meta(output_dir, meta)

    print(f"\n{'=' * 60}\nDone!")
    print(f"  Train tokens : {human(meta['total_tokens'])} ({meta['total_tokens']:,}) in {len(meta['shards'])} shards")
    print(f"  Val tokens   : {human(meta['val_tokens'])}")
    print(f"  Elapsed      : {meta['elapsed_seconds'] / 60:.1f} min")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
