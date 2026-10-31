"""M3.T2: model-free drafting baselines.

    python -m drafting.baselines --data_dir $DATA --out ngram/climbmix_300m.npz --tokens 300e6

- ``PromptLookupDrafter`` (PLD): match the last 3, then 2, then 1 tokens of the verified
  context against an earlier occurrence in the context (the most recent one) and propose
  the tokens that followed it.  Copying is LZ77-style, so a match close to the end keeps
  copying from its own drafts.  No match -> -1 (never accepted).
- ``TokenNgramDrafter`` (REST-style): next-token top-k counts for contexts of the last
  3 / 2 / 1 tokens from training text (the same token stream as the sketches), backoff
  3 -> 2 -> 1, then the most frequent token; drafts autoregressively over its own drafts.

N-gram tables: contexts whose exact id space is larger than the table (``buckets``) are
hashed into buckets; each slot also stores a 16-bit fingerprint of its context, so a
lookup only uses entries of its own context (colliding contexts share a bucket's K slots;
the fingerprint keeps them from answering for each other, up to 1/65536).  Per bucket the
top-K (context, next-token) pairs are kept with the same mergeable summary as the sketches.
File: ``ids_<n>`` uint16 / ``fps_<n>`` uint16 / ``counts_<n>`` uint32 [rows_n, K], ``orders``,
``buckets``, ``vocab``, ``tok_bits``, ``fallback``, ``tokens_seen``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from drafting.build_sketches import TopKSummary, stream_windows
from drafting.common import Drafter, SeqRecord

_BUCKET_MUL, _BUCKET_MOD = 40503, 2147483647
_FP_MUL, _FP_MOD = 69069, 2147483629


# ------------------------------------------------------------------ PLD
def previous_occurrence(tokens: np.ndarray, n: int) -> np.ndarray:
    """out[e] = latest e' < e whose n-gram tokens[e'-n+1 : e'+1] equals the one ending at e (-1 if none)."""
    t = len(tokens)
    out = np.full(t, -1, dtype=np.int64)
    last: dict = {}
    for e in range(n - 1, t):
        key = tokens[e - n + 1: e + 1].tobytes()
        out[e] = last.get(key, -1)
        last[key] = e
    return out


class PromptLookupDrafter(Drafter):
    name = "pld"
    prefix_consistent = True

    def __init__(self, max_ngram: int = 3, min_ngram: int = 1):
        self.orders = list(range(max_ngram, min_ngram - 1, -1))

    def drafts(self, rec: SeqRecord, gamma: int) -> np.ndarray:
        s = np.asarray(rec.tokens, dtype=np.int64)
        p, n_cont = rec.prompt_len, rec.n_cont
        prev = {n: previous_occurrence(s, n) for n in self.orders}
        out = np.full((n_cont, gamma), -1, dtype=np.int64)
        for j in range(n_cont):
            e = p + j - 1  # last verified position
            src = -1
            for n in self.orders:
                if e - n + 1 >= 0 and prev[n][e] >= 0:
                    src = int(prev[n][e])
                    break
            if src < 0:
                continue
            for k in range(gamma):
                i = src + 1 + k
                out[j, k] = s[i] if i <= e else out[j, i - e - 1]
        return out


# ------------------------------------------------------------------ token n-gram
def tok_bits_for(vocab: int) -> int:
    b = max(1, (vocab - 1).bit_length())
    if b > 15:
        raise ValueError("token n-gram hashing assumes vocab <= 32768")
    return b


def context_keys(ctx, tok_bits: int):
    """Exact context id from [..., n] tokens (oldest first): sum ctx[i] << (tok_bits * i). numpy or torch int64."""
    key = ctx[..., 0] * 1
    for i in range(1, ctx.shape[-1]):
        key = key + (ctx[..., i] << (tok_bits * i))
    return key


def hashed(key, buckets: int, exact: bool):
    """(row, 16-bit fingerprint) of context ids; same arithmetic in numpy and torch (no int64 overflow)."""
    if exact:
        return key, key * 0
    return ((key * _BUCKET_MUL) % _BUCKET_MOD) % buckets, ((key * _FP_MUL) % _FP_MOD) % 65536


class TokenNgramModel:
    def __init__(self, vocab: int, orders, buckets: dict, k: int, ids: dict, fps: dict, counts: dict,
                 fallback: int = 0, tokens_seen: int = 0):
        self.vocab, self.orders, self.k = int(vocab), tuple(int(n) for n in orders), int(k)
        self.tok_bits = tok_bits_for(self.vocab)
        self.buckets = {int(n): int(b) for n, b in buckets.items()}
        self.ids, self.fps, self.counts = ids, fps, counts
        self.fallback, self.tokens_seen = int(fallback), int(tokens_seen)

    def exact(self, n: int) -> bool:
        return (1 << (self.tok_bits * n)) <= self.buckets[n]

    def rows(self, n: int) -> int:
        return (1 << (self.tok_bits * n)) if self.exact(n) else self.buckets[n]

    def predict(self, ctx: np.ndarray, min_count: int = 1) -> np.ndarray:
        """ctx [N, C] int64, oldest first, -1 = unavailable -> next token [N] (backoff, fallback token)."""
        c_len = ctx.shape[1]
        out = np.full(len(ctx), self.fallback, dtype=np.int64)
        done = np.zeros(len(ctx), dtype=bool)
        for n in sorted(self.orders, reverse=True):
            if n > c_len:
                continue
            c = ctx[:, c_len - n:]
            avail = (c >= 0).all(1) & ~done
            if not avail.any():
                continue
            key = context_keys(np.where(c >= 0, c, 0), self.tok_bits)
            row, fp = hashed(key, self.buckets[n], self.exact(n))
            cnt = self.counts[n][row].astype(np.int64)
            ok = (cnt >= min_count) & (self.fps[n][row].astype(np.int64) == fp[:, None])
            score = np.where(ok, cnt, 0)
            best = score.argmax(1)
            sel = avail & (score.max(1) > 0)
            out[sel] = self.ids[n][row[sel], best[sel]]
            done |= sel
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for n in self.orders:
            arrays.update({f"ids_{n}": self.ids[n], f"fps_{n}": self.fps[n], f"counts_{n}": self.counts[n]})
        np.savez(path, vocab=self.vocab, orders=np.array(self.orders), k=self.k,
                 buckets=np.array([self.buckets[n] for n in self.orders]), fallback=self.fallback,
                 tokens_seen=self.tokens_seen, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "TokenNgramModel":
        z = np.load(path)
        orders = tuple(int(n) for n in z["orders"])
        return cls(int(z["vocab"]), orders, dict(zip(orders, (int(b) for b in z["buckets"]))), int(z["k"]),
                   {n: z[f"ids_{n}"] for n in orders}, {n: z[f"fps_{n}"] for n in orders},
                   {n: z[f"counts_{n}"] for n in orders}, int(z["fallback"]), int(z["tokens_seen"]))


@torch.no_grad()
def build_ngram(windows, vocab: int, orders=(3, 2, 1), buckets: dict | None = None, k: int = 8,
                device: str | torch.device = "cpu", mode: str = "mg", max_buffer: int = 1 << 26,
                log_every: int = 0) -> TokenNgramModel:
    """Counts next tokens after every context of length n in the windows (int64 [b, T+1])."""
    tb = tok_bits_for(vocab)
    buckets = dict(buckets or {3: 1 << 22, 2: 1 << 20, 1: 1 << tb})
    for n in orders:
        buckets.setdefault(n, 1 << 22)
    shell = TokenNgramModel(vocab, orders, buckets, k, {}, {}, {})
    summ = {n: TopKSummary(shell.rows(n), k, 16 + tb, device, mode, max_buffer) for n in orders}
    freq = torch.zeros(vocab, dtype=torch.int64, device=device)
    seen, t0 = 0, time.time()
    for i, w in enumerate(windows):
        x = torch.as_tensor(w, device=device)
        s = x.shape[1] - 1
        freq += torch.bincount(x[:, 1:].reshape(-1), minlength=vocab)
        for n in orders:
            if s - n + 1 <= 0:
                continue
            ctx = torch.stack([x[:, i_: i_ + s - n + 1] for i_ in range(n)], -1)
            row, fp = hashed(context_keys(ctx, tb), buckets[n], shell.exact(n))
            nxt = x[:, n: s + 1]
            summ[n].add(row.reshape(-1), ((fp << tb) | nxt).reshape(-1))
        seen += x[:, 1:].numel()
        if log_every and (i + 1) % log_every == 0:
            print(f"[ngram] {seen:,} tokens  {seen / (time.time() - t0):,.0f} tok/s", flush=True)
    ids, fps, counts = {}, {}, {}
    mask = (1 << tb) - 1
    for n in orders:
        items, c = summ[n].numpy()
        ids[n] = (items & mask).astype(np.uint16)
        fps[n] = (items >> tb).astype(np.uint16)
        counts[n] = np.minimum(c, np.iinfo(np.uint32).max).astype(np.uint32)
    return TokenNgramModel(vocab, orders, buckets, k, ids, fps, counts, int(freq.argmax()), seen)


class TokenNgramDrafter(Drafter):
    name = "ngram"
    prefix_consistent = True

    def __init__(self, model: TokenNgramModel | str | Path, min_count: int = 1):
        self.model = TokenNgramModel.load(model) if isinstance(model, (str, Path)) else model
        self.min_count = min_count

    def drafts(self, rec: SeqRecord, gamma: int) -> np.ndarray:
        s = np.asarray(rec.tokens, dtype=np.int64)
        p, n_cont = rec.prompt_len, rec.n_cont
        c = max(self.model.orders)
        pos = (p + np.arange(n_cont))[:, None] - c + np.arange(c)[None]  # last c verified positions
        ctx = np.where(pos >= 0, s[np.clip(pos, 0, None)], -1)
        out = np.zeros((n_cont, gamma), dtype=np.int64)
        for k in range(gamma):
            out[:, k] = self.model.predict(ctx, self.min_count)
            ctx = np.concatenate((ctx[:, 1:], out[:, k: k + 1]), 1)
        return out


# ------------------------------------------------------------------ CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokens", type=float, default=300e6)
    p.add_argument("--seq_len", type=int, default=2048, help="same windows as build_sketches")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--orders", type=int, nargs="+", default=[3, 2, 1])
    p.add_argument("--buckets3", type=int, default=1 << 22)
    p.add_argument("--buckets2", type=int, default=1 << 20)
    p.add_argument("--merge", default="mg", choices=["mg", "topk"])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--log_every", type=int, default=200)
    return p.parse_args(argv)


def main(argv=None) -> dict:
    from ffnmem.data.loader import read_meta

    a = parse_args(argv)
    dev = "cuda" if (a.device == "auto" and torch.cuda.is_available()) or a.device == "cuda" else "cpu"
    vocab = int(read_meta(a.data_dir)["vocab_size"])
    t0 = time.time()
    model = build_ngram(stream_windows(a.data_dir, a.seq_len, int(a.tokens), a.batch), vocab, tuple(a.orders),
                        {3: a.buckets3, 2: a.buckets2, 1: 1 << tok_bits_for(vocab)}, a.k, dev, a.merge,
                        log_every=a.log_every)
    model.save(a.out)
    info = {"tokens_seen": model.tokens_seen, "orders": list(model.orders), "k": model.k,
            "buckets": model.buckets, "build_seconds": time.time() - t0}
    Path(a.out).with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2), flush=True)
    return info


if __name__ == "__main__":
    main()
