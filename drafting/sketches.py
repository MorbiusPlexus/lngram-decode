"""Count sketches keyed by memory rows (M3.T1 format; level 0 of the ladder).

For every (route, order, row) of a memory layer, the top-K (K = 16) next tokens
with counts.  Rows use v2's address convention (``r * 16^n + addr``, oldest code
least significant -- the same index as ``tables["ngram_n"]``), so a sketch row is
the table row the memory layer reads at position t, and its tokens are what
followed t (the token at t+1) in training text.

File ``sketches/<run>.npz``: ``ids_<n>`` uint16 [R * 16^n, K], ``counts_<n>``
uint32 [R * 16^n, K] (count 0 = empty slot), plus ``routes``, ``bits``, ``ngrams``,
``layer`` (memory block position), ``tokens_seen``, ``vocab``.

The level-0 decoder scores token v at position t as the sum over routes (and over
the orders that fit) of v's count in the rows read at t; ``decode_dense`` turns
that into a smoothed distribution for KL / sum-min metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass
class Sketch:
    ids: dict            # order -> uint16 [R * 16^n, K]
    counts: dict         # order -> uint32 [R * 16^n, K]
    routes: int
    bits: int
    ngrams: tuple
    layer: int
    tokens_seen: int = 0
    vocab: int = 32768

    @property
    def k(self) -> int:
        return next(iter(self.ids.values())).shape[1]

    @classmethod
    def empty(cls, routes: int, bits: int, ngrams, layer: int, k: int = 16, vocab: int = 32768) -> "Sketch":
        a = 1 << bits
        ids = {n: np.zeros((routes * a ** n, k), np.uint16) for n in ngrams}
        counts = {n: np.zeros((routes * a ** n, k), np.uint32) for n in ngrams}
        return cls(ids, counts, routes, bits, tuple(ngrams), layer, 0, vocab)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"ids_{n}": self.ids[n] for n in self.ngrams}
        arrays.update({f"counts_{n}": self.counts[n] for n in self.ngrams})
        np.savez(path, routes=self.routes, bits=self.bits, ngrams=np.array(self.ngrams), layer=self.layer,
                 tokens_seen=self.tokens_seen, vocab=self.vocab, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Sketch":
        z = np.load(path)
        ngrams = tuple(int(n) for n in z["ngrams"])
        return cls({n: z[f"ids_{n}"] for n in ngrams}, {n: z[f"counts_{n}"] for n in ngrams}, int(z["routes"]),
                   int(z["bits"]), ngrams, int(z["layer"]), int(z["tokens_seen"]), int(z["vocab"]))

    # ------------------------------------------------------------- addressing
    def rows_at(self, codes: np.ndarray, order: int) -> np.ndarray:
        """Global rows [T, R] of the n-grams ending at each t (-1 where t < order-1). codes [T, R]."""
        t, r = codes.shape
        out = np.full((t, r), -1, dtype=np.int64)
        if t >= order:
            c = codes.astype(np.int64)
            addr = c[: t - order + 1].copy()
            for e in range(1, order):
                addr |= c[e: e + t - order + 1] << (e * self.bits)
            out[order - 1:] = np.arange(r)[None, :] * (1 << self.bits) ** order + addr
        return out

    # --------------------------------------------------------------- decoding
    def _tensors(self, order: int) -> tuple[Tensor, Tensor]:
        """int64 ids / float32 counts of one order as tensors (converted once, then cached)."""
        cache = self.__dict__.setdefault("_tensor_cache", {})
        if order not in cache:
            cache[order] = (torch.from_numpy(self.ids[order].astype(np.int64)),
                            torch.from_numpy(self.counts[order].astype(np.float32)))
        return cache[order]

    def scores_torch(self, rows: dict, device=None) -> tuple[Tensor, Tensor]:
        """Sparse scores for a batch of positions. rows: order -> int64 [N, R] (-1 = none).
        Returns (token ids [N, M], summed counts [N, M]) with M = R * len(orders) * K
        (duplicates not merged; use ``topk_tokens`` / ``decode_dense``)."""
        ids, cnts = [], []
        for n, r in rows.items():
            r = torch.as_tensor(r)
            valid = r >= 0
            rr = r.clamp_min(0)
            ids_n, counts_n = self._tensors(n)
            i = ids_n[rr]                                                  # [N, R, K]
            c = counts_n[rr] * valid[..., None]
            ids.append(i.flatten(1))
            cnts.append(c.flatten(1))
        ids_t, c_t = torch.cat(ids, 1), torch.cat(cnts, 1)
        if device is not None:
            ids_t, c_t = ids_t.to(device), c_t.to(device)
        return ids_t, c_t

    def decode_dense(self, rows: dict, smoothing: float = 1e-3, device=None) -> Tensor:
        """Distribution [N, vocab]: summed counts, normalised, mixed with ``smoothing`` uniform mass."""
        ids, c = self.scores_torch(rows, device)
        dense = torch.zeros(ids.shape[0], self.vocab, device=ids.device).scatter_add_(1, ids, c)
        tot = dense.sum(-1, keepdim=True)
        p = torch.where(tot > 0, dense / tot.clamp_min(1e-30), torch.full_like(dense, 1.0 / self.vocab))
        return (1 - smoothing) * p + smoothing / self.vocab

    def topk_tokens(self, rows: dict, k: int = 1, device=None) -> Tensor:
        """Top-k tokens [N, k] by summed count (tie order unspecified)."""
        return self.decode_dense(rows, 0.0, device).topk(k, dim=-1).indices
