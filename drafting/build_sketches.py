"""M3.T1: count sketches keyed by memory rows (format: ``drafting/sketches.py``).

    python -m drafting.build_sketches --target runs/w3_sb_1b --data_dir $DATA \
        --out sketches/w3_sb_1b.npz --tokens 300e6 --batch 16
    python -m drafting.build_sketches --target runs/w3_sb_1b --data_dir $DATA \
        --out sketches/w3_sb_1b_randrouter.npz --code_source random_router --seed 0   # ladder control

Streams training tokens (shards in order, the training windows of ``seq_len + 1``
tokens; never val) and computes the memory layer's codes:

- unmodified targets (router reads the hidden state): blocks before the memory
  layer + its router only (``memory_codes``), never the rest of the backbone;
- SD-B / SD-C (token-only router): router features from the embeddings (+ SSM), no backbone.

``--code_source random_router`` replaces the router weight by a freshly initialised
one of the same shape (seeded), same input features.

For each (order, route, row) it keeps the top-K next tokens with counts.  Per chunk
the (row, next-token) pairs are counted on the device with ``torch.unique``; chunks are
merged into a running per-row top-K summary (``TopKSummary``): Misra-Gries merge by
default (subtract the (K+1)-th count, keep positive ones; exact while a row has at most
K distinct tokens, error <= N_row / (K+1) otherwise), or plain top-K truncation
(``--merge topk``).  Recall@16 of the true next token on a val slice is printed and
written next to the sketch (``<out>.json``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from drafting.sketches import Sketch
from ffnmem.data.loader import ShardSet, ValWindows
from ffnmem.model import Transformer


# ------------------------------------------------------------------ data
def stream_windows(data_dir: str | Path, seq_len: int, max_tokens: int, batch: int,
                   skip_windows: int = 0) -> Iterator[np.ndarray]:
    """Training windows int64 [b, seq_len + 1] in shard order (windows start at multiples of
    ``seq_len`` inside a shard, as in training) until ``max_tokens`` predicted tokens."""
    shards = ShardSet(data_dir)
    seen, buf, skipped = 0, [], 0
    for s in shards.meta["shards"]:
        toks = shards.memmap(s["file"])
        for j in range(max(0, (len(toks) - 1) // seq_len)):
            if skipped < skip_windows:
                skipped += 1
                continue
            if seen >= max_tokens:
                break
            buf.append(np.asarray(toks[j * seq_len: j * seq_len + seq_len + 1], dtype=np.int64))
            seen += seq_len
            if len(buf) == batch:
                yield np.stack(buf)
                buf = []
        if seen >= max_tokens:
            break
    if buf:
        yield np.stack(buf)


# ------------------------------------------------------------------ summaries
class TopKSummary:
    """Per-row top-K items with counts, built from (row, item) pairs on the device.

    ``add`` counts a chunk exactly (``torch.unique``) into a buffer; ``flush`` merges the
    buffer into the summary.  ``mode='mg'``: Misra-Gries merge (mergeable summary);
    ``'topk'``: keep the K largest summed counts.  Slots of a row are sorted by count
    (descending, ties by item id); count 0 = empty."""

    def __init__(self, n_rows: int, k: int, item_bits: int, device: str | torch.device = "cpu",
                 mode: str = "mg", max_buffer: int = 1 << 26):
        if mode not in ("mg", "topk"):
            raise ValueError(f"mode must be mg | topk, got {mode!r}")
        if (n_rows - 1).bit_length() + item_bits > 62:
            raise ValueError("row/item key does not fit in int64")
        self.n_rows, self.k, self.item_bits, self.mode = n_rows, k, item_bits, mode
        self.device = torch.device(device)
        self.ids = torch.zeros(n_rows, k, dtype=torch.int64, device=self.device)
        self.counts = torch.zeros(n_rows, k, dtype=torch.int64, device=self.device)
        self.max_buffer = max_buffer
        self._buf: list[tuple[Tensor, Tensor]] = []
        self._buf_len = 0

    def add(self, rows: Tensor, items: Tensor) -> None:
        keys = (rows.to(self.device, torch.int64) << self.item_bits) | items.to(self.device, torch.int64)
        uk, cnt = torch.unique(keys, return_counts=True)
        self._buf.append((uk, cnt))
        self._buf_len += uk.numel()
        if self._buf_len >= self.max_buffer:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        keys = torch.cat([b[0] for b in self._buf])
        cnts = torch.cat([b[1] for b in self._buf])
        self._buf, self._buf_len = [], 0
        if len(keys):
            uk, inv = torch.unique(keys, return_inverse=True)
            self._merge(uk, torch.zeros(len(uk), dtype=torch.int64, device=self.device).index_add_(0, inv, cnts))

    def _merge(self, keys: Tensor, cnts: Tensor) -> None:
        ib, k = self.item_bits, self.k
        mask = (1 << ib) - 1
        touched = torch.unique_consecutive(keys >> ib)  # keys are sorted
        ex_c = self.counts[touched]
        live = ex_c > 0
        ex_k = ((touched[:, None] << ib) | self.ids[touched])[live]
        uk, inv = torch.unique(torch.cat((keys, ex_k)), return_inverse=True)
        s = torch.zeros(len(uk), dtype=torch.int64, device=self.device).index_add_(0, inv, torch.cat((cnts, ex_c[live])))
        rows = uk >> ib
        o = torch.argsort(-s, stable=True)
        o = o[torch.argsort(rows[o], stable=True)]           # row ascending, count descending
        rows, items, s = rows[o], uk[o] & mask, s[o]
        urows, per = torch.unique_consecutive(rows, return_counts=True)
        rinv = torch.repeat_interleave(torch.arange(len(urows), device=self.device), per)
        rank = torch.arange(len(rows), device=self.device) - (torch.cumsum(per, 0) - per)[rinv]
        if self.mode == "mg":
            thr = torch.zeros(len(urows), dtype=torch.int64, device=self.device)
            at = rank == k
            thr[rinv[at]] = s[at]
            s = s - thr[rinv]
        keep = (rank < k) & (s > 0)
        self.ids[urows] = 0
        self.counts[urows] = 0
        self.ids[rows[keep], rank[keep]] = items[keep]
        self.counts[rows[keep], rank[keep]] = s[keep]

    def numpy(self) -> tuple[np.ndarray, np.ndarray]:
        self.flush()
        return self.ids.cpu().numpy(), self.counts.cpu().numpy()


# ------------------------------------------------------------------ codes
def memory_layer(model: Transformer, layer: int | None = None) -> int:
    if layer is not None:
        return int(layer)
    return [i for i, b in enumerate(model.blocks) if b.memory is not None][0]


def random_router_weight(model: Transformer, layer: int, seed: int) -> Tensor:
    """A freshly initialised router weight of the same shape (normal, the model's router init std)."""
    mem = model.blocks[layer].memory
    std = float((model.cfg.get("memory") or {}).get("router_init_std", model.cfg["init_std"]))
    g = torch.Generator().manual_seed(int(seed))
    w = torch.randn(mem.router.weight.shape, generator=g) * std
    return w.to(mem.router.weight.device)


@torch.no_grad()
def memory_codes(model: Transformer, idx: Tensor, layer: int, router_weight: Tensor | None = None) -> Tensor:
    """The memory layer's route codes [B, T, R] for token windows ``idx``.

    Token-only routers (SD-B / SD-C) need no backbone; hidden-state routers run only the
    blocks before ``layer``.  ``router_weight`` overrides the router (random-router control)."""
    mem = model.blocks[layer].memory
    w = mem.router.weight if router_weight is None else router_weight
    emb = model.tok_emb(idx)
    s = model.ssm(emb.detach()) if model.ssm is not None else None

    def route(x: Tensor) -> Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return mem.module.quantize_route_logits(F.linear(x.float(), w.float()))

    if mem.router_input != "hidden":
        return route(mem.router_features(emb, s))
    h = emb
    for pos, block in enumerate(model.blocks[:layer]):
        router_in = None
        if block.memory is not None and block.memory.router_input != "hidden":
            router_in = block.memory.router_features(emb, s)
        h = block(h, model.rope_cos, model.rope_sin, router_in)
    return route(model.blocks[layer].attn_norm(h))


def seq_rows(codes: Tensor, order: int, bits: int) -> Tensor:
    """Global sketch / table rows [B, T, R] of the n-grams ending at each t (-1 for t < order-1).
    Same convention as ``Sketch.rows_at`` (oldest code least significant)."""
    b, t, r = codes.shape
    c = codes.long()
    out = torch.full((b, t, r), -1, dtype=torch.long, device=codes.device)
    if t >= order:
        addr = c[:, : t - order + 1].clone()
        for e in range(1, order):
            addr |= c[:, e: e + t - order + 1] << (e * bits)
        out[:, order - 1:] = torch.arange(r, device=codes.device) * (1 << (bits * order)) + addr
    return out


class SketchTensors:
    """A ``Sketch`` on a device: ids int64 / counts fp32 per order, dense summed scores."""

    def __init__(self, sketch: Sketch, device: str | torch.device = "cpu"):
        self.sketch = sketch
        self.ids = {n: torch.from_numpy(sketch.ids[n].astype(np.int64)).to(device) for n in sketch.ngrams}
        self.counts = {n: torch.from_numpy(sketch.counts[n].astype(np.float32)).to(device) for n in sketch.ngrams}

    def scores(self, rows: dict) -> Tensor:
        """rows: order -> int64 [N, R] (-1 = none) -> summed counts [N, vocab] (fp32)."""
        first = next(iter(rows.values()))
        dense = torch.zeros(first.shape[0], self.sketch.vocab, device=first.device)
        for n, r in rows.items():
            valid = (r >= 0)[..., None]
            rr = r.clamp_min(0)
            dense.scatter_add_(1, self.ids[n][rr].flatten(1), (self.counts[n][rr] * valid).flatten(1))
        return dense


# ------------------------------------------------------------------ build
def _amp(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is None:
        return torch.autocast(device.type, enabled=False)
    return torch.autocast(device.type, dtype=amp_dtype)


@torch.no_grad()
def build_sketch(model: Transformer, windows, *, layer: int | None = None, k: int = 16,
                 router_weight: Tensor | None = None, route_chunk: int = 0, mode: str = "mg",
                 amp_dtype: torch.dtype | None = None, max_buffer: int = 1 << 25,
                 log_every: int = 0) -> Sketch:
    """Sketch from an iterable of token windows (int64 [b, T+1] arrays or tensors)."""
    layer = memory_layer(model, layer)
    mem = model.blocks[layer].memory
    m = mem.module
    dev = model.tok_emb.weight.device
    R, bits, ngrams, vocab = mem.routes, mem.bits, tuple(m.ngrams), int(model.cfg["vocab"])
    item_bits = max(1, (vocab - 1).bit_length())
    rc = route_chunk or R
    summ = {n: TopKSummary(R * (1 << (bits * n)), k, item_bits, dev, mode, max_buffer) for n in ngrams}
    seen, t0 = 0, time.time()
    for i, w in enumerate(windows):
        x = torch.as_tensor(w, device=dev)
        inp, tgt = x[:, :-1], x[:, 1:]
        with _amp(dev, amp_dtype):
            codes = memory_codes(model, inp, layer, router_weight)
        for n in ngrams:
            rows = seq_rows(codes, n, bits)
            valid = rows[:, :, 0] >= 0
            rv, tv = rows[valid], tgt[valid]
            for r0 in range(0, R, rc):
                part = rv[:, r0:r0 + rc]
                summ[n].add(part.reshape(-1), tv[:, None].expand_as(part).reshape(-1))
        seen += tgt.numel()
        if log_every and (i + 1) % log_every == 0:
            print(f"[sketch] {seen:,} tokens  {seen / (time.time() - t0):,.0f} tok/s", flush=True)
    ids, counts = {}, {}
    for n in ngrams:
        a, c = summ[n].numpy()
        ids[n] = a.astype(np.uint16)
        counts[n] = np.minimum(c, np.iinfo(np.uint32).max).astype(np.uint32)
    return Sketch(ids, counts, R, bits, ngrams, layer, seen, vocab)


@torch.no_grad()
def sketch_recall(model: Transformer, sketch: Sketch, batches, *, router_weight: Tensor | None = None,
                  amp_dtype: torch.dtype | None = None, pos_chunk: int = 512) -> dict:
    """Recall of the true next token on val windows (positions with every order valid):
    ``recall16_row`` = mean over (route, order) of 'next token in the row's top-K';
    ``recall16`` / ``top1`` = level-0 decoder (counts summed over routes and orders)."""
    dev = model.tok_emb.weight.device
    st = SketchTensors(sketch, dev)
    k = sketch.k
    hit_row = n_row = hit16 = hit1 = n_pos = 0.0
    for x, y in batches:
        x, y = x.to(dev), y.to(dev)
        with _amp(dev, amp_dtype):
            codes = memory_codes(model, x, sketch.layer, router_weight)
        start = max(sketch.ngrams) - 1
        rows = {n: seq_rows(codes, n, sketch.bits)[:, start:].reshape(-1, sketch.routes) for n in sketch.ngrams}
        tgt = y[:, start:].reshape(-1)
        for p0 in range(0, len(tgt), pos_chunk):
            sl = slice(p0, p0 + pos_chunk)
            t = tgt[sl]
            for n, r in rows.items():
                ids, cnt = st.ids[n][r[sl]], st.counts[n][r[sl]]          # [P, R, K]
                hit_row += float(((ids == t[:, None, None]) & (cnt > 0)).any(-1).sum())
                n_row += r[sl].numel()
            sc = st.scores({n: r[sl] for n, r in rows.items()})
            top = sc.topk(k, dim=-1)
            ok = (top.indices == t[:, None]) & (top.values > 0)
            hit16 += float(ok.any(-1).sum())
            hit1 += float(ok[:, 0].sum())
            n_pos += len(t)
    return {"recall16_row": hit_row / max(n_row, 1), f"recall{k}": hit16 / max(n_pos, 1),
            "top1": hit1 / max(n_pos, 1), "val_positions": int(n_pos)}


# ------------------------------------------------------------------ CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True, help="checkpoint file or run dir")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True, help="sketches/<run>.npz")
    p.add_argument("--tokens", type=float, default=300e6)
    p.add_argument("--seq_len", type=int, help="window length (default: the model's seq_len)")
    p.add_argument("--batch", type=int, default=16, help="windows per forward")
    p.add_argument("--layer", type=int, help="memory block position (default: first memory layer)")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--code_source", default="model", choices=["model", "random_router"])
    p.add_argument("--seed", type=int, default=0, help="random_router seed")
    p.add_argument("--merge", default="mg", choices=["mg", "topk"])
    p.add_argument("--route_chunk", type=int, default=0, help="routes per unique() call (0 = all)")
    p.add_argument("--max_buffer", type=int, default=1 << 25, help="buffered (row, token) keys before a merge")
    p.add_argument("--val_tokens", type=float, default=1e6, help="val tokens for recall@16 (0 = skip)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--precision", default="auto", choices=["auto", "fp16", "fp32"])
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args(argv)


def main(argv=None) -> dict:
    from ffnmem.infer import load_model

    a = parse_args(argv)
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) or a.device == "cuda" else "cpu")
    amp = torch.float16 if (a.precision == "fp16" or (a.precision == "auto" and dev.type == "cuda")) else None
    model, cfg = load_model(a.target, device=dev)
    layer = memory_layer(model, a.layer)
    seq_len = a.seq_len or int(model.cfg["seq_len"])
    rw = random_router_weight(model, layer, a.seed) if a.code_source == "random_router" else None
    t0 = time.time()
    sk = build_sketch(model, stream_windows(a.data_dir, seq_len, int(a.tokens), a.batch), layer=layer, k=a.k,
                      router_weight=rw, route_chunk=a.route_chunk, mode=a.merge, amp_dtype=amp,
                      max_buffer=a.max_buffer, log_every=a.log_every)
    build_s = time.time() - t0
    sk.save(a.out)
    info = {"target": cfg.get("_checkpoint"), "code_source": a.code_source, "seed": a.seed, "merge": a.merge,
            "tokens_seen": sk.tokens_seen, "layer": layer, "k": a.k, "seq_len": seq_len, "build_seconds": build_s}
    if a.val_tokens > 0:
        val = ValWindows(a.data_dir, seq_len, int(a.val_tokens))
        info["val"] = sketch_recall(model, sk, val.batches(max(1, a.batch // 4)), router_weight=rw, amp_dtype=amp)
        info["val"]["tokens"] = int(a.val_tokens)
    Path(a.out).with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2), flush=True)
    return info


if __name__ == "__main__":
    main()
