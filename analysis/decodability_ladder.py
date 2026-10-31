"""M2.T4 (RQ1) decodability ladder: how much of the model's next-token distribution
can be read off the memory layer, at increasing decoder capacity?

    python -m analysis.decodability_ladder ngram --data /data/climbmix --tokens 300000000 --out sketches/tok_ngram.npz
    python -m analysis.decodability_ladder train --ckpt runs/w3_sb_1b --data /data/climbmix --run sb \
        --train_tokens 20000000 --out results/lens_sb.pt
    python -m analysis.decodability_ladder eval --ckpt runs/w3_sb_1b --data /data/climbmix --run sb \
        --sketch sketches/sb.npz --tok_sketch sketches/tok_ngram.npz --random_sketch sketches/sb_random.npz \
        --lens results/lens_sb.pt [--drafter runs/posthoc_sb/drafter.pt] --max_tokens 2000000

Every level q is scored against the model's own next-token distribution p,
teacher-forced on val windows: top-1 agreement (argmax q == argmax p), KL(p || q),
sum_v min(p, q) (and, for reference, accuracy against the true next token).

- L0 ``sketch``: ``Sketch.decode_dense`` on the model's codes at the sketch's layer
  (counts summed over routes and orders, ``--smoothing`` uniform mass).
- L0 controls: ``tok_ngram`` -- the same decoder keyed by the raw token bigram
  (x[t-1], x[t]) and trigram (x[t-2..t]) (one "route", top-16 next-token counts per
  row from training tokens, hashed to 2^22 rows per order when the key space is
  larger; built by the ``ngram`` subcommand); ``random_router`` -- a sketch built
  from a randomly initialised router of the same shape (``--random_sketch``), with
  val codes from that router (``router_weight`` stored in the npz if present, else
  ``random_router_weight(model, layer, seed)``).
- L1 ``zero_query``: v2 readout with a zero query, i.e. every real route gets
  exp(0) / (R + exp(b0)) and the Sink the rest (``_real_route_probabilities`` on zero
  scores), then ``readout_o_proj``, final RMSNorm, tied unembedding.  The readout's
  short causal conv is NOT included (it mixes neighbouring positions' readouts; the
  draft readout also omits it).  ``zero_query_conv`` is the same plus the conv over
  the window's zero-query readouts, reported as a variant.
- L2 ``draft``: ``model.draft_logits`` for SD models, else a ``PosthocDrafter``
  checkpoint (``--drafter``; drafter's own codes, the live rule).
- L3 ``tuned_lens``: one affine map d -> d on the residual stream after the memory
  layer's block, applied before the final RMSNorm + unembedding, trained with
  KL(p || q) on training tokens (frozen model; ``train`` subcommand).
- L4 ``model``: p itself (agreement 1, KL 0, sum-min 1).

Outputs: ``results/ladder_<run>.json`` / ``.md`` and ``results/figures/ladder_<run>.png``.
CIs: percentile bootstrap over val windows.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from analysis.code_predictability import bootstrap_weights, summarize
from analysis.dump_codes import Capture, default_device, default_precision, memory_positions, val_windows
from drafting.sketches import Sketch
from ffnmem.data.loader import ShardSet
from ffnmem.infer import load_model
from ffnmem.model import Transformer
from ffnmem.train import amp_context

LEVELS = ("L0_sketch", "L0_tok_ngram", "L0_random_router", "L1_zero_query", "L1_zero_query_conv", "L2_draft",
          "L3_tuned_lens", "L4_model")
METRICS = ("top1_agree", "kl", "sum_min", "acc_true")


# ------------------------------------------------------------------ metrics
def compare(logp: Tensor, logq: Tensor, y: Tensor) -> Tensor:
    """Per position [n, 4]: (argmax agreement, KL(p||q), sum min(p, q), argmax q == y)."""
    logp, logq = logp.float(), logq.float()
    p, q = logp.exp(), logq.exp()
    top_q = logq.argmax(-1)
    kl = torch.where(p > 0, p * (logp - logq), torch.zeros_like(p)).sum(-1)
    return torch.stack(((top_q == logp.argmax(-1)).float(), kl, torch.minimum(p, q).sum(-1),
                        (top_q == y).float()), -1)


def log_probs(probs: Tensor) -> Tensor:
    return probs.float().clamp_min(1e-30).log()


# ------------------------------------------------------------------ level 1
def zero_query_readout(m, codes: Tensor) -> Tensor:
    """v2 readout output (after ``readout_o_proj``, no conv) with a zero query. codes [B, T, R] -> [B, T, d]."""
    tokens = m._lookup_route_chunk(codes, None, 0, m.num_routes, None, None)
    _, value = m._project_memory(tokens)  # [B, T, Hkv, R, hd]
    b, t = codes.shape[:2]
    scores = torch.zeros(b, t, m.readout_num_kv_heads, m.readout_groups, m.num_routes, device=codes.device)
    probs = m._real_route_probabilities(scores)  # uniform over the R rows, Sink in the denominator
    head = torch.matmul(probs.to(value.dtype), value)  # [B, T, Hkv, G, hd]
    return m.readout_o_proj(head.reshape(b, t, m.readout_inner_dim))


def zero_query_hidden(m, codes: Tensor, conv: bool) -> Tensor:
    mixed = zero_query_readout(m, codes)
    return mixed + m.short_conv(mixed) if conv else mixed


# ------------------------------------------------------------------ level 3
class TunedLens(nn.Module):
    """Affine map on the residual stream after block ``layer`` (identity init)."""

    def __init__(self, d_model: int, layer: int):
        super().__init__()
        self.layer = layer
        self.proj = nn.Linear(d_model, d_model)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def logits(self, model: Transformer, h: Tensor) -> Tensor:
        with torch.autocast(device_type=h.device.type, enabled=False):
            return model.output(model.norm(self.proj(h.float())))

    def save(self, path: str | Path, **extra) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.state_dict(), "d_model": self.proj.in_features, "layer": self.layer, **extra}, path)

    @classmethod
    def load(cls, path: str | Path, map_location="cpu") -> "TunedLens":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        lens = cls(int(blob["d_model"]), int(blob["layer"]))
        lens.load_state_dict(blob["state"])
        return lens


class BlockTap:
    """Residual stream after block ``layer`` and the memory router's input at that block."""

    def __init__(self, model: Transformer, layer: int):
        self.resid = None
        self.router_in = None
        blk = model.blocks[layer]
        self._h = [blk.register_forward_hook(self._out)]
        if blk.memory is not None:
            self._h.append(blk.memory.register_forward_pre_hook(self._pre))

    def _out(self, mod, args, out):
        self.resid = out

    def _pre(self, mem, args):
        h_norm = args[1]
        router_in = args[2] if len(args) > 2 else None
        self.router_in = h_norm if mem.router_input == "hidden" else router_in

    def close(self):
        for h in self._h:
            h.remove()


def train_windows(data_dir: str | Path, seq_len: int, n_tokens: int, skip_tokens: int = 0):
    """Contiguous ``seq_len`` windows from the training shards in order (never val)."""
    shards = ShardSet(data_dir)
    need, skip = int(n_tokens) // seq_len, int(skip_tokens)
    for s in shards.meta["shards"]:
        toks = shards.memmap(s["file"])
        n_win = len(toks) // seq_len
        for w in range(n_win):
            if skip >= seq_len:
                skip -= seq_len
                continue
            if need <= 0:
                return
            yield np.asarray(toks[w * seq_len:(w + 1) * seq_len], dtype=np.int64)
            need -= 1


def train_lens(model: Transformer, data_dir: str | Path, *, layer: int, train_tokens: int, batch: int = 8,
               positions: int = 256, lr: float = 1e-3, seed: int = 0, device: torch.device | None = None,
               precision: str | None = None, log_every: int = 50) -> tuple[TunedLens, list]:
    device = device or default_device()
    precision = precision or default_precision(device)
    model.to(device).eval().requires_grad_(False)
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    seq_len = int(model.cfg["seq_len"])
    lens = TunedLens(model.cfg["d_model"], layer).to(device)
    n_steps = max(1, (int(train_tokens) // seq_len) // batch)
    opt = torch.optim.Adam(lens.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, n_steps // 20)))
    tap = BlockTap(model, layer)
    hist, buf, step = [], [], 0
    try:
        for win in train_windows(data_dir, seq_len, train_tokens):
            buf.append(win)
            if len(buf) < batch:
                continue
            x = torch.from_numpy(np.stack(buf)).to(device)
            buf = []
            b, t = x.shape
            with torch.no_grad(), amp_context(device, precision):
                logits = model(x)
            k = min(positions, t)
            t_idx = torch.stack([torch.randperm(t, generator=gen)[:k] for _ in range(b)]).to(device)
            b_idx = torch.arange(b, device=device)[:, None].expand(b, k)
            logp = logits[b_idx, t_idx].float().log_softmax(-1).reshape(b * k, -1)
            logq = lens.logits(model, tap.resid[b_idx, t_idx].reshape(b * k, -1)).log_softmax(-1)
            loss = (logp.exp() * (logp - logq)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            hist.append(float(loss.detach()))
            step += 1
            if log_every and (step % log_every == 0 or step == 1):
                print(f"[lens] step {step}/{n_steps} KL {np.mean(hist[-log_every:]):.4f}", flush=True)
    finally:
        tap.close()
    return lens.eval(), hist


# ------------------------------------------------------------------ level 0 controls
def random_router_weight(model: Transformer, layer: int, seed: int = 0) -> Tensor:
    """The random router ``drafting/build_sketches.py --code_source random_router`` used (same
    function, so the sketch and the val codes come from the same router; on CPU)."""
    from drafting.build_sketches import random_router_weight as builder_weight

    return builder_weight(model, layer, seed).cpu()


def token_ngram_keys(tokens: np.ndarray, n: int, vocab: int) -> np.ndarray:
    """Keys of the n-grams ending at t = n-1 .. T-1 (oldest token least significant). tokens [..., T]."""
    T = tokens.shape[-1]
    key = tokens[..., : T - n + 1].astype(np.uint64)
    for e in range(1, n):
        key = key + tokens[..., e: e + T - n + 1].astype(np.uint64) * np.uint64(vocab ** e)
    return key


def ngram_rows(keys: np.ndarray, n: int, vocab: int, rows_log2: int) -> np.ndarray:
    """Row ids of n-gram keys: the key itself if V^n <= 2^rows_log2, else a multiplicative hash."""
    if vocab ** n <= (1 << rows_log2):
        return keys.astype(np.int64)
    with np.errstate(over="ignore"):
        h = keys.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    return (h >> np.uint64(64 - rows_log2)).astype(np.int64)


def merge_topk(ids: np.ndarray, counts: np.ndarray, rows: np.ndarray, toks: np.ndarray, vocab: int) -> None:
    """Add (row, next token) observations to top-K tables in place (exact within a call, top-K across calls)."""
    k = ids.shape[1]
    pair = rows.astype(np.int64) * vocab + toks.astype(np.int64)
    u, c = np.unique(pair, return_counts=True)
    orow, oslot = np.nonzero(counts)
    allp = np.concatenate((u, orow.astype(np.int64) * vocab + ids[orow, oslot].astype(np.int64)))
    allc = np.concatenate((c.astype(np.float64), counts[orow, oslot].astype(np.float64)))
    up, inv = np.unique(allp, return_inverse=True)
    tot = np.bincount(inv, weights=allc)
    r, t = up // vocab, up % vocab
    order = np.lexsort((-tot, r))
    r, t, tot = r[order], t[order], tot[order]
    idx = np.arange(len(r))
    first = np.r_[True, r[1:] != r[:-1]]
    rank = idx - np.maximum.accumulate(np.where(first, idx, 0))
    keep = rank < k
    touched = r[first]
    ids[touched] = 0
    counts[touched] = 0
    ids[r[keep], rank[keep]] = t[keep]
    counts[r[keep], rank[keep]] = np.minimum(tot[keep], np.iinfo(np.uint32).max)


def build_token_ngram_sketch(data_dir: str | Path, n_tokens: int, *, ngrams=(2, 3), rows_log2: int = 22,
                             k: int = 16, chunk_tokens: int = 50_000_000) -> Sketch:
    """Top-K next-token counts keyed by token n-grams from the first ``n_tokens`` training tokens."""
    shards = ShardSet(data_dir)
    V = int(shards.vocab_size)
    ids = {n: np.zeros((min(V ** n, 1 << rows_log2), k), np.uint16) for n in ngrams}
    counts = {n: np.zeros_like(ids[n], dtype=np.uint32) for n in ngrams}
    seen = 0
    for s in shards.meta["shards"]:
        toks = shards.memmap(s["file"])
        for a in range(0, len(toks), chunk_tokens):
            if seen >= n_tokens:
                break
            chunk = np.asarray(toks[a: min(a + chunk_tokens, a + n_tokens - seen)], dtype=np.int64)
            seen += len(chunk)
            for n in ngrams:
                if len(chunk) <= n:
                    continue
                keys = token_ngram_keys(chunk[:-1], n, V)
                merge_topk(ids[n], counts[n], ngram_rows(keys, n, V, rows_log2), chunk[n:], V)
            print(f"[ngram] {seen:,} tokens", flush=True)
    sk = Sketch(ids, counts, routes=1, bits=0, ngrams=tuple(ngrams), layer=-1, tokens_seen=seen, vocab=V)
    sk.rows_log2 = rows_log2
    return sk


def load_token_sketch(path: str | Path) -> Sketch:
    sk = Sketch.load(path)
    z = np.load(path)
    sk.rows_log2 = int(z["rows_log2"]) if "rows_log2" in z.files else 22
    return sk


def save_token_sketch(sk: Sketch, path: str | Path) -> None:
    sk.save(path)
    z = dict(np.load(path))
    z["rows_log2"] = np.array(sk.rows_log2)
    np.savez(path, **z)


class SketchDecoder:
    """``Sketch.decode_dense`` with the tables converted to tensors once (on ``device``), not per call."""

    def __init__(self, sk: Sketch, device: torch.device | str = "cpu"):
        self.sk, self.vocab, self.device = sk, sk.vocab, torch.device(device)
        self.ids = {n: torch.from_numpy(sk.ids[n].astype(np.int32)).to(self.device) for n in sk.ngrams}
        self.counts = {n: torch.from_numpy(sk.counts[n].astype(np.float32)).to(self.device) for n in sk.ngrams}

    def probs(self, rows: dict, smoothing: float = 1e-3) -> Tensor:
        ids, cnts = [], []
        for n, r in rows.items():
            r = torch.as_tensor(r, device=self.device)
            valid = r >= 0
            rr = r.clamp_min(0)
            ids.append(self.ids[n][rr].flatten(1).long())
            cnts.append((self.counts[n][rr] * valid[..., None]).flatten(1))
        ids_t, c_t = torch.cat(ids, 1), torch.cat(cnts, 1)
        dense = torch.zeros(ids_t.shape[0], self.vocab, device=self.device).scatter_add_(1, ids_t, c_t)
        tot = dense.sum(-1, keepdim=True)
        p = torch.where(tot > 0, dense / tot.clamp_min(1e-30), torch.full_like(dense, 1.0 / self.vocab))
        return (1 - smoothing) * p + smoothing / self.vocab


def token_rows(sk: Sketch, x: np.ndarray) -> dict:
    """Rows [B*T, 1] per order for windows x [B, T] (-1 where the n-gram crosses the window start)."""
    b, t = x.shape
    out = {}
    for n in sk.ngrams:
        r = np.full((b, t), -1, np.int64)
        if t >= n:
            r[:, n - 1:] = ngram_rows(token_ngram_keys(x, n, sk.vocab), n, sk.vocab, sk.rows_log2)
        out[n] = r.reshape(-1, 1)
    return out


def code_rows(sk: Sketch, codes: np.ndarray) -> dict:
    """Rows [B*T, R] per order for codes [B, T, R] (window-local n-grams)."""
    return {n: np.concatenate([sk.rows_at(c, n) for c in codes]) for n in sk.ngrams}


# ------------------------------------------------------------------ ladder
@torch.no_grad()
def ladder(model: Transformer, data_dir: str | Path, *, layer: int | None = None, sketch: Sketch | None = None,
           tok_sketch: Sketch | None = None, random_sketch: Sketch | None = None, random_weight: Tensor | None = None,
           lens: TunedLens | None = None, drafter=None, max_tokens: int = 2_000_000, batch: int = 4,
           chunk: int = 2048, smoothing: float = 1e-3, n_boot: int = 1000, seed: int = 0,
           device: torch.device | None = None, precision: str | None = None, val_tokens: int | None = None,
           levels=LEVELS) -> dict:
    device = device or default_device()
    precision = precision or default_precision(device)
    model.to(device).eval()
    layer = memory_positions(model)[0] if layer is None else int(layer)
    mem = model.blocks[layer].memory
    m = mem.module
    seq_len = int(model.cfg["seq_len"])
    V = int(model.cfg["vocab"])
    val = val_windows(data_dir, seq_len, max_tokens, val_tokens)
    for sk in (sketch, tok_sketch, random_sketch):
        if sk is not None and sk.vocab != V:
            raise ValueError(f"sketch vocab {sk.vocab} != model vocab {V}")
    sk_layer = sketch.layer if sketch is not None else layer
    rnd_layer = random_sketch.layer if random_sketch is not None else layer
    if random_sketch is not None:
        random_weight = (random_weight if random_weight is not None else random_router_weight(model, rnd_layer)).to(device)
    if drafter is None and getattr(model, "draft", None) is not None:
        drafter = "joint"
    avail = {
        "L0_sketch": sketch is not None, "L0_tok_ngram": tok_sketch is not None,
        "L0_random_router": random_sketch is not None, "L1_zero_query": True, "L1_zero_query_conv": True,
        "L2_draft": drafter is not None, "L3_tuned_lens": lens is not None, "L4_model": True,
    }
    use = [lv for lv in levels if avail[lv]]
    if lens is not None:
        lens.to(device).eval()
        if lens.layer != layer:
            print(f"[ladder] lens trained after block {lens.layer}; ladder layer is {layer}", flush=True)
    sums = {lv: [] for lv in use}
    decoders = {lv: SketchDecoder(sk, device) for lv, sk in
                (("L0_sketch", sketch), ("L0_tok_ngram", tok_sketch), ("L0_random_router", random_sketch))
                if lv in use}
    taps ={layer: BlockTap(model, layer)}
    if lens is not None and lens.layer not in taps:
        taps[lens.layer] = BlockTap(model, lens.layer)
    if rnd_layer not in taps:
        taps[rnd_layer] = BlockTap(model, rnd_layer)
    t0, done = time.time(), 0
    try:
        with Capture(model, sink=False) as cap:
            for x, y in val.batches(batch):
                x, y = x.to(device), y.to(device)
                b, t = x.shape
                N = b * t
                with amp_context(device, precision):
                    logits = model(x).reshape(N, V)
                    fns = {}
                    if "L0_sketch" in use:
                        rows = code_rows(sketch, cap.codes[sk_layer].cpu().numpy())
                        fns["L0_sketch"] = lambda sl, rows=rows: log_probs(
                            decoders["L0_sketch"].probs({n: r[sl] for n, r in rows.items()}, smoothing))
                    if "L0_tok_ngram" in use:
                        trows = token_rows(tok_sketch, x.cpu().numpy())
                        fns["L0_tok_ngram"] = lambda sl, rows=trows: log_probs(
                            decoders["L0_tok_ngram"].probs({n: r[sl] for n, r in rows.items()}, smoothing))
                    if "L0_random_router" in use:
                        with torch.autocast(device_type=device.type, enabled=False):
                            q = F.linear(taps[rnd_layer].router_in.float(), random_weight.float())
                        rc = model.blocks[rnd_layer].memory.module.quantize_route_logits(q)
                        rrows = code_rows(random_sketch, rc.cpu().numpy())
                        fns["L0_random_router"] = lambda sl, rows=rrows: log_probs(
                            decoders["L0_random_router"].probs({n: r[sl] for n, r in rows.items()}, smoothing))
                    codes = cap.codes[layer]
                    for lv, conv in (("L1_zero_query", False), ("L1_zero_query_conv", True)):
                        if lv in use:
                            hid = torch.cat([zero_query_hidden(m, codes[i:i + 1], conv) for i in range(b)]).reshape(N, -1)
                            fns[lv] = lambda sl, hid=hid: model.output(model.norm(hid[sl])).float().log_softmax(-1)
                    if "L2_draft" in use:
                        b_all = torch.arange(b, device=device).repeat_interleave(t)
                        t_all = torch.arange(t, device=device).repeat(b)
                        if drafter == "joint":
                            fns["L2_draft"] = lambda sl: model.draft_logits(x, b_all[sl], t_all[sl]).float().log_softmax(-1)
                        else:
                            fns["L2_draft"] = lambda sl: drafter(x, b_all[sl], t_all[sl]).float().log_softmax(-1)
                    if "L3_tuned_lens" in use:
                        res = taps[lens.layer].resid.reshape(N, -1)
                        fns["L3_tuned_lens"] = lambda sl, res=res: lens.logits(model, res[sl]).float().log_softmax(-1)
                    acc = {lv: torch.zeros(b, 4, dtype=torch.float64, device=device) for lv in use}
                    yf = y.reshape(N)
                    for s in range(0, N, chunk):
                        sl = slice(s, min(N, s + chunk))
                        lp = logits[sl].float().log_softmax(-1)
                        win = torch.arange(sl.start, sl.stop, device=device) // t
                        for lv in use:
                            lq = lp if lv == "L4_model" else fns[lv](sl)
                            acc[lv].index_add_(0, win, compare(lp, lq, yf[sl]).double())
                for lv in use:
                    sums[lv].append(acc[lv].cpu().numpy())
                done += N
                if done % (50 * N) == 0:
                    print(f"[ladder] {done:,} positions ({time.time() - t0:.0f}s)", flush=True)
    finally:
        for tp in taps.values():
            tp.close()
    W = bootstrap_weights(val.num_windows, n_boot, seed)
    denom = W.sum(1) * seq_len
    out = {"levels": {}, "layer": layer, "n_windows": val.num_windows, "n_positions": val.num_windows * seq_len,
           "smoothing": smoothing, "precision": precision,
           "skipped": [lv for lv in levels if not avail[lv]],
           "level2": None if drafter is None else ("joint draft head" if drafter == "joint" else "post-hoc drafter"),
           "ci": "95% percentile bootstrap over val windows"}
    for lv in use:
        S = np.concatenate(sums[lv])  # [n_windows, 4]
        A = W @ S
        out["levels"][lv] = {k: summarize(A[:, j] / denom) for j, k in enumerate(METRICS)}
    return out


# ------------------------------------------------------------------ reporting
DESCRIPTIONS = {
    "L0_sketch": "count sketch keyed by the model's codes",
    "L0_tok_ngram": "control: sketch keyed by token bigram/trigram",
    "L0_random_router": "control: sketch keyed by a random router's codes",
    "L1_zero_query": "v2 readout, zero query (uniform rows + Sink), no conv",
    "L1_zero_query_conv": "as L1 + the readout's short conv",
    "L2_draft": "draft head (joint for SD, post-hoc otherwise)",
    "L3_tuned_lens": "tuned lens after the memory block",
    "L4_model": "the model itself",
}


def markdown(res: dict) -> str:
    fmt = lambda s: f"{s['value']:.4f} [{s['lo']:.4f}, {s['hi']:.4f}]"  # noqa: E731
    lines = [f"# Decodability ladder: {res.get('run', '')}", "",
             f"checkpoint `{res.get('checkpoint')}` (step {res.get('step')}), memory layer {res['layer']}, "
             f"{res['n_positions']:,} val positions; 95% bootstrap CIs over windows.", "",
             "| level | decoder | top-1 agree | KL(p‖q) | Σ min(p,q) | acc (true next) |",
             "|---|---|---|---|---|---|"]
    for lv, r in res["levels"].items():
        lines.append(f"| {lv} | {DESCRIPTIONS[lv]} | " + " | ".join(fmt(r[k]) for k in METRICS) + " |")
    if res.get("skipped"):
        lines += ["", "Skipped (no input given): " + ", ".join(res["skipped"])]
    return "\n".join(lines) + "\n"


def plot(res: dict, path: str | Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ladder] matplotlib not installed; no figure", flush=True)
        return False
    lv = list(res["levels"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    y = np.arange(len(lv))
    for a, keys in ((ax[0], ("top1_agree", "sum_min")), (ax[1], ("kl",))):
        for j, k in enumerate(keys):
            v = np.array([res["levels"][x][k]["value"] for x in lv])
            err = np.array([[res["levels"][x][k]["value"] - res["levels"][x][k]["lo"],
                             res["levels"][x][k]["hi"] - res["levels"][x][k]["value"]] for x in lv]).T
            a.barh(y + 0.4 * j - 0.2 * (len(keys) - 1), v, height=0.38, xerr=err, capsize=2, label=k)
        a.set_yticks(y, lv)
        a.invert_yaxis()
        a.legend()
    ax[0].set(xlim=(0, 1), title=f"{res.get('run', '')}: agreement with the model")
    ax[1].set(title="KL(p || q) (nats)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


# ------------------------------------------------------------------ CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--ckpt", required=True)
        p.add_argument("--data", required=True)
        p.add_argument("--run", required=True)
        p.add_argument("--layer", type=int, default=None, help="memory block position (default: first)")
        p.add_argument("--batch", type=int, default=4)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", default=None)
        p.add_argument("--precision", choices=["fp16", "fp32"], default=None)

    tr = sub.add_parser("train", help="train the level-3 tuned lens")
    common(tr)
    tr.add_argument("--train_tokens", type=int, default=20_000_000)
    tr.add_argument("--positions", type=int, default=256, help="random positions per window in the KL loss")
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--out", default=None, help="default results/lens_<run>.pt")

    ng = sub.add_parser("ngram", help="build the token bigram/trigram control sketch")
    ng.add_argument("--data", required=True)
    ng.add_argument("--tokens", type=int, default=300_000_000)
    ng.add_argument("--rows_log2", type=int, default=22)
    ng.add_argument("--out", required=True)

    ev = sub.add_parser("eval", help="score every available level on val")
    common(ev)
    ev.add_argument("--sketch", default=None, help="level-0 sketch of this model (drafting/build_sketches.py)")
    ev.add_argument("--tok_sketch", default=None, help="token n-gram control sketch (ngram subcommand)")
    ev.add_argument("--random_sketch", default=None, help="sketch built from a random router")
    ev.add_argument("--random_router_seed", type=int, default=0, help="used if the npz has no router_weight")
    ev.add_argument("--lens", default=None, help="tuned lens from the train subcommand")
    ev.add_argument("--drafter", default=None, help="PosthocDrafter checkpoint (unmodified models)")
    ev.add_argument("--levels", nargs="*", default=list(LEVELS))
    ev.add_argument("--max_tokens", type=int, default=2_000_000)
    ev.add_argument("--val_tokens", type=int, default=20_000_000)
    ev.add_argument("--chunk", type=int, default=2048)
    ev.add_argument("--smoothing", type=float, default=1e-3)
    ev.add_argument("--n_boot", type=int, default=1000)
    ev.add_argument("--out_dir", default="results")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.cmd == "ngram":
        sk = build_token_ngram_sketch(args.data, args.tokens, rows_log2=args.rows_log2)
        save_token_sketch(sk, args.out)
        print(f"[ngram] wrote {args.out} ({sk.tokens_seen:,} tokens)", flush=True)
        return {"tokens_seen": sk.tokens_seen, "out": args.out}
    device = torch.device(args.device) if args.device else default_device()
    model, cfg = load_model(args.ckpt, device=device)
    layer = memory_positions(model)[0] if args.layer is None else args.layer
    if args.cmd == "train":
        lens, hist = train_lens(model, args.data, layer=layer, train_tokens=args.train_tokens, batch=args.batch,
                                positions=args.positions, lr=args.lr, seed=args.seed, device=device,
                                precision=args.precision)
        out = args.out or f"results/lens_{args.run}.pt"
        lens.save(out, checkpoint=cfg.get("_checkpoint"), train_tokens=args.train_tokens, kl_history=hist)
        print(f"[lens] wrote {out} (final KL {np.mean(hist[-20:]):.4f})", flush=True)
        return {"out": out, "kl_history": hist}
    sketch = Sketch.load(args.sketch) if args.sketch else None
    tok = load_token_sketch(args.tok_sketch) if args.tok_sketch else None
    rnd, rnd_w = None, None
    if args.random_sketch:
        rnd = Sketch.load(args.random_sketch)
        z = np.load(args.random_sketch)
        if "router_weight" in z.files:
            rnd_w = torch.from_numpy(z["router_weight"]).float()
        else:
            rnd_w = random_router_weight(model, rnd.layer if rnd.layer >= 0 else layer, args.random_router_seed)
            print("[ladder] random sketch has no router_weight; using random_router_weight(seed="
                  f"{args.random_router_seed}) -- it must match the router the sketch was built with", flush=True)
    lens = TunedLens.load(args.lens) if args.lens else None
    drafter = None
    if args.drafter:
        from drafting.posthoc_model import PosthocDrafter

        drafter = PosthocDrafter.load(args.drafter, model).eval()
    res = ladder(model, args.data, layer=layer, sketch=sketch, tok_sketch=tok, random_sketch=rnd,
                 random_weight=rnd_w, lens=lens, drafter=drafter, max_tokens=args.max_tokens, batch=args.batch,
                 chunk=args.chunk, smoothing=args.smoothing, n_boot=args.n_boot, seed=args.seed, device=device,
                 precision=args.precision, val_tokens=args.val_tokens, levels=args.levels)
    res.update(run=args.run, checkpoint=cfg.get("_checkpoint"), step=cfg.get("_step"),
               inputs={k: getattr(args, k) for k in ("sketch", "tok_sketch", "random_sketch", "lens", "drafter")})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"ladder_{args.run}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out_dir / f"ladder_{args.run}.md").write_text(markdown(res), encoding="utf-8")
    plot(res, out_dir / "figures" / f"ladder_{args.run}.png")
    print(markdown(res), flush=True)
    return res


if __name__ == "__main__":
    main()
