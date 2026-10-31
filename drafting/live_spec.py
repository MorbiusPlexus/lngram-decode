"""Live greedy speculative decoding: target on GPU, drafter on CPU (M5.T3, chain).

    python -m drafting.live_spec --ckpt runs/w3_sdb_1b --bundle bundles/sdb.bin --val_bin data/val.bin \
        --n_prompts 500 --prompt_len 256 --n_new 128 --gamma 1 3 4 --verifier graph --run sdb_fp32
    python -m drafting.live_spec ... --dtype fp16 --run sdb_fp16          # match rate + near-tie check
    tokens, stats = speculative_generate(model, drafter, prompt, n_new=128, gamma=4)

Per block (batch 1).  The drafter has committed every known token; the target has processed
all but the last one (the "pending" token).

1. Submit to a worker thread (the C++ drafter releases the GIL): the main branch
   ``draft(gamma + 1, hyp=D)`` (assumes full acceptance; its first token guesses the
   verifier's bonus token) and the repair branch ``draft(gamma, hyp=[r1])`` (r1 = drafter's
   runner-up at the first draft position).
2. Launch verification of ``[pending, d1..d_gamma]`` with ``decode_step`` (eager), or the
   fixed-shape ``static_verify`` (``--verifier static``), captured as a CUDA graph
   (``--verifier graph``; falls back to eager replay if capture fails).
3. Wait for the verifier's argmax; accept the longest matching prefix k plus the verifier's
   token, ``state.truncate`` to roll back the rejected positions, commit to the drafter.
4. Next block: the main branch if k == gamma and its bonus guess was right, the repair
   branch if k == 0 and the verifier picked r1, else draft synchronously (a miss).

Memory rows for verification: by default the model's own (GPU) table (B0).  A ``RowSource``
(``provider(state, block_tokens, ctx_codes) -> row_provider | None``) can supply rows through
``decode_step``'s ``row_provider`` hook -- the seam for M4's exact block staging
(serving/prefetch.py).  ``HostExactRows`` is a simple reference implementation (rows gathered
on the CPU from the drafter's codes, checked against the target's codes, synchronous fallback).

Exactness: fp32 tokens must equal plain greedy decoding; under fp16 autocast the match rate
is reported and every first mismatch is checked to be a near-tie of the plain decode
(top-2 logit margin < 1e-2).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch
from torch import Tensor

from ffnmem.infer import DecodeState, decode_step, greedy_generate, rope_at, ssm_continue

NEAR_TIE = 1e-2


# ---------------------------------------------------------------------- fixed-shape verify
def _gqa_capturable(m, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """v2 ``_manual_gqa`` (eval) with the Sink logit as a Python scalar: the vendored
    ``scores.new_tensor(sink)`` is a host-to-device copy, which CUDA graph capture rejects.
    Same arithmetic: max(amax, sink), exp(s - max) / (sum + exp(sink - max))."""
    b, t = query.shape[:2]
    grouped = query.view(b, t, m.readout_num_kv_heads, m.readout_groups, m.readout_head_dim)
    scores = torch.matmul(grouped, key.transpose(-1, -2)).float() * m.softmax_scale
    sink = float(m.readout_sink_logit)
    mx = scores.amax(dim=-1, keepdim=True).clamp_min(sink)
    ex = torch.exp(scores - mx)
    probs = ex / (ex.sum(dim=-1, keepdim=True) + torch.exp(sink - mx))
    return torch.matmul(probs.to(value.dtype), value).reshape(b, t, m.readout_num_heads, m.readout_head_dim)


@torch.no_grad()
def static_verify(model, state: DecodeState, tokens: Tensor, L_t: Tensor) -> Tensor:
    """``decode_step`` with shapes independent of the cache length (CUDA-graph capturable).

    tokens [B, n] (device buffer), L_t 0-d long device tensor = ``state.length``.  Attention
    spans all ``max_len`` cache slots with a mask, caches are written with ``index_copy_``
    and history is gathered with clamped indices + validity masks.  The caller sets
    ``state.length = L + n`` afterwards.  Not supported: ``row_provider``, tree masks."""
    b, n = tokens.shape
    dev = tokens.device
    T = state.max_len
    pos_idx = L_t + torch.arange(n, device=dev)
    positions = pos_idx.expand(b, n)
    has_prev = L_t > 0
    prev_idx = (L_t - 1).clamp_min(0).view(1)
    emb = model.tok_emb(tokens)
    s_new = None
    if model.ssm is not None:
        s_prev = state.ssm.index_select(1, prev_idx)[:, 0] * has_prev.to(state.ssm.dtype)
        s_new = ssm_continue(model, emb.detach(), s_prev)
    h = emb
    cos, sin = model.rope_cos, model.rope_sin
    mask = torch.arange(T, device=dev)[None, :] <= pos_idx[:, None]  # [n, T]
    for pos, block in enumerate(model.blocks):
        if block.memory is not None:
            mem = block.memory
            m = mem.module
            if mem.router_input == "hidden":
                q_logits = mem.route_logits(block.attn_norm(h))
            else:
                prev_emb = model.tok_emb(state.tokens.index_select(1, prev_idx)).to(emb.dtype)
                start = mem.router_features.start.to(emb.dtype).view(1, 1, -1)
                prev_emb = torch.where(has_prev, prev_emb, start)
                s_cat = None if s_new is None else torch.cat((s_new[:, :1], s_new), 1)
                q_logits = mem.route_logits(mem.router_features(torch.cat((prev_emb, emb), 1), s_cat)[:, 1:])
            codes = m.quantize_route_logits(q_logits)
            hist = m.route_history_length
            hidx = L_t - hist + torch.arange(hist, device=dev)
            hcodes = state.codes[pos].index_select(1, hidx.clamp_min(0)).to(codes.dtype)
            ctx = torch.cat((hcodes, codes), 1)
            seg = torch.cat(((hidx >= 0).to(torch.int32), torch.ones(n, dtype=torch.int32, device=dev)))
            state.codes[pos].index_copy_(1, pos_idx, codes.to(state.codes[pos].dtype))
            rt = m._lookup_route_chunk(ctx, None, 0, m.num_routes, seg[None].expand(b, -1), None, target_length=n)
            key, value = m._project_memory(rt)
            head = _gqa_capturable(m, m._project_query(h), key, value)
            mixed = m.readout_o_proj(head.reshape(b, n, m.readout_inner_dim))
            conv = m.short_conv
            ci = state.conv_in[pos]
            ci.index_copy_(1, pos_idx, conv.norm(mixed).to(ci.dtype))
            lp = conv.left_padding
            widx = L_t - lp + torch.arange(lp + n, device=dev)
            window = ci.index_select(1, widx.clamp_min(0)) * (widx >= 0).to(ci.dtype)[None, :, None]
            conv_out = conv.act(conv.conv(window.transpose(1, 2).to(mixed.dtype))).transpose(1, 2)
            h = h + (mixed + conv_out)
        a = block.attn
        x = block.attn_norm(h)
        q = rope_at(a.wq(x).view(b, n, a.n_heads, a.head_dim), positions, cos, sin).transpose(1, 2)
        k = rope_at(a.wk(x).view(b, n, a.n_kv_heads, a.head_dim), positions, cos, sin).transpose(1, 2)
        v = a.wv(x).view(b, n, a.n_kv_heads, a.head_dim).transpose(1, 2)
        state.k[pos].index_copy_(2, pos_idx, k.to(state.k[pos].dtype))
        state.v[pos].index_copy_(2, pos_idx, v.to(state.v[pos].dtype))
        kk, vv = state.k[pos], state.v[pos]
        g = a.n_heads // a.n_kv_heads
        if g > 1:
            kk, vv = kk.repeat_interleave(g, 1), vv.repeat_interleave(g, 1)
        out = torch.nn.functional.scaled_dot_product_attention(q, kk.to(q.dtype), vv.to(q.dtype), attn_mask=mask)
        h = h + a.wo(out.transpose(1, 2).reshape(b, n, a.n_heads * a.head_dim))
        h = h + block.ffn(block.ffn_norm(h))
    if s_new is not None:
        state.ssm.index_copy_(1, pos_idx, s_new)
    state.tokens.index_copy_(1, pos_idx, tokens)
    return model.output(model.norm(h))


class Verifier:
    """Verification of a fixed-size block: ``eager`` (decode_step), ``static`` (static_verify,
    eager) or ``graph`` (static_verify captured as a CUDA graph; static eager if capture fails)."""

    def __init__(self, model, state: DecodeState, n: int, mode: str = "eager"):
        self.model, self.state, self.n, self.mode = model, state, n, mode
        self.graph = None
        self.error: str | None = None
        dev = state.tokens.device
        if mode == "eager":
            return
        self.tok = torch.zeros(1, n, dtype=torch.long, device=dev)
        self.L = torch.full((), state.length, dtype=torch.long, device=dev)
        if mode == "graph":
            if dev.type != "cuda":
                self.error, self.mode = "CUDA graphs need a CUDA device", "static"
                return
            try:  # warm-up / capture write past state.length only (overwritten before use)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        static_verify(model, state, self.tok, self.L)
                torch.cuda.current_stream().wait_stream(side)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self.out = static_verify(model, state, self.tok, self.L)
                self.graph = g
            except Exception as exc:  # noqa: BLE001 -- fall back, report why
                self.error, self.mode, self.graph = f"{type(exc).__name__}: {exc}", "static", None

    def __call__(self, block: list, row_provider=None) -> Tensor:
        st = self.state
        L = st.length
        if self.mode == "eager" or row_provider is not None:
            t = torch.tensor([block], dtype=torch.long, device=st.tokens.device)
            return decode_step(self.model, st, t, row_provider=row_provider)
        self.tok.copy_(torch.tensor([block], dtype=torch.long), non_blocking=True)
        self.L.fill_(L)
        if self.graph is not None:
            self.graph.replay()
            logits = self.out
        else:
            logits = static_verify(self.model, st, self.tok, self.L)
        st.length = L + self.n
        return logits


# ---------------------------------------------------------------------- memory rows (seam for M4)
class RowSource(Protocol):
    def provider(self, state: DecodeState, block_tokens: list, ctx_codes: np.ndarray | None) -> Callable | None:
        """Called before verifying ``block_tokens`` (positions L .. L+n-1).  ``ctx_codes``
        [h + n, R] are the drafter's codes for positions max(0, L-H) .. L+n-1 (H = n-gram
        history).  Return a ``decode_step`` row_provider or None (rows from the model's table)."""


class HostExactRows:
    """Reference row source: host copy of the v2 tables, rows staged from the drafter's codes,
    checked against the target's codes inside the callback (synchronous gather on a miss).
    Correct but not fast (the check syncs); M4's serving/prefetch.py replaces it for timing."""

    def __init__(self, model, layer: int):
        self.layer = layer
        self.m = model.blocks[layer].memory.module
        pin = torch.cuda.is_available() and model.tok_emb.weight.is_cuda
        self.tables = {}
        for n in self.m.ngrams:
            w = self.m.tables[f"ngram_{n}"].weight.detach().cpu()
            self.tables[n] = w.pin_memory() if pin else w
        self.hits = self.misses = 0

    def _gather(self, ctx: np.ndarray, n: int) -> Tensor:
        """Route tokens [1, n, R, Dt] for the last n positions of ``ctx`` (codes from the sequence start
        or with at least H positions of history)."""
        m = self.m
        r = np.arange(m.num_routes, dtype=np.int64)
        c = ctx.astype(np.int64)
        tot = len(c)
        parts = []
        for o in m.ngrams:
            out = torch.zeros(n, m.num_routes, m.memory_dim, dtype=self.tables[o].dtype)
            for i in range(n):
                j = tot - n + i
                if j < o - 1:
                    continue
                addr = np.zeros(m.num_routes, np.int64)
                for k in range(o):
                    addr |= c[j - o + 1 + k] << (k * m.bits_per_route)
                out[i] = self.tables[o][torch.from_numpy(r * (1 << m.bits_per_route) ** o + addr)]
            parts.append(out)
        return torch.cat(parts, -1)[None]

    def provider(self, state: DecodeState, block_tokens: list, ctx_codes: np.ndarray | None):
        if ctx_codes is None:
            return None
        n = len(block_tokens)
        staged = self._gather(ctx_codes, n)
        hist = self.m.route_history_length

        def cb(pos: int, st: DecodeState, codes_new: Tensor):
            if pos != self.layer:
                return None
            L = st.length
            actual = torch.cat((st.codes[pos][0, max(0, L - hist):L].to(torch.long),
                                codes_new[0].to(torch.long)), 0).cpu().numpy()
            if actual.shape == ctx_codes.shape and np.array_equal(actual, ctx_codes):
                self.hits += 1
                rows = staged
            else:
                self.misses += 1
                rows = self._gather(actual, n)
            return rows.to(st.tokens.device, non_blocking=True)

        return cb


# ---------------------------------------------------------------------- the loop
def _device_sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _amp(dev: torch.device, amp_dtype):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(dev.type, dtype=amp_dtype, cache_enabled=False)


@torch.no_grad()
def speculative_generate(model, drafter, prompt, n_new: int, gamma: int, *, row_source: RowSource | None = None,
                         target_codes: bool = False, branches: bool = True, verifier: str = "eager",
                         amp_dtype=None, executor: ThreadPoolExecutor | None = None) -> tuple[list, dict]:
    """Greedy speculative continuation of one prompt -> (n_new tokens, stats).

    ``target_codes``: the drafter's committed history uses the target's codes at the drafter's
    memory layer (post-hoc drafters on unmodified targets); drafted positions use its own."""
    dev = model.tok_emb.weight.device
    prompt = [int(x) for x in (prompt.view(-1).tolist() if isinstance(prompt, Tensor) else prompt)]
    P = len(prompt)
    layer = int(drafter.cfg["layer"])
    hist = model.blocks[layer].memory.module.route_history_length
    state = DecodeState.empty(model, 1, P + n_new + gamma + 1)
    own_pool = executor is None and branches
    pool = ThreadPoolExecutor(1) if own_pool else executor
    use_cuda = dev.type == "cuda"
    need_codes = row_source is not None
    st = {"blocks": 0, "accepted": [], "branch": {"main": 0, "repair": 0, "miss": 0}, "verify_span_ms": [],
          "draft_sync_ms": [], "wait_ms": []}
    try:
        with _amp(dev, amp_dtype):
            ver = Verifier(model, state, gamma + 1, verifier)
            st["verifier"] = ver.mode
            st["verifier_error"] = ver.error
            if P > 1:
                decode_step(model, state, torch.tensor([prompt[:-1]], device=dev))
            drafter.reset()
            if target_codes and P > 1:
                drafter.commit(prompt[:-1], codes=state.codes[layer][0, :P - 1].cpu().numpy())
                drafter.commit(prompt[-1:])
            else:
                drafter.commit(prompt)
            _device_sync(dev)
            t_start = time.perf_counter()
            t0 = time.perf_counter()
            cur = drafter.draft(gamma, last_codes=need_codes)
            st["draft_sync_ms"].append((time.perf_counter() - t0) * 1e3)
            D, r1, dcodes = cur.tokens, cur.runner_up, cur.codes
            pending = prompt[-1]
            out: list = []
            y_host = torch.empty(gamma + 1, dtype=torch.long, pin_memory=True) if use_cuda else None
            while len(out) < n_new:
                L = state.length
                block = [pending] + D
                f_main = f_rep = None
                if branches:
                    f_main = pool.submit(drafter.draft, gamma + 1, D, None, need_codes)
                    if r1 >= 0:
                        f_rep = pool.submit(drafter.draft, gamma, [r1], None, need_codes)
                rp = None
                if row_source is not None:
                    cc = drafter.committed_codes()
                    ctx = np.concatenate((cc[max(0, L - hist):L + 1], dcodes[-gamma:] if gamma else dcodes[:0]), 0)
                    rp = row_source.provider(state, block, ctx)
                t0 = time.perf_counter()
                if use_cuda:
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                logits = ver(block, rp)
                y_dev = logits[0].argmax(-1)
                if use_cuda:
                    e1.record()
                    y_host.copy_(y_dev, non_blocking=True)
                    done = torch.cuda.Event()
                    done.record()
                    tw = time.perf_counter()
                    done.synchronize()
                    st["wait_ms"].append((time.perf_counter() - tw) * 1e3)
                    st["verify_span_ms"].append(e0.elapsed_time(e1))
                    y = y_host.tolist()
                else:
                    y = y_dev.tolist()
                    st["verify_span_ms"].append((time.perf_counter() - t0) * 1e3)
                k = 0
                while k < gamma and D[k] == y[k]:
                    k += 1
                new = D[:k] + [y[k]]
                state.truncate(L + k + 1)
                out.extend(new)
                st["blocks"] += 1
                st["accepted"].append(k)
                main = f_main.result() if f_main is not None else None
                rep = f_rep.result() if f_rep is not None else None
                if target_codes:
                    n_c = drafter.length
                    drafter.truncate(n_c - 1)
                    drafter.commit([pending] + D[:k], codes=state.codes[layer][0, L:L + k + 1].cpu().numpy())
                    drafter.commit([y[k]])
                else:
                    drafter.commit(new)
                pending = y[k]
                if len(out) >= n_new:
                    break
                if main is not None and k == gamma and y[gamma] == main.tokens[0]:
                    D, r1, dcodes = main.tokens[1:], int(main.topk_ids[1, 1]) if main.topk_ids.shape[1] > 1 else -1, \
                        main.codes[gamma + 1:] if need_codes else main.codes
                    st["branch"]["main"] += 1
                elif rep is not None and k == 0 and y[0] == r1:
                    D, r1, dcodes = rep.tokens, rep.runner_up, rep.codes[1:] if need_codes else rep.codes
                    st["branch"]["repair"] += 1
                else:
                    t0 = time.perf_counter()
                    cur = drafter.draft(gamma, last_codes=need_codes)
                    st["draft_sync_ms"].append((time.perf_counter() - t0) * 1e3)
                    D, r1, dcodes = cur.tokens, cur.runner_up, cur.codes
                    st["branch"]["miss"] += 1
            _device_sync(dev)
            st["seconds"] = time.perf_counter() - t_start
    finally:
        if own_pool:
            pool.shutdown(wait=True)
    return out[:n_new], st


@torch.no_grad()
def greedy_with_margins(model, prompt, n_new: int, amp_dtype=None) -> tuple[list, np.ndarray]:
    """Plain greedy decoding (same steps as ``greedy_generate``) + the top-2 logit margin per step."""
    dev = model.tok_emb.weight.device
    p = torch.as_tensor(prompt, device=dev).view(1, -1)
    state = DecodeState.empty(model, 1, p.shape[1] + n_new)
    toks, margins = [], []
    with _amp(dev, amp_dtype):
        logits = decode_step(model, state, p)
        for i in range(n_new):
            top = logits[0, -1].float().topk(2)
            toks.append(top.indices[0])
            margins.append(top.values[0] - top.values[1])
            if i == n_new - 1:
                break
            logits = decode_step(model, state, top.indices[:1].view(1, 1))
    return [int(t) for t in torch.stack(toks).tolist()], torch.stack(margins).cpu().numpy()


@torch.no_grad()
def timed_greedy(model, prompt, n_new: int, amp_dtype=None) -> tuple[list, float]:
    dev = model.tok_emb.weight.device
    p = torch.as_tensor(prompt, device=dev).view(1, -1)
    _device_sync(dev)
    t0 = time.perf_counter()
    gen, _ = greedy_generate(model, p, n_new, amp_dtype=amp_dtype)
    _device_sync(dev)
    return gen[0].tolist(), time.perf_counter() - t0


def compare_exact(spec: list, ref: list, margins: np.ndarray) -> dict:
    """First mismatch (if any) and whether it is a near-tie of the plain decode."""
    for i, (a, b) in enumerate(zip(spec, ref)):
        if a != b:
            return {"match": False, "first_mismatch": i, "margin": float(margins[i]),
                    "near_tie": bool(margins[i] < NEAR_TIE)}
    return {"match": len(spec) == len(ref), "first_mismatch": -1, "margin": None, "near_tie": None}


def summarize(stats: list, exact: list, spec_tokens: int, spec_s: float, greedy_tokens: int, greedy_s: float) -> dict:
    acc = np.concatenate([np.asarray(s["accepted"], float) for s in stats]) if stats else np.zeros(0)
    br = {k: sum(s["branch"][k] for s in stats) for k in ("main", "repair", "miss")}
    nb = max(sum(br.values()), 1)
    span = sum(sum(s["verify_span_ms"]) for s in stats) / 1e3
    sync_draft = sum(sum(s["draft_sync_ms"]) for s in stats) / 1e3
    mism = [e for e in exact if not e["match"]]
    return {
        "tokens_per_s": spec_tokens / max(spec_s, 1e-9),
        "greedy_tokens_per_s": greedy_tokens / max(greedy_s, 1e-9),
        "speedup": (spec_tokens / max(spec_s, 1e-9)) / (greedy_tokens / max(greedy_s, 1e-9)),
        "mean_accepted": float(acc.mean()) if len(acc) else 0.0,
        "mean_tokens_per_block": float(acc.mean() + 1) if len(acc) else 0.0,
        "blocks": int(len(acc)),
        "branch_hit_rate": (br["main"] + br["repair"]) / nb,
        "branch_counts": br,
        "gpu_idle_fraction": (1.0 - span / spec_s) if any(s.get("on_cuda") for s in stats) else None,
        "draft_sync_s": sync_draft,
        "match_rate": 1.0 - len(mism) / max(len(exact), 1),
        "mismatches": len(mism),
        "mismatches_near_tie": sum(1 for e in mism if e["near_tie"]),
        "all_mismatches_near_tie": all(e["near_tie"] for e in mism),
        "verifier": stats[0].get("verifier") if stats else None,
        "verifier_error": stats[0].get("verifier_error") if stats else None,
    }


# ---------------------------------------------------------------------- CLI
def _prompts(args, vocab: int) -> list:
    if args.val_bin:
        toks = np.memmap(args.val_bin, dtype="<u2", mode="r")
        stride = max(args.prompt_len + args.n_new, len(toks) // max(args.n_prompts, 1))
        return [[int(x) for x in toks[i * stride: i * stride + args.prompt_len]] for i in range(args.n_prompts)
                if i * stride + args.prompt_len <= len(toks)]
    rng = np.random.default_rng(args.seed)
    return [[int(x) for x in rng.integers(0, vocab, args.prompt_len)] for _ in range(args.n_prompts)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bundle", default=None, help="exported drafter; default: export from --ckpt/--posthoc")
    ap.add_argument("--posthoc", default=None)
    ap.add_argument("--sketch", default=None)
    ap.add_argument("--impl", choices=("auto", "cpp", "ref"), default="auto")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--gamma", type=int, nargs="+", default=[4])
    ap.add_argument("--n_prompts", type=int, default=500)
    ap.add_argument("--prompt_len", type=int, default=256)
    ap.add_argument("--n_new", type=int, default=128)
    ap.add_argument("--val_bin", default=None, help="uint16 token file; default random prompts")
    ap.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    ap.add_argument("--verifier", choices=("eager", "static", "graph"), default="graph")
    ap.add_argument("--rows", choices=("b0", "host"), default="b0", help="b0: GPU table; host: HostExactRows")
    ap.add_argument("--target_codes", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--no_branches", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", default="run")
    args = ap.parse_args(argv)

    from ffnmem.infer import load_model

    from .cpu_drafter import make_drafter
    from .export_drafter import export, load_bundle

    model, cfg = load_model(args.ckpt, device=args.device, seq_len=args.prompt_len + args.n_new + max(args.gamma) + 2)
    bundle_path = args.bundle
    if bundle_path is None:
        posthoc = None
        if args.posthoc:
            from .posthoc_model import PosthocDrafter

            posthoc = PosthocDrafter.load(args.posthoc, model).eval()
        sketch = None
        if args.sketch:
            from .sketches import Sketch

            sketch = Sketch.load(args.sketch)
        bundle_path = f"bundles/{args.run}.bin"
        export(bundle_path, model, posthoc, sketch, source={"ckpt": cfg["_checkpoint"]})
    bundle = load_bundle(bundle_path)
    drafter = make_drafter(bundle, args.impl, threads=args.threads)
    tc = {"on": True, "off": False}.get(args.target_codes, bundle.config["kind"] == "posthoc")
    amp = torch.float16 if args.dtype == "fp16" else None
    dev = torch.device(args.device)
    prompts = _prompts(args, int(cfg["model"]["vocab"]))
    rows = HostExactRows(model, int(bundle.config["layer"])) if args.rows == "host" else None
    res = {"run": args.run, "ckpt": cfg["_checkpoint"], "bundle": bundle_path, "drafter": drafter.name,
           "threads": args.threads, "dtype": args.dtype, "rows": args.rows, "target_codes": tc,
           "n_prompts": len(prompts), "prompt_len": args.prompt_len, "n_new": args.n_new,
           "torch": torch.__version__, "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
           "by_gamma": {}}
    greedy = []
    g_tokens, g_secs = 0, 0.0
    if prompts:
        timed_greedy(model, prompts[0], 8, amp)  # warm-up (allocator, kernels)
    for p in prompts:
        ref, margins = greedy_with_margins(model, p, args.n_new, amp)
        gen, secs = timed_greedy(model, p, args.n_new, amp)
        greedy.append((ref, margins))
        g_tokens, g_secs = g_tokens + len(gen), g_secs + secs
    with ThreadPoolExecutor(1) as pool:
        for gamma in args.gamma:
            stats, exact = [], []
            s_tokens, s_secs = 0, 0.0
            for p, (ref, margins) in zip(prompts, greedy):
                toks, st = speculative_generate(model, drafter, p, args.n_new, gamma, row_source=rows,
                                                target_codes=tc, branches=not args.no_branches,
                                                verifier=args.verifier, amp_dtype=amp, executor=pool)
                st["on_cuda"] = dev.type == "cuda"
                stats.append(st)
                exact.append(compare_exact(toks, ref, margins))
                s_tokens, s_secs = s_tokens + len(toks), s_secs + st["seconds"]
            summ = summarize(stats, exact, s_tokens, s_secs, g_tokens, g_secs)
            summ["mismatch_detail"] = [dict(e, prompt=i) for i, e in enumerate(exact) if not e["match"]][:50]
            res["by_gamma"][str(gamma)] = summ
            print(json.dumps({"gamma": gamma, **{k: v for k, v in summ.items() if k != "mismatch_detail"}}))
    if rows is not None:
        res["host_rows"] = {"hits": rows.hits, "misses": rows.misses}
    out = Path(f"results/live_{args.run}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
