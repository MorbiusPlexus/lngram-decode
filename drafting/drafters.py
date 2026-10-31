"""Model-based drafters for offline acceptance (M3.T4); contract in ``drafting/common.py``.

    d = SketchDrafter(Sketch.load("sketches/w3_sdb_1b.npz"), target)          # level 0
    d = PosthocNeuralDrafter(PosthocDrafter.load("runs/ph_sb/drafter.pt", target))
    d = SDHeadDrafter(target, online=True)                                     # SD-B/C/A head + online counts
    d = OracleDrafter(target)                                                  # the target itself (tests)
    drafts = d.drafts(rec, gamma)                                              # int64 [n_cont, gamma]

All rows of a record are drafted together: for row j the verified prefix is
``tokens[:L_j]`` (``L_j = prompt_len + j``) and the drafter runs gamma autoregressive steps
on [n_cont]-batched tensors.  Codes (memory-layer route codes):

- verified positions (< L_j), ``verified='exact'`` (default): the target's codes --
  recomputed from tokens for token-router targets (SD-B / SD-C), ``rec.codes`` for
  hidden-router targets (unmodified models, SD-A).  ``verified='own'``: the drafter's
  router everywhere (what SD-A's head / a post-hoc drafter was trained on).
- drafted positions (>= L_j): codes from a token router the drafter owns: the target's
  token router (SD-B / SD-C), SD-A's imitating router, or the post-hoc drafter's router.
  ``SketchDrafter`` on an unmodified target uses the post-hoc router when one is given
  (``router=``); without one it has no codes for drafted positions and drafts only the
  first token (later slots are -1, never accepted).
- verified-position quantities (codes, SSM states) are computed over the whole record and
  sliced at L_j - 1; all of them are causal, so row j never sees tokens >= L_j.

Online sketch updates (``online=True``): an in-context count table keyed by the same
(order, route, row) as the sketch, filled from verified (row at t, token t+1) pairs with
t + 1 < L_j.  Its distribution is mixed in: ``p = (1 - lam) p_base + lam p_online``,
``lam = n / (n + kappa)``, n = matched pairs per (route, order) lookup.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from drafting.build_sketches import SketchTensors, seq_rows
from drafting.common import Drafter, SeqRecord
from drafting.posthoc_model import PosthocDrafter
from drafting.sketches import Sketch
from ffnmem.model import Transformer


# ------------------------------------------------------------------ token routers
class TokenCoder:
    """Token-only route codes (+ the SSM state that feeds them) for one memory layer."""

    def __init__(self, features, weight: Tensor, module, ssm, embed):
        self.features, self.weight, self.module, self.ssm, self.embed = features, weight, module, ssm, embed

    def _route(self, x: Tensor) -> Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.linear(x.float(), self.weight.float())

    def seq(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        """x [T] -> codes [T, R] (long), states [T, S] or None."""
        emb = self.embed(x[None])
        s = self.ssm(emb) if self.ssm is not None else None
        codes = self.module.quantize_route_logits(self._route(self.features(emb, s)))
        return codes[0].long(), None if s is None else s[0]

    def step(self, prev: Tensor, cur: Tensor, s_prev: Tensor | None) -> tuple[Tensor, Tensor | None]:
        """Codes [N, R] at a new position with token ``cur`` after ``prev``; new SSM state."""
        e_cur = self.embed(cur)
        s = self.ssm.step(s_prev, e_cur) if self.ssm is not None else None
        q = self._route(self.features.step(self.embed(prev), e_cur, s))
        return self.module.quantize_route_logits(q[None])[0].long(), s


def target_coder(model: Transformer, layer: int) -> TokenCoder | None:
    """The target's token-only router (SD-B / SD-C main router, SD-A imitating router) or None."""
    mem = model.blocks[layer].memory
    embed = lambda t: model.tok_emb(t).detach()  # noqa: E731
    if mem.router_input != "hidden":
        return TokenCoder(mem.router_features, mem.router.weight, mem.module, model.ssm, embed)
    if mem.token_router is not None:
        return TokenCoder(mem.token_router.features, mem.token_router.proj.weight, mem.module, None, embed)
    return None


def posthoc_coder(drafter: PosthocDrafter) -> TokenCoder:
    return TokenCoder(drafter.features, drafter.router.weight, drafter.memory_module, drafter.ssm, drafter.embed)


def exact_codes(model: Transformer, layer: int, rec: SeqRecord, x: Tensor) -> Tensor:
    """The target's own codes [T, R] at every position of the record."""
    mem = model.blocks[layer].memory
    if mem.router_input != "hidden":
        return target_coder(model, layer).seq(x)[0]
    if layer not in rec.codes:
        raise KeyError(f"record has no target codes for memory block {layer}")
    return torch.as_tensor(np.asarray(rec.codes[layer]), device=x.device).long()


# ------------------------------------------------------------------ helpers
def window_rows(win: Tensor, real: Tensor, order: int, bits: int) -> Tensor:
    """Global sketch rows [N, R] of the n-gram ending at the window's last position (-1 if invalid)."""
    n, w, r = win.shape
    c = win[:, w - order:]
    addr = c[:, 0].clone()
    for e in range(1, order):
        addr |= c[:, e] << (e * bits)
    rows = torch.arange(r, device=win.device) * (1 << (bits * order)) + addr
    return torch.where(real[:, w - order:].all(1, keepdim=True), rows, torch.full_like(rows, -1))


def window_route_tokens(m, win: Tensor, real: Tensor) -> Tensor:
    """Memory route tokens [N, R, D] of the n-grams ending at the window's last position
    (v2 lookup on the window, positions before the sequence start as their own segment)."""
    tokens = m._lookup_route_chunk(win * real[..., None], None, 0, m.num_routes, real.to(torch.int32), None)
    return tokens[:, -1]


class OnlineCounts:
    """In-context (order, route, row) -> next-token counts from one record's verified pairs."""

    def __init__(self, codes: Tensor, x: Tensor, orders, bits: int, vocab: int, kappa: float = 1.0,
                 route_chunk: int = 32):
        self.keys = {n: seq_rows(codes[None], n, bits)[0, :-1] for n in orders}   # pairs t = 0 .. T-2
        self.nxt = x[1:]
        self.orders, self.bits, self.vocab, self.kappa, self.rc = tuple(orders), bits, vocab, kappa, route_chunk
        self.routes = codes.shape[1]

    def mix(self, p: Tensor, win: Tensor, real: Tensor, deadline: Tensor) -> Tensor:
        """p [N, V]; only pairs t <= deadline[j] (token t+1 verified for row j) count."""
        n_rows, t = win.shape[0], len(self.nxt)
        ok_t = torch.arange(t, device=win.device)[None] <= deadline[:, None]
        w = torch.zeros(n_rows, t, device=win.device)
        for n in self.orders:
            q = window_rows(win, real, n, self.bits)
            for r0 in range(0, self.routes, self.rc):
                qq, kk = q[:, None, r0:r0 + self.rc], self.keys[n][None, :, r0:r0 + self.rc]
                w += ((kk == qq) & (qq >= 0)).sum(-1).float()
        w = w * ok_t
        tot = w.sum(1, keepdim=True)
        sc = torch.zeros(n_rows, self.vocab, device=win.device).scatter_add_(1, self.nxt[None].expand(n_rows, -1), w)
        n_eff = tot / (self.routes * len(self.orders))
        lam = n_eff / (n_eff + self.kappa)
        return (1 - lam) * p + lam * sc / tot.clamp_min(1e-9)


# ------------------------------------------------------------------ base loop
class WindowDrafter(Drafter):
    """Shared batched loop: a window of the last ``W`` codes per row, gamma steps."""

    prefix_consistent = True

    def __init__(self, target: Transformer, layer: int, coder: TokenCoder | None, verified: str = "exact",
                 own_coder: TokenCoder | None = None, online: bool = False, kappa: float = 1.0,
                 amp_dtype: torch.dtype | None = None):
        if verified not in ("exact", "own"):
            raise ValueError("verified must be exact | own")
        self.target, self.layer, self.coder = target, layer, coder
        self.verified, self.online, self.kappa, self.amp_dtype = verified, online, kappa, amp_dtype
        mem = target.blocks[layer].memory
        self.m, self.bits, self.routes = mem.module, mem.bits, mem.routes
        self.orders = tuple(self.m.ngrams)
        self.window = max(self.orders)
        self.vocab = int(target.cfg["vocab"])
        self.device = target.tok_emb.weight.device

    def _amp(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self.amp_dtype)

    def _verified(self, rec: SeqRecord, x: Tensor) -> tuple[Tensor, Tensor | None]:
        own = self.coder.seq(x) if self.coder is not None else (None, None)
        if self.verified == "own":
            if own[0] is None:
                raise ValueError(f"{self.name}: verified='own' needs a token router")
            return own
        return exact_codes(self.target, self.layer, rec, x), own[1]

    def _dist(self, win: Tensor, real: Tensor, cur: Tensor, s_cur: Tensor | None) -> Tensor:
        """Next-token probabilities [N, V] (rows may be all-zero = no draft)."""
        raise NotImplementedError

    @torch.no_grad()
    def drafts(self, rec: SeqRecord, gamma: int) -> np.ndarray:
        x = torch.as_tensor(np.asarray(rec.tokens), device=self.device).long()
        p0, n = rec.prompt_len, rec.n_cont
        codes, states = self._verified(rec, x)
        last = torch.arange(p0 - 1, p0 - 1 + n, device=self.device)
        pos = last[:, None] - (self.window - 1) + torch.arange(self.window, device=self.device)
        real = pos >= 0
        win = codes[pos.clamp_min(0)] * real[..., None]
        cur = x[last]
        s_cur = states[last] if states is not None else None
        online = OnlineCounts(codes, x, self.orders, self.bits, self.vocab, self.kappa) if self.online else None
        out = torch.full((n, gamma), -1, dtype=torch.long, device=self.device)
        for k in range(gamma):
            with self._amp():
                p = self._dist(win, real, cur, s_cur)
            if online is not None:
                p = online.mix(p, win, real, last - 1)
            best = p.max(-1)
            out[:, k] = torch.where(best.values > 0, best.indices, torch.full_like(best.indices, -1))
            if k == gamma - 1 or self.coder is None:
                break
            prev, cur = cur, out[:, k].clamp_min(0)
            c, s_cur = self.coder.step(prev, cur, s_cur)
            win = torch.cat((win[:, 1:], c[:, None]), 1)
            real = torch.cat((real[:, 1:], torch.ones_like(real[:, :1])), 1)
        return out.cpu().numpy()


# ------------------------------------------------------------------ drafters
class SketchDrafter(WindowDrafter):
    """Level 0: next token = argmax of sketch counts summed over routes and orders."""

    name = "sketch"

    def __init__(self, sketch: Sketch, target: Transformer, router: PosthocDrafter | None = None,
                 verified: str = "exact", online: bool = False, kappa: float = 1.0):
        coder = target_coder(target, sketch.layer)
        if coder is None and router is not None:
            coder = posthoc_coder(router)
        super().__init__(target, sketch.layer, coder, verified, online=online, kappa=kappa)
        self.st = SketchTensors(sketch, self.device)

    def _dist(self, win, real, cur, s_cur):
        sc = self.st.scores({n: window_rows(win, real, n, self.bits) for n in self.orders})
        return sc / sc.sum(-1, keepdim=True).clamp_min(1e-9)


class _NeuralDrafter(WindowDrafter):
    readout = None
    module = None

    def _query_state(self, s_cur):
        return s_cur if self.readout.w_sq is not None else None

    def _dist(self, win, real, cur, s_cur):
        rows = window_route_tokens(self.module, win, real)
        emb = self.target.tok_emb(cur).detach()
        logits = self.readout(self.module, emb, rows, self.target.tok_emb.weight.detach(), self._query_state(s_cur))
        return torch.softmax(logits.float(), -1)


class PosthocNeuralDrafter(_NeuralDrafter):
    """Post-hoc drafter (token router + draft readout on the frozen target's table)."""

    name = "posthoc"

    def __init__(self, drafter: PosthocDrafter, verified: str = "exact", online: bool = False,
                 kappa: float = 1.0, amp_dtype: torch.dtype | None = None):
        super().__init__(drafter.target, drafter.layer, posthoc_coder(drafter), verified, online=online,
                         kappa=kappa, amp_dtype=amp_dtype)
        self.readout, self.module = drafter.draft, drafter.memory_module


class SDHeadDrafter(_NeuralDrafter):
    """Jointly trained draft head of an SD-B / SD-C / SD-A model (``model.draft_logits`` path)."""

    name = "sdhead"

    def __init__(self, model: Transformer, verified: str = "exact", online: bool = False, kappa: float = 1.0,
                 amp_dtype: torch.dtype | None = None):
        if model.draft is None:
            raise ValueError("model has no draft head")
        super().__init__(model, model.draft_pos, target_coder(model, model.draft_pos), verified, online=online,
                         kappa=kappa, amp_dtype=amp_dtype)
        self.readout, self.module = model.draft, model.blocks[model.draft_pos].memory.module


class OracleDrafter(Drafter):
    """The target itself (greedy, cached decoding): accepted length = gamma everywhere.
    Replays the generation path (prompt prefill, then one token per step); one row at a time."""

    name = "oracle"
    prefix_consistent = True

    def __init__(self, target: Transformer, amp_dtype: torch.dtype | None = None):
        self.target, self.amp_dtype = target, amp_dtype

    @torch.no_grad()
    def drafts(self, rec: SeqRecord, gamma: int) -> np.ndarray:
        from ffnmem.infer import DecodeState, decode_step

        dev = self.target.tok_emb.weight.device
        x = torch.as_tensor(np.asarray(rec.tokens), device=dev).long()[None]
        p0, n = rec.prompt_len, rec.n_cont
        out = np.zeros((n, gamma), dtype=np.int64)
        st = DecodeState.empty(self.target, 1, len(rec.tokens) + gamma)
        ctx = (torch.autocast(dev.type, dtype=self.amp_dtype) if self.amp_dtype
               else torch.autocast(dev.type, enabled=False))
        with ctx:
            logits = decode_step(self.target, st, x[:, :p0])
            for j in range(n):
                length = st.length
                nxt = logits[:, -1].argmax(-1)
                for k in range(gamma):
                    out[j, k] = int(nxt)
                    if k < gamma - 1:
                        nxt = decode_step(self.target, st, nxt[:, None])[:, -1].argmax(-1)
                st.truncate(length)
                if j < n - 1:
                    logits = decode_step(self.target, st, x[:, p0 + j: p0 + j + 1])
        return out
