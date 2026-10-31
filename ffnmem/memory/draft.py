"""Draft readout: next-token logits from the memory table alone (Wave 3, M1.T2).

Inputs per drafted position t (no backbone activations):

- route tokens of the n-grams ending at t, i.e. the rows the memory layer reads
  (``[N, R, len(ngrams) * mem_dim]``), through v2's shared memory norm + KV projection;
- the query ``W_q . RMSNorm(E[x[t]].detach())`` (+ ``W_sq . s[t]`` with the SSM), 4 heads x 64;
- attention over the R route slots plus v2's fixed zero-value Sink (same
  denominator as the v2 readout), output ``W_o`` -> d_model -> RMSNorm;
- logits against the tied, detached embedding matrix.

``lookup_rows_at`` gathers the rows for a set of (batch, position) pairs through
the vendored ``_lookup_route_chunk`` on 3-token windows, so the surrogate route
gradient (draft loss -> router) is the v2 one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .lngram_v2 import RouteNgramMemoryV2


def lookup_rows_at(m: RouteNgramMemoryV2, codes: Tensor, q_logits: Tensor | None,
                   b_idx: Tensor, t_idx: Tensor) -> Tensor:
    """Route tokens of the n-grams ending at (b_idx[i], t_idx[i]).

    codes: [B, T, R]; q_logits: [B, T, R*bits] or None (no surrogate).
    Returns [N, R, len(ngrams) * mem_dim], identical to the full-sequence lookup at those positions.
    """
    hist = m.route_history_length
    b, t, r = codes.shape
    offs = torch.arange(hist + 1, device=codes.device)
    win_t = t_idx[:, None] - hist + offs[None, :]  # [N, hist+1], may be negative at the start
    real = win_t >= 0
    win_t = win_t.clamp_min(0)
    win_codes = codes[b_idx[:, None], win_t] * real[..., None].to(codes.dtype)
    # Window positions before the sequence start are their own segment, so n-grams
    # that would cross t = 0 are invalid -- exactly the zero rows of the full lookup.
    seg = real.to(torch.int32)
    q_win = None
    if q_logits is not None:
        q_win = q_logits.view(b, t, r, m.bits_per_route)[b_idx[:, None], win_t]
    tokens = m._lookup_route_chunk(win_codes, q_win, 0, m.num_routes, seg, None)
    return tokens[:, -1]


class DraftReadout(nn.Module):
    def __init__(self, d_model: int, memory_module: RouteNgramMemoryV2, q_heads: int = 4, head_dim: int = 64,
                 ssm_dim: int = 0, eps: float = 1e-5, init_std: float = 0.02):
        super().__init__()
        from ..model import RMSNorm

        kv_heads = memory_module.readout_num_kv_heads
        if head_dim != memory_module.readout_head_dim:
            raise ValueError(f"draft head_dim {head_dim} != memory readout head_dim {memory_module.readout_head_dim}")
        if q_heads % kv_heads:
            raise ValueError(f"draft q_heads {q_heads} not a multiple of memory kv_heads {kv_heads}")
        self.q_heads, self.head_dim, self.kv_heads = q_heads, head_dim, kv_heads
        self.groups = q_heads // kv_heads
        inner = q_heads * head_dim
        self.q_norm = RMSNorm(d_model, eps)
        self.w_q = nn.Linear(d_model, inner, bias=False)
        self.w_sq = nn.Linear(ssm_dim, inner, bias=False) if ssm_dim else None
        self.w_o = nn.Linear(inner, d_model, bias=False)
        self.out_norm = RMSNorm(d_model, eps)
        for lin in (self.w_q, self.w_sq, self.w_o):
            if lin is not None:
                nn.init.normal_(lin.weight, mean=0.0, std=init_std)

    def hidden(self, m: RouteNgramMemoryV2, emb_cur: Tensor, route_tokens: Tensor,
               ssm_state: Tensor | None = None) -> Tensor:
        """emb_cur [N, d] (detached E[x[t]]), route_tokens [N, R, D] -> draft hidden [N, d]."""
        n = emb_cur.shape[0]
        q = self.w_q(self.q_norm(emb_cur))
        if self.w_sq is not None:
            q = q + self.w_sq(ssm_state.to(q.dtype))
        key, value = m._project_memory(route_tokens[:, None])  # [N, 1, kv, R, hd]
        key, value = key[:, 0], value[:, 0]
        q = q.view(n, self.kv_heads, self.groups, self.head_dim)
        scores = torch.matmul(q.to(key.dtype), key.transpose(-1, -2)).float() * m.softmax_scale
        probs = m._real_route_probabilities(scores)  # [N, kv, G, R], Sink in the denominator
        out = torch.matmul(probs.to(value.dtype), value).reshape(n, self.q_heads * self.head_dim)
        return self.out_norm(self.w_o(out))

    def forward(self, m: RouteNgramMemoryV2, emb_cur: Tensor, route_tokens: Tensor, unembed: Tensor,
                ssm_state: Tensor | None = None) -> Tensor:
        """Draft logits [N, V] (in the autocast dtype); ``unembed`` is the detached tied embedding."""
        return F.linear(self.hidden(m, emb_cur, route_tokens, ssm_state), unembed)
