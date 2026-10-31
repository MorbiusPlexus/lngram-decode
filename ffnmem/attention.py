"""Causal attention backends.  The model calls ``attend`` only.

Inputs are ``q: [B, H, T, D]`` and ``k, v: [B, Hkv, T, D]``; output ``[B, H, T, D]``.

- ``sdpa``  : torch SDPA.  On CUDA the memory-efficient kernel is pinned so every
              GPU type (sm75 / sm86) runs the same kernel; on CPU the math kernel.
- ``naive`` : explicit softmax attention in fp32 (reference for tests).
- ``flash`` : flash-attn 2 (upstream on Ampere, the 2080 Ti fork on Turing).
              Optional; imported lazily; enable only after the equivalence test
              and benchmark in REPORT.md pass.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

BACKENDS = ("sdpa", "naive", "flash")


def _expand_kv(k: torch.Tensor, v: torch.Tensor, n_heads: int):
    groups = n_heads // k.shape[1]
    if groups == 1:
        return k, v
    return k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)


def _naive(q, k, v):
    t = q.shape[-2]
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    mask = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


def _sdpa(q, k, v):
    kernel = SDPBackend.EFFICIENT_ATTENTION if q.is_cuda else SDPBackend.MATH
    with sdpa_kernel([kernel]):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def _flash(q, k, v):
    from flash_attn import flash_attn_func  # noqa: PLC0415 - optional dependency

    # flash-attn wants [B, T, H, D] and handles GQA natively.
    out = flash_attn_func(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True
    )
    return out.transpose(1, 2)


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backend: str = "sdpa") -> torch.Tensor:
    if backend == "flash":
        return _flash(q, k, v)
    k, v = _expand_kv(k, v, q.shape[1])
    if backend == "sdpa":
        return _sdpa(q, k, v)
    if backend == "naive":
        return _naive(q, k, v)
    raise ValueError(f"unknown attention backend {backend!r}; expected one of {BACKENDS}")
