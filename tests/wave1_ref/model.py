"""Llama-style dense decoder with per-layer FFN width, block deletion and
Lngram v2 memory insertion.

Layer ids:
- ``layers`` lists the *original* layer ids that are kept (deletion = omit ids).
- ``d_ff`` is indexed by original layer id.
- ``memory_layers`` is indexed in the kept-layer numbering (position in ``layers``).

Block forward (memory before attention, as in both Lngram papers)::

    h = h + mem(h, attn_norm(h))        # memory layers only
    h = h + attn(attn_norm(h))
    h = h + ffn(ffn_norm(h))
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ffnmem.attention import attend
from .wrapper import LngramMemory


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def rope_cache(seq_len: int, head_dim: int, theta: float) -> tuple[Tensor, Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv)  # [T, D/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: [B, T, H, D]; rotate pairs (even, odd) -- llama2.c convention."""
    t = x.shape[1]
    xf = x.float().reshape(*x.shape[:-1], -1, 2)
    x0, x1 = xf[..., 0], xf[..., 1]
    c = cos[:t].view(1, t, 1, -1)
    s = sin[:t].view(1, t, 1, -1)
    out = torch.stack((x0 * c - x1 * s, x0 * s + x1 * c), dim=-1).flatten(-2)
    return out.type_as(x)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        if n_heads % n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.backend = "sdpa"

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, t, _ = x.shape
        q = apply_rope(self.wq(x).view(b, t, self.n_heads, self.head_dim), cos, sin)
        k = apply_rope(self.wk(x).view(b, t, self.n_kv_heads, self.head_dim), cos, sin)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim)
        out = attend(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), self.backend)
        return self.wo(out.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: dict, layer_id: int, d_ff: int, memory: dict | None):
        super().__init__()
        self.layer_id = layer_id
        d, eps = cfg["d_model"], cfg["norm_eps"]
        self.attn_norm = RMSNorm(d, eps)
        self.attn = Attention(d, cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"])
        self.ffn_norm = RMSNorm(d, eps)
        self.ffn = FeedForward(d, d_ff)
        self.memory = LngramMemory(d, memory, cfg["init_std"], eps) if memory else None

    def forward(self, h: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        if self.memory is not None:
            h = h + self.memory(h, self.attn_norm(h))
        h = h + self.attn(self.attn_norm(h), cos, sin)
        return h + self.ffn(self.ffn_norm(h))


def memory_config_for(cfg: dict, pos: int) -> dict:
    """Shared ``memory`` config, updated by ``memory_overrides[pos]`` (kept-layer position) if given."""
    mem = dict(cfg["memory"])
    override = (cfg.get("memory_overrides") or {}).get(pos)
    if override:
        mem.update(override)
    return mem


class Transformer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.tok_emb = nn.Embedding(cfg["vocab"], d)
        mem_set = set(cfg["memory_layers"])
        self.blocks = nn.ModuleList(
            Block(cfg, lid, cfg["d_ff"][lid], memory_config_for(cfg, pos) if pos in mem_set else None)
            for pos, lid in enumerate(cfg["layers"])
        )
        self.norm = RMSNorm(d, cfg["norm_eps"])
        self.output = nn.Linear(d, cfg["vocab"], bias=False)
        self.output.weight = self.tok_emb.weight  # tied
        cos, sin = rope_cache(cfg["seq_len"], cfg["head_dim"], cfg["rope_theta"])
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.act_ckpt = False
        self._init_weights()

    # ------------------------------------------------------------------ init
    def _init_weights(self) -> None:
        std = self.cfg["init_std"]
        proj_std = std / math.sqrt(2 * len(self.blocks))
        for block in self.blocks:
            for lin in (block.attn.wq, block.attn.wk, block.attn.wv, block.ffn.w1, block.ffn.w3):
                nn.init.normal_(lin.weight, mean=0.0, std=std)
            for lin in (block.attn.wo, block.ffn.w2):
                nn.init.normal_(lin.weight, mean=0.0, std=proj_std)
            # Memory modules keep the Appendix E init set in LngramMemory.
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=std)

    def set_attention_backend(self, backend: str) -> None:
        for block in self.blocks:
            block.attn.backend = backend

    def memory_modules(self) -> list[LngramMemory]:
        return [b.memory for b in self.blocks if b.memory is not None]

    # --------------------------------------------------------------- forward
    def forward(self, idx: Tensor, targets: Tensor | None = None):
        h = self.tok_emb(idx)
        cos, sin = self.rope_cos, self.rope_sin
        for block in self.blocks:
            if self.act_ckpt and self.training and torch.is_grad_enabled():
                # Non-reentrant: reentrant checkpointing runs the forward under
                # no_grad, which would silently disable the Lngram surrogate.
                h = checkpoint(block, h, cos, sin, use_reentrant=False)
            else:
                h = block(h, cos, sin)
        logits = self.output(self.norm(h))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss


# ---------------------------------------------------------------- counting
COMPONENTS = ("embedding", "attention", "ffn", "norms", "memory_tables", "memory_readout")


def component_of(name: str) -> str:
    if name.startswith("tok_emb.") or name.startswith("output."):
        return "embedding"
    if ".memory.module.tables." in name or ".memory.rank1_tables." in name:
        return "memory_tables"
    if ".memory." in name:
        return "memory_readout"
    if ".attn." in name:
        return "attention"
    if ".ffn." in name:
        return "ffn"
    if "norm" in name:
        return "norms"
    raise KeyError(f"unclassified parameter {name}")


def count_params(model: nn.Module) -> dict:
    """Parameter counts by component (tied embedding counted once)."""
    counts: dict = defaultdict(int)
    seen: set[int] = set()
    for name, p in model.named_parameters(remove_duplicate=False):
        if id(p) in seen:
            continue
        seen.add(id(p))
        counts[component_of(name)] += p.numel()
    out = {c: int(counts.get(c, 0)) for c in COMPONENTS}
    out["total"] = sum(out[c] for c in COMPONENTS)
    out["non_embedding"] = out["total"] - out["embedding"]
    mem = [m for m in model.modules() if isinstance(m, LngramMemory)]
    out["active_non_embedding_per_token"] = (
        out["non_embedding"] - out["memory_tables"] + sum(m.active_table_params_per_token() for m in mem)
    )
    return out

