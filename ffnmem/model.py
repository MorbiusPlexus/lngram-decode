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

Wave 3 (self-drafting; all off unless ``model.draft`` / ``model.ssm`` / ``memory.router_input``
are set): a token-only router input built from the embeddings (and SSM state), and a
draft readout over the memory layer's rows that never reads backbone activations.
``forward(idx, targets, draft_sel=...)`` adds ``lambda * KL(sg(p) || q)`` on the
selected positions to the returned loss; the parts are in ``model.last_aux``.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .attention import attend
from .losses import draft_kl_chunk
from .memory.draft import DraftReadout, lookup_rows_at
from .memory.ssm import DiagonalSSM
from .memory.wrapper import LngramMemory


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
        ssm_dim = int(cfg["ssm"]["state_dim"]) if cfg.get("ssm") else 0
        self.memory = LngramMemory(d, memory, cfg["init_std"], eps, ssm_dim) if memory else None

    def forward(self, h: Tensor, cos: Tensor, sin: Tensor, router_in: Tensor | None = None) -> Tensor:
        if self.memory is not None:
            h = h + self.memory(h, self.attn_norm(h), router_in)
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
        self._build_self_drafting(cfg)
        self._init_weights()

    def _build_self_drafting(self, cfg: dict) -> None:
        """Wave 3 modules; nothing is created (and no RNG consumed) for Wave 1 configs."""
        self.ssm = None
        self.draft = None
        self.draft_pos = None
        self.draft_cfg: dict = {}
        self.last_aux: dict = {}
        d, eps, std = cfg["d_model"], cfg["norm_eps"], cfg["init_std"]
        ssm_cfg = cfg.get("ssm")
        if ssm_cfg:
            self.ssm = DiagonalSSM(d, int(ssm_cfg["state_dim"]), int(ssm_cfg.get("chunk", 128)), eps, std)
        draft_cfg = cfg.get("draft")
        if draft_cfg:
            self.draft_cfg = dict(draft_cfg)
            self.draft_pos = int(draft_cfg.get("layer", cfg["memory_layers"][0]))
            mem = self.blocks[self.draft_pos].memory
            if mem is None:
                raise ValueError(f"draft.layer {self.draft_pos} has no memory")
            if mem.router_input == "hidden" and mem.token_router is None:
                raise ValueError("the draft path needs token-only codes: router_input embed2* or token_router")
            if mem.module.execution_mode != "full":
                raise ValueError("draft readout supports memory execution_mode: full only")
            if not draft_cfg.get("detach_embeddings", True) or not draft_cfg.get("share_kv_proj", True):
                raise ValueError("only detach_embeddings: true and share_kv_proj: true are implemented")
            feeds = (ssm_cfg or {}).get("feeds", [])
            ssm_dim = int(ssm_cfg["state_dim"]) if (ssm_cfg and "draft_query" in feeds) else 0
            self.draft = DraftReadout(d, mem.module, int(draft_cfg.get("q_heads", 4)),
                                      int(draft_cfg.get("head_dim", 64)), ssm_dim, eps, std)

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
    def forward(self, idx: Tensor, targets: Tensor | None = None,
                draft_sel: tuple[Tensor, Tensor] | None = None):
        """``draft_sel = (b_idx, t_idx)``: positions for the draft loss (needs targets).

        The returned loss is ``L_LM + lambda * KL [+ mu * BCE]``; ``self.last_aux`` holds
        the detached parts (``lm``, ``draft_kl``, ``draft_top1_agree``, ``draft_accept_proxy``,
        ``route_bce``) of the last call with ``draft_sel``.
        """
        emb = self.tok_emb(idx)
        h = emb
        # The SSM reads detached embeddings: neither the router nor the draft path trains E through it.
        s = self.ssm(emb.detach()) if self.ssm is not None else None
        cos, sin = self.rope_cos, self.rope_sin
        for block in self.blocks:
            router_in = None
            if block.memory is not None and block.memory.router_input != "hidden":
                router_in = block.memory.router_features(emb, s)
            if self.act_ckpt and self.training and torch.is_grad_enabled():
                # Non-reentrant: reentrant checkpointing runs the forward under
                # no_grad, which would silently disable the Lngram surrogate.
                h = checkpoint(block, h, cos, sin, router_in, use_reentrant=False)
            else:
                h = block(h, cos, sin, router_in)
        logits = self.output(self.norm(h))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.reshape(-1))
        if draft_sel is not None and self.draft is not None:
            loss = self._add_draft_loss(loss, emb, s, logits, draft_sel)
        return logits, loss

    # ------------------------------------------------------------ self-drafting
    def draft_route_logits(self, emb_d: Tensor, s: Tensor | None) -> Tensor:
        """Token-only route logits used by the draft path (fp32, [B, T, R*bits])."""
        mem = self.blocks[self.draft_pos].memory
        if mem.token_router is not None:
            return mem.token_router(emb_d)
        return mem.route_logits(mem.router_features(emb_d, s))

    def _draft_chunk(self, emb_d: Tensor, s: Tensor | None, q_logits: Tensor, codes: Tensor,
                     b_idx: Tensor, t_idx: Tensor) -> Tensor:
        m = self.blocks[self.draft_pos].memory.module
        rows = lookup_rows_at(m, codes, q_logits, b_idx, t_idx)
        s_sel = s[b_idx, t_idx] if (s is not None and self.draft.w_sq is not None) else None
        return self.draft(m, emb_d[b_idx, t_idx], rows, self.tok_emb.weight.detach(), s_sel)

    def draft_logits(self, idx: Tensor, b_idx: Tensor, t_idx: Tensor) -> Tensor:
        """Draft logits at (b_idx, t_idx) from token ids alone (no backbone forward)."""
        emb_d = self.tok_emb(idx).detach()
        s = self.ssm(emb_d) if self.ssm is not None else None
        q = self.draft_route_logits(emb_d, s)
        codes = self.blocks[self.draft_pos].memory.module.quantize_route_logits(q)
        return self._draft_chunk(emb_d, s, q, codes, b_idx, t_idx)

    def _add_draft_loss(self, loss: Tensor, emb: Tensor, s: Tensor | None, logits: Tensor,
                        draft_sel: tuple[Tensor, Tensor]) -> Tensor:
        dc = self.draft_cfg
        mem = self.blocks[self.draft_pos].memory
        emb_d = emb.detach()
        q = self.draft_route_logits(emb_d, s)
        codes = mem.module.quantize_route_logits(q)
        b_idx, t_idx = draft_sel
        n = b_idx.numel()
        chunk = int(dc.get("loss_chunk", 512))
        use_ckpt = self.act_ckpt and self.training and torch.is_grad_enabled()

        def one(emb_d, s, q, codes, b, t, p):
            return draft_kl_chunk(self._draft_chunk(emb_d, s, q, codes, b, t), p)

        kl = agree = accept = 0.0
        for start in range(0, n, chunk):
            b, t = b_idx[start:start + chunk], t_idx[start:start + chunk]
            p = logits[b, t].detach()
            if use_ckpt:
                parts = checkpoint(one, emb_d, s, q, codes, b, t, p, use_reentrant=False)
            else:
                parts = one(emb_d, s, q, codes, b, t, p)
            kl, agree, accept = kl + parts[0], agree + parts[1], accept + parts[2]
        kl = kl / max(n, 1)
        aux = {"lm": loss.detach(), "draft_kl": kl.detach(),
               "draft_top1_agree": agree / max(n, 1), "draft_accept_proxy": accept / max(n, 1)}
        total = loss + float(dc.get("loss_weight", 0.1)) * kl
        if mem.token_router is not None:
            target = (mem.last_route_logits > 0).to(q.dtype)
            bce = F.binary_cross_entropy_with_logits(q, target)
            aux["route_bce"] = bce.detach()
            total = total + float((mem.cfg_token_router or {}).get("loss_weight", 0.1)) * bce
        self.last_aux = aux
        return total


# ---------------------------------------------------------------- counting
COMPONENTS = ("embedding", "attention", "ffn", "norms", "memory_tables", "memory_readout", "draft", "ssm")


def component_of(name: str) -> str:
    if name.startswith("tok_emb.") or name.startswith("output."):
        return "embedding"
    if name.startswith("draft."):
        return "draft"
    if name.startswith("ssm."):
        return "ssm"
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
