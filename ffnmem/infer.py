"""Inference: checkpoint loading and cached (KV + memory side state) decoding.

    model, cfg = load_model("runs/w3_sdb_1b/checkpoints/step_0001907.pt", device="cuda")
    state = DecodeState.empty(model, batch=4, max_len=2048)
    logits = decode_step(model, state, prompt)            # prefill: [B, P, V]
    logits = decode_step(model, state, next_tok[:, None])  # one token
    state.truncate(L)                                     # roll back (speculative decoding)

Everything the cache holds is indexed by absolute position (attention K/V, the
memory layers' route codes and normalised readout outputs for the short conv,
the SSM states and the processed tokens), so rolling back is just moving
``state.length``.  ``decode_step`` accepts any number of new positions, so the
same function serves prefill, single-token decoding and block verification
(the default mask is causal; tree verification passes ``attn_mask`` and
``positions``).  All sequences in a batch share one length.

The memory readout uses the same kernels as the full forward (``_manual_gqa`` +
``short_conv``), so cached logits match ``model(idx)`` to float rounding.
Not supported: the Wave 2 rank-1 path, memory ``execution_mode: streaming``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from . import checkpoint as ckpt
from .model import Transformer


# ------------------------------------------------------------------ loading
def load_model(path: str | Path, device: str | torch.device = "cpu", dtype: torch.dtype | None = None,
               seq_len: int | None = None) -> tuple[Transformer, dict]:
    """Model from a checkpoint file or a run dir (latest checkpoint); eval mode.

    ``seq_len`` enlarges the RoPE cache (e.g. prompt + generation > training length)."""
    path = Path(path)
    if path.is_dir():
        found = ckpt.latest_checkpoint(path)
        if found is None:
            raise FileNotFoundError(f"no checkpoint in {path}")
        path = found
    state = ckpt.load(path)
    cfg = state["config"]
    mcfg = dict(cfg["model"])
    if seq_len is not None:
        mcfg["seq_len"] = max(int(seq_len), int(mcfg["seq_len"]))
    model = Transformer(mcfg)  # checkpoints store the resolved model config
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if unexpected or [k for k in missing if not k.startswith("rope_")]:
        raise RuntimeError(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    model.to(device)
    if dtype is not None:
        model.to(dtype)
    model.eval()
    cfg = dict(cfg, _checkpoint=str(path), _step=int(state.get("step", 0)))
    return model, cfg


# ------------------------------------------------------------------ state
@dataclass
class DecodeState:
    batch: int
    max_len: int
    length: int = 0
    tokens: Tensor | None = None                       # [B, max_len]
    k: list = field(default_factory=list)              # per block [B, Hkv, max_len, D]
    v: list = field(default_factory=list)
    codes: dict = field(default_factory=dict)          # block pos -> [B, max_len, R] uint8
    conv_in: dict = field(default_factory=dict)        # block pos -> [B, max_len, d] normalised readout output
    ssm: Tensor | None = None                          # [B, max_len, N] fp32
    epoch: int = 0                                     # bumped on truncate (prefetchers key staging on it)

    @classmethod
    def empty(cls, model: Transformer, batch: int, max_len: int, dtype: torch.dtype | None = None) -> "DecodeState":
        dev = model.tok_emb.weight.device
        dtype = dtype or model.tok_emb.weight.dtype
        s = cls(batch=batch, max_len=max_len)
        s.tokens = torch.zeros(batch, max_len, dtype=torch.long, device=dev)
        for pos, block in enumerate(model.blocks):
            a = block.attn
            s.k.append(torch.zeros(batch, a.n_kv_heads, max_len, a.head_dim, dtype=dtype, device=dev))
            s.v.append(torch.zeros(batch, a.n_kv_heads, max_len, a.head_dim, dtype=dtype, device=dev))
            if block.memory is not None:
                if block.memory.rank1_dim:
                    raise NotImplementedError("cached decoding of the rank-1 memory path")
                if block.memory.module.execution_mode == "streaming":
                    raise NotImplementedError("cached decoding with streaming memory readout")
                r = block.memory.routes
                s.codes[pos] = torch.zeros(batch, max_len, r, dtype=torch.uint8, device=dev)
                s.conv_in[pos] = torch.zeros(batch, max_len, model.cfg["d_model"], dtype=dtype, device=dev)
        if model.ssm is not None:
            s.ssm = torch.zeros(batch, max_len, model.ssm.state_dim, dtype=torch.float32, device=dev)
        return s

    def truncate(self, length: int) -> None:
        if not 0 <= length <= self.length:
            raise ValueError(f"cannot truncate to {length} (length {self.length})")
        self.length = length
        self.epoch += 1

    def index_select(self, idx: Tensor) -> "DecodeState":
        """Rows ``idx`` of every cache (e.g. to drop finished sequences)."""
        sel = lambda t: t.index_select(0, idx.to(t.device))  # noqa: E731
        return DecodeState(len(idx), self.max_len, self.length, sel(self.tokens), [sel(t) for t in self.k],
                           [sel(t) for t in self.v], {p: sel(t) for p, t in self.codes.items()},
                           {p: sel(t) for p, t in self.conv_in.items()},
                           None if self.ssm is None else sel(self.ssm))


# ------------------------------------------------------------------ pieces
def rope_at(x: Tensor, positions: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x [B, n, H, D]; positions [B, n] absolute positions (llama2.c pair rotation)."""
    xf = x.float().reshape(*x.shape[:-1], -1, 2)
    x0, x1 = xf[..., 0], xf[..., 1]
    c = cos[positions].unsqueeze(2)  # [B, n, 1, D/2]
    s = sin[positions].unsqueeze(2)
    return torch.stack((x0 * c - x1 * s, x0 * s + x1 * c), dim=-1).flatten(-2).type_as(x)


def ssm_continue(model: Transformer, emb: Tensor, s_prev: Tensor | None) -> Tensor:
    """SSM states for new positions given the state before them ([B, N] or None)."""
    from .memory.ssm import chunked_scan

    ssm = model.ssm
    u = ssm.inputs(emb)
    a = ssm.decay()
    out = chunked_scan(a, u, ssm.chunk)
    if s_prev is not None:
        n = emb.shape[1]
        powers = torch.exp(torch.arange(1, n + 1, device=emb.device, dtype=torch.float32)[:, None] * torch.log(a))
        out = out + powers[None] * s_prev[:, None]
    return out


def router_input_at(model: Transformer, mem, state: DecodeState, emb: Tensor, s_new: Tensor | None) -> Tensor:
    """Token-only router features for the new positions (needs the previous token)."""
    L = state.length
    if L == 0:
        return mem.router_features(emb, s_new)
    prev = model.tok_emb(state.tokens[:, L - 1:L]).to(emb.dtype)
    s_cat = None
    if s_new is not None:
        s_cat = torch.cat((state.ssm[:, L - 1:L], s_new), dim=1)
    return mem.router_features(torch.cat((prev, emb), dim=1), s_cat)[:, 1:]


def memory_step(model: Transformer, pos: int, state: DecodeState, h: Tensor, codes_new: Tensor,
                route_codes_ctx: Tensor | None = None, route_tokens: Tensor | None = None) -> Tensor:
    """Memory output for the new positions. ``codes_new`` [B, n, R] are written into the cache.

    ``route_codes_ctx`` (optional) replaces the cached history for the lookup (tree
    verification: codes along each node's ancestor path); shape [B, n + H, R].
    ``route_tokens`` (optional) [B, n, R, len(ngrams)*mem_dim]: rows supplied by the
    caller (host-RAM serving / prefetch); the table lookup is skipped."""
    mem = model.blocks[pos].memory
    m = mem.module
    L, n = state.length, h.shape[1]
    state.codes[pos][:, L:L + n] = codes_new
    hist = m.route_history_length
    if route_codes_ctx is None:
        start = max(0, L - hist)
        route_codes_ctx = state.codes[pos][:, start:L + n]
    if route_tokens is None:
        tokens = m._lookup_route_chunk(route_codes_ctx, None, 0, m.num_routes, None, None, target_length=n)
    else:
        tokens = route_tokens.to(h.device, m.readout_kv_proj.weight.dtype)
    key, value = m._project_memory(tokens)
    head = m._manual_gqa(m._project_query(h), key, value)
    mixed = m.readout_o_proj(head.reshape(h.shape[0], n, m.readout_inner_dim))
    conv = m.short_conv
    state.conv_in[pos][:, L:L + n] = conv.norm(mixed).to(state.conv_in[pos].dtype)
    lp = conv.left_padding
    window = state.conv_in[pos][:, max(0, L - lp):L + n]
    if window.shape[1] < lp + n:
        window = F.pad(window, (0, 0, lp + n - window.shape[1], 0))
    conv_out = conv.act(conv.conv(window.transpose(1, 2).to(mixed.dtype))).transpose(1, 2)
    return mixed + conv_out


def _attend_cached(block, state: DecodeState, pos: int, x: Tensor, positions: Tensor,
                   cos: Tensor, sin: Tensor, attn_mask: Tensor | None) -> Tensor:
    a = block.attn
    b, n, _ = x.shape
    L = state.length
    q = rope_at(a.wq(x).view(b, n, a.n_heads, a.head_dim), positions, cos, sin).transpose(1, 2)
    k = rope_at(a.wk(x).view(b, n, a.n_kv_heads, a.head_dim), positions, cos, sin).transpose(1, 2)
    v = a.wv(x).view(b, n, a.n_kv_heads, a.head_dim).transpose(1, 2)
    state.k[pos][:, :, L:L + n] = k.to(state.k[pos].dtype)
    state.v[pos][:, :, L:L + n] = v.to(state.v[pos].dtype)
    kk, vv = state.k[pos][:, :, :L + n], state.v[pos][:, :, :L + n]
    g = a.n_heads // a.n_kv_heads
    if g > 1:
        kk, vv = kk.repeat_interleave(g, 1), vv.repeat_interleave(g, 1)
    if attn_mask is None:
        attn_mask = torch.ones(n, L + n, dtype=torch.bool, device=x.device).tril(diagonal=L)
    out = F.scaled_dot_product_attention(q, kk.to(q.dtype), vv.to(q.dtype), attn_mask=attn_mask)
    return a.wo(out.transpose(1, 2).reshape(b, n, a.n_heads * a.head_dim))


# ------------------------------------------------------------------ step
@torch.no_grad()
def decode_step(model: Transformer, state: DecodeState, new_tokens: Tensor, *,
                positions: Tensor | None = None, attn_mask: Tensor | None = None,
                route_codes_ctx: dict | None = None, row_provider=None, return_hidden: bool = False):
    """Process ``new_tokens`` [B, n] after the cached ``state.length`` positions.

    Returns logits [B, n, V] (and the final hidden states if ``return_hidden``).
    ``positions`` [B, n] (RoPE) default to ``L .. L+n-1``; ``attn_mask`` [n, L+n] (bool,
    True = attend) defaults to causal.  ``route_codes_ctx`` {block pos: [B, n+H, R]}
    overrides memory n-gram context (tree verification).  ``row_provider(pos, state,
    codes_new) -> route tokens [B, n, R, D] | None`` supplies memory rows from elsewhere
    (host RAM / prefetch buffers); it is called after the new codes are known and
    before they are written to the cache (``state.length`` is still L)."""
    b, n = new_tokens.shape
    L = state.length
    if L + n > state.max_len:
        raise ValueError(f"cache full: {L} + {n} > {state.max_len}")
    if positions is None:
        positions = torch.arange(L, L + n, device=new_tokens.device).expand(b, n)
    emb = model.tok_emb(new_tokens)
    s_new = None
    if model.ssm is not None:
        s_new = ssm_continue(model, emb.detach(), state.ssm[:, L - 1] if L > 0 else None)
    h = emb
    cos, sin = model.rope_cos, model.rope_sin
    for pos, block in enumerate(model.blocks):
        if block.memory is not None:
            mem = block.memory
            h_norm = block.attn_norm(h)
            if mem.router_input == "hidden":
                q_logits = mem.route_logits(h_norm)
            else:
                q_logits = mem.route_logits(router_input_at(model, mem, state, emb, s_new))
            codes = mem.module.quantize_route_logits(q_logits)
            ctx = None if route_codes_ctx is None else route_codes_ctx.get(pos)
            rows = None if row_provider is None else row_provider(pos, state, codes)
            h = h + memory_step(model, pos, state, h, codes, ctx, rows)
        h = h + _attend_cached(block, state, pos, block.attn_norm(h), positions, cos, sin, attn_mask)
        h = h + block.ffn(block.ffn_norm(h))
    if s_new is not None:
        state.ssm[:, L:L + n] = s_new
    state.tokens[:, L:L + n] = new_tokens
    state.length = L + n
    hidden = model.norm(h)
    logits = model.output(hidden)
    return (logits, hidden) if return_hidden else logits


@torch.no_grad()
def greedy_generate(model: Transformer, prompts: Tensor, n_new: int, *, max_len: int | None = None,
                    amp_dtype: torch.dtype | None = None) -> tuple[Tensor, DecodeState]:
    """Greedy continuation of equal-length prompts [B, P] -> tokens [B, n_new] and the final state."""
    b, p = prompts.shape
    state = DecodeState.empty(model, b, max_len or (p + n_new))
    ctx = torch.autocast(prompts.device.type, dtype=amp_dtype) if amp_dtype else torch.autocast(
        prompts.device.type, enabled=False)
    out = []
    with ctx:
        logits = decode_step(model, state, prompts)
        nxt = logits[:, -1].argmax(-1)
        for i in range(n_new):
            out.append(nxt)
            if i == n_new - 1:
                break
            logits = decode_step(model, state, nxt[:, None])
            nxt = logits[:, -1].argmax(-1)
    return torch.stack(out, 1), state
