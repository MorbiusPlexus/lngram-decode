"""Tree verification for speculative decoding (M5.T3 stretch).

    spec = default_tree()                                   # 8 nodes, parent indices + top-k ranks
    toks, stats = tree_generate(model, drafter, prompt, n_new=64, spec=spec)

A tree is ``parent`` (node 0 = root = the pending token, ``parent[0] = -1``, ``parent[i] < i``)
plus ``rank`` (node i takes the drafter's rank-th candidate after its parent's path).  Node i
sits at position ``L + depth(i)`` and attends to the cache and to its ancestors (and itself).

Everything that depends on "the previous positions" must follow each node's ancestors, not
the block order, so ``tree_verify_step`` is a tree-aware ``decode_step``:

- SSM state: ``s[i] = a * s[parent(i)] + (1 - a) * u[i]`` (root continues the cache);
- token router features: ``E[x[parent(i)]]`` is the previous token;
- memory n-grams: ``tree_ngram_codes(parent_idx, codes)`` builds each node's code window from
  its ancestors (then the cached history) -> rows via v2's lookup (``tree_route_tokens``);
- the readout's short causal conv: window over the ancestors' conv inputs;
- attention: ``tree_attention_mask`` + ``tree_positions`` (RoPE by depth).

After verification ``tree_accept`` returns the longest root-to-node path whose drafted tokens
equal the verifier's argmax at their parent, and ``compact_path`` moves that path's cache
entries to consecutive positions, so the cache equals chain decoding of the accepted tokens.
(``decode_step``'s ``attn_mask``/``positions``/``route_codes_ctx`` alone cannot express this:
its SSM, embed2 router input and short conv follow block order.)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ffnmem.infer import DecodeState, _attend_cached, decode_step


@dataclass(frozen=True)
class TreeSpec:
    parent: tuple
    rank: tuple

    def __post_init__(self):
        if len(self.parent) != len(self.rank) or not self.parent or self.parent[0] != -1:
            raise ValueError("parent[0] must be -1 and parent/rank have equal length")
        for i, p in enumerate(self.parent[1:], 1):
            if not 0 <= p < i:
                raise ValueError(f"node {i}: parent {p} must precede it")

    @property
    def n(self) -> int:
        return len(self.parent)


def default_tree() -> TreeSpec:
    """8 nodes: top-1 chain of depth 4, runner-up at the first position (+ its top-1 child),
    runner-up at the second position."""
    return TreeSpec((-1, 0, 1, 2, 3, 0, 5, 1), (0, 0, 0, 0, 0, 1, 0, 1))


def chain_tree(n: int) -> TreeSpec:
    return TreeSpec((-1,) + tuple(range(n - 1)), (0,) * n)


# ---------------------------------------------------------------------- structure
def tree_depths(parent) -> list:
    d = []
    for p in parent:
        d.append(0 if p < 0 else d[p] + 1)
    return d


def tree_path(parent, i: int) -> list:
    """Root .. i."""
    out = []
    while i >= 0:
        out.append(i)
        i = parent[i]
    return out[::-1]


def tree_positions(parent, L: int = 0, device=None) -> Tensor:
    return L + torch.tensor(tree_depths(parent), dtype=torch.long, device=device)


def tree_attention_mask(parent, L: int = 0, device=None) -> Tensor:
    """bool [n, L + n]: every cached position, plus ancestors and self."""
    n = len(parent)
    m = torch.zeros(n, L + n, dtype=torch.bool, device=device)
    m[:, :L] = True
    for i in range(n):
        for a in tree_path(parent, i):
            m[i, L + a] = True
    return m


def tree_slots(parent, L: int, back: int) -> tuple[Tensor, Tensor]:
    """Slots [n, back+1] (oldest first) of each node's ``back`` predecessors along its ancestors,
    then the ``L`` positions before the root; slot L + a = node a.  Validity mask alongside."""
    n = len(parent)
    idx = torch.zeros(n, back + 1, dtype=torch.long)
    ok = torch.zeros(n, back + 1, dtype=torch.bool)
    for i in range(n):
        path = tree_path(parent, i)
        for j in range(back + 1):          # j steps back from node i
            if j < len(path):
                s = L + path[len(path) - 1 - j]
            else:
                s = L - (j - len(path) + 1)
            if s >= 0:
                idx[i, back - j], ok[i, back - j] = s, True
    return idx, ok


def tree_ngram_codes(parent_idx, codes: Tensor, history: Tensor | None = None,
                     hist: int = 2) -> tuple[Tensor, Tensor]:
    """Each node's n-gram code window from its ancestors (not the block order).

    codes [B, n, R] of the tree nodes; history [B, h, R] codes of the positions before the root
    (oldest first).  Returns (ctx [B, n, hist + 1, R], valid [n, hist + 1]); ctx[:, i, -1] is
    node i, ctx[:, i, -2] its parent, ... continuing into ``history``; invalid slots hold 0."""
    b, n, r = codes.shape
    h = 0 if history is None else history.shape[1]
    allc = codes if history is None else torch.cat((history.to(codes.dtype), codes), 1)
    idx, ok = tree_slots(parent_idx, h, hist)
    idx, ok = idx.to(codes.device), ok.to(codes.device)
    ctx = allc[:, idx.clamp_min(0)] * ok[None, :, :, None].to(codes.dtype)
    return ctx, ok


def tree_route_tokens(m, ctx: Tensor, valid: Tensor) -> Tensor:
    """Route tokens [B, n, R, Dt] of the n-grams ending at each node (v2 lookup over the windows;
    n-grams that reach an invalid slot are zero, as at the sequence start)."""
    b, n, w, r = ctx.shape
    seg = valid.to(torch.int32)[None].expand(b, n, w).reshape(b * n, w)
    rt = m._lookup_route_chunk(ctx.reshape(b * n, w, r), None, 0, m.num_routes, seg, None)
    return rt[:, -1].reshape(b, n, r, -1)


# ---------------------------------------------------------------------- verification
@torch.no_grad()
def tree_verify_step(model, state: DecodeState, tokens: Tensor, parent) -> Tensor:
    """Logits [B, n, V] for the tree nodes ``tokens`` [B, n] after the cached positions.
    Writes caches at L .. L+n-1 in node order and sets ``state.length = L + n``."""
    b, n = tokens.shape
    L = state.length
    if L + n > state.max_len:
        raise ValueError(f"cache full: {L} + {n} > {state.max_len}")
    dev = tokens.device
    positions = tree_positions(parent, L, dev).expand(b, n)
    mask = tree_attention_mask(parent, L, dev)
    emb = model.tok_emb(tokens)
    s_new = None
    if model.ssm is not None:
        u = model.ssm.inputs(emb.detach())
        a = model.ssm.decay()
        root_prev = state.ssm[:, L - 1] if L > 0 else torch.zeros_like(u[:, 0])
        ss = []
        for i in range(n):
            sp = root_prev if parent[i] < 0 else ss[parent[i]]
            ss.append(a * sp + (1 - a) * u[:, i])
        s_new = torch.stack(ss, 1)
    h = emb
    cos, sin = model.rope_cos, model.rope_sin
    for pos, block in enumerate(model.blocks):
        if block.memory is not None:
            mem = block.memory
            m = mem.module
            if mem.router_input == "hidden":
                q_logits = mem.route_logits(block.attn_norm(h))
            else:
                feats = mem.router_features
                if L > 0:
                    root_prev_emb = model.tok_emb(state.tokens[:, L - 1]).to(emb.dtype)
                else:
                    root_prev_emb = feats.start.to(emb.dtype).expand(b, -1)
                prev = torch.stack([root_prev_emb if parent[i] < 0 else emb[:, parent[i]] for i in range(n)], 1)
                q_logits = mem.route_logits(feats.step(prev, emb, s_new))
            codes = m.quantize_route_logits(q_logits)
            hist = m.route_history_length
            ctx, valid = tree_ngram_codes(parent, codes, state.codes[pos][:, max(0, L - hist):L], hist)
            rt = tree_route_tokens(m, ctx, valid)
            state.codes[pos][:, L:L + n] = codes
            key, value = m._project_memory(rt)
            head = m._manual_gqa(m._project_query(h), key, value)
            mixed = m.readout_o_proj(head.reshape(b, n, m.readout_inner_dim))
            conv = m.short_conv
            ci = state.conv_in[pos]
            ci[:, L:L + n] = conv.norm(mixed).to(ci.dtype)
            lp = conv.left_padding
            idx, ok = tree_slots(parent, L, lp)
            idx, ok = idx.to(dev), ok.to(dev)
            window = ci[:, idx] * ok[None, :, :, None].to(ci.dtype)          # [B, n, lp+1, d]
            d = window.shape[-1]
            conv_out = conv.act(conv.conv(window.reshape(b * n, lp + 1, d).transpose(1, 2).to(mixed.dtype)))
            h = h + (mixed + conv_out.transpose(1, 2).reshape(b, n, d))
        h = h + _attend_cached(block, state, pos, block.attn_norm(h), positions, cos, sin, mask)
        h = h + block.ffn(block.ffn_norm(h))
    if s_new is not None:
        state.ssm[:, L:L + n] = s_new
    state.tokens[:, L:L + n] = tokens
    state.length = L + n
    return model.output(model.norm(h))


def tree_accept(parent, node_tokens, argmax) -> tuple[list, list]:
    """Longest root-to-node path whose tokens equal the verifier's argmax at their parent
    (first matching child in index order).  Returns (path nodes incl. root, new tokens =
    accepted drafted tokens + the verifier's token after the path)."""
    children: dict = {}
    for i, p in enumerate(parent):
        if p >= 0:
            children.setdefault(p, []).append(i)
    path, cur = [0], 0
    while True:
        nxt = next((c for c in children.get(cur, []) if int(node_tokens[c]) == int(argmax[cur])), None)
        if nxt is None:
            break
        path.append(nxt)
        cur = nxt
    return path, [int(node_tokens[c]) for c in path[1:]] + [int(argmax[cur])]


def compact_path(state: DecodeState, L: int, path) -> None:
    """Move the cache entries of tree nodes ``path`` (slots L + path) to L, L+1, ...; length = L + len(path)."""
    dev = state.tokens.device
    src = L + torch.as_tensor(path, dtype=torch.long, device=dev)
    dst = L + torch.arange(len(path), device=dev)

    def mv(t: Tensor, dim: int) -> None:
        t.index_copy_(dim, dst, t.index_select(dim, src))

    mv(state.tokens, 1)
    for t in state.k + state.v:
        mv(t, 2)
    for t in list(state.codes.values()) + list(state.conv_in.values()):
        mv(t, 1)
    if state.ssm is not None:
        mv(state.ssm, 1)
    state.length = L + len(path)


# ---------------------------------------------------------------------- drafting + loop
def draft_tree(drafter, spec: TreeSpec) -> list:
    """Node tokens: root = the drafter's last committed token; node i = rank[i]-th candidate
    after its parent's path (the drafter's own codes along the path)."""
    k = max(spec.rank) + 1
    toks = [drafter.tokens[-1]]
    cache: dict = {}
    for i in range(1, spec.n):
        p = spec.parent[i]
        if p not in cache:
            hyp = [toks[a] for a in tree_path(spec.parent, p)[1:]]
            cache[p] = drafter.draft(1, hyp, topk=k).topk_ids[0]
        ids = cache[p]
        t = int(ids[spec.rank[i]])
        toks.append(t if t >= 0 else int(ids[0]))
    return toks


@torch.no_grad()
def tree_generate(model, drafter, prompt, n_new: int, spec: TreeSpec | None = None) -> tuple[list, dict]:
    """Greedy speculative decoding with tree verification (eager, batch 1)."""
    spec = spec or default_tree()
    dev = model.tok_emb.weight.device
    prompt = [int(x) for x in (prompt.view(-1).tolist() if isinstance(prompt, Tensor) else prompt)]
    P = len(prompt)
    state = DecodeState.empty(model, 1, P + n_new + spec.n + 1)
    if P > 1:
        decode_step(model, state, torch.tensor([prompt[:-1]], device=dev))
    drafter.reset()
    drafter.commit(prompt)
    out: list = []
    stats = {"blocks": 0, "accepted": []}
    while len(out) < n_new:
        L = state.length
        toks = draft_tree(drafter, spec)
        logits = tree_verify_step(model, state, torch.tensor([toks], device=dev), spec.parent)
        y = logits[0].argmax(-1).tolist()
        path, new = tree_accept(spec.parent, toks, y)
        compact_path(state, L, path)
        drafter.commit(new)
        out.extend(new)
        stats["blocks"] += 1
        stats["accepted"].append(len(path) - 1)
    return out[:n_new], stats
