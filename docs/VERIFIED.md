# VERIFIED — answers to the plan's open items (Section 8)

Source checked: `ffnmem/memory/lngram_v2.py`, vendored byte-identical from
Lngram commit `c8a53ca110dabe7903f8c6b623b3054561f36647` (see `ffnmem/memory/VENDORED.md`),
and `ffnmem/memory/wrapper.py`. Line numbers refer to `lngram_v2.py`.

## 1. N-gram direction and row index (M4.T1 / gate G1) — assumption (b) is WRONG

- **(a) Suffix ending at t: yes.** `_lookup_route_chunk` (L1047) computes addresses for
  windows `codes[t-n+1 .. t]` and left-pads with `n-1` zero rows, so position `t`
  reads the n-gram that ends at `t`. Positions `t < n-1` get a zero row (masked by
  `_valid_mask`). No document-boundary masking in our use (`segment_ids=None`).
- **(b) Digit order: the OLDEST code is least significant, the CURRENT code is most significant.**
  `_aligned_ngram_address` (L314) starts from `codes[:, :valid]` (the oldest code of
  each window) and ORs in `codes[exp]` shifted by `exp*bits`. With 4 bits:
  - 2-gram row = `r*256 + c[t-1] + 16*c[t]`
  - 3-gram row = `r*4096 + c[t-2] + 16*c[t-1] + 256*c[t]`

  The plan assumed `16*c[t-1] + c[t]`. **Consequence for prefetch:** once step t's codes
  are known, the 16 candidate rows for step t+1 are
  `r*256 + c[t] + 16*k` (stride 16) and `r*4096 + c[t-1] + 16*c[t] + 256*k`
  (stride 256), k = 0..15. They are **not contiguous**, so G1 fails as designed.
  Fix per plan: permute rows once at load time to put the current code in the
  least-significant digit (`row' = r*16^n + reverse_digits(addr)`), then check the
  outputs match exactly (`tests/test_index_layout.py`, owner M4).
- **(c) Layout:** one `nn.Embedding` per order (`tables["ngram_2"]`, `tables["ngram_3"]`),
  each `[R * 16^n, mem_dim]`. Route `r` owns the contiguous block `[r*16^n, (r+1)*16^n)`.

## 2. Router input and the surrogate gradient

- The vendored module does not own the router. `wrapper.LngramMemory` computes
  `q_logits = router(h_norm)` in fp32 and calls `module(h, codes, q_logits=...)`.
  `h_norm` is the block's pre-attention RMSNorm output, so the router reads the
  layer-1 residual stream.
- The surrogate (`_LngramSurrogate`, L328) has an identity forward on the looked-up
  rows. Its backward returns a gradient **for `q_logits` only** (approximate mode:
  one-bit counterfactual flips against the detached table). It is active only when
  `module.training and torch.is_grad_enabled() and q_logits is not None`; the wrapper
  passes `q_logits=None` in eval.
- Because the gradient goes to `q_logits`, it does not matter what the router reads.
  With `router_input: embed2` it flows through the router weight into the token
  embeddings. `tests/test_router_input.py` checks that the router and the
  embedding get non-zero gradients.
- Non-reentrant checkpointing is still required: under reentrant checkpointing,
  `torch.is_grad_enabled()` is False during the forward, so the surrogate turns off.

## 3. Shared KV projection output and Sink (needed by the draft readout)

- `_lookup_route_chunk(codes, q_btrm, 0, R, seg, valid)` returns **route tokens
  `[B, T, R, len(ngrams)*mem_dim]` = `[B, T, R, 256]`**. The 2-gram and 3-gram rows
  of a route are **concatenated in the feature dimension**, so there are R slots per
  position, not R·orders.
- `_project_memory(tokens)` (L1120): `readout_memory_norm` (RMSNorm over 256) →
  `readout_kv_proj` (256 → 2·kv_heads·head_dim = 512) → split →
  `key, value: [B, T, kv_heads=4, R, head_dim=64]`.
- Sink: there is no tensor and no parameter. A fixed scalar logit
  `b0 = log(R*(1-m)/m)` (m = `sink_initial_active_mass` = 0.5, so b0 = log R) is added
  to the softmax denominator with a zero value
  (`RouteNgramMemoryV2._real_route_probabilities`, L1887). The draft readout reuses
  this function, so it sees the same Sink.
- After `readout_o_proj` the module adds a depthwise causal conv over time
  (`_finish`, L1354: `mixed + short_conv(mixed)`). The draft readout does **not** use
  it, because it would need neighbouring positions' readouts.

## 4. Query heads in v2's readout

16 query heads × 64 (the module requires `q_heads*head_dim == d_model`), 4 KV heads,
4 groups. The draft readout uses 4 query heads (1 per KV head).

## 5. Host RAM, PCIe generation, NUMA — OPEN

Needs `free -g` and `nvidia-smi topo -m` on the serving machine (see `serving/TIMING.md`).

## 6. `w2_mem_super1` — PARTLY OPEN

There is no config for it in the Wave 1 repo. `configs/w2_mem_super1.yaml` rebuilds
it as Run 2's recipe (24/4 attention, `d_ff` cut in layers 2–5 and 12–15) with memory
at layer 1 only and R = 184. This gives exactly **369,777,920** non-embedding
params, the pivot plan's number. Still unknown: whether the 2B run finished, which
hardware it ran on, and its final val loss.

## 7. Block-deletion spec — OPEN (needs team agreement; no config written)

## 8. Code subset in ClimbMix shards — OPEN
