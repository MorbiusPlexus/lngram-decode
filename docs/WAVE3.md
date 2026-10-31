# Wave 3 — status and how to run

## M2–M6 (all code written; CPU-tested on tiny models — nothing has run on real data or GPUs)

| Task | Files | Tests |
|---|---|---|
| Shared: cached decoding (KV + memory + SSM caches, rollback = `truncate`), checkpoint loading | `ffnmem/infer.py` | `tests/test_infer.py` |
| Shared: drafter contract, sketch format, post-hoc drafter model | `drafting/common.py`, `drafting/sketches.py`, `drafting/posthoc_model.py` | via M2/M3/M5 tests |
| M2.T1–T4 dumps, RQ2 predictability, substitution, ladder (+ tuned lens) | `analysis/dump_codes.py`, `code_predictability.py`, `substitution.py`, `decodability_ladder.py` | `tests/test_analysis_*.py` |
| M3.T1–T4 sketches, PLD + token n-gram, post-hoc distillation, offline acceptance, SSM/live gates | `drafting/build_sketches.py`, `baselines.py`, `drafters.py`, `posthoc_distill.py`, `offline_acceptance.py`, `gates.py` | `tests/test_drafting_*.py` |
| M4.T1 serving row layout (G1 fix) | `serving/layout.py` | `tests/test_index_layout.py` |
| M4.T2–T4 B0/B1/B4 prefetch, exactness, bench (+ CUDA graphs), scaled tables, env capture, timing protocol | `serving/prefetch.py`, `exactness.py`, `bench_prefetch.py`, `scaled_tables.py`, `env_capture.py`, `TIMING.md` | `tests/test_serving_*.py` |
| M5 export bundle, CPU drafter (PyTorch reference + C++), live chain loop, tree verification | `drafting/export_drafter.py`, `cpu_drafter.py`, `cpu/drafter.cpp`, `cpu/setup.py`, `live_spec.py`, `tree.py` | `tests/test_live_*.py` |
| M6.T1–T3 lm-eval adapter + script + bootstrap table, FFN importance, D270 vs Run 2 | `eval/lm_eval_adapter.py`, `run_lmeval.sh`, `bootstrap_lmeval.py`, `ffn_importance.py`, `table_worth.py` | `tests/test_eval_*.py` |
| M6.T4 figures (+ `figures.json` input provenance) | `analysis/plot_figures.py` | `tests/test_figures.py` |

Things to know before running on GPUs:
- **The C++ drafter has never been compiled.** There is no compiler on the dev laptop. Build it first with `python drafting/cpu/setup.py build_ext --inplace`, then run `python -m drafting.cpu_drafter --check <bundle>` (parity with the reference) and `tests/test_live_cpu_ext*`. Everything else falls back to the PyTorch reference (`make_drafter(impl="auto")`).
- **CUDA streams, events, pinned memory and CUDA-graph capture** (`serving/`, `live_spec.py --verifier graph`) are written but only tested on CPU. Graph capture falls back to eager execution and records `graph_error` if it fails.
- **lm-eval:** run it as `python -m eval.lm_eval_adapter lm_eval --model morbius ...`. lm-eval 0.4.13 has no plugin hook, so plain `lm_eval --model morbius` does not work. MMLU uses the `mmlu_continuation` task (cloze form).
- **Code text type:** offline acceptance skips it unless `--code_doc_ids` is given (VERIFIED §8 is still open).
- **Random-router control:** `analysis/decodability_ladder.py` reuses `drafting.build_sketches.random_router_weight`, so the control sketch and the val codes come from the same router. Keep `--seed` equal in both (default 0).
- **Speculative decoding with SD-B:** the CPU drafter's fp32 codes can differ from the GPU fp16 router's near 0. The live loop checks `blk.codes_match()` (serving) and falls back to a synchronous gather.

## M1

## What is in this repo (M1)

| Task | Where | Test |
|---|---|---|
| T0 Wave 1 regression | `tests/wave1_ref/` (frozen pre-Wave-3 model), `tests/data/wave1_reference.npz` | `tests/test_wave1_unchanged.py` |
| T1 router input `hidden / embed2 / embed2_ssm` | `ffnmem/memory/router.py`, `wrapper.py` | `tests/test_router_input.py` |
| T2 draft readout | `ffnmem/memory/draft.py`, `Transformer.draft_logits` | `tests/test_draft_isolation.py` |
| T3 diagonal SSM | `ffnmem/memory/ssm.py` | `tests/test_ssm_scan.py` |
| T4 draft loss | `ffnmem/losses.py`, `Transformer._add_draft_loss`, `train.py` | `tests/test_stopgrad.py`, `tests/test_sd_train.py` |
| T5 configs + params | `configs/w3_*.yaml`, `configs/w2_mem_super1.yaml`, `scripts/print_params.py` | `print_params.py` exit code |
| T6 launch | `scripts/launch_4gpu.sh`, `scripts/launch_2gpu.sh` | — |

Parameter matching (from `scripts/print_params.py`):

| config | R | non-embedding | vs SB |
|---|---|---|---|
| w3_sb_1b | 184 | 369,777,920 | — |
| w3_sdb_1b | 182 | 369,930,496 | +0.041% |
| w3_sdc_1b | 181 | 369,879,552 | +0.028% |
| w3_sda_1b | 180 | 369,537,280 | −0.065% |
| w3_dense270 | — | 269,796,352 | +0.093% vs Run 2 active |

## Implementation decisions to know

- **Config keys live under `model:`** (`model.draft`, `model.ssm`, `model.memory.router_input`, `model.memory.token_router`). `memory.routes: auto` is solved when the config loads, against `params.match_config`.
- **1B runs use `tokens_total: 999,817,216`**, which is exactly 1,907 steps. Setting 1e9 rounds up to 1,908.
- **Warmup is 100 steps, not 38.** `base.yaml` keeps `warmup_min_steps: 100`, so warmup is max(100, 2%), the same as in Wave 1. Use `--set train.warmup_min_steps=0` for a strict 2%. Apply it to all three SB/SD runs or to none.
- **Smoke configs run the target's own schedule and stop at step 95** (`train.max_steps`). Their loss at step 90 is then comparable to SB's log step for step, which gate G0 needs.
- **Logged `loss` is the LM loss** for SD runs, so it compares directly with SB. The extra metrics are `loss/draft_kl`, `draft/top1_agree`, `draft/accept_proxy`, and `loss/route_bce` (SD-A only). Eval adds `val_draft_*` on the diagnostic batches.
- **Draft-loss positions** are seeded by (data_seed, step, global window id). They don't depend on GPU count, micro-batch, or resume (tested).
- **The draft path keeps its own gradients off the embeddings.** It reads `E.detach()` for its query, its unembedding, and its own copy of the router features. The LM path's router (`embed2`) still trains `E`, as the plan requires. The SSM always reads `E.detach()`.
- **The draft path gathers rows separately** at the selected positions. It runs v2's `_lookup_route_chunk` over 3-token windows, so the v2 surrogate carries the draft gradient to the router. The rows equal the full-sequence lookup bitwise (tested).
- **No short conv in the draft path.** v2's readout adds a causal depthwise conv after `o_proj`; the draft does not use it.
- **Memory rows are not contiguous by current code** (see `docs/VERIFIED.md` §1), so gate G1 fails as designed. Permute rows at load time (M4).
- **SD configs require memory `execution_mode: full`**, which Wave 1 already uses.
- `train.strip_old_optimizer: true` removes optimizer state from every checkpoint except the newest. `train.save_steps` keeps those steps permanently.

## Commands

```bash
python -m pytest -q tests                                    # 231 pass, 17 skip on CPU (C++ ext + GPU-only)
python scripts/print_params.py                               # breakdown + match checks
python -m ffnmem.fitcheck --config configs/w3_sdb_1b.yaml --nproc 4 --micro_batch 1 2 --seq_len 2048 --zero1 on --out runs/fitcheck/w3_2080ti.jsonl
DATA_DIR=/path/climbmix PY=/path/python nohup scripts/launch_4gpu.sh w3_sb_1b > runs/w3_sb_1b.log 2>&1 &
DATA_DIR=/path/climbmix PY=/path/python nohup scripts/launch_2gpu.sh w3_smoke_sdb > runs/w3_smoke_sdb.log 2>&1 &
```

## Not covered by the CPU test suite

- The peak-memory fit for SD-B at mb 2 on a 2080 Ti (fitcheck above). If it OOMs, use `--set train.micro_batch=1`.
- The fp16 GPU tests. They skip on this laptop (CPU torch).
- VERIFIED items 5–8: RAM/PCIe/NUMA, the `w2_mem_super1` run status, the BD spec, the code subset.
- BD config: not written until the spec is agreed.
