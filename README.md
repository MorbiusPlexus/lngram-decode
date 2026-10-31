# Self-Drafting Latent Memory

**Can an n-gram memory draft tokens for its own language model?** This repository holds the code, training logs, evaluation results and paper for a study of *self-drafting* route n-gram memory ([Lngram](https://github.com/zyaaa-ux/Lngram) v2).

At each position, a route n-gram memory quantises its input into short discrete codes, and the codes of the last few positions address rows of a large table. We make those codes depend on tokens only. That lets a small *draft head* read the same table without running the transformer backbone, and propose tokens for speculative decoding of its own model. We train the draft head jointly with the language model.

## Main findings

| Finding | Evidence |
|---|---|
| **Self-drafting is free.** Token-only routing plus a jointly trained draft head costs nothing measurable. | Validation loss +0.0004 to +0.003 nats vs. the baseline (two baseline runs differ by 0.002). No significant difference on six zero-shot benchmarks. |
| **It is free because layer-1 routing is already a near per-token lookup.** | The current token explains 86% of the baseline's code entropy. Codes predicted from three tokens keep 99.8% of the memory's 0.28-nat contribution. |
| **Joint training makes a much better drafter than post-hoc fitting.** | Top-1 agreement with the target 0.45 vs 0.30. Accepted length roughly doubles. A small document-level SSM adds 25–30%. |
| **Code-keyed online counts are strong.** | 2.5–2.6 accepted tokens per verification step at γ=4. That beats prompt lookup on open-ended text. |
| **Prompt lookup's lead comes from greedy self-repetition.** | 49% of greedy continuation tokens copy the model's own earlier output. In the first 16 tokens, the self-drafting heads accept 40–65% more than prompt lookup. |
| **A hybrid beats prompt lookup.** | Prompt lookup when it matches, else the learned head: +0.05 to +0.13 tokens per step, CIs excluding 0. A per-row oracle shows ≈ +0.3 of headroom for a learned switch. |
| **The drafter knows when it's right.** | Its own confidence predicts agreement (r = 0.50). Gating at 0.5 gives 88% agreement on 18% of positions. |
| **What the memory learns.** | Emergent part-of-speech-like code classes. Code assignments are not identifiable across runs (37% agreement at equal loss). Hidden-state routes are 2.7× more redundant than token-routed ones. The memory helps most on frequent, local and formatting tokens. |
| **Serving and correctness.** | Serving the table from host RAM is bit-exact and costs ~1% latency. The C++ CPU drafter matches the reference to 1e-5. Live speculative decoding is exact in fp32. |

**Open problems.** The current live decoder is slower than greedy decoding (0.28–0.58×), because the C++ drafter isn't optimised yet. At 370M parameters and 1B tokens, a dense model with equal active parameters matches the memory model. The paper's §10 lists the directions the evidence points to.

Paper: [`paper/main.tex`](paper/main.tex) (ACL format; figures in `paper/figures/`). Detailed results: [`docs/RESULTS.md`](docs/RESULTS.md).

## Models

All models are 24-layer, d=1024 Llama-style decoders trained on the same 1B ClimbMix tokens with the same schedule (see `configs/`).

| Name | Config | Description |
|---|---|---|
| SB | `w3_sb_1b` | Baseline: one Lngram v2 layer (184 routes, 102M-parameter table) at layer 1, hidden-state router. Trained twice (`sb_run1`, `sb_run2`). |
| SD-B | `w3_sdb_1b` | Token-embedding router + jointly trained draft head (182 routes, parameter-matched). |
| SD-C | `w3_sdc_1b` | SD-B + a 256-channel diagonal SSM feeding the router and the draft query (181 routes). |
| D270 | `w3_dense270` | Dense control with no memory and the same active parameters per token; 2B tokens. |

## Weights and training logs

Checkpoints are on the Hugging Face Hub at [afoolishman/self-drafting-latent-memory](https://huggingface.co/afoolishman/self-drafting-latent-memory), with a model card per folder. Live training curves are in the W&B project [morbius-wave3](https://forge.coreweave.com/wandb/kishor-iiit-hyderabad/morbius-wave3). The same logs are also in `results/training/<run>/train.jsonl`.

| Model | Weights | W&B run | Final val loss |
|---|---|---|---|
| SB run 1 | [`sb-run1/`](https://huggingface.co/afoolishman/self-drafting-latent-memory/tree/main/sb-run1) | [w3_sb_1b-415a3242](https://forge.coreweave.com/wandb/kishor-iiit-hyderabad/morbius-wave3/runs/w3_sb_1b-415a3242) | 2.7127 |
| SB run 2 | [`sb-run2/`](https://huggingface.co/afoolishman/self-drafting-latent-memory/tree/main/sb-run2) | [w3_sb_1b-7f059808](https://forge.coreweave.com/wandb/kishor-iiit-hyderabad/morbius-wave3/runs/w3_sb_1b-7f059808) | 2.7107 |
| SD-B | [`sd-b/`](https://huggingface.co/afoolishman/self-drafting-latent-memory/tree/main/sd-b) | [w3_sdb_1b-fb508de3](https://forge.coreweave.com/wandb/kishor-iiit-hyderabad/morbius-wave3/runs/w3_sdb_1b-fb508de3) | 2.7159 |
| SD-C | [`sd-c/`](https://huggingface.co/afoolishman/self-drafting-latent-memory/tree/main/sd-c) | [w3_sdc_1b-b2100336](https://forge.coreweave.com/wandb/kishor-iiit-hyderabad/morbius-wave3/runs/w3_sdc_1b-b2100336) | 2.7111 |
| D270 (2B tokens) | [`d270/`](https://huggingface.co/afoolishman/self-drafting-latent-memory/tree/main/d270) | [w3_dense270-448b0ee5](https://forge.coreweave.com/wandb/kishor-iiit-hyderabad/morbius-wave3/runs/w3_dense270-448b0ee5) | 2.5147 |
| Post-hoc drafter | [`posthoc-drafters/posthoc_sb.pt`](https://huggingface.co/afoolishman/self-drafting-latent-memory/tree/main/posthoc-drafters) | [ph_sb-73624b3a](https://forge.coreweave.com/wandb/kishor-iiit-hyderabad/morbius-wave3/runs/ph_sb-73624b3a) | — |
| Post-hoc drafter + SSM | [`posthoc-drafters/posthoc_sb_ssm.pt`](https://huggingface.co/afoolishman/self-drafting-latent-memory/tree/main/posthoc-drafters) | [ph_sb_ssm-4e4d104e](https://forge.coreweave.com/wandb/kishor-iiit-hyderabad/morbius-wave3/runs/ph_sb_ssm-4e4d104e) | — |

The W&B project also holds smoke tests and two incomplete runs that aren't used in the paper.

## Repository layout

| Path | Contents |
|---|---|
| `ffnmem/` | Model and trainer. Includes the vendored Lngram v2 (`memory/lngram_v2.py`, MIT, unmodified), token routers (`memory/router.py`), SSM (`memory/ssm.py`), draft readout (`memory/draft.py`), and cached decoding (`infer.py`). |
| `drafting/` | Count sketches, prompt-lookup and n-gram baselines, post-hoc distillation, offline acceptance harness, drafter export, C++ CPU drafter (`cpu/`), live speculative loop, tree verification. |
| `serving/` | Host-RAM serving layout (row permutation), prefetch modes B0/B1/B4, exactness and latency benchmarks. |
| `analysis/` | Code dumps, code predictability, substitution, decodability ladder (incl. tuned lens), interpretability studies (`interp_gpu.py`, `interp_studies.py`). |
| `eval/` | lm-eval adapter and bootstrap tables, FFN importance. |
| `configs/`, `scripts/`, `slurm/` | Run configs, 2- and 4-GPU launchers, Slurm job scripts. |
| `tests/` | 231 CPU tests (`pytest -q tests`); a further 17 need a GPU or the C++ build. |
| `results/training/<run>/` | `config.yaml`, `params.json`, and `train.jsonl` (every logged step and evaluation) for SB (two runs), SD-B, SD-C, D270, and both post-hoc drafters. |
| `results/evaluation/` | `acceptance/` (offline acceptance, gates), `lmeval/` (per-model results, paired-CI table, per-sample archive), `predictability/`, `ladder/`, `serving/` (exactness, latency), `drafter/` (C++ parity and benchmark, live decoding, speedup model). |
| `results/interp/` | Interpretability studies (see `docs/RESULTS.md` §5). |
| `paper/` | LaTeX source, bibliography, figures, and `make_figures.py`, which regenerates every figure from `results/`. |
| `docs/` | `RESULTS.md` (all numbers), `WAVE3.md` (implementation notes), `VERIFIED.md` (facts about Lngram v2's layout). |

Checkpoints aren't in this repository (1.2–4.8 GB each). They are on the [Hugging Face Hub](https://huggingface.co/afoolishman/self-drafting-latent-memory).

## Reproduce

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -m pytest -q tests

python data/download_dataset.py --tokenizer data/tokenizer --output_dir data/climbmix \
    --target_tokens 2200000000 --val_tokens 50000000
python -m ffnmem.data.build_index --data data/climbmix
DATA_DIR=data/climbmix PY=python scripts/launch_4gpu.sh w3_sdc_1b      # or launch_2gpu.sh

python paper/make_figures.py
```

Evaluation needs `requirements-eval.txt` (lm-eval 0.4.13) in its own environment. `docs/WAVE3.md` lists the command for every analysis and drafting script.

## Notes

- **Licence:** `ffnmem/memory/lngram_v2.py` is vendored from Lngram (MIT; see `ffnmem/memory/LICENSE.lngram`). Choose a licence for the rest before publishing.
- **Held-out set:** held-out drafting and interpretability analyses use 1,000 validation documents of ≥2,048 tokens. Training-side artefacts (sketches, controls, tuned lens) were built from text with every document overlapping that set removed.
