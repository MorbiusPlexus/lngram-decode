# Results

Every number below comes from a file under `results/`.

- **Held-out set:** 1,000 validation documents of ≥2,048 tokens. Long prompts are the first 1,536 tokens, open prompts the first 256, each followed by 128 greedy tokens. The analyses cover 1.536M positions.
- **Validation loss:** measured on a fixed 20M-token slice.
- **CIs:** 95% percentile bootstrap.

## 1. Language modelling (`results/training/*/train.jsonl`)

| Step (tokens) | SB run 1 | SB run 2 | SD-B | SD-C | D270 (2B schedule) |
|---|---|---|---|---|---|
| 400 (0.21B) | 3.6320 | 3.6329 | 3.6630 | 3.6284 | 3.6523 |
| 1000 (0.52B) | 2.9299 | 2.9282 | 2.9307 | 2.9234 | 2.9406 |
| 1400 (0.73B) | 2.7908 | 2.7894 | 2.7941 | 2.7881 | 2.8063 |
| 1800 (0.94B) | 2.7220 | 2.7199 | 2.7254 | 2.7204 | 2.7167 |
| **1907 (1.0B)** | **2.7127** | **2.7107** | **2.7159** | **2.7111** | — |
| 3815 (2.0B) | | | | | 2.5147 |

- **Replicate difference:** 0.0020.
- **Self-drafting cost:** SD-B − SB run 1 = +0.0032; SD-C − SB run 2 = +0.0004. Each SD model is compared with the SB run that used the same training configuration.
- **Stability:** no run had a non-finite loss. SD-B has a reproducible grad-norm spike at step 20 that recovers by step 30.

Parameters (non-embedding):

| | SB | SD-B | SD-C | D270 |
|---|---|---|---|---|
| Total | 369.78M | 369.93M | 369.88M | 269.80M |
| Routes | 184 | 182 | 181 | — |
| Active per token | 267.3M | 268.6M | 269.1M | 269.8M |

## 2. Downstream (`results/evaluation/lmeval/`)

| Model | HellaSwag | MMLU-cloze | SciQ | WinoGrande | PIQA | ARC-E |
|---|---|---|---|---|---|---|
| SB run 1 | 28.7 | 24.7 | 60.1 | 51.7 | 58.9 | 38.9 |
| SB run 2 | 29.3 | 24.6 | 60.4 | 51.8 | 60.0 | 38.0 |
| SD-B | 28.6 | 24.7 | 58.6 | 50.4 | 59.7 | 39.0 |
| SD-C | 28.6 | 24.3 | 60.3 | 51.6 | 58.3 | 40.0 |
| D270 (2B tokens) | 31.0 | 25.0 | 68.1 | 52.6 | 61.5 | 41.2 |

None of the SD − SB paired differences excludes 0. Full paired CIs are in `lmeval_table.md`.

## 3. Drafting

### 3.1 Decodability ladder (`results/evaluation/ladder/`)

Top-1 agreement with the target:

| Decoder | Agreement |
|---|---|
| Token n-gram counts (control) | 0.341 |
| Random-router code counts (control) | 0.162 |
| SB code counts | 0.174 |
| SB rows, zero query | 0.039 |
| SB post-hoc head | 0.296 |
| SB tuned lens after layer 1 | 0.333 |
| **SD-B joint head** | **0.410** |
| **SD-C joint head** | **0.449** |

### 3.2 Offline acceptance at γ=4 (`results/evaluation/acceptance/`)

Accepted tokens per verification step:

| Target | Drafter | Open | Long |
|---|---|---|---|
| SB | prompt lookup | 2.38 | 2.47 |
| SB | token n-gram | 0.84 | 0.75 |
| SB | post-hoc / + SSM | 0.42 / 0.55 | 0.39 / 0.51 |
| SB | sketch / + online | 0.21 / 2.28 | 0.21 / 2.07 |
| SD-B | prompt lookup | 2.46 | 2.54 |
| SD-B | head / + online | 0.81 / **2.59** | 0.73 / 2.46 |
| SD-C | prompt lookup | 2.34 | 2.46 |
| SD-C | head / + online | 1.04 / **2.52** | 0.94 / 2.19 |

- **Gates:** the SSM gives +32.3% \[29.4, 35.2\] acceptance for the post-hoc drafter on long text. The live gate passes: head + online beats the n-gram baseline by +1.72.
- **Paired vs prompt lookup** (head + online): open +0.133 (SD-B) and +0.183 (SD-C); long −0.082 and −0.267.

### 3.3 Hybrid drafter and acceptance along the continuation (`results/interp/accept.json`)

| Target, text | Prompt lookup | PLD → head | Δ (95% CI) | Oracle |
|---|---|---|---|---|
| SD-C open | 2.341 | 2.471 | +0.130 \[0.123, 0.138\] | 2.700 |
| SD-C long | 2.460 | 2.516 | +0.057 \[0.052, 0.061\] | 2.747 |
| SD-B open | 2.458 | 2.569 | +0.111 \[0.104, 0.118\] | 2.737 |
| SD-B long | 2.541 | 2.587 | +0.045 \[0.042, 0.049\] | 2.753 |

An n-gram fallback comes within 0.01 of the head.

In the first 16 continuation tokens (open text), prompt lookup accepts 0.65, the SD-C head 0.92 and the SD-C head + online 1.07. By tokens 96–124 those are 3.26, 1.07 and 3.34.

## 4. Serving and speed (`results/evaluation/serving/`, `results/evaluation/drafter/`)

- **Exactness:** B0 = B1 = B4 (identical tokens, max logit diff 0) in fp32 and fp16, for SD-C and SB.
- **C++ drafter vs reference:** max |Δlogit| 9.5e-6; argmax agreement 1.0.
- **Decode step**, 0.2 GB table, V100 fp16:

  | Batch | B0 | B1 | B4 |
  |---|---|---|---|
  | 1 | 25.3 ms | 25.7 ms | 25.7 ms |
  | 16 | 26.7 ms | 26.6 ms | 26.9 ms |

- **Drafter cost per token:** C++ 10.3 / 5.7 / 3.6 ms (1/2/4 threads); PyTorch reference 1.3 ms.
- **Speedup model** E[A+1]/(1+γc), SD-C head + online, PyTorch drafter (c = 0.059): 2.86× open, 2.59× long at γ=4. Prompt lookup with c ≈ 0: 3.34×, 3.46×.
- **Live loop** (head only, C++ drafter): 0.49× (γ=2) and 0.28× (γ=4) of greedy speed. fp32 output matches greedy 100%. fp16 matches 89–91%, and every mismatch is a near-tie.

## 5. Interpretability (`results/interp/`)

| Study | Result |
|---|---|
| Emergent classes (`semantics.json`) | SB route 137 separates prepositions/conjunctions, determiners, subjects, copulas, possessives and punctuation. A code carries 31% of token-class entropy in SB, 22% in SD-B, 21% in SD-C. |
| Utilization (`utilization.json`) | 100% of rows used. Effective rows: 77% (2-gram) and 61% (3-gram) for SB. The busiest 1% of rows take 5–9% of hits. |
| Redundancy (`redundancy.json`) | Mean pairwise NMI between routes: SB 0.231, SD-B 0.087, SD-C 0.085. |
| Identifiability (`replicate.json`) | The two SB runs assign the same code at the same position 36.8% of the time (chance 6.3%; bits 77%). |
| Memory benefit (`benefit.json`) | Zeroing memory costs SB 0.283 nats and SD-C 0.216. 72% of the gain comes from 10% of tokens. Gain rises with token frequency (0.15 → 0.34). Largest on whitespace/newline (0.73), digits (0.35), word continuations (0.32), and at position 1 (0.37). |
| Dense comparison by type (`benefit.json`) | D270 (2B) leads SB by 0.18 overall, but by only 0.035 on whitespace and 0.089 on punctuation, against 0.21 on word starts. |
| Draft agreement (`draftacc.json`) | SD-C agreement rises from 23% (target confidence < 0.2) to 77% (> 0.9). By the target's next token: whitespace 83%, punctuation 55%, digits 21%. The drafter's own confidence is informative and conservative (r = 0.50): a gate at 0.3 gives 35% coverage at 75% agreement; at 0.5, 18% coverage at 88%. |
| Self-repetition (`repetition.json`) | 59–62% of continuation tokens repeat an earlier 5-gram; ~49% are copies of the model's own output. This rises from 9–16% in the first 16 tokens to 87–92% in the last 32. |
