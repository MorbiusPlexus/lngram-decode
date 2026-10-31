# Decodability ladder: sdc

checkpoint `checkpoints/w3_sdc_1b\checkpoints\step_0001907.pt` (step 1907), memory layer 1, 1,536,000 val positions; 95% bootstrap CIs over windows.

| level | decoder | top-1 agree | KL(p‖q) | Σ min(p,q) | acc (true next) |
|---|---|---|---|---|---|
| L0_tok_ngram | control: sketch keyed by token bigram/trigram | 0.3412 [0.3381, 0.3440] | 6.1180 [6.0663, 6.1746] | 0.3066 [0.3040, 0.3090] | 0.2406 [0.2369, 0.2440] |
| L1_zero_query | v2 readout, zero query (uniform rows + Sink), no conv | 0.0428 [0.0419, 0.0436] | 6.4605 [6.4351, 6.4875] | 0.0608 [0.0598, 0.0617] | 0.0270 [0.0264, 0.0276] |
| L1_zero_query_conv | as L1 + the readout's short conv | 0.0440 [0.0431, 0.0449] | 6.3803 [6.3540, 6.4082] | 0.0623 [0.0613, 0.0633] | 0.0275 [0.0269, 0.0281] |
| L2_draft | draft head (joint for SD, post-hoc otherwise) | 0.4491 [0.4462, 0.4520] | 1.4591 [1.4475, 1.4706] | 0.4652 [0.4632, 0.4671] | 0.2977 [0.2933, 0.3016] |
| L4_model | the model itself | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 0.4645 [0.4595, 0.4693] |

Skipped (no input given): L0_sketch, L0_random_router, L3_tuned_lens
