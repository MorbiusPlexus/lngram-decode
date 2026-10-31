# Decodability ladder: sdb

checkpoint `checkpoints/w3_sdb_1b\checkpoints\step_0001907.pt` (step 1907), memory layer 1, 1,536,000 val positions; 95% bootstrap CIs over windows.

| level | decoder | top-1 agree | KL(p‖q) | Σ min(p,q) | acc (true next) |
|---|---|---|---|---|---|
| L0_tok_ngram | control: sketch keyed by token bigram/trigram | 0.3410 [0.3378, 0.3439] | 6.1183 [6.0657, 6.1757] | 0.3064 [0.3038, 0.3088] | 0.2406 [0.2369, 0.2440] |
| L1_zero_query | v2 readout, zero query (uniform rows + Sink), no conv | 0.0514 [0.0506, 0.0523] | 6.1553 [6.1319, 6.1780] | 0.0681 [0.0671, 0.0691] | 0.0343 [0.0337, 0.0348] |
| L1_zero_query_conv | as L1 + the readout's short conv | 0.0530 [0.0522, 0.0539] | 6.0406 [6.0170, 6.0637] | 0.0699 [0.0690, 0.0709] | 0.0351 [0.0345, 0.0357] |
| L2_draft | draft head (joint for SD, post-hoc otherwise) | 0.4099 [0.4070, 0.4126] | 1.7148 [1.7024, 1.7275] | 0.4208 [0.4191, 0.4225] | 0.2780 [0.2741, 0.2817] |
| L4_model | the model itself | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 0.4641 [0.4591, 0.4688] |

Skipped (no input given): L0_sketch, L0_random_router, L3_tuned_lens
