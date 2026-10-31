# Decodability ladder: sb

checkpoint `checkpoints/w3_sb_1b\checkpoints\step_0001907.pt` (step 1907), memory layer 1, 1,536,000 val positions; 95% bootstrap CIs over windows.

| level | decoder | top-1 agree | KL(p‖q) | Σ min(p,q) | acc (true next) |
|---|---|---|---|---|---|
| L0_sketch | count sketch keyed by the model's codes | 0.1743 [0.1725, 0.1760] | 7.5621 [7.5307, 7.5918] | 0.1747 [0.1734, 0.1760] | 0.1080 [0.1068, 0.1092] |
| L0_tok_ngram | control: sketch keyed by token bigram/trigram | 0.3410 [0.3379, 0.3437] | 6.1261 [6.0736, 6.1826] | 0.3063 [0.3037, 0.3086] | 0.2406 [0.2369, 0.2440] |
| L0_random_router | control: sketch keyed by a random router's codes | 0.1617 [0.1601, 0.1633] | 8.0941 [8.0615, 8.1253] | 0.1474 [0.1464, 0.1486] | 0.0991 [0.0979, 0.1002] |
| L1_zero_query | v2 readout, zero query (uniform rows + Sink), no conv | 0.0390 [0.0384, 0.0396] | 7.8075 [7.7818, 7.8335] | 0.0470 [0.0464, 0.0476] | 0.0241 [0.0236, 0.0247] |
| L1_zero_query_conv | as L1 + the readout's short conv | 0.0391 [0.0385, 0.0397] | 7.7782 [7.7524, 7.8042] | 0.0473 [0.0467, 0.0479] | 0.0242 [0.0236, 0.0247] |
| L2_draft | draft head (joint for SD, post-hoc otherwise) | 0.2956 [0.2932, 0.2978] | 2.5869 [2.5695, 2.6039] | 0.2980 [0.2962, 0.2998] | 0.1995 [0.1965, 0.2022] |
| L3_tuned_lens | tuned lens after the memory block | 0.3333 [0.3307, 0.3355] | 2.1610 [2.1461, 2.1764] | 0.3492 [0.3473, 0.3510] | 0.2248 [0.2218, 0.2277] |
| L4_model | the model itself | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 0.4642 [0.4592, 0.4690] |
