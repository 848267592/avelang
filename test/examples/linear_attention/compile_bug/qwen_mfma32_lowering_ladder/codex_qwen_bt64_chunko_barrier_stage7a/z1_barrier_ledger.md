# Stage 7A Z1 Barrier Provenance Ledger

| ISA | PC | source | execution | phase | hazard | classification | LLVM |
|---:|:---|:---|:---|:---|:---|:---|---:|
| 1 | `0x000000002124` | A.stage_qh:90 | A k_stage=0 | A inter Q/H K32 stage | CTA Q/H LDS producer -> MFMA LDS consumer (RAW) | required for this shared staging schedule | 270 |
| 2 | `0x000000002160` | A.pack_frag:96 | A k_stage=0, kt=0 | A per-lane frag_words pack | frag_words[tid] write -> same-lane load | candidate source-scheduling barrier | 278 |
| 3 | `0x000000002178` | A.reuse_frag:103 | A k_stage=0, kt=0 | A next packed fragment | same-lane frag_words reuse after MFMA | candidate source-scheduling barrier | 293 |
| 4 | `0x0000000021a8` | A.pack_frag:96 | A k_stage=0, kt=1 | A per-lane frag_words pack | frag_words[tid] write -> same-lane load | candidate source-scheduling barrier | 303 |
| 5 | `0x0000000021c0` | A.reuse_frag:103 | A k_stage=0, kt=1 | A next packed fragment | same-lane frag_words reuse after MFMA | candidate source-scheduling barrier | 318 |
| 6 | `0x00000000266c` | A.stage_qh:90 | A k_stage=1 | A inter Q/H K32 stage | CTA Q/H LDS producer -> MFMA LDS consumer (RAW) | required for this shared staging schedule | 409 |
| 7 | `0x00000000269c` | A.pack_frag:96 | A k_stage=1, kt=0 | A per-lane frag_words pack | frag_words[tid] write -> same-lane load | candidate source-scheduling barrier | 416 |
| 8 | `0x0000000026b4` | A.reuse_frag:103 | A k_stage=1, kt=0 | A next packed fragment | same-lane frag_words reuse after MFMA | candidate source-scheduling barrier | 431 |
| 9 | `0x0000000026e4` | A.pack_frag:96 | A k_stage=1, kt=1 | A per-lane frag_words pack | frag_words[tid] write -> same-lane load | candidate source-scheduling barrier | 438 |
| 10 | `0x0000000026fc` | A.reuse_frag:103 | A k_stage=1, kt=1 | A next packed fragment | same-lane frag_words reuse after MFMA | candidate source-scheduling barrier | 453 |
| 11 | `0x000000002ba8` | A.stage_qh:90 | A k_stage=2 | A inter Q/H K32 stage | CTA Q/H LDS producer -> MFMA LDS consumer (RAW) | required for this shared staging schedule | 544 |
| 12 | `0x000000002bd8` | A.pack_frag:96 | A k_stage=2, kt=0 | A per-lane frag_words pack | frag_words[tid] write -> same-lane load | candidate source-scheduling barrier | 551 |
| 13 | `0x000000002bf0` | A.reuse_frag:103 | A k_stage=2, kt=0 | A next packed fragment | same-lane frag_words reuse after MFMA | candidate source-scheduling barrier | 566 |
| 14 | `0x000000002c20` | A.pack_frag:96 | A k_stage=2, kt=1 | A per-lane frag_words pack | frag_words[tid] write -> same-lane load | candidate source-scheduling barrier | 573 |
| 15 | `0x000000002c38` | A.reuse_frag:103 | A k_stage=2, kt=1 | A next packed fragment | same-lane frag_words reuse after MFMA | candidate source-scheduling barrier | 588 |
| 16 | `0x000000003174` | A.stage_qh:90 | A k_stage=3 | A inter Q/H K32 stage | CTA Q/H LDS producer -> MFMA LDS consumer (RAW) | required for this shared staging schedule | 679 |
| 17 | `0x0000000031a4` | A.pack_frag:96 | A k_stage=3, kt=0 | A per-lane frag_words pack | frag_words[tid] write -> same-lane load | candidate source-scheduling barrier | 686 |
| 18 | `0x0000000031bc` | A.reuse_frag:103 | A k_stage=3, kt=0 | A next packed fragment | same-lane frag_words reuse after MFMA | candidate source-scheduling barrier | 701 |
| 19 | `0x000000003268` | A.pack_frag:96 | A k_stage=3, kt=1 | A per-lane frag_words pack | frag_words[tid] write -> same-lane load | candidate source-scheduling barrier | 708 |
| 20 | `0x000000003348` | A.reuse_frag:103 | A k_stage=3, kt=1 | A next packed fragment | same-lane frag_words reuse after MFMA | candidate source-scheduling barrier | 718 |
| 21 | `0x000000003364` | B.stage_qk:125 | B source_half=0, dynamic k_stage body x4 | B score Q/K K32 stage | CTA Q/K LDS producer -> owner-wave MFMA consumer (RAW) | required for this shared staging schedule | 881 |
| 22 | `0x0000000038dc` | B.pack_frag:131 | B source_half=0, dynamic k_stage body x4, kt=0 | B per-lane frag_words pack | frag_words[tid] write -> same-lane load | candidate source-scheduling barrier | 888 |
| 23 | `0x00000000398c` | B.reuse_frag:139 | B source_half=0, dynamic k_stage body x4, kt=0 | B next packed fragment | same-lane frag_words reuse after MFMA | candidate source-scheduling barrier | 910 |
| 24 | `0x000000003a44` | B.pack_frag:131 | B source_half=0, dynamic k_stage body x4, kt=1 | B per-lane frag_words pack | frag_words[tid] write -> same-lane load | candidate source-scheduling barrier | 917 |
| 25 | `0x000000003a74` | B.reuse_frag:139 | B source_half=0, dynamic k_stage body x4, kt=1 | B next packed fragment | same-lane frag_words reuse after MFMA | candidate source-scheduling barrier | 939 |
| 26 | `0x000000004d70` | B.serialize_score:153 | B source_half=0 serialize score | B score half store | score store -> later score/V consume; next stage is physically disjoint | candidate phase-merge barrier | 1304 |
| 27 | `0x000000004d8c` | B.stage_qk:125 | B source_half=1, dynamic k_stage body x4 | B score Q/K K32 stage | CTA Q/K LDS producer -> owner-wave MFMA consumer (RAW) | required for this shared staging schedule | 1509 |
| 28 | `0x000000005300` | B.pack_frag:131 | B source_half=1, dynamic k_stage body x4, kt=0 | B per-lane frag_words pack | frag_words[tid] write -> same-lane load | candidate source-scheduling barrier | 1516 |
| 29 | `0x0000000053b0` | B.reuse_frag:139 | B source_half=1, dynamic k_stage body x4, kt=0 | B next packed fragment | same-lane frag_words reuse after MFMA | candidate source-scheduling barrier | 1538 |
| 30 | `0x000000005468` | B.pack_frag:131 | B source_half=1, dynamic k_stage body x4, kt=1 | B per-lane frag_words pack | frag_words[tid] write -> same-lane load | candidate source-scheduling barrier | 1545 |
| 31 | `0x000000005498` | B.reuse_frag:139 | B source_half=1, dynamic k_stage body x4, kt=1 | B next packed fragment | same-lane frag_words reuse after MFMA | candidate source-scheduling barrier | 1567 |
| 32 | `0x000000006528` | B.serialize_score:153 | B source_half=1 serialize score | B score half store | score store -> later score/V consume; next stage is physically disjoint | candidate phase-merge barrier | 1984 |
| 33 | `0x000000006940` | C.stage_v:164 | C V-new transpose | C V-new transpose | CTA V-new LDS producer + earlier score store -> intra MFMA consumer (RAW) | required for this shared staging schedule | 2342 |
| 34 | `0x000000006978` | C.pack_frag:172 | C source_half=0, kt=0 | C score/V per-lane frag_words pack | frag_words[tid] write -> same-lane load | candidate source-scheduling barrier | 2361 |
| 35 | `0x000000006998` | C.reuse_frag:179 | C source_half=0, kt=0 | C next score/V packed fragment | same-lane frag_words reuse after MFMA | candidate source-scheduling barrier | 2376 |
| 36 | `0x0000000069d0` | C.pack_frag:172 | C source_half=0, kt=1 | C score/V per-lane frag_words pack | frag_words[tid] write -> same-lane load | candidate source-scheduling barrier | 2385 |
| 37 | `0x0000000069e8` | C.reuse_frag:179 | C source_half=0, kt=1 | C next score/V packed fragment | same-lane frag_words reuse after MFMA | candidate source-scheduling barrier | 2400 |
| 38 | `0x000000006a28` | C.pack_frag:172 | C source_half=1, kt=0 | C score/V per-lane frag_words pack | frag_words[tid] write -> same-lane load | candidate source-scheduling barrier | 2415 |
| 39 | `0x000000006a40` | C.reuse_frag:179 | C source_half=1, kt=0 | C next score/V packed fragment | same-lane frag_words reuse after MFMA | candidate source-scheduling barrier | 2430 |
| 40 | `0x000000006a78` | C.pack_frag:172 | C source_half=1, kt=1 | C score/V per-lane frag_words pack | frag_words[tid] write -> same-lane load | candidate source-scheduling barrier | 2439 |
| 41 | `0x000000006a98` | C.reuse_frag:179 | C source_half=1, kt=1 | C next score/V packed fragment | same-lane frag_words reuse after MFMA | candidate source-scheduling barrier | 2454 |
