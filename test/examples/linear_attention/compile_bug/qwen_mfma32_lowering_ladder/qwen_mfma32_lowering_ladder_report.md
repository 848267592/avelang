# Qwen MFMA32 Lowering Ladder Report

## Purpose

This ladder follows the negative `mfma_region_lifetime_v3` result: independent sequential MFMA regions do not additively accumulate AccVGPR. The goal here is Qwen-shaped delta debugging for v28/v29-like MFMA32 pred/update source patterns.

## Summary Conclusion

Biggest weighted transition: `L4 -> L5` (k staging/shared view cost).

The data below should be read as lowering/resource attribution, not full Qwen correctness. Each level uses fixed Qwen-like BT64/BV32/Hv8/K128 tensors and deterministic inputs.

Manual interpretation:

- The first large runtime/source-size jump is `L4 -> L5`, adding K staging for update: trace `+24.196 us`, VGPR `+64`, VALU `+82752`, SALU `+5120`, VMEM `+15360`, LDS inst `+8768`.
- The largest AccVGPR jump is the next transition, `L5 -> L6`, adding the first dependent update MFMA: AccVGPR `+192`, LDS block `+16384`, MFMA `+512`.
- `L8 -> L9` h/vn global materialization is measurable but smaller: trace `+6.49 us`, AccVGPR `+16`, VMEM `+4096`.
- The two tried source workarounds did **not** improve latency: token-major K staging reduced VALU/VGPR but increased trace by `+5.168 us`; grouped-v4 materialization increased trace by `+2.925 us`.

Conclusion: the ladder finds no single small source tweak that rescues v29. The issue is mixed: Qwen-shaped staging/update dataflow is large, and the `L4 -> L5`/`L5 -> L6` patterns are the best places to inspect lowering helpers, but current evidence does not justify calling this a narrow compiler bug.

## Ladder Variants

- `L0_pred32_mfma_only`
- `L1_pred32_unpack_to_pred_partial`
- `L2_pred32_unpack_reduce`
- `L3_pred32_ucorr_epilogue_no_store`
- `L4_pred32_vdecay_shared_stage`
- `L5_pred32_vdecay_plus_k_stage`
- `L5_alt_token_major_k_stage`
- `L6_pred32_vdecay_update16_one_ktile`
- `L7_pred32_vdecay_update16_full_k_no_feedback`
- `L8_pred32_update16_full_k_state_writeback`
- `L9_pred32_update16_with_h_vn_global_materialization`
- `L9_alt_grouped_v4_materialization`

## Smoke/Checksum

| level | latency_ms | finite | checksum |
|:---|---:|:---:|---:|
| L5_alt_token_major_k_stage | 0.069543 | True | 22279 |
| L0_pred32_mfma_only | 0.046589 | True | 434.674 |
| L1_pred32_unpack_to_pred_partial | 0.045628 | True | 485.339 |
| L2_pred32_unpack_reduce | 0.045868 | True | 5209.53 |
| L3_pred32_ucorr_epilogue_no_store | 0.047771 | True | 8355.67 |
| L4_pred32_vdecay_shared_stage | 0.047851 | True | 8391.76 |
| L5_pred32_vdecay_plus_k_stage | 0.073629 | True | 21529 |
| L6_pred32_vdecay_update16_one_ktile | 0.063834 | True | 7556.48 |
| L7_pred32_vdecay_update16_full_k_no_feedback | 0.065698 | True | 65279 |
| L8_pred32_update16_full_k_state_writeback | 0.078717 | True | 0 |
| L9_pred32_update16_with_h_vn_global_materialization | 0.084947 | True | 0 |
| L9_alt_grouped_v4_materialization | 0.088592 | True | 0 |

## Rocprof Resource Table

| level | trace_median_us | VGPR_Count | Accum_VGPR_Count | SGPR_Count | Scratch_Size | LDS_Block_Size | SQ_INSTS_MFMA | SQ_INSTS_VALU | SQ_INSTS_SALU | SQ_INSTS_VMEM | SQ_INSTS_LDS | OccupancyPercent |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0_pred32_mfma_only | 9.3340 | 96.0000 | 144.0000 | 112.0000 | 0 | 16384 | 1024 | 90048 | 1536 | 6528 | 7168 | 0.297547 |
| L1_pred32_unpack_to_pred_partial | 9.7750 | 92.0000 | 172.0000 | 112.0000 | 0 | 24576 | 1024 | 86208 | 1536 | 6528 | 9344 | 0.320898 |
| L2_pred32_unpack_reduce | 9.7340 | 28.0000 | 236.0000 | 112.0000 | 0 | 24576 | 1024 | 80960 | 1344 | 6528 | 9344 | 0.321915 |
| L3_pred32_ucorr_epilogue_no_store | 13.8610 | 64.0000 | 200.0000 | 112.0000 | 0 | 24576 | 1024 | 92992 | 2048 | 8448 | 10240 | 0.37061 |
| L4_pred32_vdecay_shared_stage | 14.1010 | 64.0000 | 200.0000 | 112.0000 | 0 | 24576 | 1024 | 101184 | 2240 | 9472 | 10240 | 0.37143 |
| L5_pred32_vdecay_plus_k_stage | 38.2970 | 128.0000 | 144.0000 | 112.0000 | 0 | 28672 | 1024 | 183936 | 7360 | 24832 | 19008 | 0.47898 |
| L5_alt_token_major_k_stage | 43.4650 | 104.0000 | 160.0000 | 112.0000 | 0 | 28672 | 1024 | 168384 | 11584 | 26880 | 19456 | 0.520545 |
| L6_pred32_vdecay_update16_one_ktile | 39.8990 | 128.0000 | 336.0000 | 112.0000 | 0 | 45056 | 1536 | 176064 | 3136 | 17152 | 21504 | 0.523166 |
| L7_pred32_vdecay_update16_full_k_no_feedback | 33.6500 | 128.0000 | 264.0000 | 112.0000 | 0 | 45056 | 5120 | 181824 | 11008 | 20736 | 28672 | 0.503087 |
| L8_pred32_update16_full_k_state_writeback | 46.2280 | 128.0000 | 272.0000 | 112.0000 | 0 | 45056 | 5120 | 204992 | 14080 | 28928 | 28544 | 0.535291 |
| L9_pred32_update16_with_h_vn_global_materialization | 52.7180 | 128.0000 | 288.0000 | 112.0000 | 0 | 45056 | 5120 | 216832 | 14336 | 33024 | 30464 | 0.545922 |
| L9_alt_grouped_v4_materialization | 55.6430 | 128.0000 | 296.0000 | 112.0000 | 0 | 45056 | 5120 | 223360 | 14336 | 35072 | 31232 | 0.552216 |

## Static ISA Table

| level | v_mfma_f32_32x32x8_bf16 | v_mfma_f32_16x16x16_bf16 | global_load | global_store | buffer_load | buffer_store | ds_read | ds_write | s_barrier | s_waitcnt | v_add | v_add3 | v_lshl | v_lshl_add | v_or | v_bfe | max_vgpr_index_static_best_effort | max_acc_index_static_best_effort |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0_pred32_mfma_only | 16.0000 | 0 | 96.0000 | 6.0000 | 2.0000 | 2.0000 | 16.0000 | 96.0000 | 5.0000 | 76.0000 | 232.0000 | 96.0000 | 357.0000 | 192.0000 | 128.0000 | 96.0000 | 220.0000 | 15.0000 |
| L1_pred32_unpack_to_pred_partial | 16.0000 | 0 | 96.0000 | 6.0000 | 2.0000 | 2.0000 | 18.0000 | 128.0000 | 7.0000 | 80.0000 | 256.0000 | 96.0000 | 358.0000 | 192.0000 | 131.0000 | 96.0000 | 218.0000 | 15.0000 |
| L2_pred32_unpack_reduce | 16.0000 | 0 | 96.0000 | 6.0000 | 2.0000 | 2.0000 | 18.0000 | 128.0000 | 7.0000 | 75.0000 | 210.0000 | 96.0000 | 325.0000 | 176.0000 | 128.0000 | 96.0000 | 153.0000 | 15.0000 |
| L3_pred32_ucorr_epilogue_no_store | 8.0000 | 0 | 72.0000 | 12.0000 | 2.0000 | 2.0000 | 16.0000 | 80.0000 | 4.0000 | 75.0000 | 191.0000 | 64.0000 | 319.0000 | 193.0000 | 107.0000 | 64.0000 | 188.0000 | 15.0000 |
| L4_pred32_vdecay_shared_stage | 8.0000 | 0 | 80.0000 | 12.0000 | 2.0000 | 2.0000 | 16.0000 | 80.0000 | 4.0000 | 83.0000 | 199.0000 | 72.0000 | 327.0000 | 201.0000 | 115.0000 | 72.0000 | 188.0000 | 15.0000 |
| L5_pred32_vdecay_plus_k_stage | 16.0000 | 0 | 140.0000 | 16.0000 | 2.0000 | 2.0000 | 44.0000 | 144.0000 | 9.0000 | 145.0000 | 321.0000 | 129.0000 | 507.0000 | 277.0000 | 155.0000 | 112.0000 | 252.0000 | 15.0000 |
| L5_alt_token_major_k_stage | 8.0000 | 0 | 92.0000 | 16.0000 | 2.0000 | 2.0000 | 20.0000 | 88.0000 | 5.0000 | 96.0000 | 196.0000 | 76.0000 | 362.0000 | 221.0000 | 120.0000 | 72.0000 | 229.0000 | 15.0000 |
| L6_pred32_vdecay_update16_one_ktile | 16.0000 | 8.0000 | 256.0000 | 12.0000 | 2.0000 | 2.0000 | 64.0000 | 272.0000 | 11.0000 | 280.0000 | 281.0000 | 120.0000 | 801.0000 | 424.0000 | 213.0000 | 113.0000 | 255.0000 | 201.0000 |
| L7_pred32_vdecay_update16_full_k_no_feedback | 8.0000 | 32.0000 | 144.0000 | 36.0000 | 2.0000 | 2.0000 | 88.0000 | 152.0000 | 6.0000 | 179.0000 | 186.0000 | 72.0000 | 721.0000 | 400.0000 | 166.0000 | 73.0000 | 255.0000 | 130.0000 |
| L8_pred32_update16_full_k_state_writeback | 8.0000 | 32.0000 | 180.0000 | 40.0000 | 2.0000 | 2.0000 | 87.0000 | 152.0000 | 6.0000 | 217.0000 | 340.0000 | 104.0000 | 727.0000 | 412.0000 | 138.0000 | 73.0000 | 255.0000 | 143.0000 |
| L9_pred32_update16_with_h_vn_global_materialization | 8.0000 | 32.0000 | 196.0000 | 56.0000 | 2.0000 | 2.0000 | 102.0000 | 152.0000 | 6.0000 | 239.0000 | 363.0000 | 104.0000 | 759.0000 | 433.0000 | 140.0000 | 72.0000 | 255.0000 | 155.0000 |
| L9_alt_grouped_v4_materialization | 8.0000 | 32.0000 | 204.0000 | 64.0000 | 2.0000 | 2.0000 | 108.0000 | 152.0000 | 6.0000 | 250.0000 | 377.0000 | 106.0000 | 775.0000 | 441.0000 | 142.0000 | 72.0000 | 255.0000 | 164.0000 |

## Delta Table

| transition | purpose | trace_median_us | VGPR_Count | Accum_VGPR_Count | SQ_INSTS_VALU | SQ_INSTS_SALU | SQ_INSTS_VMEM | SQ_INSTS_LDS | LDS_Block_Size |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 -> L1 | accumulator unpack cost | 0.441 | -4.0000 | 28.0000 | -3840 | 0 | 0 | 2176 | 8192 |
| L1 -> L2 | pred_partial LDS read/reduction cost | -0.041 | -64.0000 | 64.0000 | -5248 | -192.0000 | 0 | 0 | 0 |
| L2 -> L3 | u load/sub cost | 4.1270 | 36.0000 | -36.0000 | 12032 | 704.0000 | 1920 | 896.0000 | 0 |
| L3 -> L4 | v_decay shared staging cost | 0.24 | 0 | 0 | 8192 | 192.0000 | 1024 | 0 | 0 |
| L4 -> L5 | k staging/shared view cost | 24.1960 | 64.0000 | -56.0000 | 82752 | 5120 | 15360 | 8768 | 4096 |
| L5 -> L6 | first dependent update MFMA cost | 1.6020 | 0 | 192.0000 | -7872 | -4224 | -7680 | 2496 | 16384 |
| L6 -> L7 | full K update cost | -6.2490 | 0 | -72.0000 | 5760 | 7872 | 3584 | 7168 | 0 |
| L7 -> L8 | state writeback cost | 12.5780 | 0 | 8.0000 | 23168 | 3072 | 8192 | -128.0000 | 0 |
| L8 -> L9 | h/vn global materialization cost | 6.4900 | 0 | 16.0000 | 11840 | 256.0000 | 4096 | 1920 | 0 |

## Disproportionate Jump Analysis

The first/strongest weighted jump is `L4 -> L5`: k staging/shared view cost.

Classification: **Mixed: source schedule growth plus possible shared-layout/index lowering cost**.

## Workaround Result

Primary workaround candidate for the L4->L5 jump: `L5_alt_token_major_k_stage`, staging K as token-major `[BT,K]` instead of update-oriented transposed `[K,BT]`.

| metric | L5 transposed | L5_alt token-major | delta |
|:---|---:|---:|---:|
| trace_median_us | 38.2970 | 43.4650 | 5.1680 |
| VGPR_Count | 128.0000 | 104.0000 | -24.0000 |
| Accum_VGPR_Count | 144.0000 | 160.0000 | 16.0000 |
| SGPR_Count | 112.0000 | 112.0000 | 0 |
| Scratch_Size | 0 | 0 | 0 |
| LDS_Block_Size | 28672 | 28672 | 0 |
| SQ_INSTS_MFMA | 1024 | 1024 | 0 |
| SQ_INSTS_VALU | 183936 | 168384 | -15552 |
| SQ_INSTS_SALU | 7360 | 11584 | 4224 |
| SQ_INSTS_VMEM | 24832 | 26880 | 2048 |
| SQ_INSTS_LDS | 19008 | 19456 | 448.0000 |
| OccupancyPercent | 0.47898 | 0.520545 | 0.0415649 |

Secondary workaround candidate: `L9_alt_grouped_v4_materialization`, using flat grouped-v4 VN stores instead of scalar 4D tensor indexing.

| metric | L9 | L9_alt | delta |
|:---|---:|---:|---:|
| trace_median_us | 52.7180 | 55.6430 | 2.9250 |
| VGPR_Count | 128.0000 | 128.0000 | 0 |
| Accum_VGPR_Count | 288.0000 | 296.0000 | 8.0000 |
| SGPR_Count | 112.0000 | 112.0000 | 0 |
| Scratch_Size | 0 | 0 | 0 |
| LDS_Block_Size | 45056 | 45056 | 0 |
| SQ_INSTS_MFMA | 5120 | 5120 | 0 |
| SQ_INSTS_VALU | 216832 | 223360 | 6528 |
| SQ_INSTS_SALU | 14336 | 14336 | 0 |
| SQ_INSTS_VMEM | 33024 | 35072 | 2048 |
| SQ_INSTS_LDS | 30464 | 31232 | 768.0000 |
| OccupancyPercent | 0.545922 | 0.552216 | 0.00629395 |

## Targeted Compiler/Lowering Diagnosis

This ladder avoids broad compiler grep. Based on the final counters:

- First inspect the K staging/update-shared layout around `k_all_t[128,BT]`, `kall_vec`, and transposed shared indexing. This is the `L4 -> L5` jump.
- Then inspect the first dependent update MFMA pattern. `L5 -> L6` increases AccVGPR by `+192`, even though trace only increases by `+1.602 us`.
- Do not prioritize global materialization first: `L8 -> L9` is real but smaller, and grouped-v4 stores regressed.
- Do not prioritize accumulator unpack first: `L0 -> L1` only adds `+0.441 us` and `+28` AccVGPR; `L1 -> L2` adds `+64` AccVGPR but no trace jump.

Potential helper-sized investigations:

1. A dedicated shared staging helper/layout for update K tiles that avoids expensive transposed index/address generation without triggering the token-major regression.
2. A source or lowering pattern for dependent update MFMA that limits AccVGPR growth when MFMA32 pred partials feed BF16 shared values consumed by MFMA16 update.
3. Only after those, revisit vectorized global materialization.

## Exact Commands

```bash
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_qwen_mfma32_lowering_ladder.py --level all --warmup 5 --repeat 20
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/profile_qwen_mfma32_lowering_ladder.py --warmup 5 --repeat 20 --rocprof-warmup 2 --rocprof-repeat 5
```
