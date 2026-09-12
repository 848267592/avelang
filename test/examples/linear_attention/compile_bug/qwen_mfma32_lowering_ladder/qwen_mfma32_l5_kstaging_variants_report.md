# Qwen MFMA32 L5 K-Staging Variants Report

## Summary

Best measured variant: `L5_direct_global_k_update_probe` with trace `13.740 us` (delta vs baseline_L5 `-22.8740`).

These variants isolate the L4->L5 K staging/shared-view jump. They are diagnostic and do not change v23/v24/full Qwen.

## Variants

- `baseline_L5`: current ladder L5, transposed `k_all_t[128,BT]` plus `kall_vec`.
- `baseline_L6`: current ladder L6, same K staging plus one dependent update MFMA.
- `L5_token_major_no_kall_vec`: stage `[BT,128]`, avoid `kall_vec`.
- `L5_transposed_prepacked_input`: read prepacked `[head,k,token]` input, skip runtime transposed shared staging.
- `L5_subtile_k_stage_16token`: stage only a 16-token transposed subtile.
- `L5_khalf_stage_64`: stage only K half `[64,BT]`.
- `L5_direct_global_k_update_probe`: avoid full K shared staging; direct global diagnostic loads.
- `L5_packed_i32_contiguous_load`: token-major shared plus packed i32 view loads.
- `L5_alt_token_major_k_stage_existing`: preserved existing token-major workaround.

## Smoke/Checksum

| variant | latency_ms | finite | checksum |
|:---|---:|:---:|---:|
| baseline_L5 | 0.053980 | True | 43288.7 |
| baseline_L6 | 0.046710 | True | 16081.7 |
| L5_token_major_no_kall_vec | 0.047451 | True | 44692.3 |
| L5_transposed_prepacked_input | 0.046349 | True | 43288.7 |
| L5_subtile_k_stage_16token | 0.039278 | True | 43151.9 |
| L5_khalf_stage_64 | 0.040400 | True | 42681 |
| L5_direct_global_k_update_probe | 0.031187 | True | 41423.9 |
| L5_packed_i32_contiguous_load | 0.042543 | False | inf |
| L5_alt_token_major_k_stage_existing | 0.045809 | True | 44692.3 |

## Rocprof Counters

| variant | trace_median_us | VGPR_Count | Accum_VGPR_Count | SGPR_Count | Scratch_Size | LDS_Block_Size | SQ_INSTS_MFMA | SQ_INSTS_VALU | SQ_INSTS_SALU | SQ_INSTS_VMEM | SQ_INSTS_LDS | OccupancyPercent |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_L5 | 36.6140 | 120.0000 | 144.0000 | 112.0000 | 0 | 28672 | 1024 | 201728 | 8704 | 25088 | 19456 | 0.490776 |
| baseline_L6 | 40.3000 | 128.0000 | 336.0000 | 112.0000 | 0 | 45056 | 1536 | 176640 | 3136 | 17408 | 20992 | 0.520599 |
| L5_token_major_no_kall_vec | 37.6960 | 112.0000 | 152.0000 | 112.0000 | 0 | 28672 | 1024 | 191936 | 11072 | 25088 | 19456 | 0.509586 |
| L5_transposed_prepacked_input | 38.0560 | 112.0000 | 152.0000 | 112.0000 | 0 | 28672 | 1024 | 192512 | 8960 | 25088 | 19456 | 0.500571 |
| L5_subtile_k_stage_16token | 32.7690 | 124.0000 | 148.0000 | 112.0000 | 0 | 28672 | 1024 | 111552 | 2432 | 12800 | 11520 | 0.501983 |
| L5_khalf_stage_64 | 23.6350 | 112.0000 | 152.0000 | 112.0000 | 0 | 28672 | 1024 | 114304 | 1920 | 16896 | 14592 | 0.451776 |
| L5_direct_global_k_update_probe | 13.7400 | 96.0000 | 168.0000 | 112.0000 | 0 | 28672 | 1024 | 101824 | 2368 | 10752 | 11392 | 0.378753 |
| L5_packed_i32_contiguous_load | 40.8610 | 128.0000 | 288.0000 | 112.0000 | 0 | 45056 | 1024 | 151104 | 6528 | 18816 | 24320 | 0.512858 |
| L5_alt_token_major_k_stage_existing | 37.8160 | 112.0000 | 152.0000 | 112.0000 | 0 | 28672 | 1024 | 191936 | 11072 | 25088 | 19456 | 0.510594 |

## Static ISA Counts

| variant | v_mfma_f32_32x32x8_bf16 | v_mfma_f32_16x16x16_bf16 | global_load | global_store | buffer_load | buffer_store | ds_read | ds_write | s_barrier | s_waitcnt | v_add | v_add3 | v_lshl | v_lshl_add | v_or | v_bfe | max_vgpr_index_static_best_effort | max_acc_index_static_best_effort |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_L5 | 16.0000 | 0 | 136.0000 | 16.0000 | 2.0000 | 2.0000 | 40.0000 | 144.0000 | 9.0000 | 145.0000 | 317.0000 | 126.0000 | 482.0000 | 262.0000 | 158.0000 | 112.0000 | 246.0000 | 15.0000 |
| baseline_L6 | 16.0000 | 8.0000 | 256.0000 | 16.0000 | 2.0000 | 2.0000 | 56.0000 | 272.0000 | 11.0000 | 280.0000 | 288.0000 | 120.0000 | 803.0000 | 424.0000 | 212.0000 | 113.0000 | 255.0000 | 202.0000 |
| L5_token_major_no_kall_vec | 16.0000 | 0 | 136.0000 | 16.0000 | 2.0000 | 2.0000 | 40.0000 | 144.0000 | 9.0000 | 144.0000 | 315.0000 | 128.0000 | 458.0000 | 244.0000 | 158.0000 | 112.0000 | 239.0000 | 15.0000 |
| L5_transposed_prepacked_input | 16.0000 | 0 | 136.0000 | 16.0000 | 2.0000 | 2.0000 | 40.0000 | 144.0000 | 9.0000 | 145.0000 | 317.0000 | 128.0000 | 466.0000 | 247.0000 | 158.0000 | 112.0000 | 237.0000 | 15.0000 |
| L5_subtile_k_stage_16token | 8.0000 | 0 | 96.0000 | 24.0000 | 2.0000 | 2.0000 | 18.0000 | 88.0000 | 5.0000 | 96.0000 | 202.0000 | 72.0000 | 419.0000 | 257.0000 | 117.0000 | 72.0000 | 249.0000 | 15.0000 |
| L5_khalf_stage_64 | 16.0000 | 0 | 192.0000 | 72.0000 | 2.0000 | 2.0000 | 84.0000 | 144.0000 | 9.0000 | 244.0000 | 297.0000 | 112.0000 | 563.0000 | 247.0000 | 154.0000 | 112.0000 | 236.0000 | 15.0000 |
| L5_direct_global_k_update_probe | 8.0000 | 0 | 88.0000 | 16.0000 | 2.0000 | 2.0000 | 17.0000 | 88.0000 | 5.0000 | 95.0000 | 192.0000 | 72.0000 | 364.0000 | 219.0000 | 115.0000 | 72.0000 | 221.0000 | 15.0000 |
| L5_packed_i32_contiguous_load | 8.0000 | 0 | 144.0000 | 23.0000 | 2.0000 | 2.0000 | 54.0000 | 152.0000 | 6.0000 | 168.0000 | 275.0000 | 85.0000 | 554.0000 | 312.0000 | 134.0000 | 72.0000 | 255.0000 | 159.0000 |
| L5_alt_token_major_k_stage_existing | 16.0000 | 0 | 136.0000 | 16.0000 | 2.0000 | 2.0000 | 40.0000 | 144.0000 | 9.0000 | 144.0000 | 315.0000 | 128.0000 | 458.0000 | 244.0000 | 158.0000 | 112.0000 | 239.0000 | 15.0000 |

## Interpretation

The strongest result is diagnostic:

- `L5_direct_global_k_update_probe`: `13.740 us`, `2.665x` faster than `baseline_L5`.
- It cuts VALU from `201728` to `101824`, VMEM from `25088` to `10752`, and LDS from `19456` to `11392`.
- This confirms that the full transposed shared K staging/view pattern is the first-order L4->L5 cost.
- It is not a production replacement because it only probes direct global loads and does not implement the full update-friendly K dataflow.

The production-relevant staging reductions are:

- `L5_khalf_stage_64`: `23.635 us`, `1.549x` faster than `baseline_L5`.
- `L5_subtile_k_stage_16token`: `32.769 us`, `1.117x` faster than `baseline_L5`.

Both are finite and preserve the same high-level pred/v_decay diagnostic path, but they are still L5 diagnostics rather than a full correctness-equivalent update kernel.  They show the likely source-helper direction: stage less K at a time, especially K-half or smaller update tiles, instead of materializing full `k_all_t[128,BT]`.

Negative/weak results:

- `L5_token_major_no_kall_vec` and `L5_alt_token_major_k_stage_existing` reduce some address-generation counters but do not improve trace.
- `L5_transposed_prepacked_input` does not improve trace; prepacking alone does not remove enough in-kernel cost.
- `L5_packed_i32_contiguous_load` is invalid as a candidate because smoke output was non-finite and AccVGPR/LDS worsened.

## Decision

The isolated ladder does identify a promising source direction, but **do not return to full Qwen yet** from this data alone.

Reason:

- The fastest result is explicitly diagnostic.
- The viable finite wins (`K-half`, `16-token subtile`) are not yet full update correctness-equivalent.
- The next safe step is a focused L6-style variant that combines `K-half`/subtile staging with actual update MFMA and verifies the update sink/correctness trend before touching full Qwen.
