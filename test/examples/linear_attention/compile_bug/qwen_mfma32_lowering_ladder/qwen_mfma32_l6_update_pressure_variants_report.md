# Qwen MFMA32 L6 Update-Pressure Variants Report

## Summary

This report isolates the L5->L6 transition by keeping the K staging source shape fixed and varying only update MFMA count/scope.

One complete Ktile update changes AccVGPR by `192.0000` versus no update.

`L6_one_update_mfma_only` AccVGPR is `320.0000`, useful for separating one intrinsic from a full Ktile branch set.

`L6_full_update_like_current` trace is `33.9710 us`.

## Variants

- `baseline_L5_no_update`
- `L6_one_update_mfma_only`
- `L6_one_ktile_update`
- `L6_two_ktile_update`
- `L6_full_update_like_current`
- `L6_update_acc_scope_split_source`
- `L6_update_acc_reinit_variant`

## Smoke/Checksum

| variant | latency_ms | finite | checksum |
|:---|---:|:---:|---:|
| baseline_L5_no_update | 0.053520 | True | 86047.7 |
| L6_one_update_mfma_only | 0.043003 | True | 3382.34 |
| L6_one_ktile_update | 0.044165 | True | 7556.48 |
| L6_two_ktile_update | 0.042864 | True | 16545.2 |
| L6_full_update_like_current | 0.046449 | True | 65279 |
| L6_update_acc_scope_split_source | 0.046229 | True | 65279 |
| L6_update_acc_reinit_variant | 0.045888 | True | 65279 |

## Rocprof Counters

| variant | trace_median_us | VGPR_Count | Accum_VGPR_Count | SGPR_Count | Scratch_Size | LDS_Block_Size | SQ_INSTS_MFMA | SQ_INSTS_VALU | SQ_INSTS_SALU | SQ_INSTS_VMEM | SQ_INSTS_LDS | OccupancyPercent |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_L5_no_update | 36.4540 | 120.0000 | 144.0000 | 112.0000 | 0 | 28672 | 1024 | 201920 | 8768 | 25600 | 19456 | 0.494572 |
| L6_one_update_mfma_only | 31.7270 | 128.0000 | 320.0000 | 112.0000 | 0 | 45056 | 1152 | 169728 | 2368 | 17920 | 20608 | 0.500302 |
| L6_one_ktile_update | 40.1400 | 128.0000 | 336.0000 | 112.0000 | 0 | 45056 | 1536 | 176512 | 3200 | 17920 | 21504 | 0.523019 |
| L6_two_ktile_update | 30.7660 | 128.0000 | 272.0000 | 112.0000 | 0 | 45056 | 2048 | 174592 | 4928 | 18432 | 22528 | 0.489922 |
| L6_full_update_like_current | 33.9710 | 128.0000 | 264.0000 | 112.0000 | 0 | 45056 | 5120 | 182016 | 11072 | 21504 | 28672 | 0.498407 |
| L6_update_acc_scope_split_source | 34.1310 | 128.0000 | 264.0000 | 112.0000 | 0 | 45056 | 5120 | 182016 | 11072 | 21504 | 28672 | 0.50868 |
| L6_update_acc_reinit_variant | 33.9300 | 128.0000 | 264.0000 | 112.0000 | 0 | 45056 | 5120 | 186112 | 11072 | 21504 | 28672 | 0.51148 |

## AccVGPR Trend

| transition | AccVGPR delta | trace delta us | MFMA delta |
|:---|---:|---:|---:|
| baseline_L5_no_update -> L6_one_update_mfma_only | 176.0000 | -4.7270 | 128.0000 |
| baseline_L5_no_update -> L6_one_ktile_update | 192.0000 | 3.6860 | 512.0000 |
| baseline_L5_no_update -> L6_two_ktile_update | 128.0000 | -5.6880 | 1024 |
| baseline_L5_no_update -> L6_full_update_like_current | 120.0000 | -2.4830 | 4096 |
| baseline_L5_no_update -> L6_update_acc_scope_split_source | 120.0000 | -2.3230 | 4096 |
| baseline_L5_no_update -> L6_update_acc_reinit_variant | 120.0000 | -2.5240 | 4096 |

## Static ISA Counts

| variant | v_mfma_f32_32x32x8_bf16 | v_mfma_f32_16x16x16_bf16 | global_load | global_store | buffer_load | buffer_store | ds_read | ds_write | s_barrier | s_waitcnt | v_add | v_add3 | v_lshl | v_lshl_add | v_or | v_bfe | max_vgpr_index_static_best_effort | max_acc_index_static_best_effort |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_L5_no_update | 16.0000 | 0 | 136.0000 | 24.0000 | 2.0000 | 2.0000 | 40.0000 | 144.0000 | 9.0000 | 145.0000 | 319.0000 | 126.0000 | 483.0000 | 263.0000 | 158.0000 | 112.0000 | 246.0000 | 15.0000 |
| L6_one_update_mfma_only | 16.0000 | 2.0000 | 256.0000 | 24.0000 | 2.0000 | 2.0000 | 50.0000 | 272.0000 | 11.0000 | 267.0000 | 284.0000 | 120.0000 | 804.0000 | 425.0000 | 212.0000 | 113.0000 | 255.0000 | 189.0000 |
| L6_one_ktile_update | 16.0000 | 8.0000 | 256.0000 | 24.0000 | 2.0000 | 2.0000 | 64.0000 | 272.0000 | 11.0000 | 280.0000 | 283.0000 | 120.0000 | 802.0000 | 425.0000 | 213.0000 | 113.0000 | 255.0000 | 203.0000 |
| L6_two_ktile_update | 8.0000 | 8.0000 | 144.0000 | 24.0000 | 2.0000 | 2.0000 | 40.0000 | 152.0000 | 6.0000 | 156.0000 | 188.0000 | 72.0000 | 722.0000 | 401.0000 | 166.0000 | 73.0000 | 255.0000 | 139.0000 |
| L6_full_update_like_current | 8.0000 | 32.0000 | 144.0000 | 48.0000 | 2.0000 | 2.0000 | 88.0000 | 152.0000 | 6.0000 | 179.0000 | 188.0000 | 72.0000 | 722.0000 | 401.0000 | 166.0000 | 73.0000 | 255.0000 | 130.0000 |
| L6_update_acc_scope_split_source | 8.0000 | 32.0000 | 144.0000 | 48.0000 | 2.0000 | 2.0000 | 88.0000 | 152.0000 | 14.0000 | 179.0000 | 188.0000 | 72.0000 | 722.0000 | 401.0000 | 166.0000 | 73.0000 | 255.0000 | 130.0000 |
| L6_update_acc_reinit_variant | 8.0000 | 32.0000 | 144.0000 | 48.0000 | 2.0000 | 2.0000 | 88.0000 | 152.0000 | 6.0000 | 179.0000 | 220.0000 | 72.0000 | 722.0000 | 401.0000 | 166.0000 | 73.0000 | 255.0000 | 130.0000 |

## Interpretation

The AccVGPR jump happens immediately:

- `baseline_L5_no_update`: AccVGPR `144`.
- `L6_one_update_mfma_only`: AccVGPR `320`.
- `L6_one_ktile_update`: AccVGPR `336`.

So the pressure is not linear in Ktile count.  One dependent update MFMA branch already creates most of the accumulator pressure.  More tiles actually settle lower in this repro (`272` for two Ktiles, `264` for full update-like current), likely because the compiler selects a different schedule/lifetime shape once the loop is larger.

Source-level workarounds tested here did not help:

- `L6_update_acc_scope_split_source`: AccVGPR remains `264`, trace slightly worse than full update-like current.
- `L6_update_acc_reinit_variant`: AccVGPR remains `264`, VALU increases from `182016` to `186112`.

Decision:

- L5->L6 AccVGPR growth looks like conservative/mixed MFMA32+MFMA16 lowering around the first dependent update, not something fixed by simple source barriers or accumulator reinit.
- The better next lever is reducing the K staging/source pressure before update, especially K-half/subtile staging from the L5 report.
