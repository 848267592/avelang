# Qwen MFMA32 L6 K-Staging Update Variants Report

## Summary

This report combines the promising L5 K-staging alternatives with the real MFMA16 update path.

- K-half full-update delta vs current: trace `-7.1710 us`, AccVGPR `-76.0000`, LDS inst `0`.

- Subtile full-update delta vs current: trace `-15.1820 us`, AccVGPR `-96.0000`.

- Minimal/direct K-tile diagnostic trace: `14.3810 us`, AccVGPR `152.0000`.

- Synthetic no-pred-dependency full update delta vs current: trace `-24.1960 us`, AccVGPR `-184.0000`.

- Minimal update fragment lower bound: trace `3.1640 us`, AccVGPR `4.0000`.


## Variants

- `L6_baseline_current_update`: current full transposed K staging plus current update-MFMA pattern.
- `L6_khalf_stage64_*`: L5 K-half staging combined with one-update, one-Ktile, and full-update-like paths.
- `L6_subtile16_stage_*`: L5 16-token subtile K staging combined with update paths.
- `L6_no_shared_k_direct_global_update_probe`: diagnostic minimal K-tile staging for exact update fragments.
- `L6_update_mfma_no_pred_dependency`: synthetic v_decay, full K staging/update, no pred/v_decay live dependency.
- `L6_update_mfma_minimal_frag`: minimal standalone MFMA16 update fragment lower bound.

## Smoke/Finite Checks

| variant | latency_ms | finite | checksum | conclusion |
|:---|---:|:---:|---:|:---|
| L6_baseline_current_update | 0.04727 | True | 129497 | valid |
| L6_khalf_stage64_one_update | 0.034932 | True | 7126 | valid |
| L6_khalf_stage64_one_ktile_update | 0.0355525 | True | 16082 | valid |
| L6_khalf_stage64_full_update_like | 0.043645 | True | 129497 | valid |
| L6_subtile16_stage_one_update | 0.0312665 | True | 7126 | valid |
| L6_subtile16_stage_one_ktile_update | 0.0309055 | True | 16082 | valid |
| L6_subtile16_stage_full_update_like | 0.0349715 | True | 129497 | valid |
| L6_no_shared_k_direct_global_update_probe | 0.0299445 | True | 16082 | valid |
| L6_update_mfma_no_pred_dependency | 0.027721 | True | 2732 | valid |
| L6_update_mfma_minimal_frag | 0.026319 | True | 671.8628 | valid |

## Rocprof Counters

| variant | trace_median_us | VGPR_Count | Accum_VGPR_Count | SGPR_Count | Scratch_Size | LDS_Block_Size | SQ_INSTS_MFMA | SQ_INSTS_VALU | SQ_INSTS_SALU | SQ_INSTS_VMEM | SQ_INSTS_LDS | OccupancyPercent |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L6_baseline_current_update | 34.3710 | 128.0000 | 264.0000 | 112.0000 | 0 | 45056 | 5120 | 182144 | 11200 | 22528 | 28672 | 0.503387 |
| L6_khalf_stage64_one_update | 20.4310 | 128.0000 | 224.0000 | 112.0000 | 0 | 36864 | 1152 | 120960 | 2496 | 14848 | 16512 | 0.426671 |
| L6_khalf_stage64_one_ktile_update | 22.5130 | 128.0000 | 232.0000 | 112.0000 | 0 | 36864 | 1536 | 122816 | 3328 | 14848 | 17408 | 0.441095 |
| L6_khalf_stage64_full_update_like | 27.2000 | 76.0000 | 188.0000 | 112.0000 | 0 | 36864 | 5120 | 244992 | 28800 | 22528 | 28672 | 0.474944 |
| L6_subtile16_stage_one_update | 15.3430 | 100.0000 | 164.0000 | 112.0000 | 0 | 32768 | 1152 | 108288 | 3200 | 12800 | 13568 | 0.38593 |
| L6_subtile16_stage_one_ktile_update | 15.6630 | 96.0000 | 168.0000 | 112.0000 | 0 | 32768 | 1536 | 109376 | 4096 | 12800 | 14336 | 0.384651 |
| L6_subtile16_stage_full_update_like | 19.1890 | 96.0000 | 168.0000 | 112.0000 | 0 | 32768 | 5120 | 120128 | 11264 | 16384 | 21504 | 0.437438 |
| L6_no_shared_k_direct_global_update_probe | 14.3810 | 112.0000 | 152.0000 | 112.0000 | 0 | 30720 | 1536 | 106944 | 3264 | 11776 | 12800 | 0.375638 |
| L6_update_mfma_no_pred_dependency | 10.1750 | 96.0000 | 80.0000 | 112.0000 | 0 | 20480 | 2048 | 69504 | 4928 | 8192 | 7168 | 0.322613 |
| L6_update_mfma_minimal_frag | 3.1640 | 12.0000 | 4.0000 | 16.0000 | 0 | 1024 | 256.0000 | 4096 | 1344 | 2432 | 512.0000 | 0.113003 |

## Deltas vs Current Baseline

| variant | trace_delta_us | AccVGPR_delta | VGPR_delta | MFMA_delta | VALU_delta | VMEM_delta | LDS_delta |
|:---|---:|---:|---:|---:|---:|---:|---:|
| L6_khalf_stage64_one_update | -13.9400 | -40.0000 | 0 | -3968 | -61184 | -7680 | -12160 |
| L6_khalf_stage64_one_ktile_update | -11.8580 | -32.0000 | 0 | -3584 | -59328 | -7680 | -11264 |
| L6_khalf_stage64_full_update_like | -7.1710 | -76.0000 | -52.0000 | 0 | 62848 | 0 | 0 |
| L6_subtile16_stage_one_update | -19.0280 | -100.0000 | -28.0000 | -3968 | -73856 | -9728 | -15104 |
| L6_subtile16_stage_one_ktile_update | -18.7080 | -96.0000 | -32.0000 | -3584 | -72768 | -9728 | -14336 |
| L6_subtile16_stage_full_update_like | -15.1820 | -96.0000 | -32.0000 | 0 | -62016 | -6144 | -7168 |
| L6_no_shared_k_direct_global_update_probe | -19.9900 | -112.0000 | -16.0000 | -3584 | -75200 | -10752 | -15872 |
| L6_update_mfma_no_pred_dependency | -24.1960 | -184.0000 | -32.0000 | -3072 | -112640 | -14336 | -21504 |
| L6_update_mfma_minimal_frag | -31.2070 | -260.0000 | -116.0000 | -4864 | -178048 | -20096 | -28160 |

## Static ISA Counts

| variant | v_mfma_f32_32x32x8_bf16 | v_mfma_f32_16x16x16_bf16 | global_load | global_store | buffer_load | buffer_store | ds_read | ds_write | s_barrier | s_waitcnt | v_add | v_add3 | v_lshl | v_lshl_add | v_or | v_bfe | max_vgpr_index_static_best_effort | max_acc_index_static_best_effort |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L6_baseline_current_update | 8.0000 | 32.0000 | 144.0000 | 64.0000 | 2.0000 | 2.0000 | 88.0000 | 152.0000 | 6.0000 | 180.0000 | 192.0000 | 72.0000 | 722.0000 | 401.0000 | 166.0000 | 73.0000 | 255.0000 | 131.0000 |
| L6_khalf_stage64_one_update | 16.0000 | 2.0000 | 192.0000 | 40.0000 | 2.0000 | 2.0000 | 50.0000 | 208.0000 | 13.0000 | 206.0000 | 293.0000 | 120.0000 | 532.0000 | 273.0000 | 177.0000 | 113.0000 | 255.0000 | 93.0000 |
| L6_khalf_stage64_one_ktile_update | 16.0000 | 8.0000 | 192.0000 | 40.0000 | 2.0000 | 2.0000 | 64.0000 | 208.0000 | 13.0000 | 215.0000 | 292.0000 | 120.0000 | 530.0000 | 273.0000 | 178.0000 | 113.0000 | 255.0000 | 97.0000 |
| L6_khalf_stage64_full_update_like | 8.0000 | 16.0000 | 84.0000 | 48.0000 | 2.0000 | 2.0000 | 56.0000 | 92.0000 | 7.0000 | 99.0000 | 200.0000 | 72.0000 | 365.0000 | 226.0000 | 134.0000 | 73.0000 | 201.0000 | 15.0000 |
| L6_subtile16_stage_one_update | 8.0000 | 1.0000 | 96.0000 | 36.0000 | 2.0000 | 2.0000 | 18.0000 | 104.0000 | 6.0000 | 99.0000 | 207.0000 | 73.0000 | 411.0000 | 243.0000 | 116.0000 | 73.0000 | 225.0000 | 15.0000 |
| L6_subtile16_stage_one_ktile_update | 8.0000 | 4.0000 | 96.0000 | 36.0000 | 2.0000 | 2.0000 | 24.0000 | 104.0000 | 6.0000 | 103.0000 | 208.0000 | 73.0000 | 410.0000 | 242.0000 | 116.0000 | 73.0000 | 223.0000 | 15.0000 |
| L6_subtile16_stage_full_update_like | 8.0000 | 32.0000 | 96.0000 | 64.0000 | 2.0000 | 2.0000 | 80.0000 | 104.0000 | 6.0000 | 131.0000 | 208.0000 | 73.0000 | 410.0000 | 242.0000 | 116.0000 | 73.0000 | 223.0000 | 15.0000 |
| L6_no_shared_k_direct_global_update_probe | 16.0000 | 8.0000 | 144.0000 | 40.0000 | 2.0000 | 2.0000 | 40.0000 | 160.0000 | 11.0000 | 149.0000 | 272.0000 | 120.0000 | 480.0000 | 275.0000 | 152.0000 | 113.0000 | 236.0000 | 15.0000 |
| L6_update_mfma_no_pred_dependency | 0 | 32.0000 | 64.0000 | 64.0000 | 2.0000 | 2.0000 | 32.0000 | 80.0000 | 2.0000 | 107.0000 | 80.0000 | 16.0000 | 406.0000 | 194.0000 | 79.0000 | 17.0000 | 93.0000 | 3.0000 |
| L6_update_mfma_minimal_frag | 0 | 4.0000 | 2.0000 | 36.0000 | 2.0000 | 2.0000 | 4.0000 | 4.0000 | 1.0000 | 16.0000 | 13.0000 | 2.0000 | 14.0000 | 4.0000 | 3.0000 | 3.0000 | 11.0000 | 3.0000 |

## Answers

1. **Does K-half staging still help after adding real update MFMA?**
   Yes, but it is no longer the best variant. `L6_khalf_stage64_full_update_like` improves trace from `34.371 us` to `27.200 us` and reduces AccVGPR from `264` to `188`. It keeps the same dynamic MFMA count (`5120`) and the same dynamic LDS/VMEM counts as the current baseline, but lowers VGPR (`128 -> 76`) and LDS block (`45056 -> 36864`). The tradeoff is much higher VALU/SALU (`VALU +62848`, `SALU +17600`), so it is useful but not clean enough to call the final answer.
2. **Does subtile staging help after update MFMA?**
   Yes. `L6_subtile16_stage_full_update_like` is the best full-update-like source variant in this pass: trace `19.189 us`, AccVGPR `168`, VGPR `96`, LDS block `32768`, VALU `120128`, VMEM `16384`, LDS inst `21504`. Compared with current baseline, that is `-15.182 us`, `-96 AccVGPR`, `-32 VGPR`, `-62016 VALU`, `-6144 VMEM`, and `-7168 LDS inst` with the same dynamic MFMA count. This is the strongest production-relevant source workaround signal.
3. **Is direct/minimal K-tile still much faster?**
   Yes. `L6_no_shared_k_direct_global_update_probe` is `14.381 us` with AccVGPR `152`, faster than both K-half and subtile full-update-like variants. It is diagnostic rather than production-safe, but it strongly supports that full shared K staging/view lowering remains a major cost even when the exact update MFMA fragment path is present.
4. **Does synthetic no-pred-dependency update reduce AccVGPR?**
   Yes, dramatically. `L6_update_mfma_no_pred_dependency` drops AccVGPR from `264` to `80` and trace from `34.371 us` to `10.175 us`. It still performs a full K staging/update-style path, but removes the MFMA32 pred/v_decay dependency live range. This is the clearest evidence that the current high AccVGPR is not intrinsic to MFMA16 update alone.
5. **What dominates AccVGPR?**
   The dominant factor is the combination of pred/v_decay live range plus the update MFMA path and heavy K staging. The minimal update fragment alone has AccVGPR `4`, and synthetic full update without pred dependency has AccVGPR `80`; current pred-dependent full update has AccVGPR `264`. So the update intrinsic shape is not by itself enough to explain the pressure. Source layout and live-range boundaries matter.

## Targeted Diagnosis

This is a compiler/backend diagnosis report, not a production-kernel report.

The measured result is closest to a combination of Case C and Case D:

- Case C: direct/minimal K-tile remains much faster, so current shared K staging/view lowering is still expensive.
- Case D: synthetic update without pred dependency has much lower AccVGPR, so the pred/v_decay live range into update is also implicated.

The best source-level candidate is `L6_subtile16_stage_full_update_like`, not K-half. It preserves the same update MFMA count as the current baseline while reducing trace by `44.2%` (`34.371 -> 19.189 us`) and AccVGPR by `36.4%` (`264 -> 168`).

Recommended helper/compiler directions:

- A narrow source/helper path for update K staging that stages only the token/K subtile needed by each update tile, avoiding `k_all_t[128,BT]` and the broad `kall_vec` view.
- A possible explicit lifetime/staging-boundary mechanism around pred/v_decay temporaries, because no-pred-dependency drops AccVGPR to `80`.

Recommended next experiment:

- Build one isolated correctness-oriented `L6_best_kstage_update_source_workaround` from the subtile full-update-like pattern if a cleaner version is needed.
- Then, and only then, test a full Qwen experimental copy using subtile K staging in chunk_gdr update.
