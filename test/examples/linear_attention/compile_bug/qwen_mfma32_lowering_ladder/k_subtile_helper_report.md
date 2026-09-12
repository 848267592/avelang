# Qwen MFMA32 K-Subtile Helper Report

## Summary

This report validates the smallest practical source-level helper/pattern for the Qwen update K staging path.

- Validated subtile source pattern delta vs bad baseline: trace `-15.0230 us`, AccVGPR `-96.0000`, VALU `-62016`.

- Helper/source-pattern delta vs bad baseline: trace `-15.3020 us`, AccVGPR `-96.0000`, VALU `-62016`.


## Bad Pattern

The bad source pattern materializes all `K[128,BT]` into shared memory and creates a broad packed view:
```python
k_all_t = al.make_shared((128, BT), al.bf16)
kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 8, 4), (8 * 4, 4, 1)))
k_all_t[kk, tok_k] = k[0, tok_k, key_head_idx, kk]
b_words = kall_vec[tile * 16 + lane_col, pack_base]
```
Counters show this pattern is expensive in trace, AccVGPR, VALU, VMEM, and LDS traffic.

## Helper Design

Chosen path: Option A, source-level helper/pattern in examples only.  No compiler or intrinsic changes were needed.
The helper stages only the 16-token K subtile consumed by the current update MFMA tile:
```python
k_sub_t = al.make_shared((128, 16), al.bf16)
ksub_vec = al.view(k_sub_t, al.i32, al.make_layout((128, 2, 4), (2 * 4, 4, 1)))
k_sub_t[kk_sub, tok_local] = k[0, tok_sub, key_head_idx, kk_sub]
b_words = ksub_vec[tile * 16 + lane_col, pack]
```
This preserves the same update MFMA math while avoiding `k_all_t[128,BT]` and the broad `kall_vec` view.

## Smoke/Finite Checks

| variant | latency_ms | finite | checksum | conclusion |
|:---|---:|:---:|---:|:---|
| L6_baseline_current_update | 0.04697 | True | 129497 | valid |
| L6_subtile16_stage_full_update_like | 0.034952 | True | 129497 | valid |
| L6_k_subtile_helper_full_update_like | 0.034631 | True | 129497 | valid |

## Rocprof Counters

| variant | trace_median_us | VGPR_Count | Accum_VGPR_Count | SGPR_Count | Scratch_Size | LDS_Block_Size | SQ_INSTS_MFMA | SQ_INSTS_VALU | SQ_INSTS_SALU | SQ_INSTS_VMEM | SQ_INSTS_LDS | OccupancyPercent |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L6_baseline_current_update | 34.3310 | 128.0000 | 264.0000 | 112.0000 | 0 | 45056 | 5120 | 182144 | 11200 | 22528 | 28672 | 0.503626 |
| L6_subtile16_stage_full_update_like | 19.3080 | 96.0000 | 168.0000 | 112.0000 | 0 | 32768 | 5120 | 120128 | 11264 | 16384 | 21504 | 0.422323 |
| L6_k_subtile_helper_full_update_like | 19.0290 | 96.0000 | 168.0000 | 112.0000 | 0 | 32768 | 5120 | 120128 | 11264 | 16384 | 21504 | 0.42594 |

## Deltas vs Current Baseline

| variant | trace_delta_us | AccVGPR_delta | VGPR_delta | MFMA_delta | VALU_delta | VMEM_delta | LDS_delta |
|:---|---:|---:|---:|---:|---:|---:|---:|
| L6_subtile16_stage_full_update_like | -15.0230 | -96.0000 | -32.0000 | 0 | -62016 | -6144 | -7168 |
| L6_k_subtile_helper_full_update_like | -15.3020 | -96.0000 | -32.0000 | 0 | -62016 | -6144 | -7168 |

## Static ISA Counts

| variant | v_mfma_f32_32x32x8_bf16 | v_mfma_f32_16x16x16_bf16 | global_load | global_store | buffer_load | buffer_store | ds_read | ds_write | s_barrier | s_waitcnt | v_add | v_add3 | v_lshl | v_lshl_add | v_or | v_bfe | max_vgpr_index_static_best_effort | max_acc_index_static_best_effort |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L6_baseline_current_update | 8.0000 | 32.0000 | 144.0000 | 64.0000 | 2.0000 | 2.0000 | 88.0000 | 152.0000 | 6.0000 | 180.0000 | 192.0000 | 72.0000 | 722.0000 | 401.0000 | 166.0000 | 73.0000 | 255.0000 | 131.0000 |
| L6_subtile16_stage_full_update_like | 8.0000 | 32.0000 | 96.0000 | 64.0000 | 2.0000 | 2.0000 | 80.0000 | 104.0000 | 6.0000 | 131.0000 | 208.0000 | 73.0000 | 410.0000 | 242.0000 | 116.0000 | 73.0000 | 223.0000 | 15.0000 |
| L6_k_subtile_helper_full_update_like | 8.0000 | 32.0000 | 96.0000 | 64.0000 | 2.0000 | 2.0000 | 80.0000 | 104.0000 | 6.0000 | 131.0000 | 208.0000 | 73.0000 | 410.0000 | 242.0000 | 116.0000 | 73.0000 | 223.0000 | 15.0000 |

## Conclusion

The helper/source-pattern row matches the validated subtile variant within measurement noise.  This supports using a source-level helper/pattern rather than a compiler rewrite for the first fix.
This is best classified as a source-lowering/helper issue: the bad broad shared view can be avoided by writing a narrower K-subtile staging pattern today.
Next action: apply only this K-subtile update staging pattern to the isolated full Qwen v29 fused chunk_gdr copy.
