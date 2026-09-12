# Qwen MFMA32 L6 Lifetime Boundary Profile Report

## Summary

This report profiles the minimal `al.end_lifetime(...)` marker placed after `v_decay_t` staging and before update MFMA.

- Broad-K baseline lifetime-marker delta: trace `-0.08 us`, AccVGPR `0`, Scratch `0`.

- Broad-K memref-only lifetime-marker delta: trace `-32.2080 us`, AccVGPR `-264.0000`, Scratch `0`.

- K-subtile lifetime-marker delta: trace `-0.16 us`, AccVGPR `0`, Scratch `0`.

- K-subtile memref-only lifetime-marker delta: trace `-17.0650 us`, AccVGPR `-168.0000`, Scratch `0`.


## Marker Placement

```python
# after v_decay_t has been written and synchronized
al.end_lifetime(pred_acc, pred_partial, state_bf16, w_bf16, state_vec, w_vec)
al.syncthreads()
# update MFMA begins here
```
The marker does not end the lifetime of `v_decay_t`, K staging buffers, or any state needed by update.

## Smoke/Finite Checks

| variant | latency_ms | finite | checksum | conclusion |
|:---|---:|:---:|---:|:---|
| L6_baseline_no_lifetime | 0.0471095 | True | 129497 | valid |
| L6_with_end_lifetime_after_vdecay | 0.046509 | True | 129497 | valid |
| L6_memref_only_end_lifetime | 0.025658 | True | 0 | valid |
| L6_subtile_no_lifetime | 0.034692 | True | 129497 | valid |
| L6_subtile_with_end_lifetime_after_vdecay | 0.034852 | True | 129497 | valid |
| L6_subtile_memref_only_end_lifetime | 0.025398 | True | 0 | valid |

## Rocprof Counters

| variant | trace_median_us | VGPR_Count | Accum_VGPR_Count | SGPR_Count | Scratch_Size | LDS_Block_Size | SQ_INSTS_MFMA | SQ_INSTS_VALU | SQ_INSTS_SALU | SQ_INSTS_VMEM | SQ_INSTS_LDS | OccupancyPercent |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L6_baseline_no_lifetime | 34.3710 | 128.0000 | 264.0000 | 112.0000 | 0 | 45056 | 5120 | 182144 | 11200 | 22528 | 28672 | 0.512611 |
| L6_with_end_lifetime_after_vdecay | 34.2910 | 128.0000 | 264.0000 | 112.0000 | 0 | 45056 | 5120 | 182144 | 11200 | 22528 | 28672 | 0.50609 |
| L6_memref_only_end_lifetime | 2.1630 | 8.0000 | 0 | 16.0000 | 0 | 0 | 0 | 576.0000 | 384.0000 | 2048 | 0 | 0.0791932 |
| L6_subtile_no_lifetime | 19.3490 | 96.0000 | 168.0000 | 112.0000 | 0 | 32768 | 5120 | 120128 | 11264 | 16384 | 21504 | 0.423275 |
| L6_subtile_with_end_lifetime_after_vdecay | 19.1890 | 96.0000 | 168.0000 | 112.0000 | 0 | 32768 | 5120 | 120128 | 11264 | 16384 | 21504 | 0.421452 |
| L6_subtile_memref_only_end_lifetime | 2.2840 | 8.0000 | 0 | 16.0000 | 0 | 0 | 0 | 576.0000 | 384.0000 | 2048 | 0 | 0.0689527 |

## Deltas

| comparison | trace_delta_us | AccVGPR_delta | VGPR_delta | Scratch_delta | MFMA_delta | VALU_delta | VMEM_delta | LDS_delta |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline + marker vs baseline | -0.08 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| baseline + memref-only marker vs baseline | -32.2080 | -264.0000 | -120.0000 | 0 | -5120 | -181568 | -20480 | -28672 |
| subtile + marker vs subtile | -0.16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| subtile + memref-only marker vs subtile | -17.0650 | -168.0000 | -88.0000 | 0 | -5120 | -119552 | -14336 | -21504 |

## Static ISA Counts

| variant | v_mfma_f32_32x32x8_bf16 | v_mfma_f32_16x16x16_bf16 | global_load | global_store | buffer_load | buffer_store | ds_read | ds_write | s_barrier | s_waitcnt | v_add | v_add3 | v_lshl | v_lshl_add | v_or | v_bfe | max_vgpr_index_static_best_effort | max_acc_index_static_best_effort |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L6_baseline_no_lifetime | 8.0000 | 32.0000 | 144.0000 | 64.0000 | 2.0000 | 2.0000 | 88.0000 | 152.0000 | 6.0000 | 180.0000 | 192.0000 | 72.0000 | 722.0000 | 401.0000 | 166.0000 | 73.0000 | 255.0000 | 131.0000 |
| L6_with_end_lifetime_after_vdecay | 8.0000 | 32.0000 | 144.0000 | 64.0000 | 2.0000 | 2.0000 | 88.0000 | 152.0000 | 6.0000 | 180.0000 | 192.0000 | 72.0000 | 722.0000 | 401.0000 | 166.0000 | 73.0000 | 255.0000 | 131.0000 |
| L6_memref_only_end_lifetime | 0 | 0 | 0 | 32.0000 | 2.0000 | 2.0000 | 0 | 0 | 0 | 9.0000 | 6.0000 | 0 | 2.0000 | 1.0000 | 0 | 0 | 11.0000 |  |
| L6_subtile_no_lifetime | 8.0000 | 32.0000 | 96.0000 | 64.0000 | 2.0000 | 2.0000 | 80.0000 | 104.0000 | 6.0000 | 131.0000 | 208.0000 | 73.0000 | 410.0000 | 242.0000 | 116.0000 | 73.0000 | 223.0000 | 15.0000 |
| L6_subtile_with_end_lifetime_after_vdecay | 8.0000 | 32.0000 | 96.0000 | 64.0000 | 2.0000 | 2.0000 | 80.0000 | 104.0000 | 6.0000 | 131.0000 | 208.0000 | 73.0000 | 410.0000 | 242.0000 | 116.0000 | 73.0000 | 223.0000 | 15.0000 |
| L6_subtile_memref_only_end_lifetime | 0 | 0 | 0 | 32.0000 | 2.0000 | 2.0000 | 0 | 0 | 0 | 9.0000 | 6.0000 | 0 | 2.0000 | 1.0000 | 0 | 0 | 11.0000 |  |

## Conclusion

Interpretation belongs in `compiler_lifetime_boundary_patch_report.md`, which combines these counters with the compiler patch behavior.
