# Qwen GDN Next Decision After L6 K-Stage Update

## Summary

The isolated L6 K-stage/update experiment is complete.  The next issue is not purely K staging and not purely the MFMA16 update intrinsic.  The measured bottleneck is the combination of:

1. full shared K staging/view lowering through `k_all_t[128,BT]` plus `kall_vec`;
2. long pred/v_decay live range feeding the update MFMA path.

The best production-relevant source pattern is now `L6_subtile16_stage_full_update_like`.

## Key Measurements

| variant | trace_us | VGPR | AccVGPR | LDS block | MFMA | VALU | VMEM | LDS inst |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `L6_baseline_current_update` | `34.371` | 128 | 264 | 45056 | 5120 | 182144 | 22528 | 28672 |
| `L6_khalf_stage64_full_update_like` | `27.200` | 76 | 188 | 36864 | 5120 | 244992 | 22528 | 28672 |
| `L6_subtile16_stage_full_update_like` | `19.189` | 96 | 168 | 32768 | 5120 | 120128 | 16384 | 21504 |
| `L6_no_shared_k_direct_global_update_probe` | `14.381` | 112 | 152 | 30720 | 1536 | 106944 | 11776 | 12800 |
| `L6_update_mfma_no_pred_dependency` | `10.175` | 96 | 80 | 20480 | 2048 | 69504 | 8192 | 7168 |
| `L6_update_mfma_minimal_frag` | `3.164` | 12 | 4 | 1024 | 256 | 4096 | 2432 | 512 |

All variants passed finite checks.

## Is The Next Issue K Staging Or Update MFMA Pressure?

Both, but the data separates the two:

- K staging is still a real problem: `L6_subtile16_stage_full_update_like` cuts trace from `34.371 us` to `19.189 us` with the same dynamic MFMA count.
- Update MFMA alone is not the main explanation: `L6_update_mfma_minimal_frag` has AccVGPR `4`.
- Pred/v_decay dependency matters strongly: `L6_update_mfma_no_pred_dependency` drops AccVGPR from `264` to `80`.

So the current source shape is causing pressure by carrying pred/v_decay state into the update path while also staging a broad transposed K matrix.

## Is There Evidence For A Small Compiler/Helper Fix?

Yes, but it should be narrow.

The most concrete helper opportunity is a source/helper pattern for update K staging:

```text
stage only the K/token subtile consumed by the update MFMA tile
instead of materializing k_all_t[128,BT] and viewing it as kall_vec
```

The second possible helper is a lifetime/staging-boundary mechanism between pred/v_decay and update.  This is supported by the no-pred-dependency variant, but it is less directly actionable than the K-subtile staging source pattern.

## Should We Test A Full Qwen Copy Next?

Yes, but only an experimental copy, not a baseline change.

The isolated trace gate is now strong enough:

- `L6_subtile16_stage_full_update_like` improves trace by `44.2%`.
- It reduces AccVGPR by `36.4%`.
- It reduces VALU, VMEM, and LDS instructions while preserving full-update-like MFMA count.

This is stronger than the previous L5-only evidence, because the update MFMA path is included.

Do not modify v23/v24/v26/v27/v28.  The full Qwen test should be a new experimental file that copies the current relevant chunk_gdr source and changes only the update K staging pattern.

## If Full Qwen Is Not Recommended, Why?

Full Qwen production replacement is not recommended yet.  The isolated variant is finite and profiler-valid, but it is still a ladder/sink test, not full state-update correctness.

Before promoting anything:

1. implement a clean `L6_best_kstage_update_source_workaround` or directly a Qwen experimental copy;
2. verify state-update correctness against the current Qwen chunk_gdr reference;
3. benchmark chunk_gdr-only before full-path benchmarking.

## Next Single Action

Create a new experimental Qwen chunk_gdr copy using subtile K staging in the update path.

Recommended scope:

- keep existing pred/v_decay computation;
- keep existing update MFMA math;
- replace broad `k_all_t[128,BT]` staging with the subtile staging pattern validated by `L6_subtile16_stage_full_update_like`;
- compare chunk_gdr-only correctness and latency against the current best production baseline.

## Evidence Files

- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_qwen_mfma32_l6_kstage_update_variants.py`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/profile_qwen_mfma32_l6_kstage_update_variants.py`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_mfma32_l6_kstage_update_variants_report.md`
- `test/examples/linear_attention/rocprof_outputs/qwen_mfma32_l6_kstage_update_variants/`
- `test/examples/linear_attention/rocprof_outputs/qwen_mfma32_l6_kstage_update_variants_hsaco/`
