# Qwen GDN v29 K-Subtile Experiment Report

## Summary

The K-subtile source pattern transferred functionally to full v29: `k_subtile_exp` produces exactly the same `h` and `final_state` as the original v29 fused full kernel for the tested inputs.

It did not transfer as a performance win.  At `T=2048`, latency regressed from `0.834399 ms` to `2.281168 ms`, and rocprof shows higher AccVGPR, scratch, VMEM, and LDS traffic.

## Changed Files

- `qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_k_subtile_exp.py`
- `bench_qwen_gdn_v29_mfma32_fused_chunk_gdr_full_k_subtile_exp.py`

No v23/v24/v26/v27/v28 baseline files were modified.

## Exact Code Change

Original broad staging:

```python
k_all_t = al.make_shared((128, BT), al.bf16)
kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 8, 4), (8 * 4, 4, 1)))
...
k_all_t[k_row_all, token_k_all] = k[0, chunk_start + token_k_all, key_head_idx, k_row_all]
...
b_words_u16 = kall_vec[base_k_u + lane_col, pack_base]
```

Experiment:

```python
k_sub_t = al.make_shared((128, 16), al.bf16)
ksub_vec = al.view(k_sub_t, al.i32, al.make_layout((128, 2, 4), (2 * 4, 4, 1)))
...
for token_sub in al.range(4):
    k_sub_t[k_row_sub, token_local_sub] = k[0, chunk_start + token_k_sub, key_head_idx, k_row_sub]
    ...
    b_words_u16 = ksub_vec[base_k_u + lane_col, 0 or 1]
```

This preserves the update MFMA math and does not change pred schedule, v_decay math, chunk size, or interface.

## Correctness

The original v29 fused full path is still numerically wrong against the torch reference for nonzero `w`, and the k-subtile experiment has the same error.  For the important transfer check, `k_subtile_exp` exactly matches original v29 output/final_state.

| T | original final_state max_abs vs ref | k_subtile final_state max_abs vs ref | k_subtile vs original h max_abs | k_subtile vs original final_state max_abs | w=0 k_subtile final_state max_abs |
|---:|---:|---:|---:|---:|---:|
| 512 | `5.842288e+04` | `5.842288e+04` | `0` | `0` | `1.525879e-05` |
| 1024 | `2.073074e+08` | `2.073074e+08` | `0` | `0` | `2.288818e-05` |
| 2048 | `2.108693e+15` | `2.108693e+15` | `0` | `0` | `3.051758e-05` |

Interpretation: the K-subtile rewrite did not worsen correctness; it is functionally equivalent to the original v29 full kernel for tested inputs.  The full v29 recurrence/correctness issue remains separate.

## Benchmark

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 bench_qwen_gdn_v29_mfma32_fused_chunk_gdr_full_k_subtile_exp.py \
  --T 512 1024 2048 --warmup 5 --repeat 20
```

| T | v24 full | grouped pred-only | fused skeleton | original fused full | k_subtile exp |
|---:|---:|---:|---:|---:|---:|
| 512 | `0.310901` | `0.119618` | `0.074931` | `0.226256` | `0.583467` |
| 1024 | `0.407706` | `0.245064` | `0.134560` | `0.437189` | `1.160183` |
| 2048 | `0.600692` | `0.462207` | `0.239776` | `0.834399` | `2.281168` |

## Rocprof T=2048

Command pattern:

```bash
/opt/rocm/bin/rocprofv3 \
  --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex fused_chunk_gdr_full \
  -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_k_subtile_exp/... \
  -o ... -f csv -- python3 ...
```

| kernel | trace_us | VGPR | AccVGPR | Scratch | LDS block | MFMA | VALU | SALU | VMEM | LDS inst | Occupancy |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original full v29 | `817.775` | 128 | 264 | 0 | 61440 | 294912 | 4977280 | 810496 | 399360 | 1242304 | 0.6445 |
| k_subtile exp | `2239.567` | 128 | 384 | 84 | 49152 | 294912 | 3370304 | 642560 | 1347008 | 2159808 | 0.6495 |

## Diagnosis

The isolated L6 result did not transfer to full Qwen.

What improved:

- VALU decreased from `4,977,280` to `3,370,304`.
- SALU decreased from `810,496` to `642,560`.
- LDS block size decreased from `61,440` to `49,152`.

What got worse:

- AccVGPR increased from `264` to `384`.
- Scratch appeared: `0 -> 84`.
- VMEM increased from `399,360` to `1,347,008`.
- LDS inst increased from `1,242,304` to `2,159,808`.
- Trace regressed from `817.775 us` to `2239.567 us`.

The likely reason is that the correctness-preserving full kernel must re-stage the K subtile inside the `token_sub` loop for every update tile/v-half context.  That repeats K loads and barriers enough to overwhelm the isolated subtile benefit.  It also appears to worsen live pressure and spills.

## Conclusion

This does not revive v29 full path.

The isolated K-subtile helper is valid as a source-lowering pattern, but direct full-kernel substitution is not enough.  Remaining blockers are:

- pred/v_decay live dependency into update;
- repeated subtile staging inside the full recurrence update loop;
- AccVGPR/scratch pressure after the rewrite.

Recommended next action is not K-subtile productionization.  If v29 continues, the next compiler-facing work should target a minimal pred/v_decay lifetime boundary or a more structural fused update schedule that avoids repeated K restaging.
