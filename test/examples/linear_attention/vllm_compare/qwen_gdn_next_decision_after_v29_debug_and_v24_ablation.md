# Qwen GDN Next Decision After v29 Debug and v24 Ablation

## Summary

The new data points to a clear decision:

- Do not continue the current v29 source-level fused full path.
- Do not spend more time on the old FullOp/MFMA16 pollution bug.
- v24/v23 store removal is useful but too small as a standalone production direction.
- The best near-term path is v24/v23 production optimization, while keeping v29 MFMA32 fusion as a backend/block-dot topic rather than a source-level patch series.

## Answers

1. Is v29 fused full failure caused by a source-level recurrence/layout bug?

No simple source-level layout/indexing bug was found.  The state writeback/readback probe through the pred layout matched exactly after BF16 cast.

2. Which tensor first diverges?

`pred` diverges in chunk0 at BF16-level scale:

```text
T=64 normal pred max_abs=2.78058201e-02
```

That then becomes larger in `delta` and `state_after`.

3. Does chunk0 pass?

Not at recurrence tolerance.  Chunk0 already has:

```text
state_after max_abs=2.62058258e-01
```

4. Does chunk1 fail because updated state is not fed back correctly?

Chunk1 fails when feedback is enabled, but the state feedback buffer/layout itself is correct.  The failure is due to feeding back the numerically perturbed updated state.

Evidence:

```text
state_readback_vs_written_bf16 max_abs=0
```

5. Is decay/g_last indexing involved?

No strong evidence.  `decay_off` still fails:

```text
T=128 decay_off pred max_abs=1.09993839e+01
```

6. Did the debugfix improve correctness?

No debugfix was implemented because Track A did not identify a fixable source-level indexing/layout bug.

7. If correctness improved, did performance remain too slow?

The previous fused full candidate remained too slow even before a fix:

```text
T=2048 v29_fused_chunk_gdr_full=0.836383 ms
```

It also had high resource pressure:

```text
trace=833.9205 us, VGPR=128, AccVGPR=264, LDS=61440
```

8. How much upper-bound gain does v24 no_vn/no_h ablation show?

Chunk_gdr-only:

| T | baseline | no_h_no_vn | speedup |
|---:|---:|---:|---:|
| 512 | `0.131876` | `0.108300` | `1.2177x` |
| 1024 | `0.213958` | `0.188079` | `1.1376x` |
| 2048 | `0.359914` | `0.314488` | `1.1444x` |
| 4096 | `0.664226` | `0.569146` | `1.1671x` |

The T=2048 rocprof trace improves from `331.012 us` to `279.495 us`.

9. Based on data, what should we do?

Return to v24/v23 production optimization for immediate work.

The current v29 source-level fused path should stop because correctness fails and full update resource pressure is worse than v24.  v29 MFMA32 fusion may still be valuable later, but it needs backend/block-dot-style lowering or a much more robust recurrence strategy, not another small source edit.

## Exact Commands

v29 debug:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 qwen_gdn_chunked_avelang_v29_mfma32_fused_full_debug.py --T 64 --mode normal
PYTHONDONTWRITEBYTECODE=1 python3 qwen_gdn_chunked_avelang_v29_mfma32_fused_full_debug.py --T 128 --mode normal
PYTHONDONTWRITEBYTECODE=1 python3 qwen_gdn_chunked_avelang_v29_mfma32_fused_full_debug.py --T 128 --mode decay_off
PYTHONDONTWRITEBYTECODE=1 python3 qwen_gdn_chunked_avelang_v29_mfma32_fused_full_debug.py --T 128 --mode feedback_disabled
```

v24 ablation:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 bench_qwen_gdn_v24_gdr_store_ablation.py \
  --T 512 1024 2048 4096 --warmup 5 --repeat 20
```

rocprof:

```bash
/opt/rocm/bin/rocprofv3 \
  --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex store_ablation \
  -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v24_store_ablation_baseline \
  -o v24_store_ablation_baseline_counters \
  -f csv \
  -- python3 test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v24_gdr_store_ablation.py \
       --T 2048 --variant baseline_copy --warmup 2 --repeat 5
```

## Files

- `test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_mfma32_fused_full_debug.py`
- `test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v29_mfma32_fused_full_debug.py`
- `test/examples/linear_attention/vllm_compare/qwen_gdn_v29_mfma32_fused_full_debug_report.md`
- `test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v24_gdr_store_ablation.py`
- `test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v24_gdr_store_ablation.py`
- `test/examples/linear_attention/vllm_compare/qwen_gdn_v24_gdr_store_ablation_report.md`
- `test/examples/linear_attention/vllm_compare/qwen_gdn_next_decision_after_v29_debug_and_v24_ablation.md`
