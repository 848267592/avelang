# Qwen GDN v29 Full K-Fragment Rewrite Experiment

## Summary

The positive isolated L6 producer-consumer rewrite did not transfer to the
full v29 BT64 chunk_gdr kernel. The experimental kernel is bit-exact relative
to original v29 for `h` and `final_state`, but regresses substantially and
introduces scratch.

At `T=2048`, normal chunk_gdr latency changed from `0.837143 ms` to
`1.338669 ms` (`0.625x` of original performance). The rocprof trace
median regressed from `830.974 us` to `1302.635 us`.

This is a negative full-kernel transfer result. It is not production-ready
and does not modify v23/v24/v26/v27/v28 or any production baseline.

## Scope

New experimental files:

- `qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_kfrag_rewrite_exp.py`
- `bench_qwen_gdn_v29_mfma32_fused_chunk_gdr_full_kfrag_rewrite_exp.py`

The source was copied from original v29 fused full. Only the broad
`kall_vec` update B-operand loads were replaced with persistent
`al.amdgpu.qwen_update_kfrag_load_bf16x4(...)`. Pred MFMA32, v-decay,
recurrence, geometry, and `h/final_state` interface remain unchanged.

A surviving persistent operation makes the dedicated pass fail compilation.
Successful smoke/benchmark compilation therefore confirms no generic
`kall_vec` fallback.

## Correctness

The initial seed mapped `token_sub` as an 8-token block. The correct
MFMA16 B fragments are `token_sub * 16 + {0,4,8,12}`. After that fix,
the rewrite exactly preserves original v29 semantics:

| T | rewrite vs original h max/mean | rewrite vs original final-state max/mean | rewrite w=0 final-state max |
|---:|---:|---:|---:|
| 512 | `0 / 0` | `0 / 0` | `1.525879e-05` |
| 1024 | `0 / 0` | `0 / 0` | `2.288818e-05` |
| 2048 | `0 / 0` | `0 / 0` | `3.051758e-05` |

The `w=0` values are BF16-level update error versus torch. Original v29's
nonzero-`w` recurrence remains incorrect against its reference (T=2048
final-state max error about `2.11e15`). Since rewrite is exactly equal to
original, that pre-existing issue is not attributable to this experiment.

## Benchmark

Warmup `3`, repeat `10`:

| T | original v29 ms | rewrite exp ms | rewrite/original |
|---:|---:|---:|---:|
| 512 | `0.231964` | `0.337081` | `0.688x` |
| 1024 | `0.448606` | `0.697176` | `0.643x` |
| 2048 | `0.837143` | `1.338669` | `0.625x` |

The earlier K-subtile full experiment is context only: it was also negative,
at about `2.24 ms` with `84 B` scratch for T=2048.

## T=2048 Rocprof

Trace medians are calculated from eight target dispatches.

| Metric | original v29 | kfrag rewrite exp |
|:---|---:|---:|
| trace median us | `830.974` | `1302.635` |
| workgroup / grid work-items | `128 / 4096` | `128 / 4096` |
| LDS block bytes | `61440` | `61440` |
| scratch bytes | `0` | `736` |
| VGPR / AccVGPR / SGPR | `128 / 264 / 112` | `128 / 384 / 112` |
| SQ_INSTS_MFMA | `294912` | `294912` |
| SQ_INSTS_VALU | `4977280` | `3180992` |
| SQ_INSTS_SALU | `810496` | `567808` |
| SQ_INSTS_VMEM | `399360` | `601984` |
| SQ_INSTS_LDS | `1242304` | `1242304` |
| OccupancyPercent | about `0.644` | about `0.643` |

The rewrite lowers VALU/SALU but raises VMEM by about 51%, raises AccVGPR by
120, and creates `736 B` scratch. MFMA and LDS instruction counts are
unchanged, so the resource regression explains the slower trace.

## ISA Evidence

HSACO was dumped under:

- `rocprof_outputs/qwen_v29_kfrag_rewrite_exp/hsaco_original/`
- `rocprof_outputs/qwen_v29_kfrag_rewrite_exp/hsaco_rewrite/`

The final `v_accvgpr_write_b32 a100..` ISA grep could not run because the
Docker execution request was rejected after rocprof completed; the host
`llvm-objdump` lacks the AMDGPU target. This report therefore does not claim
that high AGPR writes disappeared. The profiler establishes the stronger
negative result: AccVGPR increased to `384` and scratch appeared.

## Conclusion

1. The L6 rewrite transferred semantically after fixing the BT64 fragment
   mapping, but did not transfer its lowering benefit.
2. High resource pressure worsened: `264 -> 384` AccVGPR and
   `0 -> 736 B` scratch.
3. Do not migrate this experiment to full forward or a production baseline.
4. The next compiler investigation should reduce this full-loop
   scratch/AccVGPR regression in an isolated loop-shaped repro before another
   Qwen migration.

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache_qwen_kfrag_rewrite python3 -m py_compile \
  test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_kfrag_rewrite_exp.py \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v29_mfma32_fused_chunk_gdr_full_kfrag_rewrite_exp.py

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v29_mfma32_fused_chunk_gdr_full_kfrag_rewrite_exp.py \
  --T 512 1024 2048 --warmup 3 --repeat 10
```
