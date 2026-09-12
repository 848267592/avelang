# Qwen gfx942 BT64 BF16 Solved Boundary Stage 6U

## Conclusion

Stage 6U-Solved completes as **CASE A**. The selected U1 path keeps solve
arithmetic and accumulators in FP32, writes the solved matrix directly as
BF16, and feeds a main-only fused W/U kernel. It is opt-in and experimental;
Stage 6S remains the unchanged production/default path.

All authoritative correctness and latency measurements use complete Eager
public API calls. CUDA/HIP Graph capture and replay were not used. Private
kernel runs and rocprof durations are diagnostic only.

At T=2048, U1 improves over Stage 6S by `122.551 us`, with paired bootstrap
95% CI `[108.741, 136.236] us`. C0 executes `1024 MFMA/CTA` and `262144
MFMA/dispatch`, exactly half F1. Public output max error is `0.001953125` and
final-state max error is `0.0152155161`, both within the frozen contract.

## Source Audit

The hierarchical solve keeps input, shared `x/work`, recurrence and all
`mfma_16x16x4_f32_f32` accumulators in FP32. In the new producer, only the
output pointer and stores change to BF16:

- output zero/strict upper: lines 72-79;
- diagonal identity: lines 107-112;
- diagonal writeback: lines 114-124;
- lower-block writeback: lines 169-177, 218-222, 262-266 and 319-323.

These references are in
`vllm_compare/qwen_gdn_solve_bt64_hierarchical_bf16_stage6u.py`.

F1 generated the main and low residual coefficients separately and executed
W-main, W-residual, U-main and U-residual. W was fully stored before U began,
so the issue was duplicated arithmetic rather than overlapping W/U
accumulator lifetime. Exact old-source references are retained in
`f1_residual_source_map.md` and `f1_mfma_source_accounting.md`.

## P0 Producer

- Kernel: `_qwen_gdn_solve_hierarchical_bt64_fp32_compute_bf16_store_stage6u`
- Wrapper: `qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u`
- P-REF: unchanged FP32 solve followed by a numeric BF16 cast
- P0 change: final global storage dtype only
- Output: contiguous BF16 `[1,T,8,64]`
- HSACO SHA256: `4bd6ddffa7812a0d66b0919963dbda064078ff7d0cfc9d9c7f348903d7fc7c07`

P0 passed 42 producer cases at T=64/128/512/1024/2048/8192, including zero,
identity-like, high-dynamic, small, cancellation and sparse-lower inputs,
non-default stream, NaN prefill and output reuse. It is BF16 bit-exact to
P-REF: mismatch count `0`, max abs `0`.

P1 packed writeback is N/A. ISA uses scattered `global_store_short` and
`global_store_short_d16_hi`; the current lane ownership does not expose a
safe contiguous four-value store without also changing ownership/layout.

## C0 Consumer

- Kernel: `_qwen_gdn_wu_kernel_bt64_fused_bf16_solved_stage6u`
- Wrapper: `qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u`
- Input: BF16 solved A/K/V, FP32 g/beta
- Output: BF16 W/U
- Launch: 256 CTAs at T=2048, WG=256
- Residual source/coefficient/MFMA: completely absent
- HSACO SHA256: `00bee81e2646ecf6d858bb92b3668e0cc4b8ff1011e4055e38c71f40c751d2cc`

The isolated T=64/512/2048 matrix passed all six W/U checks. Maximum absolute
error against the BF16-coefficient matrix reference was `0.0009765625`.

| implementation | W main | W residual | U main | U residual | MFMA/CTA | T=2048 MFMA |
|:--|--:|--:|--:|--:|--:|--:|
| F1 | 512 | 512 | 512 | 512 | 2048 | 524288 |
| C0 | 512 | 0 | 512 | 0 | 1024 | 262144 |
| native vLLM | 64 | 0 | 64 | 0 | 128 | 32768 |

## Full APIs And Correctness

- U0: `qwen_gdn_full_bt64_stage6u_casted_bf16_solved_eager`
- U1: `qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager`
- U2: N/A

U1 does not materialize FP32 solved A, FP32 W or FP32 U. It has no solved
cast and no W/U cast. The recurrence HSACO remains
`632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e`.
KKT, recurrence, V-new cast, chunk-o and final output cast are unchanged.

| coverage | result |
|:--|:--|
| Full matrix T=64/128/512/1024/2048/8192 | 7/7 accepted |
| T=2048 expanded stability | 30 seeds accepted |
| T=8192 expanded stability | 10 seeds accepted |
| T=16384 smoke | 3 seeds accepted |
| U0 versus U1 | output and state bit-exact |
| Max public BF16 output abs | `0.001953125` <= `0.0078125` |
| Max FP32 final-state abs | `0.0152155161` <= `0.02` |
| Non-default stream | pass |

## Eager Public Performance

Each cell is the median of five independent session medians. Every session
uses warmup=30, repeat=200, synchronized HIP events, wall-clock confirmation,
the same current stream and position-balanced order. Each sample includes a
complete public call and its internal allocations/casts/wrapper glue.

| T | Stage 6S ms | F1 ms | U0 ms | U1 ms | vLLM ms |
|--:|--:|--:|--:|--:|--:|
| 512 | 0.361336 | 0.448805 | 0.408086 | 0.367786 | 0.403258 |
| 1024 | 0.293695 | 0.289290 | 0.278994 | 0.267397 | 0.381506 |
| 2048 | 1.236274 | 1.142135 | 1.106523 | 1.064339 | 0.884972 |
| 4096 | 0.602694 | 0.585548 | 0.540381 | 0.537717 | 0.582104 |
| 8192 | 2.387603 | 2.426420 | 2.299091 | 2.269348 | 1.759431 |
| 16384 | 2.104422 | 1.961651 | 1.852589 | 1.844657 | 1.449531 |

The machine showed substantial cross-session clock/load variation, including
non-monotonic absolute medians at T=2048/4096. Therefore the promotion gate
uses paired samples from the same runs, not comparisons to old reports.

| T | U1 gain vs Stage 6S us | paired 95% CI us | U1 gain vs F1 us |
|--:|--:|:--|--:|
| 512 | 26.904 | [-29.576, 59.235] | 30.817 |
| 1024 | 29.277 | [26.564, 31.980] | 17.400 |
| 2048 | 122.551 | [108.741, 136.236] | 84.903 |
| 4096 | 34.294 | [-21.641, 64.345] | 36.303 |
| 8192 | 181.001 | [158.452, 203.506] | 99.289 |
| 16384 | 232.026 | [163.207, 278.647] | 97.352 |

Wall-clock medians confirm the same selected direction:

| T | Stage 6S ms | F1 ms | U0 ms | U1 ms | vLLM ms |
|--:|--:|--:|--:|--:|--:|
| 512 | 0.406648 | 0.472201 | 0.447394 | 0.428326 | 0.431095 |
| 1024 | 0.312053 | 0.306816 | 0.297021 | 0.285915 | 0.399608 |
| 2048 | 1.291911 | 1.226600 | 1.185214 | 1.129205 | 0.950711 |
| 4096 | 0.622513 | 0.604567 | 0.558714 | 0.556476 | 0.602018 |
| 8192 | 2.410462 | 2.450797 | 2.320153 | 2.293549 | 1.783061 |
| 16384 | 2.125860 | 1.983714 | 1.874607 | 1.866695 | 1.471073 |

U1 slope is `6.771890 us/chunk`, versus Stage 6S `7.623539` and vLLM
`4.719342`. Its vLLM gap slope is `2.052548 us/chunk`, improving Stage 6S by
`0.851649 us/chunk` and exceeding the 0.5 target. U1 remains slower than vLLM
at the stable long-text points.

## Resources And ISA

These counters come from complete public API profiler entry but are
diagnostic only. Profiler trace duration is explicitly not used for latency.

| W/U | VGPR | AccVGPR | SGPR | LDS B | scratch | MFMA | VALU | SALU | VMEM | LDS inst |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F1 | 64 | 8 | 48 | 3072 | 0 | 524288 | 4846592 | 696320 | 491520 | 1114112 |
| C0 | 68 | 12 | 32 | 3072 | 0 | 262144 | 2087936 | 316416 | 311296 | 589824 |
| native vLLM | 60 | 164 | 112 | 0 | 0 | 32768 | 2176000 | 223232 | 92160 | 75776 |

C0 code-object metadata reports private segment `0`, VGPR spills `0`, SGPR
spills `0`; ISA contains `v_mfma_f32_16x16x16_bf16` and no residual phase.
P0 similarly reports private segment/spills `0` and retains
`v_mfma_f32_16x16x4_f32`.

U1 has eight dispatches: cumsum, KKT, P0, C0, recurrence, V-new cast,
chunk-o and final cast. U0 has nine due to solved cast; Stage 6S has eleven;
native vLLM has seven. The U1 trace confirms no solved or W/U cast dispatch.

## Remaining Gap And Decision

After residual removal:

```text
C0 1024 MFMA/CTA = 4x lane-group predicate x 2x MFMA16 geometry x
                   native vLLM 128 MFMA/CTA
```

The 4x factor comes from four divergent `lane_group` call regions in each W
and U source loop. The 2x factor is ideal MFMA16 versus MFMA32 tile geometry.
U2 was not implemented because branch-free operand selection has not yet
passed an isolated lowering/resource gate; changing predicates and geometry
together would violate the one-variable rule.

U1 passes the promotion gate and is the current best **experimental** Eager
candidate. Stage 6S remains production/default. The next single action is an
isolated C0 predicate-collapse experiment that proves one wave-uniform MFMA
call after per-lane fragment selection. No compiler or assembly change is
needed for that experiment.

## Evidence

All raw samples, correctness matrices, CSV comparisons, LLVM IR, ISA,
code-object metadata and exact commands are under
`codex_qwen_bt64_bf16_solved_boundary_stage6u/`.

