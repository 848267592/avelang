# Qwen gfx942 BT64 Native W/U + chunk-o Stage 3

## Summary

Stage 3 replaces the Stage 2 **generic scalar BT64** W/U and chunk-o fallbacks
with opt-in 64-thread MFMA implementations derived from v24/v14's validated
BT16 microkernel.  It does not modify v23/v24/v26/v27/v28, the frozen asm-v0
recurrence, compiler lowering, LLVM, AMDGPU RA, or default dispatch.

The native stages pass standalone numerical gates, produce real
`v_mfma_f32_16x16x16_bf16` instructions, have zero scratch, and pass the
frozen 37-case BT64 full contract.  At T=2048 the full candidate is
`1.0922 ms`, a `14.69x` reduction from Stage 2's `16.0413 ms`.

It is **not** a production promotion: the valid BT64 candidate is still slower
than v24 BT16 (`0.5882 ms`) and the two-session vLLM reference (`0.3613 ms`).
The next measured bottleneck is generic BT64 KKT.

## Scope and Source Audit

The detailed source audit is in
`codex_qwen_bt64_wu_chunko_stage3/source_audit.md`; the machine-readable map
is `source_map.json`.

- v24 imports its W/U and chunk-o fast kernels from v17, not from a scalar
  fallback.  They use 64 threads and `mfma_16x16x16_bf16_f32`.
- Stage 2 uses v6 W/U and chunk-o only to establish the BT64 numerical bridge
  to asm-v0.  Their T=2048 rocprof workgroup is one thread, with no MFMA.
- Stage 3 creates
  `qwen_gdn_bt64_native_wu_chunko_mfma_v1.py` and the explicit opt-in public
  entry `qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1`.
- BT64 is four 16-token source subtiles.  W/U accumulate four subtiles; the
  chunk-o source loop handles the full 4x4 lower-triangular tile relation.

The contracts are frozen in `wu_contract.md/json` and `chunko_contract.md/json`.
W/U output FP32 for the asm ABI; native chunk-o consumes BF16 asm H directly
and yields FP32 before public BF16 conversion.

## Standalone Correctness

GPU pytest results:

| gate | cases | maximum abs error | result |
|:--|:--|--:|:--|
| native W vs v6 BT64 | T=64,128 | `2.9802322e-08` | pass |
| native U vs v6 BT64 | T=64,128 | `2.9802322e-08` | pass |
| native chunk-o vs v6 BF16-H consumer | T=64,128 | `8.353265e-06` | pass |
| early-source to later-output causal tile check | T=64 | `2.5258523e-06` | pass |
| native full graph vs frozen vLLM contract | T=64,128,512 | output `<=0.001953125`, state `<=0.017289162` | 3 passed |

The standalone suite was `5 passed in 13.63s`; the fast full smoke suite was
`3 passed in 28.07s` on the MI300 environment.

## Full BT64 Contract Matrix

The local Stage 3 runner pins the entire 37-case plan so it does not depend on
the version of Stage 2 helper code copied into a Docker workspace.  It covers
30 random cases, neutral-gate, high-dynamic, cancellation, small-value, and
two T=8192 cases with and without initial state.

| quantity | observed maximum abs | frozen threshold | result |
|:--|--:|--:|:--|
| public BF16 output | `0.001953125` | `0.0078125` | pass |
| FP32 final state | `0.013978422` | `0.020000000` | pass |

`full_correctness.json` and `full_correctness.csv` record all 37 accepted
cases.  The largest public-output/final-state pair occurred in the T=8192
neutral-gate case; both remain under the predeclared contract.

## Microbenchmark

HIP-event median, warmup 10/repeat 50, actual MI300 runs:

| T | stage | generic Stage 2 ms | native Stage 3 ms | speedup |
|--:|:--|--:|--:|--:|
| 512 | W/U | `1.447091` | `0.194790` | `7.43x` |
| 512 | chunk-o | `3.694571` | `0.067280` | `54.91x` |
| 2048 | W/U | `4.005133` | `0.294919` | `13.58x` |
| 2048 | chunk-o | `12.151661` | `0.256241` | `47.42x` |

The requested T=2048 W/U and chunk-o gates therefore pass comfortably.  This
table compares the same Stage 2 generic Avelang fallbacks to native Stage 3;
it is not a direct vLLM per-stage measurement.

## Full Pipeline Timing

Each Stage 3/v24 row is the median of three independent Python sessions,
warmup 10/repeat 50.  vLLM completed two independent sessions before the
environment rejected the third Docker execution on quota grounds; its `n=2`
median is labelled accordingly and is not represented as a three-session
result.

| T | Stage 3 BT64 native ms | v24 BT16 ms | vLLM ms (`n=2`) |
|--:|--:|--:|--:|
| 512 | `0.489788` | `0.304152` | `0.307978` |
| 2048 | `1.092222` | `0.588173` | `0.361317` |
| 8192 | `3.659696` | `2.124136` | `0.747861` |
| 16384 | `7.223932` | `4.256944` | `1.263817` |

At T=2048 Stage 3 is `14.69x` faster than Stage 2 (`16.041299 ms`), but
`1.86x` slower than v24 and about `3.02x` slower than the two-session vLLM
reference.  The raw sessions are in `full_pipeline_benchmark_stage3_[abc].csv`
and `full_pipeline_benchmark_vllm_[ab].csv`.

## Stage Breakdown

The following T=2048 HIP-event stage medians are from independent Stage 3
session `stage3_a` (warmup 10/repeat 50):

| stage | ms |
|:--|--:|
| cumsum | `0.031327` |
| KKT generic BT64 | `0.350221` |
| solve | `0.125367` |
| native W/U | `0.235329` |
| frozen asm recurrence, preallocated | `0.201620` |
| native chunk-o | `0.227899` |
| full native graph, cached allocator | `1.089960` |

KKT is the largest individual stage.  Native W/U and chunk-o are no longer
the scalar disasters from Stage 2, but their remaining composition with KKT,
solve, and the frozen recurrence is still above v24/vLLM.

## T=2048 rocprof Resources

Tail median of five matching dispatches.  Stage 2 figures are the prior
targeted profile, while native figures are measured from this Stage 3 source.

| kernel/group | trace us | WG | grid work-items | scratch | VGPR | AccVGPR | LDS B | MFMA | VALU | VMEM |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Stage 2 generic W/U | `3611.807` | 1 | 16384 | 0 | 128 | 144 | 0 | 0 | 800227328 | 34603008 |
| native W | `131.034` | 64 | 524288 | 0 | 44 | 4 | 1024 | 131072 | 55279616 | 6127616 |
| native U | `88.572` | 64 | 524288 | 0 | 44 | 4 | 1024 | 131072 | 28295168 | 4521984 |
| Stage 2 generic chunk-o | `11761.158` | 1 | 16384 | 260 | 128 | 136 | 0 | 0 | 1079058432 | 215351296 |
| native chunk-o | `203.422` | 64 | 524288 | 0 | 12 | 140 | 13312 | 458752 | 18096128 | 1933312 |

The native W+U trace is `219.606 us`, roughly `16.45x` below the Stage 2 W/U
trace.  Native chunk-o is `57.82x` below the Stage 2 generic chunk-o trace.
The detailed counters, including SALU, LDS instructions, SGPR, and occupancy,
are in `resource_profile.json/csv`.

## ISA Evidence

All three native HSACOs were captured from the MI300 container and disassembled
with ROCm `llvm-objdump`:

- native W: 16 static `v_mfma_f32_16x16x16_bf16` instructions;
- native U: 16 static `v_mfma_f32_16x16x16_bf16` instructions;
- native chunk-o: 56 static `v_mfma_f32_16x16x16_bf16` instructions;
- all three: zero `v_mfma_f32_32x32x8_bf16` instructions.

The files are under `native_wu/isa/` and `native_chunko/isa/`.  This is the
intended v24 microkernel port, not an accidental scalar fallback.  The profile
also proves zero scratch for all three native kernels.

## Tests and Reproduction

New source/test/benchmark files:

- `vllm_compare/qwen_gdn_bt64_native_wu_chunko_mfma_v1.py`
- `vllm_compare/test_qwen_gdn_bt64_native_wu_chunko_mfma_v1.py`
- `vllm_compare/test_qwen_gdn_full_bt64_native_wu_o_v1.py`
- `vllm_compare/bench_qwen_gdn_bt64_native_wu_chunko_mfma_v1.py`
- `codex_qwen_bt64_wu_chunko_stage3/stage3_runner.py`
- `codex_qwen_bt64_wu_chunko_stage3/bench_stage3.py`
- `codex_qwen_bt64_wu_chunko_stage3/profile_native_stages.py`

Exact commands are in `codex_qwen_bt64_wu_chunko_stage3/commands.sh`.
The immutable asm-v0/P16 regression suites were not re-run in this pass after
the environment exhausted Docker execution quota; they were untouched, and
the new full matrix repeatedly dispatches the unchanged asm recurrence.

## Decision

The Stage 2 W/U and chunk-o fallbacks should be replaced **only through the
new opt-in Stage 3 API** for further BT64 experimentation.  Do not change any
production/default dispatch.  The next isolated optimization should be a
native BT64 KKT: at T=2048 it is the largest measured stage (`0.350221 ms`) and
remains the Stage 2 generic scalar implementation.  Re-evaluate full BT64
after that gate; do not alter the validated asm recurrence merely because it is
part of the remaining cost.

