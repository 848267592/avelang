# Qwen v31 Active-V16 Pred and Recurrence Audit

## Result

The D1 padded-MFMA32 pred path is semantically wrong. The first material divergence is its active-quadrant extraction, not generic lowering or partial state feedback. Replacing only that primitive with direct MFMA16 (D2) makes the one-BT64 chunk result correct, cuts pred-only trace by 2.49x, and removes padded work. D2 still fails multi-chunk reference equivalence because a small BF16 update numerical-order delta compounds across the recurrence.

No compiler change is warranted: D1/D2 final MIR has zero spill/scratch and does not recreate padded work or high-AGPR parking.

## First Divergence

With deterministic seed `20260712`, active logical pred is `W[16,128] @ state[16,128].T` after the same BF16 staging used by both paths.

| logical output | reference | D1/P32 | P16 |
|:--|--:|--:|--:|
| first >1e-3: `[0,0]` | see artifact | divergent | exact |
| maximum: `[11,8]` | `0.0090225022` | `-0.0126709975` | `0.0090225022` |

P32 max/mean error: `2.16935e-2 / 6.14372e-3`. The root cause is the padded MFMA32 lane/quadrant predicate, not decay, V16 state ownership, or K16 update.

The D2 MFMA16 primitive fixes this at T=64: final-state max error is `3.81e-6`. At T=128, chunk 1's input h differs by only `5.72e-6`, but the next state error becomes `4.01e-2`; this is the remaining recurrent BF16 update numerical-order issue. D2R deferred writeback was tested separately and did not remove that divergence, so early token16 state feedback is not the primary cause.

## P32 Versus P16

| metric | P32 padded MFMA32 | P16 direct MFMA16 |
|:--|--:|--:|
| reference correct | no | yes |
| primitive trace | `8.172 us` | `3.285 us` |
| MFMA | 16 | 8 |
| VALU / SALU | 2292 / 490 | 386 / 4 |
| VMEM / LDS | 132 / 156 | 68 / 88 |
| VGPR / AccVGPR | 24 / 240 | 48 / 88 |
| scratch | 0 | 0 |
| LDS block | 24576 B | 10240 B |

`test_qwen_v31_active_v16_pred_primitives.py` ran 50 deterministic random cases plus four extreme scales: `55 passed`. P16 is mathematically correct for the exact BF16 predicate and has no padded 32-row output quadrant.

## D2 Full Kernel

D2 changes only D1's pred microkernel: padded MFMA32 plus lane-major inverse map becomes two K64 MFMA16 partials. BT64/BV32, workgroup 128, V16/token16/K16 schedule, K staging, update code, and public interface remain otherwise the same. D2 exact LTO has no `SI_SPILL_AV*` saves.

| metric | D1 | D2 |
|:--|--:|--:|
| T=2048 normal latency | `2.327338 ms` | `1.585316 ms` |
| us per BT64 chunk | `72.729` | `49.541` |
| scratch | 0 | 0 |
| D2 rocprof AccVGPR | 164 | 228 |
| D2 T=2048 trace | - | `1546.299 us` |
| D2 T=8192 trace | - | `6162.339 us` |

D2 improves D1 by about 1.47x at T=2048, below the required 2x threshold, and cannot be promoted because T>=128 reference correctness is not preserved.

## Long Sequence (Experimental Only)

| T | D2 ms | us/BT64 chunk |
|--:|--:|--:|
| 512 | `0.411391` | `51.424` |
| 1024 | `0.811686` | `50.730` |
| 2048 | `1.585316` | `49.541` |
| 4096 | `3.133317` | `48.958` |
| 8192 | `6.224110` | `48.626` |
| 16384 | `12.487999` | `48.781` |

Fit: `latency_ms = 0.022713 + 0.000760092 * T`. These are actual timings, but invalid for v24/Triton crossover claims because D2 is not multi-chunk reference correct. No full-forward benchmark was run for the same reason.

## MIR / Compiler Decision

P16 and D2 exact LTO artifacts are in:

- `rocprof_outputs/qwen_v31_active_v16/exact_lto_p32/`
- `rocprof_outputs/qwen_v31_active_v16/exact_lto_p16/`
- `rocprof_outputs/qwen_v31_active_v16/exact_lto_d2/`

They contain no spills. D2 does not reintroduce MFMA32 padded work, high AGPR parking, or scratch. A compiler-owned block op would therefore target the wrong layer and was not implemented.

## Recommendation

Keep P16 as the correct active-V16 pred primitive. Do not run staging reuse yet: it would confound the unresolved recurrence numerical semantics. The next smallest valid step is an isolated, reference-matching BT64 update primitive that reproduces the exact 64-token accumulation order without reintroducing the broad v29 live region. Only after that passes multi-chunk recurrence should P16 be reconsidered for a production BT64 path.

## Artifacts

- `codex_v31_active_v16_audit/first_divergence.{json,md}`
- `codex_v31_active_v16_audit/pred_primitive_results.csv`
- `codex_v31_active_v16_audit/pred_primitive_rocprof.json`
- `codex_v31_active_v16_audit/d2_benchmark.csv`
- `codex_v31_active_v16_audit/d2_rocprof_t{2048,8192}.json`
- `codex_v31_active_v16_audit/long_sequence_slope.json`

No production baseline was modified or committed.
