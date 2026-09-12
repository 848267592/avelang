# Qwen GDN Next Decision After K-Fragment Helper Lowering

## Decision

Do not migrate the current `qwen_update_kfrag_load_bf16x4` helper to original full v29.

The isolated helper is bit-identical to baseline, but it does not survive as a distinct fixed-offset operation. It becomes a normal vector load before LLVM optimization, reproducing the same addrspace(3) GEP/staging graph and the same register-allocation result.

## Gate Results

| gate | result |
|:---|:---|
| baseline/helper equivalence | pass, `max_abs=0` |
| finite smoke | pass |
| MFMA count unchanged | pass, `5120` dynamic |
| Scratch remains zero | pass |
| AccVGPR clearly below 264 | fail, remains `264` |
| high `a100..a131` copies reduced | fail, max index remains `131` |
| trace clearly below 34 us | fail, `34.572 us` vs `34.251 us` |

The subtile control remains `19.188 us`, VGPR `96`, AccVGPR `168`, but prior full-Qwen evidence already showed that a source-level K-subtile copy does not transfer safely. Do not reopen that source-tuning path.

## What The Experiment Proved

Removing `kall_vec` syntax alone is insufficient. The broad shared producer remains:

```text
global K -> full [128,64] LDS staging -> generic addrspace(3) GEPs -> MFMA16 B
```

The helper changed only the final source expression. Optimized LLVM and hsaco remained effectively identical, including 155 explicit AGPR writes and AGPR indices up to 131.

This is not evidence for hard-disabling AGPR allocation. It is evidence that the attempted frontend helper was erased too early and never supplied RA with a smaller graph.

## Next Single Action

Implement a compiler-only producer-consumer rewrite around a persistent Qwen K-fragment dialect op:

1. keep the op alive through AveLang-to-memref lowering and GPU outlining;
2. identify the full `[128,64]` K staging producer plus fixed MFMA16 fragment consumers;
3. replace that pair with a compact physical LDS staging region and direct 64-bit fragment loads;
4. verify that broad stores/GEPs disappear before RA.

The acceptance gate remains isolated L6:

- exact baseline equivalence;
- MFMA count unchanged;
- Scratch `0`;
- AccVGPR materially below `264`;
- high AGPR copies materially reduced;
- trace materially below `34 us`.

Only after that gate passes should any compiler helper be tested in an experimental full v29 copy. Do not modify v23/v24/v26/v27/v28 and do not return to source-level subtile, streaming, lifetime-marker, or FullOp experiments.

## Evidence

- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_kfrag_helper_lowering_report.md`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/profile_qwen_kfrag_helper_lowering.py`
- `test/examples/linear_attention/rocprof_outputs/qwen_kfrag_helper_lowering/`

