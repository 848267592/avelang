# Qwen GDN Next Decision After Late Lifetime Lowering

## Summary

The late lifetime-boundary experiment is complete.  The marker now survives past AveLang-to-memref lowering and is erased only after GPU outlining, but correctness-equivalent L6 counters are unchanged.

Conclusion:

- `ave.end_lifetime` surviving later is not enough.
- LLVM `lifetime.end` was not emitted.
- AMDGPU resource allocation did not receive a usable lifetime signal.
- Do not create a full Qwen v29 lifetime-boundary copy from this result.

## What Changed

Compiler changes:

- `ave.end_lifetime` is no longer erased in `lower_ave_lang_to_memref_pass.cc`.
- A late `EraseAveLangEndLifetimePass` was added in `lower_to_llvm.cc` after GPU outlining and before backend lowering.

Experiment/report files:

- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/lifetime_operand_audit.md`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/compiler_late_lifetime_lowering_report.md`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/l6_lifetime_boundary_profile_report.md`

## Did Late Lifetime Lowering Affect AccVGPR/VGPR/Trace?

No material effect.

| variant | trace_us | VGPR | AccVGPR | Scratch | MFMA | VALU | VMEM | LDS inst |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `L6_baseline_no_lifetime` | `34.371` | 128 | 264 | 0 | 5120 | 182144 | 22528 | 28672 |
| `L6_with_end_lifetime_after_vdecay` | `34.291` | 128 | 264 | 0 | 5120 | 182144 | 22528 | 28672 |
| `L6_subtile_no_lifetime` | `19.349` | 96 | 168 | 0 | 5120 | 120128 | 16384 | 21504 |
| `L6_subtile_with_end_lifetime_after_vdecay` | `19.189` | 96 | 168 | 0 | 5120 | 120128 | 16384 | 21504 |

Deltas:

- broad-K marker: trace `-0.080 us`, AccVGPR `0`, VGPR `0`, Scratch `0`.
- K-subtile marker: trace `-0.160 us`, AccVGPR `0`, VGPR `0`, Scratch `0`.

The two memref-only variants are invalid/profiling-only because their checksum is `0` and MFMA count is `0`; they are not valid optimization evidence.

## Did LLVM Lifetime.End Get Emitted?

No.

The current pass only keeps `ave.end_lifetime` alive longer, then erases it before AMDGPU lowering.  It does not lower to LLVM lifetime intrinsics.

Why:

- `pred_acc` is a post-MFMA SSA/vector accumulator value, not a private allocation with a stable base pointer.
- `pred_partial`, `state_bf16`, and `w_bf16` are shared/workgroup memrefs.  Emitting stack lifetime or dealloc semantics for them is unsafe.
- `state_vec` and `w_vec` are views of shared memrefs, so they require base recovery and still map to shared memory.

## Did AMDGPU Resource Allocation Use It?

No evidence that it did.

Because the marker is erased before backend lowering and no LLVM lifetime intrinsic is emitted, AMDGPU register allocation has no new information.  The unchanged AccVGPR/VGPR counters confirm that.

## Is The Remaining Problem Expressible Through Lifetime Markers?

Not by the current marker form.

The L6 operand audit suggests the pressure is not a simple private-allocation lifetime issue:

- pure SSA accumulators already have last-use information;
- shared buffers need allocation reuse or phase scheduling, not ordinary `lifetime.end`;
- the pred-to-update pressure appears tied to Qwen-shaped dataflow and MFMA/update lowering.

## Decision

Do not create:

- `qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_late_lifetime_exp.py`

Do not continue:

- K-subtile source-only tuning;
- streaming variants;
- v23/v24 local changes;
- Triton-like block-dot lowering in this thread.

## Next Compiler Action

The next useful compiler step is not another source marker.  It should be one of:

1. a backend-visible pred-to-update phase boundary that survives into AMDGPU lowering/register allocation;
2. conservative shared allocation reuse/diagnostics for non-overlapping workgroup buffers;
3. direct investigation of Qwen-shaped accumulator/dataflow lowering around MFMA32 pred and MFMA16 update.

Current recommendation: pursue option 3 first, because both marker-only and late-survive marker experiments left valid L6 counters unchanged.
