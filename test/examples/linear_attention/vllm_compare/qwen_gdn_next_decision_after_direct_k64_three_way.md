# Next Decision After Direct-K64 Three-Way Audit

## Decision

Do not alter allocator/RA, broad-K, compact-K or a production recurrence
path. The direct-K64 recurrence suffix audit passed its diagnostic gates but
did not produce an Avelang implementation ready to replace the external
current-vLLM recurrence bridge.

At T=2048, the direct Avelang MFMA32 suffix is `1.133085 ms`; the current
Triton W0 control is `0.116653 ms`, so Avelang is `9.714x` slower despite
matching native's `2,048` dynamic MFMAs per CTA. Avelang MFMA16 is `2.163236
ms`, `1.909x` slower than the MFMA32 control and is closed as a geometry
candidate.

## Why MFMA16 Stops

MFMA16 keeps the requested ABI and H1/H2 state layout, but it needs a
`delta_stage[64,128]` FP32 transposition buffer. That raises LDS from
`6,144 B` to `34,304 B`, dynamic MFMA from `32,768` to `262,144`, and
AccVGPR from `144` to `212`, with no scratch/spill. It diagnoses a layout
cost; it is not a solution.

## What MFMA32 Establishes

The MFMA32 arm has zero scratch and zero code-object register spills. Its
per-CTA VMEM/VALU/SALU counts are respectively `12.14x`, `9.08x` and
`33.49x` the native W0 control. This points to immediate operand staging,
scalar address generation and fragment-view lowering as the next evidence
target, not generic register allocation.

The comparison is not yet compiler-only: native uses BV32/32 CTAs and
continues to execute its fused pred pipeline, while the Avelang arms use
BV64/16 CTAs and update-only math. It is therefore inappropriate to claim
that the backend alone is proven responsible.

## Single Allowed Next Action

Create an experimental `block_dot_bf16_f32` high-level operation and a
same-source generic-vs-gfx942-specialized lowering A/B. Keep source ABI,
CTA ownership, shared allocation, barriers, global traffic and math exactly
the same; only the expansion of the logical block dot may differ. Require
pre-branch MLIR identity, bit-exactness, and per-CTA counter comparison.

Only if this control demonstrates a reduction in VMEM/VALU/SALU without a
resource cliff should it be considered for a recurrence update implementation.

## Evidence

- [Three-way report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_direct_k64_three_way_recurrence_update_report.md)
- `repro_qwen_gdn_direct_k64_update_current_abi.py`
- `repro_qwen_gdn_direct_k64_update_mfma16_current_abi.py`
- `bench_qwen_gdn_direct_k64_three_way.py`
- `profile_qwen_gdn_direct_k64_three_way.py`
