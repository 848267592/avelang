# Next Decision After Direct-K64 BV32 Ownership

## Decision

Use the experimental `BV32`, 32-CTA, two-wave-cooperative
`block_dot_bf16_f32` direct-K64 suffix as the ownership baseline for the next
native-recurrence diagnostic.  Do not change production dispatch or full-v29.

At T=2048 it is `0.430079 ms`, versus `0.518670 ms` for BV64 and `0.119898 ms`
for the current Triton W=0 control.  BV32 therefore improves BV64 by 17.1%
and reduces the native ratio from 4.33x to 3.59x.  It has zero scratch and
zero spills, with VGPR 64 and occupancy about 64%.

## What This Rules In And Out

- `BV64` is no longer the better isolated ownership choice for this ABI.
- The improvement is not a hidden spill tradeoff: both LTO MIR and code-object
  metadata report zero spills/scratch.
- MFMA32 is present and MFMA16 absent.  Total dynamic Avelang MFMA remains
  32768, so the speedup is not a math reduction.
- BV32 increases total VMEM/LDS work because it doubles CTAs and makes the
  direct-K/g staging less amortized.  It is a state-pressure/occupancy gain,
  not proof that the input pipeline is optimal.
- The remaining 3.59x gap cannot be assigned solely to block-dot lowering:
  native W=0 still executes a fused pred pipeline, while Avelang is update
  suffix only.  Nevertheless, native is faster despite more total MFMA,
  pointing toward operand staging, decay/address work, and pipeline fusion.

## Single Next Action

Do not retune BV, WG, MFMA shape, allocator/RA, or `block_dot_bf16_f32` again.
Audit the BV32 direct-K64 dataflow against the current Triton W=0 HSACO at the
operand level: BF16 V-decay staging/reuse, direct K block consumption, and
global/LDS address formation.  Preserve the exact current ABI and ownership.
Only after that audit should a same-source generic/specialized experiment be
proposed for a specific remaining staging or address mechanism.

The supporting report is
`compile_bug/qwen_mfma32_lowering_ladder/qwen_direct_k64_block_dot_bv32_ownership_report.md`.
