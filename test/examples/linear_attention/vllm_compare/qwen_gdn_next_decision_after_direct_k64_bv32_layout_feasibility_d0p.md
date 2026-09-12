# Direct-K64 BV32 D0-P Next Decision

## Decision

Stop the local LDS-layout route.  D0-P proved that public AveLang source can
emit a compact register-transpose chain, but the full direct-K64 BV32 update
arm fails the mandatory T=64 semantic check.  Do not profile, benchmark,
promote, or integrate that arm.

## What Passed

- Triton audit found `amd_rotating_shared`, `swizzled_shared`,
  `amdg.in_thread_transpose`, typed `dot_op`, MFMA32 and `v_perm_b32`.
- AveLang source emitted `buffer_load_dwordx4 -> ds_bpermute_b32 ->
  v_perm_b32 -> ds_write_b128 -> ds_read_b128 -> MFMA32` in the minimal
  8x8 register-transpose probe.
- The same source route works after BF16 V-new is scaled in FP32.

## What Failed

- Token-major packed LDS to non-contiguous typed MFMA fragment still fails
  LLVM translation with `builtin.unrealized_conversion_cast`.
- The executable full register-transpose update has non-finite H values at
  T=64 and final-state max error `61.041603`.

## Next Action

Do not add more LDS swizzles or ping-pong buffers.  Move to a matched fused
recurrence-pipeline design.  A future layout revisit requires a first-class
typed operand/fragment representation with a lowering-time lane-to-LDS and
lane-to-fragment map; it must not be another source-only swizzle sweep.
