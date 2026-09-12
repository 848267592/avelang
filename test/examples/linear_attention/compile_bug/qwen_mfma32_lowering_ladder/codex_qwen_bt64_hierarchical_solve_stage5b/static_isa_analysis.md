# Stage 5B Static ISA Analysis

No Stage 5B solve HSACO exists.  The live MI300 JIT gate rejected all tested
high-level FP32 `16x16x4` names before lowering, so an ISA audit of S0 would
be fabricated and was not attempted.

The source audit and live JIT agree:

- `amdgpu_mfma_signatures.h` has no FP32 `16x16x4` configuration.
- `amdgpu_intrinsics.mlir` has no ROCDL FP32 `16x16x4` wrapper.
- The live JIT reports `Symbol not found` for all three plausible names; see
  `standalone/fp32_mfma16_feature_gate.stdout_stderr.txt`.

Stage 5A independently retains vLLM ISA evidence for
`v_mfma_f32_16x16x4_f32`.  That proves the gfx942 instruction is appropriate,
not that AveLang currently exposes it as source-level functionality.
