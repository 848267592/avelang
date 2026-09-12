# Primitive Reuse Audit

## Available and audited

- The v18 FP32 BT64 wrapper and its `[1,T,8,64]` contract are available in
  `vllm_compare/qwen_gdn_chunked_avelang_v18_bt64_layout_fixed.py`.
- Stage 5A captures the installed vLLM 4x16 source and its emitted
  `v_mfma_f32_16x16x4_f32` ISA.
- AveLang exposes BF16/F16 `16x16x16` and BF16 `32x32x8` MFMA high-level
  operations.  These cannot substitute for an all-FP32 solve.

## Required but absent in the current source tree

`lib/IR/Intrinsics/amdgpu_mfma_signatures.h` is the authoritative high-level
MFMA registry used by `AMDGPUIntrinsic::Initialize`.  Its current entries are
only:

```text
mfma_16x16x16_f16_f32
mfma_16x16x16_bf16_f32
mfma_f32_16x16x16_bf16
mfma_32x32x8_bf16_f32
mfma_f32_32x32x8_bf16
```

There is no FP32 `16x16x4` entry, no matching textual intrinsic wrapper in
`amdgpu_intrinsics.mlir`, and no public AveLang call site.  The Stage 5B
availability repro checks plausible source names in the live JIT; it is the
final authority for whether an already-built environment contains an
untracked extension.

## Consequence

The requested FP32-only hierarchical candidate cannot legally be replaced by
BF16 MFMA, inline assembly, or a compiler patch.  Until the availability
repro succeeds, no S0 solve, benchmark, rocprof success path, or full-pipeline
integration may be claimed.
