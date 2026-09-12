# Failure Classification

## Classification: E - required high-level FP32 MFMA is unavailable

This is not a numerical, lane-mapping, LDS, resource, or performance failure.
The candidate cannot be emitted because the required operation does not exist
in the current AveLang source registry or live JIT environment.

Live MI300 attempts:

| attempted source symbol | result |
|:--|:--|
| `al.amdgpu.mfma_16x16x4_f32_f32` | `Symbol not found` |
| `al.amdgpu.mfma_f32_16x16x4_f32` | `Symbol not found` |
| `al.amdgpu.mfma_16x16x4_f32` | `Symbol not found` |

The exact diagnostics are saved in
`standalone/fp32_mfma16_feature_gate.stdout_stderr.txt`; structured results
are in `standalone/fp32_mfma16_feature_gate.json`.

## Consequence

`full_pipeline_integrated=false`.  There is no S0 ISA, rocprof trace,
correctness, residual, W/U consumer result, or performance result.  Reusing
BF16 MFMA would violate the frozen all-FP32 solve contract, and adding an
intrinsic/lowering would violate this Stage 5B request's compiler-change ban.
