# FP32 16x16x4 MFMA Intrinsic Enablement Contract

## Goal

Expose the existing ROCm/ROCDL FP32 MFMA instruction through one Avelang
source-level intrinsic:

```python
al.amdgpu.mfma_16x16x4_f32_f32(a, b, acc)
```

The target instruction is `v_mfma_f32_16x16x4_f32` on gfx942.

## Exact Scope

- Register one MFMA signature: `M=16`, `N=16`, `K=4`, `A/B=f32`, `C=f32`.
- Add one inline ROCDL wrapper around `rocdl.mfma.f32.16x16x4f32`.
- Add a frontend MLIR-generation regression test.
- Add a one-wave JIT correctness/ISA repro using all-one fragments.

Per wave, the ABI is:

| operand | Avelang type | elements per lane |
|:--|:--|--:|
| A | `Tensor((1,), f32)` | 1 |
| B | `Tensor((1,), f32)` | 1 |
| C/result | `Tensor((4,), f32)` | 4 |

The JIT repro obtains the one-element A/B fragment with a rank-two
`al.view(..., (64, 1))` followed by one index.  A rank-one `al.full((1,), x)`
looks vector-like in source but is currently represented as a scalar
expression by the frontend, so it is deliberately not used for A/B.

The wrapper extracts element zero before the ROCDL op because the public
Avelang ABI uses uniform vector fragments while LLVM declares the FP32
16x16x4 A/B operands as scalar `float` values.

## Explicit Non-Goals

- No LLVM/AMDGPU register-allocation changes.
- No hand-written assembly or HSACO modification.
- No change to v18, v23, v24, v29, or the Stage 4 production candidate.
- No hierarchical-solve S0 kernel or dispatch integration in this patch.

## Why FP32

The BT64 solve matrix and its solved output are FP32 today.  This intrinsic
preserves that mathematical contract for the first hierarchical-solve
experiment; it does not require public Q/K/V inputs to become FP32.  A later
mixed-precision solve experiment can be evaluated separately against the same
full-operator correctness contract.
