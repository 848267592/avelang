# Stage 5B Implementation Contract

## Scope and status

Stage 5B is an experimental, opt-in FP32 BT64 solve.  It may replace neither
v18 nor the Stage 4 pipeline unless every standalone gate passes.  The first
gate is source-level availability of the required FP32 MFMA16 intrinsic.

## Input and output

- Input `A`: contiguous CUDA/HIP FP32 tensor `[1, T, 8, 64]` with strides
  `(T * 8 * 64, 8 * 64, 64, 1)`.
- `T > 0` and `T % 64 == 0`.
- Output `X`: contiguous FP32 tensor with the same shape and strides.
- A program owns one `(chunk, value_head)` matrix.  The intended launch is
  `grid=(T / 64 * 8, 1, 1)`, `workgroup=(256, 1, 1)`.

For each local matrix, `A` is strictly lower triangular and the required
result is `X = (I + A)^-1`.  The diagonal of `A` is ignored; the diagonal of
`X` is exactly one.  No BF16 conversion is permitted in the solve body.

## Fixed 4x16 block algebra

Split the local 64x64 matrix into 16x16 blocks `Aij`, `i,j in [1,4]`.  Let
`D_i = I + Aii` and `Xii = inverse(D_i)`.  The dependency DAG is:

```text
L0: X11, X22, X33, X44
L1: X21 = -X22 A21 X11
    X32 = -X33 A32 X22
    X43 = -X44 A43 X33
L2: X31 = -X33 (A31 X11 + A32 X21)
    X42 = -X44 (A42 X22 + A43 X32)
L3: X41 = -X44 (A41 X11 + A42 X21 + A43 X31)
```

The signs agree with the frozen Stage 5A contract: v18 first stages `-A` and
solves `(I + A) X = I`.

## Numerical and resource contract

- Diagonal inverses use an FP32 16-row strict-lower recurrence.
- Every off-diagonal product uses FP32 `16x16x4` MFMA, accumulated in FP32.
- The intended ISA evidence is `v_mfma_f32_16x16x4_f32`.
- The intended explicit LDS budget is at most 8192 B, with lifetime reuse
  between dependency levels.
- Scratch, private segment, VGPR spills, and SGPR spills must all be zero.
- This is one solve dispatch, not a sequence of per-block kernels.

The required standalone T=2048 body-only target is at most 0.060 ms.  No full
pipeline integration is allowed before correctness, residual, ISA, resource,
consumer, and performance gates pass.
