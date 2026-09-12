# Native BT64 W/U Contract

Inputs are contiguous device tensors with `k:[1,T,4,128]` BF16,
`v:[1,T,8,128]` BF16, `g/beta:[1,T,8]` FP32, and
`a_solved:[1,T,8,64]` FP32.  `T` is a positive multiple of 64.  For value head
`h`, key head is `h // 2`; source and destination tokens are in the same
64-token chunk.

For every local destination token `t`, source token `s`, and output column
`d`, native W/U compute:

```text
W[t,h,d] = sum_s a_solved[t,h,s] * beta[s,h] * exp(g[s,h]) * K[s,h//2,d]
U[t,h,d] = sum_s a_solved[t,h,s] * beta[s,h] * V[s,h,d]
```

The four 16-wide source reductions use BF16 staged coefficients and BF16 K/V,
then a FP32 correction restores the scalar FP32 coefficient contribution.
Outputs are contiguous FP32 `[1,T,8,128]`, exactly the frozen asm-v0 ABI.

