# C0 Mathematics

For every BT64 chunk and value head:

```text
A_w[t,s] = bf16(fp32(A_bf16[t,s]) * beta[s] * exp(g[s]))
A_u[t,s] = bf16(fp32(A_bf16[t,s]) * beta[s])
W = bf16(A_w @ K_bf16)
U = bf16(A_u @ V_bf16)
```

This is exactly the BF16-solved main path. The FP32-solved low residual and
both residual MFMA phases from F1 are absent.
