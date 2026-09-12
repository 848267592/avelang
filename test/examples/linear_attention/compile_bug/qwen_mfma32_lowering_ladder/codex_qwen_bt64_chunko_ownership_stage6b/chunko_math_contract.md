# Chunk-O Math and Dtype Contract

For each chunk, value head, token `t`, and value column `v`, Stage 6B preserves
the frozen Stage 4 operation:

```text
inter[t,v] = scale * exp(g[t]) * sum_k q[t,k] * h[v,k]
score[t,s] = scale * exp(g[t] - g[s]) * sum_k q[t,k] * k[s,k], s <= t
intra[t,v] = sum_{s<=t} score[t,s] * v_new[s,v]
output_fp32[t,v] = inter[t,v] + intra[t,v]
```

Inputs remain BF16 `q/k`, BF16 recurrence `h`, FP32 `g` and `v_new`; the
accumulators and output staging remain FP32. The public BF16 conversion is the
existing, separate cast kernel. No output dtype, scale, gate, cast placement,
or recurrence ABI changed.

O0/O1 are bit-exact to the Stage 4 FP32-staging output for all executed
Avelang-to-Avelang cases. Against vLLM `chunk_fwd_o` with the same direct
body inputs, Stage 4's maximum discrepancy was `1.1281809e-05`; O0/O1 inherit
that discrepancy because they are bit-exact to Stage 4.
