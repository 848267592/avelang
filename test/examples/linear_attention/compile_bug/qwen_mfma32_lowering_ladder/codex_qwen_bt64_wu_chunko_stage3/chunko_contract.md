# Native BT64 chunk-o Contract

Inputs are contiguous `q/k:[1,T,4,128]` BF16, `v_new:[1,T,8,128]` FP32,
`h_bf16:[1,T/64,8,128,128]` BF16 from the immutable recurrence, and
`g:[1,T,8]` FP32.  The output is FP32 `[1,T,8,128]`; the full wrapper alone
converts it to public BF16.

For output token `t` and value head `h`, each 16-token output tile computes:

```text
inter = exp(g[t,h]) * (scale * Q[t,h//2,:]) @ H[chunk,h,:,:]^T
intra = sum_{s <= t within the chunk}
          ((scale * Q[t,h//2,:]) @ K[s,h//2,:])
          * exp(g[t,h] - g[s,h]) * V_new[s,h,:]
out = inter + intra
```

The source loop covers all four BT64 token-16 subtiles.  The absolute causal
predicate is applied before score BF16 staging.  This preserves contribution
from early source tiles to later output tiles.

