# Qwen GDN Next Decision After Stage 6Y Updated Gap Audit

## Decision

Stage 6X X2 remains frozen as the current Avelang BT64 experimental baseline.
Stage 6Y does not modify any kernel. Its current-X2 Eager body diagnostic and
final replay graph identify one dominant, recoverable gap:

```text
Stage 6Z: native BT64 chunk-o ownership redesign
```

This is a single downstream experiment. It must not modify the recurrence
HSACO, KKT+solve X2, W/U, compiler, default selector, or the public contract.

## Evidence

At T=16384, wrapper-level chunk-o is 0.481958 ms for X2 and 0.153148 ms for
native vLLM. Its fitted slopes are 1.713 versus 0.427 us/chunk, leaving a
1.286 us/chunk X2 deficit. That is larger than the W/U deficit of 0.257
us/chunk. Cumsum is tied, recurrence uses the identical HSACO symbol and has
no standalone kernel-level deficit, and X2 KKT+solve has a smaller slope gap
than chunk-o.

The final T=2048 trace confirms a structural ownership mismatch:

| chunk-o | workgroup | global grid | workgroups | LDS | VGPR / AccVGPR |
|:--|--:|:--|--:|--:|:--|
| X2 Stage 6W body | 256 | 524288 x 1 x 1 | 2048 | 27136 B | 112 / 64 |
| native vLLM | 256 | 512 x 32 x 8 | 512 | 0 B | 100 / 36 |

This is a 4x CTA-count difference at the same workgroup size, combined with
an X2-only 27 KiB LDS tile and a 3.15x body latency at T=16384. It is enough
to prioritize chunk-o, but it does not prescribe an implementation before a
new X0 ownership/source audit.

## Stage 6Z Guardrails

1. Start with a source/trace ownership audit of native `chunk_fwd_kernel_o`:
   its program-id decomposition, vector/tile ownership, BF16 input/output
   formats, and how it covers the 64-token chunk.
2. Implement only a minimal isolated BT64 chunk-o ownership prototype. Do not
   fuse recurrence, change BF16 boundaries, or touch X2/WU.
3. Require output bit-exactness to Stage 6X where the existing BF16 contract
   permits it, then run full contract tests before a full Eager benchmark.
4. Reject a prototype that regains work by scratch/spill, a major VGPR/AccVGPR
   cliff, or a worse long-text slope even if its T=2048 body looks good.
5. Promote only after X2-vs-6Z-vLLM Eager public API comparisons at
   T=512--16384, using the existing paired Williams protocol.

W/U is the documented runner-up. It should not be changed in parallel; its
smaller body-slope deficit can be revisited only if the chunk-o ownership gate
fails.
