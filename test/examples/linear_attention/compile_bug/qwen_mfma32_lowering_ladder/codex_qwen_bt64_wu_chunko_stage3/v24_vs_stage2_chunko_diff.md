# v24 Versus Stage 2 chunk-o

v24 chunk-o is a 64-lane 16-token by 16-value-tile MFMA kernel.  It computes
the entrance-state term `Q @ H^T`, then causal score/value contributions using
`Q @ K^T` and `score @ V_new`.  It stages Q, K, H, score, and V-new in LDS and
uses BF16 MFMA operands with FP32 accumulators.

Stage 2 used v6 generic chunk-o at BT64.  One thread computes one output value,
including the full K=128 dot product and the whole local chunk loop.  It has
no MFMA and needs 260 B scratch.

The Stage 3 port retains v24's 16x16 MFMA math, but loops source token tiles
`0:16`, `16:32`, `32:48`, and `48:64`.  The causal predicate uses absolute
positions, so a later output tile sees earlier source tiles while future ones
are zeroed.  H remains directly BF16 from asm instead of Stage 2's temporary
BF16-to-FP32 materialization.

