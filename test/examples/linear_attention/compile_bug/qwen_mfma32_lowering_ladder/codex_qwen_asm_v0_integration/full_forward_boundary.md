# Full-Forward Boundary Decision

The raw asm-v0 recurrence gate is intentionally separate from v24 production.
The source contract audit found no safe direct full-forward integration in this
stage:

1. v24 uses BT16 for cumsum, KKT, solve, w/u, chunk_gdr, and chunk_o.
2. asm v0 requires BT64 and materializes `h` as BF16.
3. its pred is XF32 while the historical Avelang BT64 experiments use a
   BF16-staged pred.
4. v24 chunk_o expects the BT16 chunk layout; simply widening asm-v0's BF16
   h would not repair the changed recurrence chunk boundaries.

Consequently, the only correct current boundary is the explicit raw FLA
stage API.  A v24 full-forward call remains on its existing production
fallback and is not silently redirected to asm v0.  A BT64 full-forward
integration needs a separately validated upstream/downstream contract before
it can be profiled.
