# Stage 6T Decision

## Decision: Case A

Keep Graph B as an **opt-in experimental full-graph candidate**. It passes the full correctness contract, retains the exact current-vLLM recurrence bridge identity, has zero recurrence scratch, is faster at T=2048 by `26.647 us` with a paired 95% interval `[26.595, 26.700] us`, and improves at every longer measured length. Its gap slope versus native vLLM drops from `4.396726` to `3.284716 us/chunk`.

It is not a production promotion: Graph B still materializes FP32 W/U and FP32 V-new for the unchanged surrounding kernels, and introduces three explicit conversion dispatches. The result is evidence for the next action, not permission to silently replace the historical default.

## Single Next Action

Propagate the BF16 storage boundary end to end: make W/U producers natively write the recurrence's BF16 contract and make chunk-o natively consume BF16 V-new, then remove the three explicit casts one at a time with the same correctness and full-graph contract. Do not modify the recurrence HSACO, compiler, asm-v0, KKT, hierarchical solve, output staging, or final public cast as part of that next action.
