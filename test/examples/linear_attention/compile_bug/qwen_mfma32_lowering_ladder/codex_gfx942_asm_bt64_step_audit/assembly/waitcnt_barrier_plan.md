# Planned BT64/BV32 Assembly Waitcnt And Barrier Plan

This plan is intentionally not applied to a nonexistent performance kernel.
Each future barrier must be justified by the named producer/consumer pair.

1. Wait `vmcnt` after W/state global loads before their LDS writes.
2. Barrier after the two-wave W/state LDS staging, before P16 MFMA reads.
3. Barrier after each wave writes its partial-pred tile, before cross-wave
   pred summation and U/decay epilogue.
4. Barrier after compact V-decay LDS writes, before update MFMA reads.
5. For each K16 window: wait after K global loads, barrier after K LDS writes,
   then consume immediately with MFMA16; do not keep multiple K fragments live.
6. Barrier only if the next window reuses the same LDS region.
7. Global state stores need no workgroup barrier after all consumers finish.

The smoke kernel uses no LDS and therefore no barrier; its result does not
validate any of these future dependencies.
