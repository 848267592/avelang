# Qwen GDN Next Decision After Qwen-Shaped Lowering Ladder

## Summary

Independent sequential MFMA lifetime was not the generic issue in `mfma_region_lifetime_v3`; this Qwen-shaped ladder attributes resource growth to concrete source features in a v29-like BT64/BV32 MFMA32 schedule.

Biggest weighted transition observed: `L4 -> L5` (k staging/shared view cost).

Manual conclusion: the issue is not a single clean compiler bug. The first large runtime jump is K staging/update shared layout (`L4 -> L5`), while the largest AccVGPR jump is first dependent update MFMA (`L5 -> L6`). Two small source workarounds did not help:

- `L5_alt_token_major_k_stage` reduced VALU/VGPR but increased trace.
- `L9_alt_grouped_v4_materialization` increased trace and counters.

So the current v29/BT64/MFMA32 source schedule is broadly too heavy unless a more targeted helper for K staging/update layout or dependent-update accumulator pressure is found.

## Decision Questions

1. Since independent sequential MFMA lifetime is not the issue, what Qwen-shaped pattern is the issue?

   - Current evidence points first to `L4 -> L5`: K staging/update shared layout cost.
   - A second important signal is `L5 -> L6`: first dependent update MFMA adds `+192` AccVGPR.

2. Is there evidence for a small compiler/helper fix?

   - Possible, but not proven. The likely helper area is shared K staging/update layout or dependent update MFMA lowering, not global-store vectorization.
   - The tried grouped-v4 global materialization workaround did not help.

3. Which source pattern should be fixed first?

   - Start with the source pattern added by `L4 -> L5`: `k_all_t[128,BT]` staging, `kall_vec` view, and transposed shared indexing for update.
   - Then inspect `L5 -> L6` dependent update MFMA AccVGPR growth.

4. Should we continue v29/BT64/MFMA32 after this?

   - Not as a source-only patch series right now. Continue v29 only if a focused helper/lowering change for K staging/update pressure shows improvement in the ladder first.
   - Otherwise v24 remains the production path.

5. Exact next action:

   - Build one focused experiment for `L4 -> L5`: an update-friendly K staging helper/layout that keeps the transposed access benefits but reduces address-generation and shared-memory overhead.
   - Gate any return to full Qwen on improvement in the isolated ladder.
