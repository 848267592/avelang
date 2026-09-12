# Stage 6C Decision

**No-Go for Stage 6C direct BF16 output.** O0 is correct and structurally
better than Stage 4, but it misses every mandatory body performance threshold:

| gate | target | O0 measured |
|---|---:|---:|
| T2048 speedup vs current | >= 1.50x | 1.227x |
| T2048 A-vLLM body gap | <= 25 us | 32.127 us |
| gap slope | <= 0.65 us/chunk | 0.775 us/chunk |

O1 is a negative control: it regresses at T2048 and long T. Direct BF16
output/cast removal would change a second structural variable and could hide
this ownership failure, so it is explicitly not ready. No compiler or
assembly action is supported by this result: all variants have zero scratch
and zero spills; the limiting trade-off is source-level LDS/AGPR/occupancy.

The single recommendation is to close this two-variant ownership experiment
without promotion. Re-rank the remaining Stage 6A body gaps before choosing a
new, separately scoped Stage 6B action; do not continue O1-style V-tile
sweeps or mix in output-cast fusion.
