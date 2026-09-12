# Stage 6B Final Summary

O0 is correct and reduces the actual repeated ownership work, but it is not a
promotion candidate. At T=2048 its V64 CTA ownership reaches the vLLM CTA
count (`512`), preserves zero scratch/spills and is bit-exact to Stage 4
FP32 staging. It lowers dynamic MFMA from `458752` to `188416`, VMEM from
`851968` to `335872`, and LDS instructions from `1343488` to `520192`.

The corresponding body latency changes from `76.073 us` to `62.012 us`
(`1.227x`), whereas the mandatory threshold is `>=1.5x`. Its vLLM gap remains
`32.127 us` at T=2048 and `0.775 us/chunk`, both above their gates. O1 V32 is
a negative result. Therefore no selected variant is integrated into full
graph replay; all full Stage 6B performance rows are intentionally N/A.

Keep FP32 output staging and the separate cast unchanged. Do not make direct
BF16 output a follow-up to this failed ownership gate. No evidence calls for
compiler or assembly work: the limiting O0 trade-off is source-level score
cache LDS/AccVGPR/occupancy pressure.
