# Intended Storage Plan

The candidate must not hold a full 64x64 FP32 matrix in private memory.

| region | intended contents | bytes | lifetime |
|:--|:--|--:|:--|
| LDS input staging | only strict-lower 16x16 operands needed by current DAG level | <= 4096 | load through last consumer of the level |
| LDS block exchange | one or two published 16x16 FP32 X blocks | <= 4096 | producer barrier through last cross-wave consumer |
| VGPR/AGPR | current MFMA A/B fragments and 4-value C fragment per lane | register | one block product |

The maximum explicit LDS allocation is 8192 B.  The design must reuse the
same LDS regions between L1, L2, and L3; it may not materialize all ten output
blocks in LDS at once.  Final output is written directly once per matrix.

This is a frozen target plan, not a claim that the source candidate exists.
