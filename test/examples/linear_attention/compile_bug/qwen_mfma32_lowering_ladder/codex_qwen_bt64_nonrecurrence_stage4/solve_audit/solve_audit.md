# BT64 Solve Audit

The current candidate uses v18's 128-thread shared-memory triangular recurrence.
It preserves the v6 FP32 layout and remains effectively exact versus the FP32
oracle (`2.98e-8` maximum absolute difference in this audit).

| T | v18 FP32 ms | vLLM FP32 ms | ratio | vLLM BF16 ms | BF16 max abs |
|--:|--:|--:|--:|--:|--:|
| 512 | 0.121140 | 0.053359 | 2.270x | 0.053720 | 0.0004882 |
| 2048 | 0.126888 | 0.053779 | 2.359x | 0.052959 | 0.0006309 |

The >=1.5x audit gate is met, but the vLLM schedule is a hierarchical block
inverse/dot construction rather than a local edit to v18's row recurrence.
Implementing it while preserving the frozen FP32 contract needs its own
correctness and machine-resource gate. Stage 4 already meets its full target,
so solve was not modified in this pass.

`solve_not_modified_reason`: a safe improvement requires a new hierarchical
solve design; a rushed change would expand the numerical and schedule risk
after KKT/W-U/chunk-o already passed all Stage 4 gates.
