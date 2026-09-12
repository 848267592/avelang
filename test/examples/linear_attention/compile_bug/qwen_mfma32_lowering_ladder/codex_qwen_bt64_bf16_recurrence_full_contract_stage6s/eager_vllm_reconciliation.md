# Native vLLM Eager vs CUDA Graph Reconciliation

At T=2048 the Stage 6S Graph-C timing is 0.188800 ms. This is CUDA Graph replay timing after JIT, module load, allocation and capture, and is used only because Graph A, B and C can all be preallocated by capture.

The older approximately 0.364 ms vLLM results used direct eager public calls. The original Stage-2 eager harness was rerun in the current container, with the same public vLLM entry, BT64 input contract and current autotune patch:

| timing mode | warmup | repeat | median ms | p10 ms | p90 ms |
|:--|--:|--:|--:|--:|--:|
| direct eager HIP-event | 20 | 100 | 0.360936 | 0.353244 | 0.373635 |
| Stage6S CUDA Graph replay | 20 | 100 x 5 sessions | 0.188800 | 0.188160 | 0.189602 |

Both results are expected. The direct eager interval contains host launch and queue gaps between device kernels after the start event is recorded. A captured graph replay removes those gaps by enqueuing the fixed graph as a replay. Do not compare the two absolute series as if they measured the same end-to-end boundary.
