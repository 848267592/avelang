# Frozen Graph Difference

The structural comparison after replacing the solve symbol/workgroup is `true`.

- GRAPH-A solve: `_qwen_gdn_solve_kernel_v18_parallel`, WG `[128, 1, 1]`
- GRAPH-B solve: `_qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1`, WG `[256, 1, 1]`
- exact common output pointer for this audit process: `0x7f7e26e00000`
- no copy, fill, or allocation occurs between solve and W/U.
