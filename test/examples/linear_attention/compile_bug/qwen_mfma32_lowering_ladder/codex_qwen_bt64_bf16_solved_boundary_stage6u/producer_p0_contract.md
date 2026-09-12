# P0 Producer Contract

P0 is `_qwen_gdn_solve_hierarchical_bt64_fp32_compute_bf16_store_stage6u`,
launched by `qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u`.

Input, local/shared values, recurrence, block DAG and MFMA remain FP32. Only
the global output pointer and final stores are BF16. The output layout remains
contiguous `[1,T,8,64]`; diagonal identity, strict lower values, strict upper
zeros and unsupported-shape rejection are unchanged. P0 has no fallback.
