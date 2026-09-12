# Direct-Out Wrapper Audit

The audit-only source is
`vllm_compare/qwen_gdn_bt64_solve_direct_out_stage5e_audit.py`.

It exports:

- `qwen_gdn_solve_v18_bt64_direct_out_audit(a, out)`;
- `qwen_gdn_solve_hierarchical_fp32_v1_direct_out_audit(a, out)`.

Each wrapper validates FP32 CUDA/HIP contiguous `[1,T,8,64]`, same device and
shape, T divisibility, distinct storage, and 16-byte output alignment. It then
passes `out` directly as the original kernel's `out_ptr` argument. It performs
no output allocation, copy, fill, fallback, synchronization, or math change.

The launches remain:

- v18: grid `T/64 * 8`, workgroup 128;
- hierarchical v1: grid `T/64 * 8`, workgroup 256.

The file is not imported by Stage 4/5C production modules.

