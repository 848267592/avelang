# Stage 5A Solve Source Call Graph

## Avelang Stage 4 path

```text
qwen_gdn_full_bt64_stage4_all_s0_stages
  [vllm_compare/qwen_gdn_bt64_nonrecurrence_mfma_v2.py:757]
  -> qwen_gdn_kkt_bt64_mfma_v2_s0
  -> qwen_gdn_solve_avelang_v18_bt64_layout
       [qwen_gdn_chunked_avelang_v18_bt64_layout_fixed.py:168]
       -> qwen_gdn_solve_avelang_v18_layout
            [:140]
            -> _qwen_gdn_solve_kernel_v18_parallel
                 [:36], grid=(num_chunks * 8, 1, 1), workgroup=(128, 1, 1)
  -> qwen_gdn_w_u_bt64_mfma_v2_s1
  -> frozen asm recurrence
  -> qwen_gdn_chunk_o_bt64_mfma_v2_s0
```

For the fixed target, `a` and `a_solved` both have contiguous FP32 layout
`[1, T, 8, 64]`, strides `[T*512, 512, 64, 1]`.  One CTA owns one
`(chunk_idx, value_head_idx)` matrix.  The wrapper allocates `torch.empty_like(a)`;
the body-only harness bypasses only that allocation and launches exactly the
same JIT kernel with a preallocated `out`.

## vLLM/FLA path

```text
chunk_gated_delta_rule source graph
  -> solve_tril(A, output_dtype=torch.float32)
       [/opt/venv/.../fla/ops/solve_tril.py:506]
       -> torch.zeros_like(A, dtype=FP32)                         [:537]
       -> merge_16x16_to_64x64_inverse_kernel[NT, B * H]          [:542-555]
  -> recompute W/U consumer in the FLA graph
```

The installed source is preserved at
`ir/vllm/solve_tril_vllm_installed.py`.  The normal BT64 configuration was
captured before profiling: `num_warps=4`, `num_stages=5`, `num_ctas=1`,
`DOT_PRECISION=ieee`, `USE_TMA=false`.  One Triton program owns one
`(chunk_idx, batch*head_idx)` 64x64 matrix.  In the body-only benchmark,
the required zeroing of unwritten upper-triangular output is performed before
each HIP event; it is not part of the timed solve body.

## Dispatch correspondence

Both public solve wrappers issue one solve dispatch for BT64.  At T=2048,
both launch 256 CTAs: `32 chunks * 8 heads`.  rocprof reports global work
items, hence Avelang `32768 / 128 = 256` and vLLM `8192 / 256 = 32` in X
times `8` in Y, also 256 CTAs.  The comparison is therefore not conflating a
multi-kernel vLLM solve with a single-kernel Avelang solve.

## Ownership that can be established from source and profiling

| implementation | CTA ownership | threads/waves | source-level lane responsibility |
|:--|:--|:--|:--|
| v18 | one 64x64 chunk/head inverse | 128 / 2 | `group_idx=tid//64` partitions the reduction for each output column; group 0 completes the sum. |
| vLLM | one 64x64 chunk/head inverse | 256 / 4 | Triton owns four 16x16 diagonal tiles and six lower off-diagonal tiles. Exact per-lane decomposition is compiler-generated and not inferred beyond the captured 4-wave launch. |

The final sentence is deliberate: Triton source and ISA prove the tile
algorithm and wave count, but do not uniquely document one semantic row per
hardware lane.
