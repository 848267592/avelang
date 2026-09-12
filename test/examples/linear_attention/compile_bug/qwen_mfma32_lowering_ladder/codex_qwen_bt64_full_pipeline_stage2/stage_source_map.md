# Stage Source Map

| Logical stage | vLLM implementation | Candidate implementation | Contract note |
|:--|:--|:--|:--|
| gate preprocessing | `ops/cumsum.py:chunk_local_cumsum` | `qwen_gdn_chunk_cumsum_avelang_v6_standalone` | BT64 FP32 local cumsum |
| KKT | `ops/chunk_scaled_dot_kkt.py:chunk_scaled_dot_kkt_fwd` | `qwen_gdn_kkt_avelang_v6_standalone` | FP32 `[1,T,8,64]` |
| triangular solve | `ops/solve_tril.py:solve_tril` | `qwen_gdn_solve_avelang_v18_bt64_layout` | candidate remains FP32; vLLM writes BF16 |
| W/U | `ops/wy_fast.py:recompute_w_u_fwd` | `qwen_gdn_w_u_avelang_v6_standalone` | candidate FP32 meets asm ABI |
| recurrence | `ops/chunk_delta_h.py:chunk_gated_delta_rule_fwd_h` | frozen `qwen_gdn_bt64_gfx942_asm_v0` | fixed gfx942 external HSACO |
| output | `ops/chunk_o.py:chunk_fwd_o` | `qwen_gdn_chunk_o_bt64_avelang_experimental` | generic v6 BF16 q/k path, BT64 not v24 BT16 |
| public adapter | `ops/chunk.py:chunk_gated_delta_rule` | `qwen_gdn_full_bt64_gfx942_asm_v0` | BF16 output, optional FP32 final state |

