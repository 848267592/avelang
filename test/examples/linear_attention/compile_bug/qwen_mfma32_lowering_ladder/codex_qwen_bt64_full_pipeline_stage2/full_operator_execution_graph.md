# Full Qwen GDN Execution Graph

The authoritative full-forward entry is
`vllm.model_executor.layers.fla.ops.chunk.chunk_gated_delta_rule` in the
installed ROCm environment.  Its implementation calls
`ChunkGatedDeltaRuleFunction.forward`, which calls
`chunk_gated_delta_rule_fwd` in the same file.

```text
q/k/v BF16, g/beta FP32, optional initial_state FP32
  |
  +-- chunk_local_cumsum(g, chunk_size=64) -> g_cumsum FP32
  +-- chunk_scaled_dot_kkt_fwd(k, beta, g_cumsum) -> A FP32 [B,T,Hv,64]
  +-- solve_tril(A, output_dtype=k.dtype) -> A_solved BF16 [B,T,Hv,64]
  +-- recompute_w_u_fwd(k,v,beta,A_solved,g_cumsum) -> w/u BF16
  +-- chunk_gated_delta_rule_fwd_h(k,w,u,g_cumsum,h0) -> h BF16,
  |      v_new BF16, final_state FP32
  +-- chunk_fwd_o(q,k,v_new,h,g_cumsum,scale) -> o BF16
  `-- public `(o.to(q.dtype), final_state)`
```

The experimental path preserves the same high-level graph but uses the
following strictly opt-in substitutions:

```text
Avelang v6 cumsum/KKT -> v18 BT64 solve -> Avelang v6 w/u (FP32)
  -> frozen qwen_gdn_bt64_gfx942_asm_v0 (BF16 h, FP32 v_new/final_state)
  -> Avelang v6 generic BT64 chunk_o -> BF16 public output
```

No candidate module imports or invokes `chunk_gated_delta_rule`.

