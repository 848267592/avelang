# Public API Contract

- Stage 6S: `qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(q, k, v, g, beta, *, initial_state=None, scale=None, output_final_state=False)`.
- F0: `qwen_gdn_full_bt64_stage6t_fused_wu_fp32_eager(q, k, v, g, beta, *, initial_state=None, scale=None, output_final_state=False)`.
- F1: `qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager(q, k, v, g, beta, *, initial_state=None, scale=None, output_final_state=False)`.
- Native reference: `vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule(q, k, v, g, beta, *, initial_state, output_final_state, scale, head_first=False, use_qk_l2norm_in_kernel=False)`.

所有 API 接收相同 `[1,T,H,D]` BF16 `q/k/v`、FP32 `g/beta/initial_state`，并在 current stream 返回 BF16 public output 和可选 FP32 final state。完整 public call 内部的输出与中间张量 allocation、cast、dispatch 与返回对象构造均计时。F0/F1 是 opt-in experimental-only；没有 selector 或 fallback 改动。
