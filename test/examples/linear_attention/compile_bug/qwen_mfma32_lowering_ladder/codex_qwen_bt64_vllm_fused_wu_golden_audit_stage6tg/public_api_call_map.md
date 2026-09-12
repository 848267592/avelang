# Public API Call Map

- Stage 6S: `qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge`.
- F1: `qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager`.
- Native: `vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule`.

The benchmark runner forms lambdas around only these APIs; it never launches a private body as a timed callable.
