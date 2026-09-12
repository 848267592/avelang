# Historical Source Map

- Stage 4 source: `vllm_compare/qwen_gdn_bt64_nonrecurrence_mfma_v2.py`
- Frozen full bridge: `vllm_compare/qwen_gdn_full_bt64_gfx942_asm_v0_experimental.py`
- Frozen recurrence bridge: `vllm_compare/qwen_gdn_bt64_gfx942_asm_v0_experimental.py`
- Stage 3 W/U and chunk-o: `vllm_compare/qwen_gdn_bt64_native_wu_chunko_mfma_v1.py`
- v24 BT16 KKT/full: `vllm_compare/qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_fixed.py`
- v14 validated W/U primitive: `vllm_compare/qwen_gdn_chunked_avelang_v14_mfma_layout_fixed.py`
- v18 parallel solve: `vllm_compare/qwen_gdn_chunked_avelang_v18_bt64_layout_fixed.py`
- v20 BT32 native experiment: `vllm_compare/qwen_gdn_chunked_avelang_v20_bt32_native_mfma_layout_fixed.py`
- Generic contract/oracle stages: `vllm_compare/qwen_gdn_chunked_avelang_v6_vllm_layout_fixed.py`

The Stage 4 public entry is `qwen_gdn_full_bt64_stage4_all_s0`. Despite the
historical `all_s0` suffix, the final graph uses KKT-S0, residual-MFMA WU-S1,
and chunk-o-S0.
