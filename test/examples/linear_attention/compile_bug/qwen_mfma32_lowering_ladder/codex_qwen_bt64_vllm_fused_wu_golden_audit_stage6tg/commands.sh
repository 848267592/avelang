#!/bin/sh
set -eu
PYTHONPYCACHEPREFIX=/tmp/pycache_stage6tg python3 -m py_compile test/examples/linear_attention/vllm_compare/audit_qwen_gdn_bt64_vllm_wu_stage6tg.py test/examples/linear_attention/vllm_compare/generate_qwen_gdn_bt64_vllm_wu_stage6tg_artifacts.py test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_vllm_wu_stage6tg_audit.py
PYTHONDONTWRITEBYTECODE=1 python3 test/examples/linear_attention/vllm_compare/audit_qwen_gdn_bt64_vllm_wu_stage6tg.py --mode benchmark --T 512 2048 8192 16384 --sessions 5 --warmup 30 --repeat 200 --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_vllm_fused_wu_golden_audit_stage6tg
PYTHONDONTWRITEBYTECODE=1 python3 test/examples/linear_attention/vllm_compare/audit_qwen_gdn_bt64_vllm_wu_stage6tg.py --mode correctness --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_vllm_fused_wu_golden_audit_stage6tg
python3 test/examples/linear_attention/vllm_compare/generate_qwen_gdn_bt64_vllm_wu_stage6tg_artifacts.py
