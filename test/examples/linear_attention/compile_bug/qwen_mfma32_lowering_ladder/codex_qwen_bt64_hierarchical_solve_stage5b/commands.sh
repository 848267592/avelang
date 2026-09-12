#!/bin/sh
set -eu

cd /workspace/project/avelang
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention/vllm_compare

python3 -m py_compile \
  test/examples/linear_attention/vllm_compare/qwen_gdn_solve_bt64_hierarchical_fp32_v1.py \
  test/examples/linear_attention/vllm_compare/repro_qwen_bt64_fp32_mfma16_feature_gate.py

python3 test/examples/linear_attention/vllm_compare/repro_qwen_bt64_fp32_mfma16_feature_gate.py \
  --output-json test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_hierarchical_solve_stage5b/standalone/fp32_mfma16_feature_gate.json
