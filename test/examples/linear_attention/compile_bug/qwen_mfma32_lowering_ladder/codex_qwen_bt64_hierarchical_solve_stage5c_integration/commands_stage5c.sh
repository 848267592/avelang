#!/usr/bin/env sh
set -eu

cd /workspace/project/avelang

PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_hierarchical_solve_stage5c.py \
  -s --tb=short

# Default selector regression: this must keep using v18 and remain unchanged.
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_nonrecurrence_mfma_v2.py \
  -s --tb=short

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_hierarchical_solve_stage5c.py \
  --T 512 2048 --warmup 10 --repeat 50 \
  --json-out /tmp/stage5c_benchmark.json

/opt/rocm/bin/rocprofv3 \
  --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1 \
  -d /tmp/stage5c_rocprof -o stage5c_solve -f csv -- \
  env PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/profile_qwen_gdn_solve_bt64_hierarchical_fp32_v1.py \
  --T 2048 --warmup 2 --repeat 5
