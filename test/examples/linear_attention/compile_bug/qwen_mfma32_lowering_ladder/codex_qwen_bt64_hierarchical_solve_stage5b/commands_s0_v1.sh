#!/bin/sh
set -eu

cd /workspace/project/avelang
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention/vllm_compare

python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_solve_bt64_hierarchical_fp32_v1.py \
  -s --tb=short

python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_solve_bt64_hierarchical_fp32_v1.py \
  --T 512 1024 2048 --warmup 5 --repeat 20 \
  --json-out /tmp/qwen_bt64_s0_benchmark.json

python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_solve_bt64_hierarchical_fp32_v1.py \
  --T 64 --warmup 1 --repeat 1 \
  --dump-hsaco-dir test/examples/linear_attention/rocprof_outputs/qwen_bt64_hierarchical_s0/isa

/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex _qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1 \
  -d test/examples/linear_attention/rocprof_outputs/qwen_bt64_hierarchical_s0/counters \
  -o s0 -f csv -- \
  python3 test/examples/linear_attention/vllm_compare/profile_qwen_gdn_solve_bt64_hierarchical_fp32_v1.py \
    --T 2048 --warmup 2 --repeat 5
