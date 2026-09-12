#!/usr/bin/env bash
set -euo pipefail
python3 test/examples/linear_attention/vllm_compare/assert_eager_public_api_contract.py test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_fused_wu_stage6t_eager_public.py
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_fused_wu_eager_stage6t.py -s
PYTHONDONTWRITEBYTECODE=1 python3 test/examples/linear_attention/vllm_compare/run_qwen_gdn_bt64_fused_wu_stage6t_eager_correctness.py
PYTHONDONTWRITEBYTECODE=1 python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_fused_wu_stage6t_eager_public.py --T 512 1024 2048 4096 8192 16384 --sessions 5 --warmup 30 --repeat 200
# rocprof commands are diagnostic only; start from the public full API runner.
