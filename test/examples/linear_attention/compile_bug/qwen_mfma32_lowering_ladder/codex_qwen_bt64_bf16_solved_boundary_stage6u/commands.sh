#!/bin/sh
set -eu

# All authoritative timing commands added by Stage 6U use complete eager public
# API calls. No CUDA/HIP Graph capture or replay is permitted.

# Existing source feature gate used before implementation.
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_fp32_mfma16_intrinsic_enablement.py \
  --warmup 1 --repeat 2 --output-json /tmp/stage6u_mfma_probe.json

# Producer matrix.
python3 test/examples/linear_attention/vllm_compare/run_qwen_gdn_stage6u_producer_correctness.py \
  --output-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_bf16_solved_boundary_stage6u

# Consumer matrix (diagnostic only).
python3 test/examples/linear_attention/vllm_compare/run_qwen_gdn_stage6u_consumer_correctness.py \
  --output test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_bf16_solved_boundary_stage6u/consumer_c0_correctness.csv \
  --T 64 512 2048

# Full eager correctness and expanded stability.
python3 test/examples/linear_attention/vllm_compare/run_qwen_gdn_bt64_bf16_solved_stage6u_correctness.py \
  --output-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_bf16_solved_boundary_stage6u

# Authoritative complete eager public benchmark. No Graph is used.
python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_bf16_solved_stage6u_eager_public.py \
  --T 512 1024 2048 4096 8192 16384 --sessions 5 --warmup 30 --repeat 200 \
  --output-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_bf16_solved_boundary_stage6u

# Diagnostic-only rocprof pattern, repeated for u1/f1 include regexes.
/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex fused_bf16_solved_stage6u \
  -d test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_bf16_solved_boundary_stage6u/rocprof/c0 \
  -o c0 -f csv -- python3 test/examples/linear_attention/vllm_compare/profile_qwen_gdn_bt64_bf16_solved_stage6u_public.py \
  --implementation u1 --T 2048 --warmup 2 --repeat 5

# Machine-readable table/decision generation.
python3 test/examples/linear_attention/vllm_compare/finalize_qwen_gdn_stage6u_artifacts.py

# Regression and hygiene.
python3 -m pytest -q test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_bf16_solved_boundary_stage6u.py -s
python3 -m py_compile \
  test/examples/linear_attention/vllm_compare/qwen_gdn_solve_bt64_hierarchical_bf16_stage6u.py \
  test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_bf16_solved_boundary_stage6u.py \
  test/examples/linear_attention/vllm_compare/run_qwen_gdn_stage6u_consumer_correctness.py
git diff --check
