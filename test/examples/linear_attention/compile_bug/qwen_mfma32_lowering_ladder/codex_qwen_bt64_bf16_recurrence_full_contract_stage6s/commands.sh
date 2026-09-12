#!/usr/bin/env sh
set -eu

ROOT=/workspace/project/avelang
OUT=$ROOT/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_bf16_recurrence_full_contract_stage6s
export PYTHONPATH=$ROOT/python:$ROOT/test/examples/linear_attention/vllm_compare
export PYTHONDONTWRITEBYTECODE=1

# Syntax, contract correctness, and inherited regression coverage.
PYTHONPYCACHEPREFIX=/tmp/pycache_stage6s python3 -m py_compile \
  $ROOT/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_bf16_recurrence_full_stage6s.py \
  $ROOT/test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_bf16_recurrence_full_stage6s.py \
  $OUT/stage6s_full_contract_audit.py \
  $OUT/stage6s_correctness_matrix.py \
  $OUT/stage6s_trace_direct.py
python3 -m pytest -q \
  $ROOT/test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_bf16_recurrence_full_stage6s.py \
  $ROOT/test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_recurrence_reconciliation_stage6r.py \
  $ROOT/test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_gfx942_asm_v0_integration.py -s
python3 -m pytest -q \
  $ROOT/test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_nonrecurrence_mfma_v2.py -s

# Diagnostic only: this direct source-JIT test is expected to fail until the
# active Python binding exports mfma_16x16x4_f32_f32. Stage 6S instead guards
# and launches the immutable prevalidated Stage-5B solve HSACO.
# python3 -m pytest -q \
#   $ROOT/test/examples/linear_attention/vllm_compare/test_qwen_gdn_solve_bt64_hierarchical_fp32_v1.py -s

# Full correctness matrix and same-process graph replay audit.
python3 $OUT/stage6s_correctness_matrix.py
python3 $OUT/stage6s_full_contract_audit.py \
  --T 512 1024 2048 4096 8192 16384 --warmup 20 --repeat 100 --sessions 5 --mode all

# Structural only: do not use rocprof timing as the latency authority.
/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  -d $OUT/rocprof/direct_b_t2048 -o stage6s -f csv -- \
  python3 $OUT/stage6s_trace_direct.py --graph b --T 2048 --warmup 2 --replay 3
/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  -d $OUT/rocprof/direct_c_t2048 -o stage6s -f csv -- \
  python3 $OUT/stage6s_trace_direct.py --graph c --T 2048 --warmup 2 --replay 3
