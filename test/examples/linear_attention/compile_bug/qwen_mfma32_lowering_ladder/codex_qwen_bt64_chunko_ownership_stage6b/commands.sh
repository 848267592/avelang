#!/bin/sh
set -eu
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare

PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 -m py_compile \
  qwen_gdn_bt64_chunko_ownership_stage6b.py \
  qwen_gdn_bt64_chunko_ownership_stage6b_o1.py \
  test_qwen_gdn_bt64_chunko_ownership_stage6b.py \
  bench_qwen_gdn_bt64_chunko_ownership_stage6b.py \
  dump_qwen_gdn_bt64_chunko_ownership_stage6b_isa.py

PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 -m pytest -q \
  test_qwen_gdn_bt64_chunko_ownership_stage6b.py -s --tb=short

PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 -m pytest -q \
  test_qwen_gdn_bt64_nonrecurrence_mfma_v2.py \
  test_qwen_gdn_solve_bt64_hierarchical_fp32_v1.py --tb=short

PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 -m pytest -q \
  test_qwen_gdn_full_bt64_gfx942_asm_v0_experimental.py --tb=short

PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 \
  bench_qwen_gdn_bt64_chunko_ownership_stage6b.py --mode body \
  --T 512 1024 2048 4096 8192 16384 --warmup 20 --repeat 100 --sessions 5 \
  --out-dir ../compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_chunko_ownership_stage6b

PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 \
  dump_qwen_gdn_bt64_chunko_ownership_stage6b_isa.py --T 2048 \
  --out-dir ../compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_chunko_ownership_stage6b

# O0 and O1 rocprof commands are recorded in the main report. Full graph replay
# is deliberately not run because neither candidate passes the body gate.
