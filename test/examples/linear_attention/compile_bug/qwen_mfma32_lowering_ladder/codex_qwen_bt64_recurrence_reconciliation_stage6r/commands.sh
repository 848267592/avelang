#!/usr/bin/env sh
set -eu

# Stage 6R is audit-only: no production stage, assembly, or compiler changes.
ROOT=/workspace/project/avelang
OUT="$ROOT/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_recurrence_reconciliation_stage6r"
cd "$ROOT"

PYTHONDONTWRITEBYTECODE=1 python3 "$OUT/stage6r_capture_current_recurrence.py" --T 2048

# Build the audit-only external launcher. It does not alter either HSACO.
sh "$OUT/build_bridge.sh"

# HIP-event timing: graph replay only; casts/allocation/module load are outside
# the timed recurrence body.
PYTHONDONTWRITEBYTECODE=1 python3 "$OUT/stage6r_recurrence_body_benchmark.py" \
  --T 512 2048 8192 16384 --warmup 20 --repeat 100 --sessions 5

# Profiler collection is only for counters/dispatch normalization, never the
# authority for the standalone body latency.
/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex qwen_gdn_bt64_gfx942_asm_v0 \
  -d "$OUT/rocprof_asm_v0" -o stage6r -f csv -- \
  python3 "$OUT/stage6r_recurrence_trace_replay.py" --implementation asm_v0_fp32 --replay 3
/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex chunk_gated_delta_rule_fwd_kernel_h_blockdim64 \
  -d "$OUT/rocprof_current_vllm" -o stage6r -f csv -- \
  python3 "$OUT/stage6r_recurrence_trace_replay.py" --implementation vllm_actual_bf16 --replay 3

PYTHONDONTWRITEBYTECODE=1 python3 "$OUT/stage6r_summarize.py"

PYTHONPATH="$ROOT/python:$ROOT/test/examples/linear_attention/vllm_compare" \
  PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  "$ROOT/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_asm_v0_integration/tests/test_qwen_gdn_bt64_gfx942_asm_v0.py" -s
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q "$OUT/test_stage6r_current_bridge.py" -s
