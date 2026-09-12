#!/bin/sh
# Stage 5F successful, audit-only commands. Run from /workspace/project/avelang.
set -eu

OUT=test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_transient_state_stage5f

PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  "$OUT/stage5f_measure.py" "$OUT/stage5f_capability_inventory.py" \
  "$OUT/stage5f_evaluate_gate.py" "$OUT/stage5f_publish.py"
PYTHONDONTWRITEBYTECODE=1 python3 "$OUT/stage5f_measure.py" --mode contract --out-dir "$OUT" --T 2048
PYTHONDONTWRITEBYTECODE=1 python3 "$OUT/stage5f_capability_inventory.py"
PYTHONDONTWRITEBYTECODE=1 python3 "$OUT/stage5f_measure.py" --mode calibrate --label hip_event \
  --out-dir "$OUT" --T 2048 8192 --operation tail full --sessions 5 --warmup 20 --repeat 200

/opt/rocm/bin/rocprofv3 --kernel-trace -d "$OUT/rocprof_trace_only" -o stage5f_trace_only -f csv -- \
  env PYTHONDONTWRITEBYTECODE=1 python3 "$OUT/stage5f_measure.py" --mode calibrate --label trace_only \
    --out-dir "$OUT" --T 2048 --operation tail --sessions 5 --warmup 20 --repeat 200
PYTHONDONTWRITEBYTECODE=1 python3 "$OUT/stage5f_evaluate_gate.py" \
  --baseline "$OUT/hip_event_tail_summary.csv" --candidate "$OUT/trace_only_tail_summary.csv" \
  --operation tail --T 2048 --out "$OUT/trace_only_gate.json"
PYTHONDONTWRITEBYTECODE=1 python3 "$OUT/stage5f_publish.py"

cd test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_direct_common_out_stage5e
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q test_direct_common_out_stage5e.py -s
