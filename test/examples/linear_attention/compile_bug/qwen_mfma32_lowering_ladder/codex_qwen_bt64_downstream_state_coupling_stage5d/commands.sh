#!/usr/bin/env sh
set -eu

cd /workspace/project/avelang
AUDIT=test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_downstream_state_coupling_stage5d
HARNESS="$AUDIT/downstream_state_coupling_harness.py"

PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile "$HARNESS" "$AUDIT/test_downstream_state_coupling_stage5d.py"
PYTHONDONTWRITEBYTECODE=1 python3 "$HARNESS" --mode contract --T 2048 --out-dir "$AUDIT"
PYTHONDONTWRITEBYTECODE=1 python3 "$HARNESS" --mode smoke --T 64 512 8192 --out-dir "$AUDIT"
PYTHONDONTWRITEBYTECODE=1 python3 "$HARNESS" --mode full --T 512 1024 2048 4096 8192 16384 --warmup 20 --repeat 200 --sessions 5 --out-dir "$AUDIT"
PYTHONDONTWRITEBYTECODE=1 python3 "$HARNESS" --mode tail --T 512 2048 8192 16384 --warmup 20 --repeat 200 --sessions 5 --out-dir "$AUDIT"
PYTHONDONTWRITEBYTECODE=1 python3 "$HARNESS" --mode controls --T 2048 8192 --warmup 20 --repeat 200 --sessions 5 --flush-mib 512 --out-dir "$AUDIT"

PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q "$AUDIT/test_downstream_state_coupling_stage5d.py" -s --tb=short

# Whole-graph profiling is intentionally a separate process per case. Add
# cache counters only after confirming them in rocprof/available_counters.txt.
PYTHONDONTWRITEBYTECODE=1 python3 "$AUDIT/profile_stage5d.py" --T 2048 8192 --out-dir "$AUDIT/rocprof"

# Read-only background observation; this never changes clocks or power caps.
PYTHONDONTWRITEBYTECODE=1 python3 "$AUDIT/observe_clock_power_stage5d.py" --case graph_a --T 8192 --out "$AUDIT/clock_power_graph_a.csv"
PYTHONDONTWRITEBYTECODE=1 python3 "$AUDIT/observe_clock_power_stage5d.py" --case graph_b --T 8192 --out "$AUDIT/clock_power_graph_b.csv"
