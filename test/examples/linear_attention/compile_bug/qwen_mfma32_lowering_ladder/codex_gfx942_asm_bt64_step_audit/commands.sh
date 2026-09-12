#!/usr/bin/env sh
set -eu

ROOT=/workspace/project/avelang
AUDIT=$ROOT/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_gfx942_asm_bt64_step_audit
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:$ROOT/python:$ROOT/test/examples/linear_attention/vllm_compare

cd "$AUDIT/smoke"
./build_commands.sh

cd "$ROOT"
RUN_GFX942_ASM_SMOKE=1 python3 -m pytest -q "$AUDIT/test_gfx942_asm_smoke.py" -s
python3 -m pytest -q test/examples/linear_attention/vllm_compare/test_qwen_v31_active_v16_pred_primitives.py -s
python3 "$AUDIT/single_step_harness.py" --warmup 10 --repeat 50 \
  --out "$AUDIT/reference_and_triton_single_step.json"

cd "$AUDIT/smoke"
/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex qwen_gfx942_asm_smoke \
  -d rocprof -o smoke_counters -f csv -- ./smoke_harness ./kernel.hsaco
