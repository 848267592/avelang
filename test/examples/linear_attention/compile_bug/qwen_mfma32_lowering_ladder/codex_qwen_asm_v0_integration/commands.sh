#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT=$(CDPATH= cd -- "$ROOT/../../../../../../" && pwd)
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:$PROJECT/python:$PROJECT/test/examples/linear_attention/vllm_compare${PYTHONPATH:+:$PYTHONPATH}
export PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-/tmp/pycache_qwen_asm_v0}

sh "$ROOT/assembly/build.sh"
sh "$ROOT/avelang_integration/build.sh"
sh "$ROOT/build_harness.sh"
sh "$ROOT/build_round_robin_harness.sh"
python3 -m py_compile \
  "$ROOT/avelang_integration/qwen_gdn_bt64_gfx942_asm_v0.py" \
  "$ROOT/tests/run_correctness.py" \
  "$ROOT/tests/test_qwen_gdn_bt64_gfx942_asm_v0.py" \
  "$ROOT/benchmark_stage.py" \
  "$ROOT/profile_context.py" \
  "$ROOT/summarize_resources.py" \
  "$PROJECT/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_gfx942_asm_v0_experimental.py"
python3 -m pytest -q "$ROOT/tests/test_qwen_gdn_bt64_gfx942_asm_v0.py" -s
python3 -m pytest -q \
  "$PROJECT/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_triton_hsaco_reuse_audit/avelang_integration/test_qwen_gdn_bt64_gfx942_external.py" \
  "$PROJECT/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_triton_fullseq_asm_opt_audit/avelang_external_full/test_qwen_gdn_bt64_gfx942_external_full.py" -s
python3 -m pytest -q \
  "$PROJECT/test/examples/linear_attention/vllm_compare/test_qwen_v31_active_v16_pred_primitives.py" -s
python3 "$ROOT/tests/run_correctness.py" --random-cases 40 --out "$ROOT"
python3 "$ROOT/benchmark_stage.py" --T 512 2048 8192 16384 --warmup 10 --repeat 50 --sessions 3 --out "$ROOT"
bash "$ROOT/profile_all.sh"
git -C "$PROJECT" diff --check
