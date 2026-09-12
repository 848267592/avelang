#!/usr/bin/env sh
set -eu

# Run from the repository root inside the ROCm container.
AUDIT=test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_triton_hsaco_reuse_audit
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention/vllm_compare
export QWEN_TRITON_HSACO_PATH=/workspace/project/avelang/$AUDIT/extracted/original_triton_kernel.hsaco
export QWEN_TRITON_EXTERNAL_BRIDGE=/workspace/project/avelang/$AUDIT/avelang_integration/libqwen_triton_external_bridge.so

python3 "$AUDIT/inspect_selected_compiled_kernel.py"
"$AUDIT/standalone_harness/build.sh"
python3 "$AUDIT/standalone_harness/run_exact_specialization.py" --cases 55 --warmup 10 --repeat 50
"$AUDIT/editable_assembly/build_original.sh"
"$AUDIT/avelang_integration/build_bridge.sh"
python3 -m pytest -q "$AUDIT/avelang_integration/test_qwen_gdn_bt64_gfx942_external.py" -s
python3 "$AUDIT/summarize_audit.py"
