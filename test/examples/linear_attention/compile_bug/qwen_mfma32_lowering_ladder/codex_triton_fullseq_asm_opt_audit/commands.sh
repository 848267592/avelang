#!/usr/bin/env bash
set -euo pipefail

# Run inside the established ROCm container from /workspace/project/avelang.
AUDIT=test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_triton_fullseq_asm_opt_audit
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention/vllm_compare:/workspace/project/avelang/$AUDIT
export TRITON_CACHE_DIR=/workspace/project/avelang/$AUDIT/triton_cache_longseq
export QWEN_TRITON_FULLSEQ_HSACO_PATH=/workspace/project/avelang/$AUDIT/golden_fullseq/shared/original.hsaco
export QWEN_TRITON_FULLSEQ_EXTERNAL_BRIDGE=/workspace/project/avelang/$AUDIT/avelang_external_full/libqwen_triton_external_full_bridge.so

python3 "$AUDIT/capture_long_sequence.py" --T 64 128 512 2048 8192 16384 --warmup 10 --repeat 50
"$AUDIT/golden_fullseq/build_golden.sh"
"$AUDIT/standalone_fullseq/build.sh"
python3 "$AUDIT/populate_golden_dirs.py"
python3 "$AUDIT/standalone_fullseq/run_fullseq_correctness.py" --T 64 128 512 2048 --random-cases 30
python3 "$AUDIT/standalone_fullseq/smoke_long_lengths.py"
"$AUDIT/avelang_external_full/build.sh"
python3 -m pytest -q "$AUDIT/avelang_external_full/test_qwen_gdn_bt64_gfx942_external_full.py" -s
python3 "$AUDIT/benchmark_fullseq.py" --T 512 2048 8192 16384 --warmup 10 --repeat 50 --sessions 3
"$AUDIT/profile_harness.sh"
python3 "$AUDIT/summarize_fullseq.py"
