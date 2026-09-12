#!/usr/bin/env bash
set -euo pipefail

# Run from /workspace/project/avelang inside ljd_qwen_vllm_avelang_rocm722.
build_dir=/tmp/avelang-build-kfrag-qwen-rocm722
audit_dir=test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_fp32_mfma16_intrinsic_enablement

/opt/rocm/llvm/bin/mlir-opt "$audit_dir/rocdl_mfma16x4_fp32_syntax_probe.mlir" \
  -o /tmp/rocdl_mfma16x4_fp32_syntax_probe.mlirbc

cmake --build "$build_dir" --target _avelang_bindings mlir_generator_test -j 16
"$build_dir/lib/IR/mlir_generator_test" --gtest_filter='*AMDGPUMFMAFP32*'

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$build_dir/python:/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention/vllm_compare"
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_fp32_mfma16_intrinsic_enablement.py \
  --warmup 5 --repeat 20 --dump-hsaco-dir "$audit_dir/hsaco" \
  --output-json "$audit_dir/live_jit.json"

/opt/rocm/llvm/bin/llvm-objdump -d "$audit_dir/hsaco/fp32_mfma16x4_intrinsic_probe.hsaco" \
  > "$audit_dir/fp32_mfma16x4_intrinsic_probe.s"
rg -n 'v_mfma_f32_16x16x4_f32|v_mfma' "$audit_dir/fp32_mfma16x4_intrinsic_probe.s"

/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  -d "$audit_dir/rocprof" -o fp32_mfma16x4_counters -f csv -- \
  python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_fp32_mfma16_intrinsic_enablement.py \
    --warmup 2 --repeat 5 --output-json "$audit_dir/live_jit_rocprof.json"
