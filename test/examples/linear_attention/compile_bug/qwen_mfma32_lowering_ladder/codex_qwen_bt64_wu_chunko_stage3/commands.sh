#!/usr/bin/env sh
# Run from /workspace/project/avelang inside ljd_qwen_vllm_avelang_rocm722.
set -eu

export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention/vllm_compare:/workspace/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2
export PYTHONDONTWRITEBYTECODE=1

python3 -m pytest -q test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_native_wu_chunko_mfma_v1.py -s
python3 -m pytest -q test/examples/linear_attention/vllm_compare/test_qwen_gdn_full_bt64_native_wu_o_v1.py -s
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_wu_chunko_stage3/stage3_runner.py --random-cases 30

python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_native_wu_chunko_mfma_v1.py --T 512 1024 2048 --warmup 10 --repeat 50
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_wu_chunko_stage3/bench_stage3.py --T 512 2048 8192 16384 --warmup 10 --repeat 50 --session stage3_a

for stage in wu_w wu_u chunk_o; do
  kernel="_qwen_gdn_${stage#wu_}_bf16_kernel_bt64_from_v24_mfma_v1"
  [ "$stage" = chunk_o ] && kernel="_qwen_gdn_chunk_o_bf16_kernel_bt64_from_v24_mfma_v1"
  rocprofv3 --kernel-trace --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent --kernel-include-regex "$kernel" -d stage3_rocprof -o "$stage" -f csv -- python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_wu_chunko_stage3/profile_native_stages.py --stage "$stage" --T 2048 --warmup 2 --repeat 5
done
