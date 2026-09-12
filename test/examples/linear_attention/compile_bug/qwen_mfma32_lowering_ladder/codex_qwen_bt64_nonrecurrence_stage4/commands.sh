#!/usr/bin/env sh
set -eu

# Run inside ljd_qwen_vllm_avelang_rocm722.
cd /workspace/project/avelang
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention/vllm_compare:/workspace/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2
ROOT=test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_nonrecurrence_stage4

python3 -m py_compile \
  test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_nonrecurrence_mfma_v2.py \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_nonrecurrence_mfma_v2.py \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_nonrecurrence_mfma_v2.py \
  "$ROOT"/stage4_runner.py "$ROOT"/bench_stage4.py "$ROOT"/profile_stage4_stages.py

python3 -m pytest -q test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_nonrecurrence_mfma_v2.py -s
python3 "$ROOT"/stage4_runner.py --random-cases 30

for tag in a b c; do
  python3 "$ROOT"/bench_stage4.py --T 512 2048 8192 16384 --warmup 10 --repeat 50 --session "$tag"
done

python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_nonrecurrence_mfma_v2.py \
  --T 512 1024 2048 --warmup 10 --repeat 50
python3 "$ROOT"/audit_solve_stage4.py --T 512 2048 --warmup 10 --repeat 50

for stage in kkt wu_w wu_u chunk_o; do
  /opt/rocm/bin/rocprofv3 --kernel-trace \
    --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
    -d "$ROOT/rocprof/$stage" -o "$stage" -f csv -- \
    python3 "$ROOT"/profile_stage4_stages.py --stage "$stage" --T 2048 --warmup 2 --repeat 5
  python3 "$ROOT"/profile_stage4_stages.py --stage "$stage" --T 2048 --dump-hsaco "$ROOT/isa/$stage"
done

python3 "$ROOT"/summarize_stage4.py
