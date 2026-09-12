#!/usr/bin/env sh
# Reproducible Stage 2 commands. Run inside ljd_qwen_vllm_avelang_rocm722.
set -eu

cd /workspace/project/avelang
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention/vllm_compare
ROOT=test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2

python3 -m py_compile "$ROOT"/stage2_runner.py "$ROOT"/profile_candidate_stage.py "$ROOT"/bench_stage2.py "$ROOT"/summarize_stage2.py \
  test/examples/linear_attention/vllm_compare/qwen_gdn_full_bt64_gfx942_asm_v0_experimental.py \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_full_bt64_gfx942_asm_v0_experimental.py

python3 "$ROOT"/stage2_runner.py --random-cases 30 --capture
python3 -m pytest -q test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_asm_v0_integration/tests/test_qwen_gdn_bt64_gfx942_asm_v0.py -s
python3 -m pytest -q test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_triton_hsaco_reuse_audit/avelang_integration/test_qwen_gdn_bt64_gfx942_external.py \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_triton_fullseq_asm_opt_audit/avelang_external_full/test_qwen_gdn_bt64_gfx942_external_full.py \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_gfx942_asm_bt64_step_audit/test_gfx942_asm_smoke.py \
  test/examples/linear_attention/vllm_compare/test_qwen_v31_active_v16_pred_primitives.py -s
python3 -m pytest -q test/examples/linear_attention/vllm_compare/test_qwen_gdn_full_bt64_gfx942_asm_v0_experimental.py -s

# Each full session uses warmup=10 and repeat=50. Keep vLLM separate to avoid
# Triton/Avelang interaction inside one Python process.
for tag in a b c; do
  python3 "$ROOT"/bench_stage2.py --T 512 2048 8192 16384 --warmup 10 --repeat 50 --session "stage2_candidate_v24_$tag" --full-only
  python3 "$ROOT"/bench_stage2.py --T 512 2048 8192 16384 --warmup 10 --repeat 50 --session "stage2_vllm_$tag" --vllm-only
done

/opt/rocm/bin/rocprofv3 --kernel-trace --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex _qwen_gdn_chunk_o_bf16_kernel_v6_standalone -d "$ROOT"/rocprof/chunk_o -o counters -f csv -- \
  python3 "$ROOT"/profile_candidate_stage.py --stage chunk_o --T 2048 --warmup 2 --repeat 5

python3 "$ROOT"/summarize_stage2.py
