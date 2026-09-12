#!/usr/bin/env sh
set -eu

cd /workspace/project/avelang
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention/vllm_compare
export PYTHONDONTWRITEBYTECODE=1

python3 -m pytest -q test/examples/linear_attention/vllm_compare/test_qwen_gdn_v30_bt64_bv32_hierarchical.py -s
python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v30_bt64_bv32_hierarchical.py \
  --T 512 1024 2048 4096 8192 16384 --warmup 5 --repeat 20

# Set AVELANG_AMDGPU_LINK_DEBUG_DIR while compiling D1, then replay the captured linker argv.
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/replay_qwen_v29_lto_mir.py \
  --argv-file test/examples/linear_attention/rocprof_outputs/qwen_v30_hierarchical/d1_link/amdgpu-link-0.argv.txt \
  --out-dir test/examples/linear_attention/rocprof_outputs/qwen_v30_hierarchical/exact_lto_d1 \
  --kernel _qwen_gdn_fused_chunk_gdr_full_bf16_kernel_v30_bt64_bv32_hierarchical

python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/analyze_qwen_v29_spill_vregs.py \
  --mir test/examples/linear_attention/rocprof_outputs/qwen_v30_hierarchical/exact_lto_d1/kernel_section_07.mir \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_v30_hierarchical_audit/d1_spills
