#!/usr/bin/env sh
# Stage 5A is audit-only. Run from the repository root inside the MI300 container.
set -eu

ROOT=/workspace/project/avelang
OUT="$ROOT/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_solve_rootcause_stage5a"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:$ROOT/python:$ROOT/test/examples/linear_attention/vllm_compare:$ROOT/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2

git status --short > "$OUT/git_before.txt"
git diff --stat >> "$OUT/git_before.txt"
git diff >> "$OUT/git_before.txt"

python3 "$OUT/solve_rootcause_harness.py" --mode correctness --out-dir "$OUT"
for session in 0 1 2; do
  python3 "$OUT/solve_rootcause_harness.py" --mode benchmark --session "$session" --out-dir "$OUT" \
    --warmup 20 --repeat 100
done
python3 "$OUT/solve_rootcause_harness.py" --mode summarize --out-dir "$OUT"
python3 "$OUT/solve_rootcause_harness.py" --mode ablation --out-dir "$OUT" --warmup 20 --repeat 100

# Profile the existing kernels only. First capture normal vLLM autotune, then
# dispatch precisely that selected config so profiling does not include trials.
python3 "$OUT/solve_rootcause_harness.py" --mode capture-vllm-config --out-dir "$OUT"
for implementation in avelang_v18 vllm; do
  for t in 512 2048 8192; do
    /opt/rocm/bin/rocprofv3 --kernel-trace \
      --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
      -d "$OUT/rocprof/${implementation}_T${t}" -o "${implementation}_T${t}" -f csv -- \
      python3 "$OUT/solve_rootcause_harness.py" --mode profile --implementation "$implementation" \
        --T "$t" --warmup 20 --repeat 8
  done
done
python3 "$OUT/summarize_rocprof.py" --root "$OUT/rocprof" --out-dir "$OUT"
python3 "$OUT/dump_solve_isa.py" --implementation all

# Frozen-path regressions and a separate full-pipeline smoke. Neither command
# imports this audit harness into the production graph.
python3 -m pytest -q "$ROOT/test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_nonrecurrence_mfma_v2.py" -s
python3 -m pytest -q "$ROOT/test/examples/linear_attention/vllm_compare/test_qwen_gdn_full_bt64_gfx942_asm_v0_experimental.py" -s
