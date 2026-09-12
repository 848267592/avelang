#!/usr/bin/env sh
set -eu

cd /workspace/project/avelang
AUDIT=test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_direct_common_out_stage5e
WRAPPER=test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_solve_direct_out_stage5e_audit.py
HARNESS="$AUDIT/direct_common_out_harness.py"

PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  "$WRAPPER" "$HARNESS" "$AUDIT/test_direct_common_out_stage5e.py" \
  "$AUDIT/profile_stage5e.py" "$AUDIT/observe_clock_power_stage5e.py" \
  "$AUDIT/analyze_stage5e_profiles.py" "$AUDIT/analyze_stage5e_telemetry.py"

PYTHONDONTWRITEBYTECODE=1 python3 "$HARNESS" --mode contract --out-dir "$AUDIT"
PYTHONDONTWRITEBYTECODE=1 python3 "$HARNESS" --mode correctness --out-dir "$AUDIT"
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q "$AUDIT/test_direct_common_out_stage5e.py" -s --tb=short

PYTHONDONTWRITEBYTECODE=1 python3 "$HARNESS" --mode benchmark \
  --warmup 20 --repeat 200 --sessions 5 --out-dir "$AUDIT"
PYTHONDONTWRITEBYTECODE=1 python3 "$HARNESS" --mode controls \
  --warmup 20 --repeat 200 --sessions 5 --flush-mib 512 --out-dir "$AUDIT"

# Conditional: execute when the T=2048 direct-common tail gap exceeds 5 us,
# preserves at least 20% of Stage 5D's 64.255 us, or remains bimodal.
PYTHONDONTWRITEBYTECODE=1 python3 "$AUDIT/profile_stage5e.py" \
  --operation tail --T 2048 8192 --control none warm perturb \
  --repeat 8 --out-dir "$AUDIT/rocprof"
PYTHONDONTWRITEBYTECODE=1 python3 "$AUDIT/profile_stage5e.py" \
  --operation full --T 2048 --control none --repeat 8 --out-dir "$AUDIT/rocprof"
PYTHONDONTWRITEBYTECODE=1 python3 "$AUDIT/profile_stage5e.py" \
  --operation tail --T 2048 --control none --repeat 5 \
  --out-dir "$AUDIT/rocprof_cache" --cache-only \
  --cache-counter TCC_HIT_sum --cache-counter TCC_MISS_sum \
  --cache-counter TCC_EA0_RDREQ_sum --cache-counter TCC_EA0_RDREQ_DRAM_sum \
  --cache-counter TCP_TOTAL_CACHE_ACCESSES_sum

# Read-only observation; these commands never set clocks or power caps.
PYTHONDONTWRITEBYTECODE=1 python3 "$AUDIT/observe_clock_power_stage5e.py" \
  --solve-impl v18 --control none --sequence single --T 8192 \
  --seconds 12 --interval 0.5 --repeat 15000 --out "$AUDIT/telemetry/a_none.csv"
PYTHONDONTWRITEBYTECODE=1 python3 "$AUDIT/observe_clock_power_stage5e.py" \
  --solve-impl hierarchical_fp32_v1 --control none --sequence single --T 8192 \
  --seconds 12 --interval 0.5 --repeat 15000 --out "$AUDIT/telemetry/b_none.csv"
PYTHONDONTWRITEBYTECODE=1 python3 "$AUDIT/observe_clock_power_stage5e.py" \
  --solve-impl v18 --control none --sequence abba --T 8192 \
  --seconds 12 --interval 0.5 --repeat 15000 --out "$AUDIT/telemetry/abba_none.csv"
PYTHONDONTWRITEBYTECODE=1 python3 "$AUDIT/observe_clock_power_stage5e.py" \
  --solve-impl v18 --control warm --T 8192 --seconds 12 --interval 0.5 \
  --repeat 8000 --out "$AUDIT/telemetry/a_warm.csv"
PYTHONDONTWRITEBYTECODE=1 python3 "$AUDIT/observe_clock_power_stage5e.py" \
  --solve-impl hierarchical_fp32_v1 --control warm --T 8192 --seconds 12 \
  --interval 0.5 --repeat 8000 --out "$AUDIT/telemetry/b_warm.csv"
PYTHONDONTWRITEBYTECODE=1 python3 "$AUDIT/observe_clock_power_stage5e.py" \
  --solve-impl v18 --control perturb --T 8192 --seconds 12 --interval 0.5 \
  --repeat 500 --out "$AUDIT/telemetry/a_perturb.csv"
PYTHONDONTWRITEBYTECODE=1 python3 "$AUDIT/observe_clock_power_stage5e.py" \
  --solve-impl hierarchical_fp32_v1 --control perturb --T 8192 --seconds 12 \
  --interval 0.5 --repeat 500 --out "$AUDIT/telemetry/b_perturb.csv"

PYTHONDONTWRITEBYTECODE=1 python3 "$AUDIT/analyze_stage5e_profiles.py"
PYTHONDONTWRITEBYTECODE=1 python3 "$AUDIT/analyze_stage5e_telemetry.py"

PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -s --tb=short \
  "$AUDIT/test_direct_common_out_stage5e.py" \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_hierarchical_solve_stage5c.py \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_nonrecurrence_mfma_v2.py \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_asm_v0_integration/tests/test_qwen_gdn_bt64_gfx942_asm_v0.py \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_triton_hsaco_reuse_audit/avelang_integration/test_qwen_gdn_bt64_gfx942_external.py
