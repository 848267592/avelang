#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT=$(CDPATH= cd -- "$ROOT/../../../../../../" && pwd)
ROCPROF=${ROCPROF:-/opt/rocm/bin/rocprofv3}
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:$PROJECT/python:$PROJECT/test/examples/linear_attention/vllm_compare${PYTHONPATH:+:$PYTHONPATH}

bash "$ROOT/profile_golden.sh" 2048
bash "$ROOT/profile_asm_v0.sh" 2048

for VARIANT in v29 v31; do
  case "$VARIANT" in
    v29) KERNEL='_qwen_gdn_fused_chunk_gdr_full_bf16_kernel_v29_mfma32' ;;
    v31) KERNEL='_qwen_gdn_fused_chunk_gdr_full_bf16_kernel_v31_bt64_bv32_hierarchical_mfma16_pred' ;;
  esac
  OUT="$ROOT/rocprof/t2048_$VARIANT"
  mkdir -p "$OUT/run"
  "$ROCPROF" --kernel-trace \
    --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
    --kernel-include-regex "$KERNEL" -d "$OUT/run" -o counters -f csv -- \
    python3 "$ROOT/profile_context.py" --variant "$VARIANT" --T 2048 --warmup 2 --repeat 5
done

python3 "$ROOT/summarize_resources.py"
