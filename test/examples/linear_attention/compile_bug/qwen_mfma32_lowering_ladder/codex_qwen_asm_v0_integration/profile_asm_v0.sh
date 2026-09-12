#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
FULLSEQ="$ROOT/../codex_triton_fullseq_asm_opt_audit"
HARNESS="$ROOT/asm_v0_harness"
PREPARE="$FULLSEQ/standalone_fullseq/prepare_case.py"
ROCPROF=${ROCPROF:-/opt/rocm/bin/rocprofv3}
SYMBOL=qwen_gdn_bt64_gfx942_asm_v0

if [[ ! -x "$HARNESS" ]]; then
  sh "$ROOT/build_harness.sh"
fi

if [[ "$#" -eq 0 ]]; then
  set -- 2048
fi

for T in "$@"; do
  INPUT="$ROOT/rocprof/input_t${T}"
  OUT="$ROOT/rocprof/t${T}_asm_v0"
  mkdir -p "$INPUT" "$OUT/run" "$OUT/out"
  PYTHONDONTWRITEBYTECODE=1 python3 "$PREPARE" --T "$T" --out "$INPUT"
  "$ROCPROF" --kernel-trace \
    --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
    --kernel-include-regex "$SYMBOL" -d "$OUT/run" -o counters -f csv -- \
    "$HARNESS" "$ROOT/assembly/qwen_gdn_bt64_gfx942_asm_v0.hsaco" "$SYMBOL" "$T" "$INPUT" "$OUT/out" 2 5
done
