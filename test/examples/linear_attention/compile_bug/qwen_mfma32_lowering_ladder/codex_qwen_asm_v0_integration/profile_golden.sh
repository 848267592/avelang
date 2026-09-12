#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
FULLSEQ="$ROOT/../codex_triton_fullseq_asm_opt_audit"
HARNESS="$ROOT/asm_v0_harness"
PREPARE="$FULLSEQ/standalone_fullseq/prepare_case.py"
ROCPROF=${ROCPROF:-/opt/rocm/bin/rocprofv3}
SYMBOL=chunk_gated_delta_rule_fwd_kernel_h_blockdim64

if [[ ! -x "$HARNESS" ]]; then
  bash "$ROOT/build_harness.sh"
fi

if [[ "$#" -eq 0 ]]; then
  set -- 2048
fi

for T in "$@"; do
  INPUT="$ROOT/rocprof/input_t${T}"
  mkdir -p "$INPUT"
  PYTHONDONTWRITEBYTECODE=1 python3 "$PREPARE" --T "$T" --out "$INPUT"
  for SPEC in \
    "golden_original:$FULLSEQ/golden_fullseq/shared/original.hsaco" \
    "golden_rebuilt:$FULLSEQ/golden_fullseq/shared/rebuilt.hsaco"; do
    NAME=${SPEC%%:*}
    HSACO=${SPEC#*:}
    OUT="$ROOT/rocprof/t${T}_${NAME}"
    mkdir -p "$OUT/run" "$OUT/out"
    "$ROCPROF" --kernel-trace \
      --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
      --kernel-include-regex "$SYMBOL" -d "$OUT/run" -o counters -f csv -- \
      "$HARNESS" "$HSACO" "$SYMBOL" "$T" "$INPUT" "$OUT/out" 2 5
  done
done
