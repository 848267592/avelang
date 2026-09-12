#!/usr/bin/env bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
HARNESS="$ROOT/standalone_fullseq/fullseq_harness"
PREPARE="$ROOT/standalone_fullseq/prepare_case.py"
KERNEL='chunk_gated_delta_rule_fwd_kernel_h_blockdim64'

for T in 512 2048 8192 16384; do
  INPUT="$ROOT/standalone_fullseq/profile_inputs/t$T"
  mkdir -p "$INPUT"
  PYTHONDONTWRITEBYTECODE=1 python3 "$PREPARE" --T "$T" --out "$INPUT"
  for VARIANT in original rebuilt; do
    HSACO="$ROOT/golden_fullseq/shared/$VARIANT.hsaco"
    OUT="$ROOT/rocprof/t${T}_${VARIANT}"
    mkdir -p "$OUT/run" "$OUT/out"
    /opt/rocm/bin/rocprofv3 --kernel-trace \
      --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
      --kernel-include-regex "$KERNEL" -d "$OUT/run" -o counters -f csv -- \
      "$HARNESS" "$HSACO" "$T" "$INPUT" "$OUT/out" 2 5
  done
done
