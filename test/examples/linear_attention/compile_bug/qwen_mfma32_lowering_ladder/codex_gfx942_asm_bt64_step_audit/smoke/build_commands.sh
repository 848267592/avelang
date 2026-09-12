#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROCM=${ROCM_PATH:-/opt/rocm}
CLANG="$ROCM/llvm/bin/clang"
LLD="$ROCM/llvm/bin/ld.lld"
OBJDUMP="$ROCM/llvm/bin/llvm-objdump"
READOBJ="$ROCM/llvm/bin/llvm-readobj"
HIPCC="$ROCM/bin/hipcc"

"$CLANG" -target amdgcn-amd-amdhsa -mcpu=gfx942 -x assembler -c \
  "$ROOT/source.s" -o "$ROOT/source.o"
"$LLD" -shared "$ROOT/source.o" -o "$ROOT/kernel.hsaco"
"$OBJDUMP" -d --no-show-raw-insn "$ROOT/kernel.hsaco" > "$ROOT/disassembly.txt"
"$READOBJ" --notes --sections --symbols "$ROOT/kernel.hsaco" > "$ROOT/metadata.txt"
"$HIPCC" -std=c++17 -O2 "$ROOT/smoke_harness.cpp" -o "$ROOT/smoke_harness"
"$ROOT/smoke_harness" "$ROOT/kernel.hsaco" > "$ROOT/correctness.json"
