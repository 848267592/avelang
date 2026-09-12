#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROCM=${ROCM_PATH:-/opt/rocm}
"$ROCM/llvm/bin/clang" -target amdgcn-amd-amdhsa -mcpu=gfx942 -x assembler \
  -c "$ROOT/original_from_triton.s" -o "$ROOT/original_from_triton.o"
"$ROCM/llvm/bin/ld.lld" -shared "$ROOT/original_from_triton.o" -o "$ROOT/original_rebuilt.hsaco"
"$ROCM/llvm/bin/llvm-objdump" -d --no-show-raw-insn "$ROOT/original_rebuilt.hsaco" > "$ROOT/rebuilt_disassembly.txt"
"$ROCM/llvm/bin/llvm-readobj" --notes --sections --symbols "$ROOT/original_rebuilt.hsaco" > "$ROOT/rebuilt_metadata.txt"
