#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SHARED="$ROOT/shared"
CLANG=${CLANG:-/opt/rocm/llvm/bin/clang}
LD_LLD=${LD_LLD:-/opt/rocm/llvm/bin/ld.lld}
"$CLANG" -target amdgcn-amd-amdhsa -mcpu=gfx942 -x assembler -c "$SHARED/original_from_triton.s" -o "$SHARED/rebuilt.o"
"$LD_LLD" -shared "$SHARED/rebuilt.o" -o "$SHARED/rebuilt.hsaco"
"${LLVM_OBJDUMP:-/opt/rocm/llvm/bin/llvm-objdump}" -d "$SHARED/original.hsaco" > "$SHARED/original.disasm"
"${LLVM_OBJDUMP:-/opt/rocm/llvm/bin/llvm-objdump}" -d "$SHARED/rebuilt.hsaco" > "$SHARED/rebuilt.disasm"
"${LLVM_READOBJ:-/opt/rocm/llvm/bin/llvm-readobj}" --notes --symbols "$SHARED/original.hsaco" > "$SHARED/original.metadata.txt"
"${LLVM_READOBJ:-/opt/rocm/llvm/bin/llvm-readobj}" --notes --symbols "$SHARED/rebuilt.hsaco" > "$SHARED/rebuilt.metadata.txt"
