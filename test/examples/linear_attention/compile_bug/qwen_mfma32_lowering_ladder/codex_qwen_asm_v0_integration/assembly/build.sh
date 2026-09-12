#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CLANG=${CLANG:-/opt/rocm/llvm/bin/clang}
LD_LLD=${LD_LLD:-/opt/rocm/llvm/bin/ld.lld}
LLVM_OBJDUMP=${LLVM_OBJDUMP:-/opt/rocm/llvm/bin/llvm-objdump}
LLVM_READOBJ=${LLVM_READOBJ:-/opt/rocm/llvm/bin/llvm-readobj}
SOURCE="$ROOT/qwen_gdn_bt64_gfx942_asm_v0.s"
OBJECT="$ROOT/qwen_gdn_bt64_gfx942_asm_v0.o"
HSACO="$ROOT/qwen_gdn_bt64_gfx942_asm_v0.hsaco"

"$CLANG" -target amdgcn-amd-amdhsa -mcpu=gfx942 -x assembler -c "$SOURCE" -o "$OBJECT"
"$LD_LLD" -shared "$OBJECT" -o "$HSACO"
"$LLVM_OBJDUMP" -d "$HSACO" > "$ROOT/disassembly.txt"
"$LLVM_READOBJ" --notes --symbols "$HSACO" > "$ROOT/metadata.txt"
sha256sum "$HSACO" > "$ROOT/qwen_gdn_bt64_gfx942_asm_v0.hsaco.sha256"
