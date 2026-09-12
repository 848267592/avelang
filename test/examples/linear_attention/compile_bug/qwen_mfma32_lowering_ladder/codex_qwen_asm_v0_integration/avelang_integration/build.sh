#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"${HIPCC:-/opt/rocm/bin/hipcc}" -std=c++17 -O3 -shared -fPIC \
  "$ROOT/qwen_gdn_bt64_gfx942_asm_v0_bridge.cpp" \
  -o "$ROOT/libqwen_gdn_bt64_gfx942_asm_v0_bridge.so"
