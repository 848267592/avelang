#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"${HIPCC:-/opt/rocm/bin/hipcc}" -std=c++17 -O3 -shared -fPIC "$ROOT/qwen_triton_external_full_bridge.cpp" -o "$ROOT/libqwen_triton_external_full_bridge.so"
