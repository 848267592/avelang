#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"${HIPCC:-/opt/rocm/bin/hipcc}" -std=c++17 -O3 "$ROOT/standalone_harness.cpp" -o "$ROOT/asm_v0_harness"
