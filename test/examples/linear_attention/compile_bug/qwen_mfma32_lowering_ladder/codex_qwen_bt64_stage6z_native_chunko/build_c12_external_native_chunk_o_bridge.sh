#!/bin/sh
set -eu

out_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
g++ -O3 -shared -fPIC -std=c++17 \
  "$out_dir/c12_native_chunk_o_bridge.cpp" \
  -o "$out_dir/libc12_native_chunk_o_bridge.so" \
  -D__HIP_PLATFORM_AMD__ \
  -I/opt/rocm/include \
  -L/opt/rocm/lib -lamdhip64
