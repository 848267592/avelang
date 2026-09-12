#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
hipcc -shared -fPIC -O2 "$ROOT/stage6s_external_solve_bridge.cpp" \
  -o "$ROOT/libstage6s_external_solve_bridge.so"
