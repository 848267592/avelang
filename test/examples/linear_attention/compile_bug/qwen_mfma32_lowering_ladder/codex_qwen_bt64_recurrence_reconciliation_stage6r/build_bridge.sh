#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
hipcc -shared -fPIC -O2 "$ROOT/stage6r_external_bridge.cpp" -o "$ROOT/libstage6r_external_bridge.so"
