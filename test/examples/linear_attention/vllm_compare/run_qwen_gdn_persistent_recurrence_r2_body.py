#!/usr/bin/env python3
"""Dispatch one preallocated R2 joint-recurrence body for rocprof collection."""

from __future__ import annotations

import argparse
import os
import sys

# rocprofv3 can prepend the container's installed AveLang package. Keep this
# runner reproducible against the build selected by the caller.
for entry in reversed(os.environ.get("AVELANG_R2_PYTHONPATH", "").split(os.pathsep)):
    if entry:
        sys.path.insert(0, entry)

import torch

import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2
import repro_qwen_gdn_persistent_recurrence_r2 as r2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    r2._set_r2_lowering()
    k, w, u, g, initial_state = p2._make_long_case(args.T, args.seed)
    launch, _, _, _ = r2.run_body(k, w, u, g, initial_state)
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    for _ in range(args.repeat):
        launch()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
