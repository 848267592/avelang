#!/usr/bin/env python3
"""Dispatch one preallocated R4 tail-issue-control body for rocprof."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for entry in reversed(os.environ.get("AVELANG_SWP_PYTHONPATH", "").split(os.pathsep)):
    if entry:
        sys.path.insert(0, entry)

import torch

import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2
import repro_qwen_gdn_persistent_recurrence_r4_tail_issue as tail_issue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    args = parser.parse_args()
    tail_issue._set_tail_issue_lowering()
    k, w, u, g, initial_state = p2._make_long_case(args.T, args.seed)
    if args.dump_hsaco_dir is not None:
        tail_issue._HSACO_DUMP_DIR = args.dump_hsaco_dir
        result = tail_issue._run_kernel(k, w, u, g, initial_state)
        launch = result["_launch"]
    else:
        launch, _, _, _ = tail_issue.run_body(k, w, u, g, initial_state)
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    for _ in range(args.repeat):
        launch()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
