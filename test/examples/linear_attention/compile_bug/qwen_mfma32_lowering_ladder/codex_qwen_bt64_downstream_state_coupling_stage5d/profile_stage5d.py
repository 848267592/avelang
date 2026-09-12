#!/usr/bin/env python3
"""Reproducible whole-graph rocprof driver for Stage 5D.

The driver is intentionally separate from the Stage 4 module.  It first
records the counters actually advertised by rocprofv3, then profiles each
control in a fresh process without inserting events inside the graph.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
HARNESS = HERE / "downstream_state_coupling_harness.py"
ROCPROF = Path("/opt/rocm/bin/rocprofv3")
BASIC_COUNTERS = (
    "SQ_INSTS_MFMA",
    "SQ_INSTS_VALU",
    "SQ_INSTS_SALU",
    "SQ_INSTS_VMEM",
    "SQ_INSTS_LDS",
    "OccupancyPercent",
)
CASES = (
    "graph_a",
    "graph_b",
    "canonical_a",
    "canonical_b",
    "same_pointer_a",
    "same_pointer_b",
    "cold_a",
    "cold_b",
    "prime_a",
    "prime_b",
)


def run(command: list[str], *, stdout: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    if stdout is None:
        subprocess.run(command, check=True)
    else:
        stdout.parent.mkdir(parents=True, exist_ok=True)
        with stdout.open("w") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, text=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--case", choices=CASES, nargs="+", default=list(CASES))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--cache-counter", action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=HERE / "rocprof")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # This file is the authority for whether a cache counter is supported.
    run([str(ROCPROF), "--list-avail"], stdout=args.out_dir / "available_counters.txt")
    counters = [*BASIC_COUNTERS, *args.cache_counter]
    manifest: list[dict[str, object]] = []
    for t in args.T:
        for case in args.case:
            destination = args.out_dir / f"t{t}_{case}"
            command = [
                str(ROCPROF),
                "--kernel-trace",
                "--pmc",
                *counters,
                "-d",
                str(destination),
                "-o",
                "stage5d",
                "-f",
                "csv",
                "--",
                sys.executable,
                str(HARNESS),
                "--mode",
                "profile",
                "--profile-case",
                case,
                "--T",
                str(t),
                "--warmup",
                str(args.warmup),
                "--repeat",
                str(args.repeat),
                "--out-dir",
                str(HERE),
            ]
            run(command)
            manifest.append({"T": t, "case": case, "directory": str(destination), "counters": counters})
    (args.out_dir / "profile_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
