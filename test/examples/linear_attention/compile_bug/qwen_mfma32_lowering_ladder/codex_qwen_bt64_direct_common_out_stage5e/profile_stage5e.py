#!/usr/bin/env python3
"""Conditional whole-graph/direct-tail rocprof driver for Stage 5E."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
HARNESS = HERE / "direct_common_out_harness.py"
ROCPROF = Path("/opt/rocm/bin/rocprofv3")
BASIC_COUNTERS = (
    "SQ_INSTS_MFMA",
    "SQ_INSTS_VALU",
    "SQ_INSTS_SALU",
    "SQ_INSTS_VMEM",
    "SQ_INSTS_LDS",
    "OccupancyPercent",
)


def run(command: list[str], output: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    if output is None:
        subprocess.run(command, check=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, text=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--control", nargs="+", choices=("none", "warm", "perturb", "prime"), default=["none", "warm", "perturb"])
    parser.add_argument("--cache-counter", action="append", default=[])
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--operation", choices=("tail", "full"), default="tail")
    parser.add_argument("--out-dir", type=Path, default=HERE / "rocprof")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run([str(ROCPROF), "--list-avail"], args.out_dir / "available_counters.txt")
    counters = list(args.cache_counter) if args.cache_only else [*BASIC_COUNTERS, *args.cache_counter]
    if not counters:
        raise ValueError("at least one rocprof counter is required")
    manifest = []
    for t in args.T:
        for control in args.control:
            for solve_impl in ("v18", "hierarchical_fp32_v1"):
                if args.operation == "full" and control != "none":
                    continue
                destination = args.out_dir / f"{args.operation}_t{t}_{control}_{solve_impl}"
                command = [
                    str(ROCPROF),
                    "--kernel-trace",
                    "--pmc",
                    *counters,
                    "-d",
                    str(destination),
                    "-o",
                    "stage5e",
                    "-f",
                    "csv",
                    "--",
                    sys.executable,
                    str(HARNESS),
                    "--mode",
                    "profile",
                    "--T",
                    str(t),
                    "--solve-impl",
                    solve_impl,
                    "--control",
                    control,
                    "--profile-repeat",
                    str(args.repeat),
                    "--profile-operation",
                    args.operation,
                ]
                run(command)
                manifest.append({"operation": args.operation, "T": t, "control": control, "solve_impl": solve_impl, "directory": str(destination), "counters": counters})
    (args.out_dir / f"profile_manifest_{args.operation}.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
