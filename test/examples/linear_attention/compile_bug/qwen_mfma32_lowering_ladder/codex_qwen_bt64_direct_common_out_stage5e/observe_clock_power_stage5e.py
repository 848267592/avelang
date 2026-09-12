#!/usr/bin/env python3
"""Read-only clock/power sampler around a long Stage 5E direct-out loop."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
HARNESS = HERE / "direct_common_out_harness.py"


def sample_metric() -> tuple[str, str]:
    if shutil.which("amd-smi"):
        command = ["amd-smi", "metric", "-g", "0", "--json"]
    elif shutil.which("rocm-smi"):
        command = ["rocm-smi", "--showclocks", "--showpower", "--showtemp", "--json"]
    else:
        return "unavailable", "{}"
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return " ".join(command), result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solve-impl", choices=("v18", "hierarchical_fp32_v1"), required=True)
    parser.add_argument("--control", choices=("none", "warm", "perturb", "prime"), default="none")
    parser.add_argument("--sequence", choices=("single", "abba"), default="single")
    parser.add_argument("--repeat", type=int, default=20000)
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    command = [
        "python3", str(HARNESS), "--mode", "profile", "--T", str(args.T),
        "--solve-impl", args.solve_impl, "--control", args.control,
        "--profile-repeat", str(args.repeat), "--profile-sequence", args.sequence,
    ]
    process = subprocess.Popen(command)
    rows = []
    deadline = time.monotonic() + args.seconds
    while process.poll() is None and time.monotonic() < deadline:
        metric_command, payload = sample_metric()
        rows.append({
            "time_ns": time.time_ns(),
            "solve_impl": args.solve_impl,
            "control": args.control,
            "sequence": args.sequence,
            "metric_command": metric_command,
            "payload_json": payload,
        })
        time.sleep(args.interval)
    process.wait()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ("time_ns", "solve_impl", "control", "sequence", "metric_command", "payload_json"))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"samples": len(rows), "returncode": process.returncode}))


if __name__ == "__main__":
    main()
