#!/usr/bin/env python3
"""Read-only clock/power sampler for a Stage 5D long-running profile case."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
HARNESS = HERE / "downstream_state_coupling_harness.py"


def sample_metric() -> tuple[str, str]:
    if shutil.which("amd-smi"):
        command = ["amd-smi", "metric", "-g", "0", "--json"]
    elif shutil.which("rocm-smi"):
        command = ["rocm-smi", "--showclocks", "--showmeminfo", "vram", "--showpower", "--showtemp", "--json"]
    else:
        return "unavailable", "{}"
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return " ".join(command), result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("graph_a", "graph_b"), required=True)
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    command = [
        "python3",
        str(HARNESS),
        "--mode",
        "profile",
        "--profile-case",
        args.case,
        "--T",
        str(args.T),
        "--warmup",
        "20",
        "--repeat",
        "20000",
        "--out-dir",
        str(HERE),
    ]
    process = subprocess.Popen(command)
    rows: list[dict[str, object]] = []
    deadline = time.monotonic() + args.seconds
    while process.poll() is None and time.monotonic() < deadline:
        metric_command, payload = sample_metric()
        rows.append({"time_ns": time.time_ns(), "case": args.case, "metric_command": metric_command, "payload_json": payload})
        time.sleep(args.interval)
    process.wait()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("time_ns", "case", "metric_command", "payload_json"))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"case": args.case, "samples": len(rows), "returncode": process.returncode}))


if __name__ == "__main__":
    main()
