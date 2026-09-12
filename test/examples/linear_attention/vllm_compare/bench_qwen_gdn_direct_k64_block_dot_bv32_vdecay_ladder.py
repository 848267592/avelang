#!/usr/bin/env python3
"""Preallocated session benchmark for the BV32 V-decay cost control."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASELINE = HERE / "repro_qwen_gdn_direct_k64_block_dot_bv32_coop.py"
PRECOMPUTED = HERE / "repro_qwen_gdn_direct_k64_block_dot_bv32_precomputed_vdecay.py"
NATIVE = HERE / "profile_qwen_gdn_direct_k64_three_way.py"


def _json_tail(output: str) -> object:
    lines = output.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].startswith("{"):
            return json.loads(lines[index])
        if lines[index] == "[":
            return json.loads("\n".join(lines[index:]))
    raise RuntimeError(f"missing JSON payload:\n{output}")


def _run(command: list[str]) -> object:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=HERE,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return _json_tail(completed.stdout)


def _row(script: Path, tokens: int, warmup: int, repeat: int, seed: int) -> float:
    rows = _run(
        [
            sys.executable,
            str(script),
            "--T",
            str(tokens),
            "--warmup",
            str(warmup),
            "--repeat",
            str(repeat),
            "--seed",
            str(seed),
            "--no-check",
            "--json",
        ]
    )
    return float(rows[0]["median_ms"])


def _native(tokens: int, warmup: int, repeat: int, seed: int) -> float:
    row = _run(
        [
            sys.executable,
            str(NATIVE),
            "--implementation",
            "native",
            "--T",
            str(tokens),
            "--warmup",
            str(warmup),
            "--repeat",
            str(repeat),
            "--seed",
            str(seed),
        ]
    )
    return float(row["median_ms"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    runners = {
        "bv32_specialized": lambda t, s: _row(BASELINE, t, args.warmup, args.repeat, s),
        "bv32_precomputed_vdecay": lambda t, s: _row(PRECOMPUTED, t, args.warmup, args.repeat, s),
        "native_w0_control": lambda t, s: _native(t, args.warmup, args.repeat, s),
    }
    orders = [
        ("bv32_specialized", "bv32_precomputed_vdecay", "native_w0_control"),
        ("bv32_precomputed_vdecay", "native_w0_control", "bv32_specialized"),
        ("native_w0_control", "bv32_specialized", "bv32_precomputed_vdecay"),
    ]
    rows: list[dict[str, object]] = []
    for tokens in args.T:
        samples = {name: [] for name in runners}
        for session in range(args.sessions):
            seed = args.seed + tokens * 101 + session
            for arm in orders[session % len(orders)]:
                samples[arm].append(runners[arm](tokens, seed))
        for arm, values in samples.items():
            rows.append(
                {
                    "T": tokens,
                    "chunks": tokens // 64,
                    "implementation": arm,
                    "median_of_session_medians_ms": statistics.median(values),
                    "session_values_ms": values,
                    "sessions": args.sessions,
                }
            )
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    for row in rows:
        print("vdecay_ladder," + ",".join(f"{key}={value}" for key, value in row.items()))


if __name__ == "__main__":
    main()
