#!/usr/bin/env python3
"""Fresh-process BV64/BV32/current-Triton direct-K64 ownership benchmark."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BV64 = HERE / "repro_qwen_gdn_direct_k64_block_dot_ab.py"
BV32 = HERE / "repro_qwen_gdn_direct_k64_block_dot_bv32_coop.py"
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


def _bv64(tokens: int, warmup: int, repeat: int, seed: int) -> float:
    rows = _run(
        [
            sys.executable,
            str(BV64),
            "--lowering",
            "specialized",
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


def _bv32(tokens: int, warmup: int, repeat: int, seed: int) -> float:
    rows = _run(
        [
            sys.executable,
            str(BV32),
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
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    runners = {
        "bv64_specialized": _bv64,
        "bv32_coop_specialized": _bv32,
        "native_w0_control": _native,
    }
    orders = [
        ("bv64_specialized", "bv32_coop_specialized", "native_w0_control"),
        ("bv32_coop_specialized", "native_w0_control", "bv64_specialized"),
        ("native_w0_control", "bv64_specialized", "bv32_coop_specialized"),
    ]
    rows: list[dict[str, object]] = []
    for tokens in args.T:
        values = {name: [] for name in runners}
        for session in range(args.sessions):
            seed = args.seed + tokens * 101 + session
            for arm in orders[session % len(orders)]:
                values[arm].append(runners[arm](tokens, args.warmup, args.repeat, seed))
        for arm, samples in values.items():
            rows.append(
                {
                    "T": tokens,
                    "chunks": tokens // 64,
                    "implementation": arm,
                    "median_of_session_medians_ms": statistics.median(samples),
                    "session_values_ms": samples,
                    "sessions": args.sessions,
                }
            )
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            print("bv32_ownership," + ",".join(f"{key}={value}" for key, value in row.items()))


if __name__ == "__main__":
    main()
