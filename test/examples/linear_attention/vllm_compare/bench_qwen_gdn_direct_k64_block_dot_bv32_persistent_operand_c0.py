#!/usr/bin/env python3
"""Fresh-process C0 benchmark: immediate typed tiles versus persistent blocks."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPRO = HERE / "repro_qwen_gdn_direct_k64_block_dot_bv32_coop.py"
NATIVE = HERE / "profile_qwen_gdn_direct_k64_three_way.py"


def _json_tail(output: str) -> object:
    lines = output.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].startswith("{"):
            return json.loads(lines[index])
        if lines[index] == "[":
            return json.loads("\n".join(lines[index:]))
    raise RuntimeError(f"missing JSON payload:\n{output}")


def _run(command: list[str], *, operand: str | None = None) -> object:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if operand is not None:
        env["AVELANG_BLOCK_DOT_OPERAND_LOWERING"] = operand
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


def _arm(tokens: int, warmup: int, repeat: int, seed: int, operand: str) -> float:
    rows = _run(
        [
            sys.executable,
            str(REPRO),
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
        ],
        operand=operand,
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
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--order-offset", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    runners = {
        "typed_immediate32": lambda t, s: _arm(t, args.warmup, args.repeat, s, "typed_vector"),
        "persistent_typed_block": lambda t, s: _arm(
            t, args.warmup, args.repeat, s, "persistent_typed_block"
        ),
        "native_triton_w0_control": lambda t, s: _native(t, args.warmup, args.repeat, s),
    }
    orders = [
        ("typed_immediate32", "persistent_typed_block", "native_triton_w0_control"),
        ("persistent_typed_block", "native_triton_w0_control", "typed_immediate32"),
        ("native_triton_w0_control", "typed_immediate32", "persistent_typed_block"),
    ]
    rows: list[dict[str, object]] = []
    for tokens in args.T:
        samples = {name: [] for name in runners}
        for session in range(args.sessions):
            seed = args.seed + tokens * 101 + session
            for arm in orders[(session + args.order_offset) % len(orders)]:
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
        print("persistent_operand_c0," + ",".join(f"{key}={value}" for key, value in row.items()))


if __name__ == "__main__":
    main()
