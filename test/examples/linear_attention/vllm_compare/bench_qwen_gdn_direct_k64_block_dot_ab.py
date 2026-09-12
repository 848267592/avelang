#!/usr/bin/env python3
"""Fresh-process body benchmark for direct-K64 block-dot generic/specialized.

Each compiler lowering is launched in its own process so the JIT cache cannot
reuse a code object compiled under the other environment selection. Native is
the existing current-Triton W=0 control and therefore retains its fused pred
work; it is a diagnostic control, not a pure update-only implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BLOCK_DOT = HERE / "repro_qwen_gdn_direct_k64_block_dot_ab.py"
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


def _block_dot(lowering: str, t: int, warmup: int, repeat: int, seed: int) -> float:
    rows = _run([
        sys.executable, str(BLOCK_DOT), "--lowering", lowering, "--T", str(t),
        "--warmup", str(warmup), "--repeat", str(repeat), "--seed", str(seed),
        "--no-check", "--json",
    ])
    return float(rows[0]["median_ms"])


def _native(t: int, warmup: int, repeat: int, seed: int) -> float:
    row = _run([
        sys.executable, str(NATIVE), "--implementation", "native", "--T", str(t),
        "--warmup", str(warmup), "--repeat", str(repeat), "--seed", str(seed),
    ])
    return float(row["median_ms"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for t in args.T:
        session_values = {name: [] for name in ("generic", "specialized", "native_w0_control")}
        # Rotate first arm between sessions. Every arm is still a fresh
        # process because AVELANG_BLOCK_DOT_LOWERING is a late compiler mode.
        orders = [
            ("generic", "specialized", "native_w0_control"),
            ("specialized", "native_w0_control", "generic"),
            ("native_w0_control", "generic", "specialized"),
        ]
        for session in range(args.sessions):
            for arm in orders[session % len(orders)]:
                seed = args.seed + t * 101 + session
                value = (
                    _native(t, args.warmup, args.repeat, seed)
                    if arm == "native_w0_control"
                    else _block_dot(arm, t, args.warmup, args.repeat, seed)
                )
                session_values[arm].append(value)
        for arm, values in session_values.items():
            median = statistics.median(values)
            rows.append({
                "T": t,
                "chunks": t // 64,
                "implementation": arm,
                "median_of_session_medians_ms": median,
                "session_values_ms": values,
                "sessions": args.sessions,
            })
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            print("block_dot_bench," + ",".join(f"{key}={value}" for key, value in row.items()))


if __name__ == "__main__":
    main()
