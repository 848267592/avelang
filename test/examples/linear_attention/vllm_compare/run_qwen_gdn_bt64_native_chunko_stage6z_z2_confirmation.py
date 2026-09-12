#!/usr/bin/env python3
"""Fresh-process paired body confirmation for the Stage 6Z Z2 candidate."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _run_one(implementation: str, t: int, warmup: int, repeat: int) -> dict[str, object]:
    command = [
        sys.executable,
        str(HERE / "bench_qwen_gdn_bt64_native_chunko_stage6z.py"),
        "--T",
        str(t),
        "--implementation",
        implementation,
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
    ]
    result = subprocess.run(command, cwd=HERE.parents[3], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{implementation} T={t} failed:\n{result.stdout}")
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=(2048, 8192))
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raw: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for t in args.T:
        paired_us: list[float] = []
        for session in range(args.sessions):
            order = ("z1", "z2") if session % 2 == 0 else ("z2", "z1")
            measurements = {implementation: _run_one(implementation, t, args.warmup, args.repeat) for implementation in order}
            z1_ms = float(measurements["z1"]["hip_ms"]["median"])
            z2_ms = float(measurements["z2"]["hip_ms"]["median"])
            paired_us.append((z1_ms - z2_ms) * 1.0e3)
            raw.append(
                {
                    "T": t,
                    "session": session,
                    "order": list(order),
                    "z1": measurements["z1"],
                    "z2": measurements["z2"],
                    "z1_minus_z2_us": paired_us[-1],
                }
            )
        summary.append(
            {
                "T": t,
                "session_count": args.sessions,
                "z1_minus_z2_us": paired_us,
                "median_gain_us": statistics.median(paired_us),
                "mean_gain_us": statistics.fmean(paired_us),
                "all_sessions_positive": all(value > 0.0 for value in paired_us),
            }
        )

    payload = {
        "scope": "caller_owned_isolated_body_diagnostic",
        "contract": {
            "fresh_process": True,
            "graph_capture": False,
            "current_stream": True,
            "preallocated_outputs": True,
            "implementations": ["z1", "z2"],
            "rotating_order": "z1,z2 / z2,z1",
        },
        "warmup": args.warmup,
        "repeat": args.repeat,
        "raw": raw,
        "summary": summary,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
