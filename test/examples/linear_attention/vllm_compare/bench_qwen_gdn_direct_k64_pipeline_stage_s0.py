#!/usr/bin/env python3
"""Fresh-process same-source A/B benchmark and byte comparison for S0."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
REPRO = HERE / "repro_qwen_gdn_direct_k64_pipeline_stage_s0.py"


def _run_worker(mode: str, *, seed: int, warmup: int, repeat: int, save: Path) -> dict[str, Any]:
    env = os.environ.copy()
    completed = subprocess.run(
        [
            sys.executable,
            str(REPRO),
            "--mode",
            mode,
            "--seed",
            str(seed),
            "--warmup",
            str(warmup),
            "--repeat",
            str(repeat),
            "--save",
            str(save),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    for line in reversed(completed.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"S0 {mode} worker produced no JSON:\n{completed.stdout}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows: dict[str, list[dict[str, Any]]] = {"immediate": [], "distributed": []}
    with tempfile.TemporaryDirectory(prefix="qwen_s0_") as raw:
        root = Path(raw)
        for session in range(args.sessions):
            order = ["immediate", "distributed"] if session % 2 == 0 else ["distributed", "immediate"]
            saved: dict[str, Path] = {}
            for mode in order:
                path = root / f"{mode}_{session}.pt"
                row = _run_worker(
                    mode,
                    seed=args.seed + session,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    save=path,
                )
                row["session"] = session
                rows[mode].append(row)
                saved[mode] = path
            immediate = torch.load(saved["immediate"], weights_only=True)
            distributed = torch.load(saved["distributed"], weights_only=True)
            equal = {name: bool(torch.equal(immediate[name], distributed[name])) for name in immediate}
            if not all(equal.values()):
                raise RuntimeError(f"S0 same-source A/B output mismatch in session {session}: {equal}")

    medians = {
        mode: statistics.median(float(row["median_ms"]) for row in values)
        for mode, values in rows.items()
    }
    result: dict[str, Any] = {
        "seed": args.seed,
        "sessions": args.sessions,
        "immediate_session_medians_ms": [row["median_ms"] for row in rows["immediate"]],
        "distributed_session_medians_ms": [row["median_ms"] for row in rows["distributed"]],
        "median_ms": medians,
        "distributed_vs_immediate_ratio": medians["distributed"] / medians["immediate"],
        "all_snapshot_checks": all(bool(row["snapshot_byte_equal"]) for values in rows.values() for row in values),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
