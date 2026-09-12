#!/usr/bin/env python3
"""Fresh-process diagnostic body benchmark for Direct-K64 BV32 B1 stream32.

This is intentionally not an Eager public-API ranking. It compares four
preallocated recurrence bodies on the current HIP stream: B1, B0, the current
Triton implementation, and the immutable Stage-6R external-HSACO bridge.
Compilation, module load, allocation, and CUDA Graph capture are excluded.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

import bench_qwen_gdn_direct_k64_bv32_full_sequence_b0 as b0bench
import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2
import repro_qwen_gdn_direct_k64_bv32_full_sequence_b1_stream32 as b1


BT = b0bench.BT


def _b1_call(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
) -> b0bench.Call:
    launch, h, v_new, final_state = b1.run_body(k, w, u, g, initial_state)
    return b0bench.Call("avelang_b1_stream32_full_sequence", launch, h, v_new, final_state)


def _checks(calls: list[b0bench.Call], reference: dict[str, torch.Tensor]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for call in calls:
        result[call.name] = {
            "h_vs_device_contract": b0bench._stats(call.h, reference["h"]),
            "v_new_vs_device_contract": b0bench._stats(call.v_new, reference["v_new"]),
            "final_state_vs_device_contract": b0bench._stats(call.final_state, reference["final_state"]),
        }
    if not all(
        check["h_vs_device_contract"]["finite"]
        and check["v_new_vs_device_contract"]["finite"]
        and check["final_state_vs_device_contract"]["finite"]
        for check in result.values()
    ):
        raise RuntimeError("non-finite output in B1 diagnostic comparison")
    return result


def _worker(t: int, *, seed: int, warmup: int, repeat: int, session: int) -> dict[str, Any]:
    if t % BT:
        raise ValueError(f"T={t} is not divisible by BT={BT}")
    b1._set_b1_lowering()
    k, w, u, g, initial_state = p2._make_long_case(t, seed)
    calls = [
        _b1_call(k, w, u, g, initial_state),
        b0bench._b0_call(k, w, u, g, initial_state),
        b0bench._direct_triton_call(k, w, u, g, initial_state),
        b0bench._bridge_call(k, w, u, g, initial_state),
    ]
    torch.cuda.synchronize()
    checks = _checks(calls, b0bench.b0._reference(k, w, u, g, initial_state))

    for call in calls:
        for _ in range(warmup):
            call.launch()
    torch.cuda.synchronize()

    # A palindromic balanced block gives every implementation each local
    # predecessor/successor relation. Rotate it across independent processes.
    base_order = [0, 1, 2, 3, 3, 2, 1, 0]
    rotation = session % len(calls)
    order = base_order[rotation:] + base_order[:rotation]
    samples = {call.name: [] for call in calls}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        for index in order:
            call = calls[index]
            samples[call.name].append(b0bench._event_ms(call, start, end))

    rows = []
    for call in calls:
        values = sorted(samples[call.name])
        rows.append(
            {
                "T": t,
                "chunks": t // BT,
                "session": session,
                "implementation": call.name,
                "median_ms": statistics.median(values),
                "p10_ms": values[max(0, len(values) // 10 - 1)],
                "p90_ms": values[min(len(values) - 1, (len(values) * 9) // 10)],
                "warmup": warmup,
                "repeat_per_impl": len(values),
                "order": order,
                "graph_capture": False,
                "current_stream": int(torch.cuda.current_stream(k.device).cuda_stream),
            }
        )
    return {"T": t, "session": session, "checks": checks, "rows": rows}


def _parent(args: argparse.Namespace) -> dict[str, Any]:
    raw: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for t in args.T:
        for session in range(args.sessions):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--T",
                str(t),
                "--seed",
                str(args.seed + t),
                "--warmup",
                str(args.warmup),
                "--repeat",
                str(args.repeat),
                "--session",
                str(session),
            ]
            output = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True
            ).stdout
            payload_line = next((line for line in reversed(output.splitlines()) if line.startswith("{")), None)
            if payload_line is None:
                raise RuntimeError(f"worker emitted no JSON payload:\n{output}")
            payload = json.loads(payload_line)
            raw.extend(payload["rows"])
            checks.append({"T": t, "session": session, "checks": payload["checks"]})
    summary, slopes = b0bench._summarize(raw)
    return {
        "raw": raw,
        "checks": checks,
        "summary": summary,
        "slopes": slopes,
        "contract": {
            "fresh_process_sessions": args.sessions,
            "graph_capture": False,
            "timing": "HIP event",
            "allocation_or_compile_in_timing": False,
            "order": "rotating palindromic block",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048, 8192])
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(_worker(args.T[0], seed=args.seed, warmup=args.warmup, repeat=args.repeat, session=args.session)))
        return
    result = _parent(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
