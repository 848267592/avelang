#!/usr/bin/env python3
"""Fresh-process R5 recurrence-body diagnostic against native and Triton controls."""

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
import repro_qwen_gdn_persistent_recurrence_r1 as r1
import repro_qwen_gdn_persistent_recurrence_r2 as r2
import repro_qwen_gdn_persistent_recurrence_r3 as r3
import repro_qwen_gdn_persistent_recurrence_r4 as r4
import repro_qwen_gdn_persistent_recurrence_r5 as r5


BT = r1.BT


def _r1_call(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, initial_state: torch.Tensor
) -> b0bench.Call:
    launch, h, v_new, final_state = r1.run_body(k, w, u, g, initial_state)
    return b0bench.Call("avelang_r1_joint_v1", launch, h, v_new, final_state)


def _r2_call(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, initial_state: torch.Tensor
) -> b0bench.Call:
    launch, h, v_new, final_state = r2.run_body(k, w, u, g, initial_state)
    return b0bench.Call("avelang_r2_joint_v2", launch, h, v_new, final_state)


def _r3_call(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, initial_state: torch.Tensor
) -> b0bench.Call:
    launch, h, v_new, final_state = r3.run_body(k, w, u, g, initial_state)
    return b0bench.Call("avelang_r3_joint_v3", launch, h, v_new, final_state)


def _r4_call(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, initial_state: torch.Tensor
) -> b0bench.Call:
    launch, h, v_new, final_state = r4.run_body(k, w, u, g, initial_state)
    return b0bench.Call("avelang_r4_joint_v4", launch, h, v_new, final_state)


def _r5_call(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, initial_state: torch.Tensor
) -> b0bench.Call:
    launch, h, v_new, final_state = r5.run_body(k, w, u, g, initial_state)
    return b0bench.Call("avelang_r5_joint_v5", launch, h, v_new, final_state)


def _worker(t: int, *, seed: int, warmup: int, repeat: int, session: int) -> dict[str, Any]:
    r5._set_r5_lowering()
    k, w, u, g, initial_state = p2._make_long_case(t, seed)
    calls = [
        b0bench._b0_call(k, w, u, g, initial_state),
        _r1_call(k, w, u, g, initial_state),
        _r2_call(k, w, u, g, initial_state),
        _r3_call(k, w, u, g, initial_state),
        _r4_call(k, w, u, g, initial_state),
        _r5_call(k, w, u, g, initial_state),
        b0bench._direct_triton_call(k, w, u, g, initial_state),
        b0bench._bridge_call(k, w, u, g, initial_state),
    ]
    torch.cuda.synchronize()
    reference = b0bench.b0._reference(k, w, u, g, initial_state)
    checks = {
        call.name: {
            "h_vs_device_contract": b0bench._stats(call.h, reference["h"]),
            "v_new_vs_device_contract": b0bench._stats(call.v_new, reference["v_new"]),
            "final_state_vs_device_contract": b0bench._stats(call.final_state, reference["final_state"]),
        }
        for call in calls
    }
    if not all(all(bool(value["finite"]) for value in check.values()) for check in checks.values()):
        raise RuntimeError("non-finite output in R5 body comparison")
    for call in calls:
        for _ in range(warmup):
            call.launch()
    torch.cuda.synchronize()
    base_order = list(range(len(calls))) + list(reversed(range(len(calls))))
    offset = session % len(calls)
    order = base_order[offset:] + base_order[:offset]
    samples = {call.name: [] for call in calls}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        for index in order:
            call = calls[index]
            samples[call.name].append(b0bench._event_ms(call, start, end))
    rows = []
    for call in calls:
        values = sorted(samples[call.name])
        rows.append({
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
        })
    return {"T": t, "session": session, "checks": checks, "rows": rows}


def _parent(args: argparse.Namespace) -> dict[str, Any]:
    raw: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for t in args.T:
        for session in range(args.sessions):
            command = [
                sys.executable, str(Path(__file__).resolve()), "--worker", "--T", str(t),
                "--seed", str(args.seed + t), "--warmup", str(args.warmup),
                "--repeat", str(args.repeat), "--session", str(session),
            ]
            output = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout
            line = next((row for row in reversed(output.splitlines()) if row.startswith("{")), None)
            if line is None:
                raise RuntimeError(f"worker emitted no JSON payload:\n{output}")
            payload = json.loads(line)
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
            "order": "rotating palindromic eight-arm",
            "interpretation": "recurrence-body diagnostic only",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048, 8192])
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(_worker(args.T[0], seed=args.seed, warmup=args.warmup, repeat=args.repeat, session=args.session)))
        return
    result = _parent(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
