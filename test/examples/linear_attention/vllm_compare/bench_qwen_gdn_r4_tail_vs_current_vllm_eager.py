#!/usr/bin/env python3
"""Fresh-process, no-graph recurrence-body timing for R4-tail and current vLLM.

The vLLM arm calls its eager Python recurrence API and is never replaced by a
captured HSACO/ctypes launcher.  The AveLang arm calls the first-class R4-tail
recurrence op.  Both use preallocated B=1,Hk=4,Hv=8,K=V=128 BF16 inputs and
the same FP32 initial state; compilation is outside HIP-event timing.  vLLM's
public API owns its returned tensors, so its allocator path remains part of
that public-Eager measurement and is reported as such.  This is deliberately
a recurrence-body measurement, not a full GDN front-end benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch


HERE = Path(__file__).resolve().parent
LADDER = HERE.parent / "compile_bug" / "qwen_mfma32_lowering_ladder"
STAGE6A = LADDER / "codex_qwen_bt64_full_graph_gap_stage6a"
STAGE2 = LADDER / "codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(STAGE6A), str(STAGE2), str(HERE)]

from stage2_runner import patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_delta_h import (  # noqa: E402
    chunk_gated_delta_rule_fwd_h,
)

import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2  # noqa: E402
import repro_qwen_gdn_persistent_recurrence_r4_tail_issue as r4_tail  # noqa: E402
import repro_qwen_gdn_persistent_recurrence_r4_tail_issue_iopacket as iopacket  # noqa: E402


BT = 64
IMPLEMENTATIONS = ("r4_tail", "r4_tail_iopacket", "current_vllm_eager")


def _event_ms(launch: Callable[[], None], start: torch.cuda.Event,
              end: torch.cuda.Event) -> float:
    start.record()
    launch()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _launch(implementation: str, t: int, seed: int) -> Callable[[], None]:
    k, w, u, g_cumsum, initial_state = p2._make_long_case(t, seed)
    if implementation == "r4_tail":
        launch, _, _, _ = r4_tail.run_body(k, w, u, g_cumsum, initial_state)
        return launch
    if implementation == "r4_tail_iopacket":
        launch, _, _, _ = iopacket.run_body(k, w, u, g_cumsum, initial_state)
        return launch
    if implementation == "current_vllm_eager":
        # This is the current-vLLM eager recurrence API, not an external
        # code-object launch.  Its output allocation is part of the API call,
        # exactly as it is in regular eager execution.
        def launch() -> None:
            chunk_gated_delta_rule_fwd_h(
                k=k, w=w, u=u, g=g_cumsum, initial_state=initial_state,
                output_final_state=True, chunk_size=BT, save_new_value=True,
                cu_seqlens=None,
            )
        return launch
    raise ValueError(f"unknown implementation: {implementation}")


def worker(args: argparse.Namespace) -> dict[str, Any]:
    patch_rocm_autotune()
    t = args.T[0]
    launch = _launch(args.implementation, t, args.seed + t)
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = [_event_ms(launch, start, end) for _ in range(args.repeat)]
    ordered = sorted(samples)
    return {
        "implementation": args.implementation,
        "T": t,
        "chunks": t // BT,
        "session": args.session,
        "median_ms": statistics.median(ordered),
        "p10_ms": ordered[max(0, len(ordered) // 10 - 1)],
        "p90_ms": ordered[min(len(ordered) - 1, len(ordered) * 9 // 10)],
        "fresh_process": True,
        "graph_capture": False,
        "timing_surface": "public_eager_recurrence_api",
        "warmup": args.warmup,
        "repeat": args.repeat,
    }


def _child(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=True)
    payload = next((line for line in reversed(result.stdout.splitlines())
                    if line.startswith("{")), None)
    if payload is None:
        raise RuntimeError(f"worker emitted no JSON payload:\n{result.stdout}")
    return json.loads(payload)


def parent(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for t in args.T:
        for session in range(args.sessions):
            order = IMPLEMENTATIONS if session % 2 == 0 else tuple(reversed(IMPLEMENTATIONS))
            for implementation in order:
                rows.append(_child([
                    sys.executable, str(HERE / Path(__file__).name), "--worker",
                    "--implementation", implementation, "--T", str(t),
                    "--seed", str(args.seed), "--session", str(session),
                    "--warmup", str(args.warmup), "--repeat", str(args.repeat),
                ]))
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((int(row["T"]), str(row["implementation"])), []).append(float(row["median_ms"]))
    summary = [{
        "T": t, "chunks": t // BT, "implementation": impl,
        "median_of_session_medians_ms": statistics.median(values),
        "session_count": len(values),
    } for (t, impl), values in sorted(grouped.items())]
    by_key = {(int(row["T"]), str(row["implementation"])): row for row in summary}
    slopes = []
    low, high = min(args.T), max(args.T)
    if low != high:
        for implementation in IMPLEMENTATIONS:
            slopes.append({
                "implementation": implementation,
                "interval": f"{low}->{high}",
                "us_per_chunk": 1000.0 * (
                    float(by_key[(high, implementation)]["median_of_session_medians_ms"])
                    - float(by_key[(low, implementation)]["median_of_session_medians_ms"])
                ) / ((high - low) // BT),
            })
    return {
        "contract": {
            "fresh_process_each_implementation_length_session": True,
            "graph_capture": False,
            "private_hsaco_launch": False,
            "compilation_in_timing": False,
            "public_vllm_output_allocation_in_timing": True,
            "shape": "B1,Hk4,Hv8,K128,V128,BF16,head_first=false,nonzero-W",
            "scope": "recurrence body only",
        },
        "raw": rows,
        "summary": summary,
        "slopes": slopes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[1024, 2048, 8192])
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if any(t % BT for t in args.T):
        parser.error("every T must be divisible by BT=64")
    if args.worker:
        if args.implementation is None:
            parser.error("--worker requires --implementation")
        result = worker(args)
    else:
        result = parent(args)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True) if args.worker else json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
