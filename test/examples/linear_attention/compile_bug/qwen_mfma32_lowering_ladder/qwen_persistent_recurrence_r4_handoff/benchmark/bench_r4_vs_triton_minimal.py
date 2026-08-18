#!/usr/bin/env python3
"""Small R4-vs-Triton recurrence-body benchmark.

It keeps only the two useful teaching arms: the extracted R4 Avelang kernel and
the current-vLLM Triton body. Compilation and allocation happen before HIP
events; this is a body diagnostic, not a public-API ranking.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


HERE = Path(__file__).resolve().parent


def _find_repo() -> Path:
    for parent in (HERE, *HERE.parents):
        if (parent / "test/examples/linear_attention/vllm_compare").is_dir():
            return parent
    raise RuntimeError("could not find the Avelang repository")


REPO = _find_repo()
sys.path.insert(0, str(HERE.parent / "avelang"))
sys.path.insert(0, str(REPO / "test/examples/linear_attention/vllm_compare"))

import bench_qwen_gdn_direct_k64_bv32_full_sequence_b0 as controls  # noqa: E402
import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2  # noqa: E402
import r4_kernel_minimal as r4  # noqa: E402


BT = 64


@dataclass
class Call:
    name: str
    launch: Callable[[], None]
    h: torch.Tensor
    v_new: torch.Tensor
    final_state: torch.Tensor


def _r4_call(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
             initial_state: torch.Tensor) -> Call:
    t = int(k.shape[1])
    chunks = t // BT
    h = torch.empty((1, chunks, 8, 128, 128), device=k.device, dtype=torch.bfloat16)
    pred_f32 = torch.empty((1, t, 8, 128), device=k.device, dtype=torch.float32)
    pred_bf16 = torch.empty_like(w)
    v_new = torch.empty_like(u)
    v_decay = torch.empty_like(u)
    state_after = torch.empty((1, chunks, 8, 128, 128), device=k.device, dtype=torch.float32)
    final_state = torch.empty_like(initial_state)

    controls.p0  # Ensure the shared reference contract is imported before JIT.
    controls.b0.p1._set_c0_lowering()
    os.environ["AVELANG_PERSISTENT_RECURRENCE_LOWERING"] = "gfx942_bt64_bv32_joint_v4"
    os.environ["AVELANG_QWEN_K64_PIPELINE_LOWERING"] = "distributed"

    def launch() -> None:
        r4._qwen_gdn_persistent_recurrence_r4_joint_v4_kernel[
            lambda: ((r4.GRID, 1, 1), (r4.WORKGROUP, 1, 1))
        ](
            k, w, u, g, initial_state, h, pred_f32, pred_bf16, v_new,
            v_decay, state_after, final_state, t, chunks,
            False, True, False, False, num_warps=2,
        )

    launch()
    return Call("avelang_r4_joint_v4_minimal", launch, h, v_new, final_state)


def _event_ms(call: Call) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    call.launch()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _worker(t: int, seed: int, warmup: int, repeat: int, session: int) -> dict:
    k, w, u, g, initial_state = p2._make_long_case(t, seed)
    r4_call = _r4_call(k, w, u, g, initial_state)
    triton_call = controls._direct_triton_call(k, w, u, g, initial_state)
    reference = controls.b0._reference(k, w, u, g, initial_state)
    checks = {
        call.name: {
            "h": controls._stats(call.h, reference["h"]),
            "v_new": controls._stats(call.v_new, reference["v_new"]),
            "final_state": controls._stats(call.final_state, reference["final_state"]),
        }
        for call in (r4_call, triton_call)
    }
    if not all(item["finite"] for check in checks.values() for item in check.values()):
        raise RuntimeError(f"non-finite output: {checks}")
    calls = (r4_call, triton_call)
    for call in calls:
        for _ in range(warmup):
            call.launch()
    torch.cuda.synchronize()
    order = (0, 1) if session % 2 == 0 else (1, 0)
    samples = {call.name: [] for call in calls}
    for _ in range(repeat):
        for index in order:
            samples[calls[index].name].append(_event_ms(calls[index]))
    return {
        "T": t,
        "session": session,
        "checks": checks,
        "median_ms": {name: statistics.median(values) for name, values in samples.items()},
        "samples_ms": samples,
        "order": order,
        "graph_capture": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048, 8192])
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--session", type=int, default=0)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(_worker(args.T[0], args.seed, args.warmup, args.repeat, args.session), sort_keys=True))
        return
    rows = []
    for t in args.T:
        if t % BT:
            raise ValueError(f"T={t} must be divisible by {BT}")
        for session in range(args.sessions):
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--T", str(t),
                       "--sessions", "1", "--warmup", str(args.warmup), "--repeat", str(args.repeat),
                       "--seed", str(args.seed + t), "--session", str(session)]
            output = subprocess.check_output(command, text=True)
            rows.append(json.loads(output.splitlines()[-1]))
    print(json.dumps({"contract": {"graph_capture": False, "timing": "HIP event"}, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
