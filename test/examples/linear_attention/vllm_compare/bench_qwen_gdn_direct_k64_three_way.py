#!/usr/bin/env python3
"""Preallocated three-way direct-K64 recurrence-update diagnostic benchmark.

The current Triton HSACO does not expose an update-only V-new entry point.
For its control arm, this benchmark feeds the supplied BF16 V-new as ``v``
and supplies BF16 ``w=0``.  Therefore native Triton has the same update math
and ABI but still executes its fused pred pipeline.  That cost is recorded as
part of the native-control measurement rather than hidden.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

import repro_qwen_gdn_direct_k64_update_current_abi as mfma32
import repro_qwen_gdn_direct_k64_update_mfma16_current_abi as mfma16


HERE = Path(__file__).resolve().parent
LADDER = HERE.parent / "compile_bug/qwen_mfma32_lowering_ladder"
STAGE6R = LADDER / "codex_qwen_bt64_recurrence_reconciliation_stage6r"
NATIVE_HSACO = STAGE6R / "current_kernels/vllm/kernel.hsaco"
NATIVE_SYMBOL = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
NATIVE_BRIDGE = STAGE6R / "libstage6r_external_bridge.so"
BT, HV, DIM, WG = 64, 8, 128, 128


def _stats(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float | bool]:
    diff = (lhs.float() - rhs.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "bitwise_equal": bool(torch.equal(lhs, rhs)),
        "finite": bool(torch.isfinite(lhs.float()).all().item() and torch.isfinite(rhs.float()).all().item()),
    }


class NativeBridge:
    def __init__(self) -> None:
        if not NATIVE_BRIDGE.is_file() or not NATIVE_HSACO.is_file():
            raise RuntimeError(f"missing Stage-6R native bridge or HSACO under {STAGE6R}")
        self.library = ctypes.CDLL(str(NATIVE_BRIDGE))
        self.library.stage6r_external_recurrence_launch.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint64,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_int32, *([ctypes.c_void_p] * 8),
        ]
        self.library.stage6r_external_recurrence_launch.restype = ctypes.c_int
        self.library.stage6r_external_last_error.restype = ctypes.c_char_p

    def launch(
        self,
        k: torch.Tensor,
        v_new_input: torch.Tensor,
        w_zero: torch.Tensor,
        g: torch.Tensor,
        initial_state: torch.Tensor,
        h: torch.Tensor,
        v_new_output: torch.Tensor,
        final_state: torch.Tensor,
    ) -> None:
        status = self.library.stage6r_external_recurrence_launch(
            str(NATIVE_HSACO).encode(), NATIVE_SYMBOL.encode(),
            int(torch.cuda.current_stream(k.device).cuda_stream),
            4, 8, WG, 40960, int(k.shape[1]),
            *[
                ctypes.c_void_p(tensor.data_ptr())
                for tensor in (k, v_new_input, w_zero, v_new_output, g, h, initial_state, final_state)
            ],
        )
        if status:
            message = self.library.stage6r_external_last_error()
            raise RuntimeError(message.decode() if message else f"native bridge failed with status={status}")


@dataclass
class Call:
    name: str
    launch: Callable[[], None]
    h: torch.Tensor
    final_state: torch.Tensor
    v_new: torch.Tensor | None


def _raw_avelang_call(module, kernel_name: str, k, v_new, g, initial_state) -> Call:
    t = int(k.shape[1])
    h = torch.empty((1, t // BT, HV, DIM, DIM), device=k.device, dtype=torch.bfloat16)
    final_state = torch.empty((1, HV, DIM, DIM), device=k.device, dtype=torch.float32)
    kernel = getattr(module, kernel_name)

    def launch() -> None:
        kernel[lambda: ((HV * 2, 1, 1), (WG, 1, 1))](
            k, v_new, g, initial_state, h, final_state, t, t // BT, True, num_warps=2
        )

    launch()  # compile and module load are intentionally outside timing.
    torch.cuda.synchronize()
    return Call(module.__name__.replace("repro_qwen_gdn_", ""), launch, h, final_state, None)


def _native_call(k, v_new, g, initial_state) -> Call:
    t = int(k.shape[1])
    bridge = NativeBridge()
    w_zero = torch.zeros_like(v_new)
    h = torch.empty((1, t // BT, HV, DIM, DIM), device=k.device, dtype=torch.bfloat16)
    v_new_output = torch.empty_like(v_new)
    final_state = torch.empty((1, HV, DIM, DIM), device=k.device, dtype=torch.float32)

    def launch() -> None:
        bridge.launch(k, v_new, w_zero, g, initial_state, h, v_new_output, final_state)

    launch()
    torch.cuda.synchronize()
    return Call("current_triton_w0_control", launch, h, final_state, v_new_output)


def _event_ms(call: Call, start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    start.record()
    call.launch()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _session_rows(calls: list[Call], warmup: int, repeat: int, session: int, t: int) -> list[dict[str, object]]:
    for call in calls:
        for _ in range(warmup):
            call.launch()
    torch.cuda.synchronize()
    # Williams-like ABCCBA order gives every implementation both early and late
    # positions in each block without adding any allocations to the measured path.
    order = [0, 1, 2, 2, 1, 0] * repeat
    samples = {call.name: [] for call in calls}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for idx in order:
        call = calls[idx]
        samples[call.name].append(_event_ms(call, start, end))
    return [
        {
            "T": t,
            "chunks": t // BT,
            "session": session,
            "implementation": call.name,
            "median_ms": statistics.median(samples[call.name]),
            "p10_ms": sorted(samples[call.name])[max(0, int(len(samples[call.name]) * 0.1) - 1)],
            "p90_ms": sorted(samples[call.name])[min(len(samples[call.name]) - 1, int(len(samples[call.name]) * 0.9))],
            "warmup": warmup,
            "repeat_per_impl": len(samples[call.name]),
            "order": "ABCCBA",
        }
        for call in calls
    ]


def _summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        buckets.setdefault((int(row["T"]), str(row["implementation"])), []).append(float(row["median_ms"]))
    return [
        {
            "T": t,
            "chunks": t // BT,
            "implementation": name,
            "median_of_session_medians_ms": statistics.median(values),
            "session_count": len(values),
        }
        for (t, name), values in sorted(buckets.items())
    ]


def run(t: int, *, seed: int, warmup: int, repeat: int, sessions: int) -> dict[str, object]:
    k, v_new, g, initial_state = mfma32._make_inputs(t, seed)
    calls = [
        _raw_avelang_call(mfma32, "_qwen_gdn_direct_k64_update_current_abi_kernel", k, v_new, g, initial_state),
        _raw_avelang_call(mfma16, "_qwen_gdn_direct_k64_update_mfma16_current_abi_kernel", k, v_new, g, initial_state),
        _native_call(k, v_new, g, initial_state),
    ]
    reference_h, reference_final = mfma32.qwen_gdn_direct_k64_update_reference(k, v_new, g, initial_state)
    torch.cuda.synchronize()
    checks: dict[str, object] = {}
    for call in calls:
        checks[call.name] = {
            "h_vs_update_reference": _stats(call.h, reference_h),
            "final_state_vs_update_reference": _stats(call.final_state, reference_final),
        }
        if call.v_new is not None:
            checks[call.name]["v_new_vs_input"] = _stats(call.v_new, v_new)
    raw: list[dict[str, object]] = []
    for session in range(sessions):
        raw.extend(_session_rows(calls, warmup, repeat, session, t))
    return {"correctness": checks, "raw": raw, "summary": _summary(raw)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = {str(t): run(t, seed=args.seed + t, warmup=args.warmup, repeat=args.repeat, sessions=args.sessions) for t in args.T}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
