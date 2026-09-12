#!/usr/bin/env python3
"""Fresh-process diagnostic body benchmark for native full-recurrence B2.

This is intentionally not an Eager public-API ranking.  It measures exactly
the recurrence body with fixed current-vLLM BF16 ABI buffers.  Each worker is
a separate Python process; allocation, JIT compilation and module load occur
before the HIP-event timing region.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p0 as p0
import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2
import repro_qwen_gdn_direct_k64_bv32_full_sequence_b0 as b0
import repro_qwen_gdn_direct_k64_bv32_full_sequence_b2_lookahead as b2


HERE = Path(__file__).resolve().parent
LADDER = HERE.parent / "compile_bug/qwen_mfma32_lowering_ladder"
STAGE6R = LADDER / "codex_qwen_bt64_recurrence_reconciliation_stage6r"
NATIVE_HSACO = STAGE6R / "current_kernels/vllm/kernel.hsaco"
NATIVE_SYMBOL = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
NATIVE_BRIDGE = STAGE6R / "libstage6r_external_bridge.so"
BT = p0.BT
H_V = p0.H_V
KDIM = p0.KDIM


def _stats(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float | bool]:
    diff = (lhs.float() - rhs.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "byte_equal": bool(torch.equal(lhs, rhs)),
        "finite": bool(torch.isfinite(lhs.float()).all().item() and torch.isfinite(rhs.float()).all().item()),
    }


class NativeBridge:
    """Existing Stage-6R bridge; B2 does not modify its code object or ABI."""

    def __init__(self) -> None:
        if not NATIVE_BRIDGE.is_file() or not NATIVE_HSACO.is_file():
            raise RuntimeError(f"missing Stage-6R native bridge or HSACO under {STAGE6R}")
        self.library = ctypes.CDLL(str(NATIVE_BRIDGE))
        self.library.stage6r_external_recurrence_launch.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_int32,
            *([ctypes.c_void_p] * 8),
        ]
        self.library.stage6r_external_recurrence_launch.restype = ctypes.c_int
        self.library.stage6r_external_last_error.restype = ctypes.c_char_p

    def launch(
        self,
        k: torch.Tensor,
        u: torch.Tensor,
        w: torch.Tensor,
        g: torch.Tensor,
        initial_state: torch.Tensor,
        h: torch.Tensor,
        v_new: torch.Tensor,
        final_state: torch.Tensor,
    ) -> None:
        status = self.library.stage6r_external_recurrence_launch(
            str(NATIVE_HSACO).encode(),
            NATIVE_SYMBOL.encode(),
            int(torch.cuda.current_stream(k.device).cuda_stream),
            4,
            8,
            128,
            40960,
            int(k.shape[1]),
            *[
                ctypes.c_void_p(value.data_ptr())
                for value in (k, u, w, v_new, g, h, initial_state, final_state)
            ],
        )
        if status:
            message = self.library.stage6r_external_last_error()
            raise RuntimeError(message.decode() if message else f"external bridge failed with status={status}")


@dataclass
class Call:
    name: str
    launch: Callable[[], None]
    h: torch.Tensor
    v_new: torch.Tensor
    final_state: torch.Tensor


def _direct_triton_call(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
) -> Call:
    import vllm.model_executor.layers.fla.ops.chunk_delta_h as triton_h

    t = int(k.shape[1])
    h = torch.empty((1, t // BT, H_V, KDIM, KDIM), device=k.device, dtype=torch.bfloat16)
    v_new = torch.empty_like(u)
    final_state = torch.empty_like(initial_state)

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return ((KDIM + int(meta["BV"]) - 1) // int(meta["BV"]), H_V)

    def launch() -> None:
        triton_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
            k=k,
            v=u,
            w=w,
            v_new=v_new,
            g=g,
            gk=None,
            h=h,
            h0=initial_state,
            ht=final_state,
            cu_seqlens=None,
            chunk_offsets=None,
            T=t,
            H=H_V,
            Hg=p0.H_K,
            K=KDIM,
            V=KDIM,
            BT=BT,
        )

    launch()
    return Call("current_triton_direct_preallocated", launch, h, v_new, final_state)


def _bridge_call(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
) -> Call:
    t = int(k.shape[1])
    bridge = NativeBridge()
    h = torch.empty((1, t // BT, H_V, KDIM, KDIM), device=k.device, dtype=torch.bfloat16)
    v_new = torch.empty_like(u)
    final_state = torch.empty_like(initial_state)

    def launch() -> None:
        bridge.launch(k, u, w, g, initial_state, h, v_new, final_state)

    launch()
    return Call("current_vllm_hsaco_external_bridge", launch, h, v_new, final_state)


def _b2_call(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
) -> Call:
    launch, h, v_new, final_state = b2.run_body(k, w, u, g, initial_state)
    return Call("avelang_b2_one_chunk_lookahead", launch, h, v_new, final_state)


def _b0_call(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
) -> Call:
    launch, h, v_new, final_state = b0.run_body(k, w, u, g, initial_state)
    return Call("avelang_b0_full_sequence", launch, h, v_new, final_state)


def _event_ms(call: Call, start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    start.record()
    call.launch()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _worker(t: int, *, seed: int, warmup: int, repeat: int, session: int) -> dict[str, Any]:
    if t % BT:
        raise ValueError(f"T={t} is not divisible by BT={BT}")
    b2.p1._set_c0_lowering()
    k, w, u, g, initial_state = p2._make_long_case(t, seed)
    calls = [
        _b0_call(k, w, u, g, initial_state),
        _b2_call(k, w, u, g, initial_state),
        _direct_triton_call(k, w, u, g, initial_state),
        _bridge_call(k, w, u, g, initial_state),
    ]
    torch.cuda.synchronize()

    reference = b2._reference(k, w, u, g, initial_state)
    checks: dict[str, Any] = {}
    for call in calls:
        checks[call.name] = {
            "h_vs_device_contract": _stats(call.h, reference["h"]),
            "v_new_vs_device_contract": _stats(call.v_new, reference["v_new"]),
            "final_state_vs_device_contract": _stats(call.final_state, reference["final_state"]),
        }
    if not all(value["h_vs_device_contract"]["finite"] and value["v_new_vs_device_contract"]["finite"] and value["final_state_vs_device_contract"]["finite"] for value in checks.values()):
        raise RuntimeError("non-finite output in B2 diagnostic comparison")

    for call in calls:
        for _ in range(warmup):
            call.launch()
    torch.cuda.synchronize()
    # Rotate the initial position across fresh processes; each implementation
    # appears twice per Williams-like block without using graph replay.
    base_order = [0, 1, 2, 3, 3, 2, 1, 0]
    order = base_order[session % 4 :] + base_order[: session % 4]
    samples = {call.name: [] for call in calls}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        for index in order:
            call = calls[index]
            samples[call.name].append(_event_ms(call, start, end))
    rows = [
        {
            "T": t,
            "chunks": t // BT,
            "session": session,
            "implementation": call.name,
            "median_ms": statistics.median(samples[call.name]),
            "p10_ms": sorted(samples[call.name])[max(0, len(samples[call.name]) // 10 - 1)],
            "p90_ms": sorted(samples[call.name])[min(len(samples[call.name]) - 1, (len(samples[call.name]) * 9) // 10)],
            "warmup": warmup,
            "repeat_per_impl": len(samples[call.name]),
            "order": order,
            "graph_capture": False,
            "current_stream": int(torch.cuda.current_stream(k.device).cuda_stream),
        }
        for call in calls
    ]
    return {"T": t, "session": session, "checks": checks, "rows": rows}


def _summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((int(row["T"]), str(row["implementation"])), []).append(float(row["median_ms"]))
    summary = [
        {
            "T": t,
            "chunks": t // BT,
            "implementation": name,
            "median_of_session_medians_ms": statistics.median(values),
            "session_count": len(values),
        }
        for (t, name), values in sorted(grouped.items())
    ]
    slopes: list[dict[str, Any]] = []
    for name in sorted({str(row["implementation"]) for row in summary}):
        points = [(float(row["chunks"]), float(row["median_of_session_medians_ms"])) for row in summary if row["implementation"] == name]
        if len(points) < 2:
            continue
        mean_x = sum(point[0] for point in points) / len(points)
        mean_y = sum(point[1] for point in points) / len(points)
        denominator = sum((x - mean_x) ** 2 for x, _ in points)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
        slopes.append({"implementation": name, "intercept_ms": mean_y - slope * mean_x, "slope_us_per_chunk": slope * 1000.0, "points": len(points)})
    return summary, slopes


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
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
            # Experimental lowering logs are emitted before the worker's final
            # machine-readable line. Keep those diagnostics without allowing
            # them to corrupt the fresh-process benchmark protocol.
            json_line = next((line for line in reversed(result.stdout.splitlines()) if line.startswith("{")), None)
            if json_line is None:
                raise RuntimeError(f"worker emitted no JSON payload:\n{result.stdout}")
            payload = json.loads(json_line)
            raw.extend(payload["rows"])
            checks.append({"T": t, "session": session, "checks": payload["checks"]})
    summary, slopes = _summarize(raw)
    return {"raw": raw, "checks": checks, "summary": summary, "slopes": slopes, "contract": {"fresh_process_sessions": args.sessions, "graph_capture": False, "timing": "HIP event", "allocation_or_compile_in_timing": False}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048, 8192])
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(_worker(args.T[0], seed=args.seed, warmup=args.warmup, repeat=args.repeat, session=args.session), sort_keys=True))
        return
    result = _parent(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
