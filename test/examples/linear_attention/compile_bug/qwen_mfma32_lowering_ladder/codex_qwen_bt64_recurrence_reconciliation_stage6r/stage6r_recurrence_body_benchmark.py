#!/usr/bin/env python3
"""Stage 6R standalone recurrence body timing and external-bridge audit."""

from __future__ import annotations

import argparse
import csv
import ctypes
import importlib.util
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


HERE = Path(__file__).resolve().parent
LADDER = HERE.parent
STAGE6A = LADDER / "codex_qwen_bt64_full_graph_gap_stage6a"
REPO = LADDER.parents[5]
COMPARE = REPO / "test/examples/linear_attention/vllm_compare"
sys.path[:0] = [str(STAGE6A), str(COMPARE)]

import stage6a_full_graph_audit as stage6a  # noqa: E402
from qwen_gdn_bt64_gfx942_asm_v0_experimental import qwen_gdn_bt64_gfx942_asm_v0  # noqa: E402
from stage2_runner import patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h  # noqa: E402


BT, HV, VDIM, KDIM = 64, 8, 128, 128
CURRENT = HERE / "current_kernels"
BRIDGE = HERE / "libstage6r_external_bridge.so"
HISTORICAL = LADDER / "codex_triton_fullseq_asm_opt_audit/golden_fullseq/shared"
ASM = LADDER / "codex_qwen_asm_v0_integration/assembly/qwen_gdn_bt64_gfx942_asm_v0.hsaco"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stats(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, object]:
    diff = (lhs.float() - rhs.float()).abs()
    return {
        "bitwise_equal": bool(torch.equal(lhs, rhs)),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "finite": bool(torch.isfinite(lhs.float()).all().item() and torch.isfinite(rhs.float()).all().item()),
    }


class Bridge:
    def __init__(self) -> None:
        self.library = ctypes.CDLL(str(BRIDGE))
        self.library.stage6r_external_recurrence_launch.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint64,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_int32, *([ctypes.c_void_p] * 8),
        ]
        self.library.stage6r_external_recurrence_launch.restype = ctypes.c_int
        self.library.stage6r_external_last_error.restype = ctypes.c_char_p

    def launch(self, hsaco: Path, symbol: str, grid: tuple[int, int], block: int, lds: int,
               k: torch.Tensor, v: torch.Tensor, w: torch.Tensor, g: torch.Tensor,
               h0: torch.Tensor, h: torch.Tensor, v_new: torch.Tensor, ht: torch.Tensor) -> None:
        status = self.library.stage6r_external_recurrence_launch(
            str(hsaco).encode(), symbol.encode(), int(torch.cuda.current_stream(k.device).cuda_stream),
            grid[0], grid[1], block, lds, int(k.shape[1]),
            *[ctypes.c_void_p(value.data_ptr()) for value in (k, v, w, v_new, g, h, h0, ht)],
        )
        if status:
            message = self.library.stage6r_external_last_error()
            raise RuntimeError(message.decode() if message else f"external bridge launch failed: {status}")


@dataclass
class GraphCall:
    name: str
    graph: torch.cuda.CUDAGraph
    result: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    @classmethod
    def create(cls, name: str, fn: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> "GraphCall":
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = fn()
        graph.replay()
        torch.cuda.synchronize()
        return cls(name, graph, result)

    def replay(self) -> None:
        self.graph.replay()


def event_ms(call: GraphCall, start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    start.record()
    call.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def external_call(bridge: Bridge, name: str, hsaco: Path, symbol: str, grid: tuple[int, int], block: int, lds: int,
                  k: torch.Tensor, v: torch.Tensor, w: torch.Tensor, g: torch.Tensor, h0: torch.Tensor,
                  vnew_dtype: torch.dtype) -> tuple[str, Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    expected_k = (1, int(k.shape[1]), 4, KDIM)
    expected_v = (1, int(k.shape[1]), HV, VDIM)
    if (not hsaco.is_file() or tuple(k.shape) != expected_k or k.dtype != torch.bfloat16 or
            tuple(v.shape) != expected_v or tuple(w.shape) != expected_v or v.dtype != w.dtype or
            v.dtype not in (torch.bfloat16, torch.float32) or tuple(g.shape) != (1, int(k.shape[1]), HV) or
            g.dtype != torch.float32 or tuple(h0.shape) != (1, HV, VDIM, KDIM) or h0.dtype != torch.float32 or
            vnew_dtype != v.dtype or any(not value.is_cuda or not value.is_contiguous() for value in (k, v, w, g, h0))):
        raise ValueError("Stage 6R bridge requires contiguous fixed Qwen tensors with matched v/w/v_new dtype")
    t = int(k.shape[1])
    h = torch.empty((1, t // BT, HV, VDIM, KDIM), dtype=torch.bfloat16, device=k.device)
    v_new = torch.empty_like(v, dtype=vnew_dtype)
    ht = torch.empty((1, HV, VDIM, KDIM), dtype=torch.float32, device=k.device)

    def run() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bridge.launch(hsaco, symbol, grid, block, lds, k, v, w, g, h0, h, v_new, ht)
        return h, v_new, ht

    return name, run


def pairs(names: dict[str, GraphCall]) -> list[tuple[str, str, str]]:
    available = set(names)
    desired = [
        ("actual_native_vs_bridge", "vllm_actual_native_bf16", "vllm_actual_bridge_bf16"),
        ("asm_native_vs_bridge", "asm_v0_native_fp32", "asm_v0_bridge_fp32"),
        ("historical_original_vs_rebuilt", "historical_original_fp32", "historical_rebuilt_fp32"),
        ("historical_original_vs_asm", "historical_original_fp32", "asm_v0_bridge_fp32"),
        ("native_abi_asm_vs_current_vllm", "asm_v0_native_fp32", "vllm_actual_native_bf16"),
        ("same_fp32_values_asm_vs_current_vllm", "asm_v0_native_fp32", "vllm_current_fp32"),
    ]
    return [item for item in desired if item[1] in available and item[2] in available]


def run_timing(t: int, warmup: int, repeat: int, sessions: int, bridge: Bridge) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    inputs = stage6a.fixed_inputs(t)
    q, k, v, g, beta, h0 = inputs
    values = stage6a.vllm_manual_stages(inputs)
    w_bf16, u_bf16, g_cumsum = values["w"], values["u"], values["g_cumsum"]
    w_fp32, u_fp32 = w_bf16.float().contiguous(), u_bf16.float().contiguous()
    calls: dict[str, GraphCall] = {}
    calls["vllm_actual_native_bf16"] = GraphCall.create(
        "vllm_actual_native_bf16",
        lambda: chunk_gated_delta_rule_fwd_h(k, w_bf16, u_bf16, g_cumsum, None, h0, True, BT, True, None),
    )
    calls["vllm_current_fp32"] = GraphCall.create(
        "vllm_current_fp32",
        lambda: chunk_gated_delta_rule_fwd_h(k, w_fp32, u_fp32, g_cumsum, None, h0, True, BT, True, None),
    )
    calls["asm_v0_native_fp32"] = GraphCall.create(
        "asm_v0_native_fp32", lambda: qwen_gdn_bt64_gfx942_asm_v0(k, w_fp32, u_fp32, g_cumsum, h0)
    )
    actual_hsaco = CURRENT / "vllm/kernel.hsaco"
    name, fn = external_call(bridge, "vllm_actual_bridge_bf16", actual_hsaco,
                             "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", (4, 8), 128, 40960,
                             k, u_bf16, w_bf16, g_cumsum, h0, torch.bfloat16)
    calls[name] = GraphCall.create(name, fn)
    for name, hsaco, symbol in (
        ("asm_v0_bridge_fp32", ASM, "qwen_gdn_bt64_gfx942_asm_v0"),
        ("historical_original_fp32", HISTORICAL / "original.hsaco", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"),
        ("historical_rebuilt_fp32", HISTORICAL / "rebuilt.hsaco", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"),
    ):
        label, fn = external_call(bridge, name, hsaco, symbol, (4, 8), 256, 57344,
                                  k, u_fp32, w_fp32, g_cumsum, h0, torch.float32)
        calls[label] = GraphCall.create(label, fn)
    correctness: list[dict[str, object]] = []
    check_pairs = [
        ("actual_native_bf16", "vllm_actual_native_bf16", "vllm_actual_native_bf16"),
        ("actual_native_bf16", "vllm_actual_bridge_bf16", "vllm_actual_native_bf16"),
        ("same_fp32", "asm_v0_native_fp32", "vllm_current_fp32"),
        ("same_fp32", "asm_v0_bridge_fp32", "asm_v0_native_fp32"),
        ("same_fp32", "historical_original_fp32", "asm_v0_bridge_fp32"),
        ("same_fp32", "historical_rebuilt_fp32", "historical_original_fp32"),
        ("bf16_native_reference", "vllm_current_fp32", "vllm_actual_native_bf16"),
        ("bf16_native_reference", "asm_v0_native_fp32", "vllm_actual_native_bf16"),
    ]
    for contract, name, reference_name in check_pairs:
        call, reference = calls[name], calls[reference_name].result
        row: dict[str, object] = {"T": t, "implementation": name, "reference": reference_name, "comparison_contract": contract}
        for field, lhs, rhs in (("h", call.result[0], reference[0]), ("v_new", call.result[1], reference[1]), ("final_state", call.result[2], reference[2])):
            result = stats(lhs, rhs)
            for key, value in result.items():
                row[f"{field}_{key}"] = value
        correctness.append(row)
    rows: list[dict[str, object]] = []
    for pair, left, right in pairs(calls):
        for session in range(sessions):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(warmup):
                calls[left].replay()
                calls[right].replay()
            torch.cuda.synchronize()
            samples: dict[str, list[float]] = {left: [], right: []}
            labels = (left, right, right, left) * repeat
            for name in labels:
                samples[name].append(event_ms(calls[name], start, end))
            for name in (left, right):
                rows.append({
                    "T": t, "chunks": t // BT, "session": session, "comparison_pair": pair,
                    "implementation": name, "median_ms": statistics.median(samples[name]),
                    "p10_ms": sorted(samples[name])[max(0, int(len(samples[name]) * 0.1) - 1)],
                    "p90_ms": sorted(samples[name])[min(len(samples[name]) - 1, int(len(samples[name]) * 0.9))],
                    "warmup": warmup, "repeat_per_impl": len(samples[name]), "order": "ABBA",
                })
    return rows, correctness


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = {}
    for row in rows:
        key = (row["T"], row["chunks"], row["comparison_pair"], row["implementation"])
        groups.setdefault(key, []).append(float(row["median_ms"]))
    return [
        {"T": key[0], "chunks": key[1], "comparison_pair": key[2], "implementation": key[3],
         "median_of_session_medians_ms": statistics.median(values), "session_count": len(values)}
        for key, values in sorted(groups.items())
    ]


def slopes(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    by_impl: dict[str, list[tuple[float, float]]] = {}
    for row in summary:
        by_impl.setdefault(str(row["implementation"]), []).append((float(row["chunks"]), float(row["median_of_session_medians_ms"])))
    for implementation, points in sorted(by_impl.items()):
        if len(points) < 2:
            continue
        n = len(points)
        mean_x = sum(x for x, _ in points) / n
        mean_y = sum(y for _, y in points) / n
        denominator = sum((x - mean_x) ** 2 for x, _ in points)
        if denominator == 0.0:
            continue
        slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
        result.append({"implementation": implementation, "intercept_ms": mean_y - slope * mean_x, "slope_us_per_chunk": slope * 1000.0, "points": n})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 2048, 8192, 16384])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--sessions", type=int, default=5)
    args = parser.parse_args()
    if not BRIDGE.is_file():
        raise RuntimeError(f"build the Stage 6R audit bridge first: {BRIDGE}")
    patch_rocm_autotune()
    bridge = Bridge()
    raw: list[dict[str, object]] = []
    correctness: list[dict[str, object]] = []
    for t in args.T:
        if t % BT:
            raise ValueError("all T values must be divisible by 64")
        rows, checks = run_timing(t, args.warmup, args.repeat, args.sessions, bridge)
        raw.extend(rows)
        correctness.extend(checks)
    summary = summarize(raw)
    write_csv(HERE / "standalone_raw.csv", raw)
    write_csv(HERE / "standalone_summary.csv", summary)
    write_csv(HERE / "standalone_slopes.csv", slopes(summary))
    write_csv(HERE / "standalone_correctness.csv", correctness)
    print(json.dumps({"rows": len(raw), "summary_rows": len(summary), "correctness_rows": len(correctness)}, indent=2))


if __name__ == "__main__":
    main()
