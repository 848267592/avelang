#!/usr/bin/env python3
"""Strict CUDA-graph A/B/C benchmark for the Stage 6B BT64 chunk-o O0 path."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_chunko_ownership_stage6b import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_ownership_o0,
    qwen_gdn_full_bt64_stage6b_chunko_o0,
)
from qwen_gdn_bt64_chunko_ownership_stage6b_o1 import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_ownership_o1,
    qwen_gdn_full_bt64_stage6b_chunko_o1,
)
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_mfma_v2_s0,
    qwen_gdn_full_bt64_stage4_all_s0,
    qwen_gdn_full_bt64_stage4_all_s0_stages,
)
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_o import chunk_fwd_o  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd  # noqa: E402
from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum  # noqa: E402
from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril  # noqa: E402
from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd  # noqa: E402


BT = 64
CURRENT = "avelang_stage4_current"
O0 = "avelang_stage6b_o0"
O1 = "avelang_stage6b_o1"
VLLM = "vllm_authoritative"
IMPLEMENTATIONS = (CURRENT, O0, O1, VLLM)
OUTPUT_ATOL = 1.0 / 128.0
STATE_ATOL = 2.0e-2


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _event_ms(fn: Callable[[], None], start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _abba(repeat: int) -> list[str]:
    labels: list[str] = []
    base = (CURRENT, O0, O1, VLLM, VLLM, O1, O0, CURRENT)
    while len(labels) < repeat * len(IMPLEMENTATIONS):
        labels.extend(base)
    return labels[: repeat * 3]


@dataclass
class CapturedGraph:
    graph: torch.cuda.CUDAGraph
    result: Any

    @classmethod
    def create(cls, fn: Callable[[], Any]) -> "CapturedGraph":
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = fn()
        graph.replay()
        torch.cuda.synchronize()
        return cls(graph=graph, result=result)

    def replay(self) -> None:
        self.graph.replay()


def _inputs(t: int) -> tuple[torch.Tensor, ...]:
    return make_inputs(t, 20260716 + t, "random", True)


def _vllm_stages(inputs: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    q, k, v, g, beta, initial_state = inputs
    g_cumsum = chunk_local_cumsum(g, chunk_size=BT)
    a = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=g_cumsum, output_dtype=torch.float32)
    a_solved = solve_tril(A=a, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=a_solved, g_cumsum=g_cumsum, cu_seqlens=None)
    h_bf16, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g_cumsum, initial_state=initial_state,
        output_final_state=True, chunk_size=BT, save_new_value=True, cu_seqlens=None,
    )
    return {"g_cumsum": g_cumsum, "h_bf16": h_bf16, "v_new": v_new, "final_state": final_state}


def _full_call(implementation: str, inputs: tuple[torch.Tensor, ...]) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    q, k, v, g, beta, h0 = inputs
    if implementation == CURRENT:
        return lambda: qwen_gdn_full_bt64_stage4_all_s0(
            q, k, v, g, beta, initial_state=h0, output_final_state=True, solve_impl="hierarchical_fp32_v1"
        )
    if implementation == O0:
        return lambda: qwen_gdn_full_bt64_stage6b_chunko_o0(
            q, k, v, g, beta, initial_state=h0, solve_impl="hierarchical_fp32_v1"
        )
    if implementation == O1:
        return lambda: qwen_gdn_full_bt64_stage6b_chunko_o1(
            q, k, v, g, beta, initial_state=h0, solve_impl="hierarchical_fp32_v1"
        )
    return lambda: vllm_full(
        q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True,
        scale=128 ** -0.5, head_first=False, use_qk_l2norm_in_kernel=False,
    )


def _body_call(implementation: str, inputs: tuple[torch.Tensor, ...], current: dict[str, torch.Tensor], vllm: dict[str, torch.Tensor]) -> Callable[[], torch.Tensor]:
    q, k, _, _, _, _ = inputs
    if implementation == CURRENT:
        return lambda: qwen_gdn_chunk_o_bt64_mfma_v2_s0(q, k, current["v_new"], current["h_bf16"], current["g_cumsum"])
    if implementation == O0:
        return lambda: qwen_gdn_chunk_o_bt64_ownership_o0(q, k, current["v_new"], current["h_bf16"], current["g_cumsum"])
    if implementation == O1:
        return lambda: qwen_gdn_chunk_o_bt64_ownership_o1(q, k, current["v_new"], current["h_bf16"], current["g_cumsum"])
    return lambda: chunk_fwd_o(
        q=q, k=k, v=vllm["v_new"], h=vllm["h_bf16"], g=vllm["g_cumsum"], scale=128 ** -0.5, chunk_size=BT
    )


def _error(actual: tuple[torch.Tensor, torch.Tensor], reference: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    output, final_state = actual
    ref_output, ref_final_state = reference
    return {
        "output_max_abs": float((output.float() - ref_output.float()).abs().max().item()),
        "output_mean_abs": float((output.float() - ref_output.float()).abs().mean().item()),
        "state_max_abs": float((final_state.float() - ref_final_state.float()).abs().max().item()),
        "state_mean_abs": float((final_state.float() - ref_final_state.float()).abs().mean().item()),
    }


def _benchmark(t: int, scope: str, args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    inputs = _inputs(t)
    if scope == "full":
        calls = {name: CapturedGraph.create(_full_call(name, inputs)) for name in IMPLEMENTATIONS}
        metadata = {
            "current_vs_o0": _error(calls[O0].result, calls[CURRENT].result),
            "current_vs_o1": _error(calls[O1].result, calls[CURRENT].result),
            "o0_vs_vllm": _error(calls[O0].result, calls[VLLM].result),
            "o1_vs_vllm": _error(calls[O1].result, calls[VLLM].result),
        }
        metadata["accepted"] = (
            metadata["o0_vs_vllm"]["output_max_abs"] <= OUTPUT_ATOL
            and metadata["o0_vs_vllm"]["state_max_abs"] <= STATE_ATOL
            and metadata["o1_vs_vllm"]["output_max_abs"] <= OUTPUT_ATOL
            and metadata["o1_vs_vllm"]["state_max_abs"] <= STATE_ATOL
        )
    else:
        current_stages = qwen_gdn_full_bt64_stage4_all_s0_stages(
            *inputs[:5], initial_state=inputs[5], solve_impl="hierarchical_fp32_v1"
        )
        vllm_stages = _vllm_stages(inputs)
        calls = {
            name: CapturedGraph.create(_body_call(name, inputs, current_stages, vllm_stages))
            for name in IMPLEMENTATIONS
        }
        body_error_o0 = (calls[O0].result - calls[CURRENT].result).abs()
        body_error_o1 = (calls[O1].result - calls[CURRENT].result).abs()
        metadata = {
            "o0_vs_current_fp32_max_abs": float(body_error_o0.max().item()),
            "o0_vs_current_fp32_mean_abs": float(body_error_o0.mean().item()),
            "o1_vs_current_fp32_max_abs": float(body_error_o1.max().item()),
            "o1_vs_current_fp32_mean_abs": float(body_error_o1.mean().item()),
            "accepted": float(body_error_o0.max().item()) <= 4.0e-3 and float(body_error_o1.max().item()) <= 4.0e-3,
        }
    rows: list[dict[str, object]] = []
    for session in range(args.sessions):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(args.warmup):
            for call in calls.values():
                call.replay()
        torch.cuda.synchronize()
        for sequence, name in enumerate(_abba(args.repeat)):
            rows.append({
                "scope": scope, "T": t, "chunks": t // BT, "session": session,
                "order": "ABCCBA", "sequence": sequence, "implementation": name,
                "latency_ms": _event_ms(calls[name].replay, start, end),
            })
    return rows, metadata


def _summary(raw: list[dict[str, object]]) -> list[dict[str, object]]:
    session_rows: list[dict[str, object]] = []
    for scope in sorted({str(row["scope"]) for row in raw}):
        for t in sorted({int(row["T"]) for row in raw if row["scope"] == scope}):
            for session in sorted({int(row["session"]) for row in raw if row["scope"] == scope and int(row["T"]) == t}):
                for implementation in IMPLEMENTATIONS:
                    values = [float(row["latency_ms"]) for row in raw if row["scope"] == scope and int(row["T"]) == t and int(row["session"]) == session and row["implementation"] == implementation]
                    session_rows.append({
                        "scope": scope, "T": t, "chunks": t // BT, "session": session, "implementation": implementation,
                        "median_ms": statistics.median(values), "p10_ms": sorted(values)[max(0, len(values) // 10 - 1)],
                        "p90_ms": sorted(values)[min(len(values) - 1, len(values) * 9 // 10)],
                    })
    aggregate: list[dict[str, object]] = []
    for scope in sorted({row["scope"] for row in session_rows}):
        for t in sorted({int(row["T"]) for row in session_rows if row["scope"] == scope}):
            medians = {name: [float(row["median_ms"]) for row in session_rows if row["scope"] == scope and int(row["T"]) == t and row["implementation"] == name] for name in IMPLEMENTATIONS}
            agg = {name: statistics.median(medians[name]) for name in IMPLEMENTATIONS}
            for name in IMPLEMENTATIONS:
                aggregate.append({"scope": scope, "T": t, "chunks": t // BT, "session": "aggregate", "implementation": name, "median_ms": agg[name], "p10_ms": min(medians[name]), "p90_ms": max(medians[name])})
            for name in (CURRENT, O0, O1):
                aggregate.append({"scope": scope, "T": t, "chunks": t // BT, "session": "aggregate", "implementation": f"{name}_minus_vllm", "median_ms": agg[name] - agg[VLLM], "gap_us": (agg[name] - agg[VLLM]) * 1000.0})
            aggregate.append({"scope": scope, "T": t, "chunks": t // BT, "session": "aggregate", "implementation": "o0_minus_current", "median_ms": agg[O0] - agg[CURRENT], "gain_us": (agg[CURRENT] - agg[O0]) * 1000.0})
            aggregate.append({"scope": scope, "T": t, "chunks": t // BT, "session": "aggregate", "implementation": "o1_minus_current", "median_ms": agg[O1] - agg[CURRENT], "gain_us": (agg[CURRENT] - agg[O1]) * 1000.0})
    return session_rows + aggregate


def _slopes(summary: list[dict[str, object]], scope: str) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    names = (*IMPLEMENTATIONS, f"{CURRENT}_minus_vllm", f"{O0}_minus_vllm", f"{O1}_minus_vllm", "o0_minus_current", "o1_minus_current")
    for name in names:
        rows = [row for row in summary if row["scope"] == scope and row["session"] == "aggregate" and row["implementation"] == name]
        if len(rows) < 2:
            continue
        xs = [float(row["chunks"]) for row in rows]
        ys = [float(row["median_ms"]) for row in rows]
        mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
        denominator = sum((x - mean_x) ** 2 for x in xs)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
        result.append({"scope": scope, "series": name, "intercept_ms": mean_y - slope * mean_x, "slope_ms_per_chunk": slope})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("body", "full", "all"), default="all")
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_chunko_ownership_stage6b")
    args = parser.parse_args()
    patch_rocm_autotune()
    if not torch.cuda.is_available():
        raise RuntimeError("requires a ROCm/CUDA GPU")
    raw: list[dict[str, object]] = []
    correctness: dict[str, object] = {"measurement": "HIP-event CUDA-graph replay", "current_stream": True, "abba": "ABCDDCBA"}
    scopes = ("body", "full") if args.mode == "all" else (args.mode,)
    for scope in scopes:
        for t in args.T:
            rows, meta = _benchmark(t, scope, args)
            raw.extend(rows)
            correctness.setdefault(scope, {})[str(t)] = meta
    summary = _summary(raw)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / ("body_raw.csv" if args.mode == "body" else "full_raw.csv" if args.mode == "full" else "stage6b_raw.csv"), raw)
    _write_csv(args.out_dir / ("body_summary.csv" if args.mode == "body" else "full_summary.csv" if args.mode == "full" else "stage6b_summary.csv"), summary)
    for scope in scopes:
        _write_csv(args.out_dir / f"{scope}_slopes.csv", _slopes(summary, scope))
    _write_json(args.out_dir / "correctness_summary.json", correctness)
    print(json.dumps(correctness, indent=2))


if __name__ == "__main__":
    main()
