#!/usr/bin/env python3
"""Stage 6A: CUDA-graph, ABBA, full-graph and body-gap audit.

This is measurement-only.  It compares the current opt-in Stage 4 BT64
Avelang graph with vLLM's authoritative public entry on identical fixed
inputs.  Stage APIs return newly allocated tensors, so each callable is
prewarmed and CUDA-graph captured before timing; replays allocate neither
outputs nor intermediates.  No kernel or production dispatch is modified.
"""

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
REPO = HERE.parents[5]
COMPARE = REPO / "test/examples/linear_attention/vllm_compare"
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(COMPARE), str(STAGE2)]

from qwen_gdn_bt64_gfx942_asm_v0_experimental import qwen_gdn_bt64_gfx942_asm_v0  # noqa: E402
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_mfma_v2_s0,
    qwen_gdn_full_bt64_stage4_all_s0,
    qwen_gdn_full_bt64_stage4_all_s0_stages,
    qwen_gdn_kkt_bt64_mfma_v2_s0,
    qwen_gdn_w_u_bt64_mfma_v2_s1,
)
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import qwen_gdn_chunk_cumsum_avelang_v6_standalone  # noqa: E402
from qwen_gdn_solve_bt64_hierarchical_fp32_v1 import qwen_gdn_solve_bt64_hierarchical_fp32_v1  # noqa: E402
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_o import chunk_fwd_o  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd  # noqa: E402
from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum  # noqa: E402
from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril  # noqa: E402
from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd  # noqa: E402


BT = 64
OUTPUT_ATOL = 1.0 / 128.0
STATE_ATOL = 2.0e-2
IMPL_A = "avelang_stage4_bt64_hierarchical_v1"
IMPL_B = "vllm_authoritative_bt64"
IMPLEMENTATIONS = (IMPL_A, IMPL_B)
STAGES = ("cumsum", "kkt", "solve", "wu", "recurrence", "chunk_o", "cast")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tensor_meta(value: torch.Tensor | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
        "contiguous": bool(value.is_contiguous()),
        "bytes": value.numel() * value.element_size(),
        "data_ptr": f"0x{value.data_ptr():x}",
    }


def flatten_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in value:
            result.extend(flatten_tensors(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(flatten_tensors(item))
        return result
    return []


def event_time_ms(fn: Callable[[], None], start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def abba_labels(repeat: int) -> list[str]:
    labels: list[str] = []
    base = (IMPL_A, IMPL_B, IMPL_B, IMPL_A)
    while len(labels) < repeat * 2:
        labels.extend(base)
    return labels[: repeat * 2]


@dataclass
class CapturedGraph:
    name: str
    fn: Callable[[], Any]
    graph: torch.cuda.CUDAGraph
    result: Any
    capture_output_ptrs: list[str]

    @classmethod
    def create(cls, name: str, fn: Callable[[], Any]) -> "CapturedGraph":
        # JIT/autotune/module load and allocator work happen before capture.
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = fn()
        graph.replay()
        torch.cuda.synchronize()
        pointers = [f"0x{tensor.data_ptr():x}" for tensor in flatten_tensors(result)]
        return cls(name=name, fn=fn, graph=graph, result=result, capture_output_ptrs=pointers)

    def replay(self) -> None:
        self.graph.replay()


def vllm_manual_stages(inputs: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    q, k, v, g, beta, initial_state = inputs
    g_cumsum = chunk_local_cumsum(g, chunk_size=BT)
    a = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=g_cumsum, output_dtype=torch.float32)
    a_solved = solve_tril(A=a, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=a_solved, g_cumsum=g_cumsum, cu_seqlens=None)
    h_bf16, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g_cumsum, initial_state=initial_state, output_final_state=True,
        chunk_size=BT, save_new_value=True, cu_seqlens=None,
    )
    output = chunk_fwd_o(q=q, k=k, v=v_new, h=h_bf16, g=g_cumsum, scale=128 ** -0.5, chunk_size=BT)
    return {
        "g_cumsum": g_cumsum, "a": a, "a_solved": a_solved, "w": w, "u": u,
        "h_bf16": h_bf16, "v_new": v_new, "final_state": final_state, "output": output,
    }


def avelang_manual_stages(inputs: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    q, k, v, g, beta, initial_state = inputs
    return qwen_gdn_full_bt64_stage4_all_s0_stages(
        q, k, v, g, beta, initial_state=initial_state, solve_impl="hierarchical_fp32_v1"
    )


def full_call(implementation: str, inputs: tuple[torch.Tensor, ...]) -> Callable[[], tuple[torch.Tensor, torch.Tensor | None]]:
    q, k, v, g, beta, initial_state = inputs
    if implementation == IMPL_A:
        return lambda: qwen_gdn_full_bt64_stage4_all_s0(
            q, k, v, g, beta, initial_state=initial_state, output_final_state=True,
            solve_impl="hierarchical_fp32_v1",
        )
    return lambda: vllm_full(
        q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state, output_final_state=True,
        scale=128 ** -0.5, head_first=False, use_qk_l2norm_in_kernel=False,
    )


def stage_calls(implementation: str, inputs: tuple[torch.Tensor, ...], stages: dict[str, torch.Tensor]) -> dict[str, Callable[[], Any] | None]:
    q, k, v, g, beta, initial_state = inputs
    if implementation == IMPL_A:
        return {
            "cumsum": lambda: qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT),
            "kkt": lambda: qwen_gdn_kkt_bt64_mfma_v2_s0(k, stages["g_cumsum"], beta),
            "solve": lambda: qwen_gdn_solve_bt64_hierarchical_fp32_v1(stages["a"]),
            "wu": lambda: qwen_gdn_w_u_bt64_mfma_v2_s1(k, v, stages["g_cumsum"], beta, stages["a_solved"]),
            "recurrence": lambda: qwen_gdn_bt64_gfx942_asm_v0(
                k, stages["w"], stages["u"], stages["g_cumsum"], initial_state
            ),
            "chunk_o": lambda: qwen_gdn_chunk_o_bt64_mfma_v2_s0(
                q, k, stages["v_new"], stages["h_bf16"], stages["g_cumsum"]
            ),
            "cast": lambda: stages["output_fp32"].to(q.dtype),
        }
    cast = None if stages["output"].dtype == q.dtype else lambda: stages["output"].to(q.dtype)
    return {
        "cumsum": lambda: chunk_local_cumsum(g, chunk_size=BT),
        "kkt": lambda: chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=stages["g_cumsum"], output_dtype=torch.float32),
        "solve": lambda: solve_tril(A=stages["a"], output_dtype=k.dtype),
        "wu": lambda: recompute_w_u_fwd(k=k, v=v, beta=beta, A=stages["a_solved"], g_cumsum=stages["g_cumsum"], cu_seqlens=None),
        "recurrence": lambda: chunk_gated_delta_rule_fwd_h(
            k=k, w=stages["w"], u=stages["u"], g=stages["g_cumsum"], initial_state=initial_state,
            output_final_state=True, chunk_size=BT, save_new_value=True, cu_seqlens=None,
        ),
        "chunk_o": lambda: chunk_fwd_o(
            q=q, k=k, v=stages["v_new"], h=stages["h_bf16"], g=stages["g_cumsum"], scale=128 ** -0.5,
            chunk_size=BT,
        ),
        "cast": cast,
    }


def compare_full_outputs(a: Any, b: Any) -> dict[str, float]:
    a_out, a_state = a
    b_out, b_state = b
    return {
        "output_max_abs": float((a_out.float() - b_out.float()).abs().max().item()),
        "output_mean_abs": float((a_out.float() - b_out.float()).abs().mean().item()),
        "state_max_abs": float((a_state.float() - b_state.float()).abs().max().item()),
        "state_mean_abs": float((a_state.float() - b_state.float()).abs().mean().item()),
    }


def fixed_inputs(t: int) -> tuple[torch.Tensor, ...]:
    # The input seed depends only on T; A/B always receive exact same tensor objects.
    return make_inputs(t, 20260816 + t, "random", True)


def full_benchmark(t: int, args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    inputs = fixed_inputs(t)
    calls = {implementation: CapturedGraph.create(implementation, full_call(implementation, inputs)) for implementation in IMPLEMENTATIONS}
    correctness = compare_full_outputs(calls[IMPL_A].result, calls[IMPL_B].result)
    rows: list[dict[str, object]] = []
    for session in range(args.sessions):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(args.warmup):
            for implementation in IMPLEMENTATIONS:
                calls[implementation].replay()
        torch.cuda.synchronize()
        for sequence, implementation in enumerate(abba_labels(args.repeat)):
            latency = event_time_ms(calls[implementation].replay, start, end)
            rows.append({
                "scope": "full", "T": t, "chunks": t // BT, "session": session, "order": "ABBA",
                "sequence": sequence, "implementation": implementation, "latency_ms": latency,
            })
    meta = {
        "T": t, "inputs": {name: tensor_meta(value) for name, value in zip(("q", "k", "v", "g", "beta", "initial_state"), inputs)},
        "cuda_graph_capture": {implementation: {"output_ptrs": call.capture_output_ptrs} for implementation, call in calls.items()},
        "correctness": correctness,
        "accepted": correctness["output_max_abs"] <= OUTPUT_ATOL and correctness["state_max_abs"] <= STATE_ATOL,
    }
    return rows, meta


def body_benchmark(t: int, args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    inputs = fixed_inputs(t)
    stage_values = {IMPL_A: avelang_manual_stages(inputs), IMPL_B: vllm_manual_stages(inputs)}
    calls: dict[str, dict[str, CapturedGraph | None]] = {}
    for implementation in IMPLEMENTATIONS:
        calls[implementation] = {}
        for stage, fn in stage_calls(implementation, inputs, stage_values[implementation]).items():
            calls[implementation][stage] = None if fn is None else CapturedGraph.create(f"{implementation}:{stage}", fn)
    rows: list[dict[str, object]] = []
    for session in range(args.sessions):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for stage in STAGES:
            active = {impl: call for impl, call in ((impl, calls[impl][stage]) for impl in IMPLEMENTATIONS) if call is not None}
            for _ in range(args.warmup):
                for call in active.values():
                    call.replay()
            torch.cuda.synchronize()
            if not active:
                rows.append({"scope": "body", "stage": stage, "T": t, "chunks": t // BT, "session": session,
                             "order": "ABBA", "sequence": -1, "implementation": IMPL_B,
                             "latency_ms": 0.0, "not_materialized": True})
                continue
            for sequence, implementation in enumerate(abba_labels(args.body_repeat)):
                call = calls[implementation][stage]
                if call is None:
                    rows.append({"scope": "body", "stage": stage, "T": t, "chunks": t // BT, "session": session,
                                 "order": "ABBA", "sequence": sequence, "implementation": implementation,
                                 "latency_ms": 0.0, "not_materialized": True})
                    continue
                rows.append({"scope": "body", "stage": stage, "T": t, "chunks": t // BT, "session": session,
                             "order": "ABBA", "sequence": sequence, "implementation": implementation,
                             "latency_ms": event_time_ms(call.replay, start, end), "not_materialized": False})
    metadata = {}
    for implementation, values in stage_values.items():
        metadata[implementation] = {name: tensor_meta(value) for name, value in values.items()}
    return rows, metadata


def summary_rows(raw: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, int], list[float]] = {}
    for row in raw:
        key = (str(row["scope"]), str(row.get("stage", "total")), int(row["T"]), int(row["session"]))
        grouped.setdefault(key + (str(row["implementation"]),), []).append(float(row["latency_ms"]))
    per_session = []
    for (scope, stage, t, session, implementation), values in grouped.items():
        per_session.append({"scope": scope, "stage": stage, "T": t, "chunks": t // BT, "session": session,
                            "implementation": implementation, "median_ms": statistics.median(values),
                            "p10_ms": sorted(values)[max(0, len(values) // 10 - 1)],
                            "p90_ms": sorted(values)[min(len(values) - 1, len(values) * 9 // 10)]})
    result = list(per_session)
    keys = sorted({(row["scope"], row["stage"], row["T"]) for row in per_session})
    for scope, stage, t in keys:
        selected = [row for row in per_session if (row["scope"], row["stage"], row["T"]) == (scope, stage, t)]
        by_impl = {impl: [float(row["median_ms"]) for row in selected if row["implementation"] == impl] for impl in IMPLEMENTATIONS}
        a = statistics.median(by_impl[IMPL_A])
        b = statistics.median(by_impl[IMPL_B])
        result.append({"scope": scope, "stage": stage, "T": t, "chunks": t // BT, "session": "aggregate",
                       "implementation": "avelang_minus_vllm", "median_ms": a - b,
                       "p10_ms": min(by_impl[IMPL_A]) - max(by_impl[IMPL_B]),
                       "p90_ms": max(by_impl[IMPL_A]) - min(by_impl[IMPL_B]),
                       "gap_us": (a - b) * 1000.0})
    return result


def fit_slope(summary: list[dict[str, object]], scope: str, stage: str) -> list[dict[str, object]]:
    result = []
    for series in (*IMPLEMENTATIONS, "avelang_minus_vllm"):
        if series == "avelang_minus_vllm":
            rows = [row for row in summary if row["scope"] == scope and row["stage"] == stage and row["session"] == "aggregate"]
        else:
            # Median the five sessions at each T before fitting an implementation curve.
            values_by_t: dict[int, list[float]] = {}
            for row in summary:
                if row["scope"] == scope and row["stage"] == stage and row["session"] != "aggregate" and row["implementation"] == series:
                    values_by_t.setdefault(int(row["T"]), []).append(float(row["median_ms"]))
            rows = [{"chunks": t // BT, "median_ms": statistics.median(values)} for t, values in values_by_t.items()]
        if len(rows) < 2:
            continue
        xs = [float(row["chunks"]) for row in rows]
        ys = [float(row["median_ms"]) for row in rows]
        mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
        denom = sum((x - mean_x) ** 2 for x in xs)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
        intercept = mean_y - slope * mean_x
        result.append({"scope": scope, "stage": stage, "series": series, "intercept_ms": intercept,
                       "slope_ms_per_chunk": slope, "points": len(rows)})
    return result


def run(args: argparse.Namespace) -> None:
    patch_rocm_autotune()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 6A requires a HIP GPU")
    all_full: list[dict[str, object]] = []
    all_bodies: list[dict[str, object]] = []
    contracts: dict[str, object] = {"capture_policy": "warm/capture outside timing; replay only inside HIP events", "current_stream": True,
                                    "internal_stage_events": False, "full_entry": {IMPL_A: "qwen_gdn_full_bt64_stage4_all_s0(solve_impl=hierarchical_fp32_v1)", IMPL_B: "vllm.chunk_gated_delta_rule"}}
    for t in args.T:
        if args.mode in ("all", "full"):
            raw, meta = full_benchmark(t, args)
            all_full.extend(raw)
            contracts[str(t)] = meta
        if args.mode in ("all", "bodies"):
            raw, meta = body_benchmark(t, args)
            all_bodies.extend(raw)
            contracts.setdefault(str(t), {})["intermediates"] = meta
    if all_full:
        write_csv(args.out_dir / "full_raw.csv", all_full)
        full_summary = summary_rows(all_full)
        write_csv(args.out_dir / "full_summary.csv", full_summary)
        slopes = fit_slope(full_summary, "full", "total")
        write_csv(args.out_dir / "full_slope.csv", slopes)
    if all_bodies:
        write_csv(args.out_dir / "body_raw.csv", all_bodies)
        body_summary = summary_rows(all_bodies)
        write_csv(args.out_dir / "body_summary.csv", body_summary)
        slopes = []
        for stage in STAGES:
            slopes.extend(fit_slope(body_summary, "body", stage))
        write_csv(args.out_dir / "body_slopes.csv", slopes)
    write_json(args.out_dir / "frozen_measurement_contract.json", contracts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("all", "full", "bodies"), default="all")
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--body-repeat", type=int, default=100)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
