#!/usr/bin/env python3
"""Stage 6S full-graph BF16-boundary integration audit.

This is intentionally an opt-in measurement harness.  Graph A and Graph B
share the same captured Stage 5B hierarchical-solve HSACO and every unchanged
Stage 4 component.  The only graph-level delta is B's FP32/BF16 boundaries
and the Stage 6R current-vLLM recurrence bridge.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
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

from qwen_gdn_bt64_bf16_recurrence_full_stage6s import (  # noqa: E402
    qwen_gdn_bt64_stage6s_recurrence_bridge,
    qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge,
    qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge_stages,
    qwen_gdn_full_bt64_stage6s_current_asm,
    qwen_gdn_full_bt64_stage6s_current_asm_stages,
    stage6s_contract,
)
from qwen_gdn_bt64_gfx942_asm_v0_experimental import qwen_gdn_bt64_gfx942_asm_v0  # noqa: E402
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402


BT = 64
OUTPUT_ATOL = 1.0 / 128.0
STATE_ATOL = 2.0e-2
IMPLEMENTATIONS = ("graph_a_current_asm", "graph_b_stage6s_bridge", "graph_c_vllm")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tensor_meta(value: torch.Tensor | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "contiguous": bool(value.is_contiguous()),
        "bytes": value.numel() * value.element_size(),
        "data_ptr": f"0x{value.data_ptr():x}",
    }


def flatten_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in flatten_tensors(item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in flatten_tensors(item)]
    return []


@dataclass
class CapturedGraph:
    name: str
    graph: torch.cuda.CUDAGraph
    result: Any
    result_ptrs: list[str]

    @classmethod
    def create(cls, name: str, fn: Callable[[], Any]) -> "CapturedGraph":
        # All JIT/module-load/allocation work is deliberately outside timing.
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = fn()
        graph.replay()
        torch.cuda.synchronize()
        return cls(name, graph, result, [f"0x{x.data_ptr():x}" for x in flatten_tensors(result)])

    def replay(self) -> None:
        self.graph.replay()


def event_ms(fn: Callable[[], None], start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def full_call(name: str, values: tuple[torch.Tensor, ...]) -> Callable[[], tuple[torch.Tensor, torch.Tensor | None]]:
    q, k, v, g, beta, h0 = values
    if name == "graph_a_current_asm":
        return lambda: qwen_gdn_full_bt64_stage6s_current_asm(
            q, k, v, g, beta, initial_state=h0, output_final_state=True
        )
    if name == "graph_b_stage6s_bridge":
        return lambda: qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(
            q, k, v, g, beta, initial_state=h0, output_final_state=True
        )
    if name == "graph_c_vllm":
        return lambda: vllm_full(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0,
            output_final_state=True, scale=128 ** -0.5,
            head_first=False, use_qk_l2norm_in_kernel=False,
        )
    raise ValueError(f"unknown graph {name}")


def pair_order(repeat: int) -> list[tuple[str, str]]:
    # Three ABBA blocks give each graph equal executions and preserve a
    # position-balanced pairwise ordering: A/B, B/C, A/C.
    labels: list[tuple[str, str]] = []
    for _ in range(repeat):
        labels.extend((
            ("ABBA", "graph_a_current_asm"), ("ABBA", "graph_b_stage6s_bridge"),
            ("ABBA", "graph_b_stage6s_bridge"), ("ABBA", "graph_a_current_asm"),
            ("BCCB", "graph_b_stage6s_bridge"), ("BCCB", "graph_c_vllm"),
            ("BCCB", "graph_c_vllm"), ("BCCB", "graph_b_stage6s_bridge"),
            ("ACCA", "graph_a_current_asm"), ("ACCA", "graph_c_vllm"),
            ("ACCA", "graph_c_vllm"), ("ACCA", "graph_a_current_asm"),
        ))
    return labels


def compare(lhs: tuple[torch.Tensor, torch.Tensor | None], rhs: tuple[torch.Tensor, torch.Tensor | None]) -> dict[str, float]:
    lhs_out, lhs_state = lhs
    rhs_out, rhs_state = rhs
    assert lhs_state is not None and rhs_state is not None
    return {
        "output_max_abs": float((lhs_out.float() - rhs_out.float()).abs().max().item()),
        "output_mean_abs": float((lhs_out.float() - rhs_out.float()).abs().mean().item()),
        "final_state_max_abs": float((lhs_state.float() - rhs_state.float()).abs().max().item()),
        "final_state_mean_abs": float((lhs_state.float() - rhs_state.float()).abs().mean().item()),
    }


def error(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float]:
    delta = (lhs.float() - rhs.float()).abs()
    return {"max_abs": float(delta.max().item()), "mean_abs": float(delta.mean().item())}


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def summarise(raw: list[dict[str, object]], *, scope: str) -> tuple[list[dict[str, object]], dict[tuple[int, str], list[float]]]:
    grouped: dict[tuple[int, int, str], list[float]] = {}
    for row in raw:
        grouped.setdefault((int(row["T"]), int(row["session"]), str(row["implementation"])), []).append(float(row["latency_ms"]))
    summaries: list[dict[str, object]] = []
    per_t_impl: dict[tuple[int, str], list[float]] = {}
    for (t, session, implementation), values in sorted(grouped.items()):
        median = statistics.median(values)
        per_t_impl.setdefault((t, implementation), []).append(median)
        summaries.append({
            "scope": scope, "T": t, "chunks": t // BT, "session": session,
            "implementation": implementation, "median_ms": median,
            "p10_ms": quantile(values, 0.1), "p90_ms": quantile(values, 0.9),
            "samples": len(values),
        })
    for (t, implementation), values in sorted(per_t_impl.items()):
        summaries.append({
            "scope": scope, "T": t, "chunks": t // BT, "session": "aggregate",
            "implementation": implementation, "median_ms": statistics.median(values),
            "p10_ms": min(values), "p90_ms": max(values), "samples": len(values),
        })
    return summaries, per_t_impl


def bootstrap_delta_us(a: list[float], b: list[float], *, seed: int = 20260717) -> tuple[float, float, float]:
    if len(a) != len(b) or not a:
        return math.nan, math.nan, math.nan
    diffs = [right - left for left, right in zip(a, b)]
    rng = random.Random(seed)
    draws = [statistics.mean([diffs[rng.randrange(len(diffs))] for _ in diffs]) * 1000.0 for _ in range(20_000)]
    return statistics.mean(diffs) * 1000.0, quantile(draws, 0.025), quantile(draws, 0.975)


def fit_lines(summary: list[dict[str, object]], scope: str, series: tuple[str, ...] | None = None) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    aggregate = [row for row in summary if row["session"] == "aggregate"]
    if series is None:
        series = IMPLEMENTATIONS
    for implementation in series:
        rows = [row for row in aggregate if row["implementation"] == implementation]
        xs = [float(row["chunks"]) for row in rows]
        ys = [float(row["median_ms"]) for row in rows]
        if len(xs) < 2:
            continue
        mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum((x - mean_x) ** 2 for x in xs)
        result.append({"scope": scope, "series": implementation, "intercept_ms": mean_y - slope * mean_x,
                       "slope_ms_per_chunk": slope, "points": len(xs)})
    if not set(IMPLEMENTATIONS).issubset({str(row["implementation"]) for row in aggregate}):
        return result
    for label, lhs, rhs in (("stage6s_minus_current", "graph_a_current_asm", "graph_b_stage6s_bridge"),
                            ("current_minus_vllm", "graph_c_vllm", "graph_a_current_asm"),
                            ("stage6s_minus_vllm", "graph_c_vllm", "graph_b_stage6s_bridge")):
        rows = []
        for t in sorted({int(row["T"]) for row in aggregate}):
            by_impl = {str(row["implementation"]): float(row["median_ms"]) for row in aggregate if int(row["T"]) == t}
            rows.append((t // BT, by_impl[rhs] - by_impl[lhs]))
        xs, ys = [x for x, _ in rows], [y for _, y in rows]
        if len(xs) < 2:
            continue
        mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum((x - mean_x) ** 2 for x in xs)
        result.append({"scope": scope, "series": label, "intercept_ms": mean_y - slope * mean_x,
                       "slope_ms_per_chunk": slope, "points": len(xs)})
    return result


def full_benchmark(t: int, args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    inputs = make_inputs(t, 2026071700 + t, "random", True)
    captures = {name: CapturedGraph.create(name, full_call(name, inputs)) for name in IMPLEMENTATIONS}
    correctness = {
        "a_vs_b": compare(captures["graph_a_current_asm"].result, captures["graph_b_stage6s_bridge"].result),
        "a_vs_c": compare(captures["graph_a_current_asm"].result, captures["graph_c_vllm"].result),
        "b_vs_c": compare(captures["graph_b_stage6s_bridge"].result, captures["graph_c_vllm"].result),
    }
    rows: list[dict[str, object]] = []
    for session in range(args.sessions):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(args.warmup):
            for captured in captures.values():
                captured.replay()
        torch.cuda.synchronize()
        for sequence, (order, name) in enumerate(pair_order(args.repeat)):
            rows.append({"scope": "full", "T": t, "chunks": t // BT, "session": session,
                         "order": order, "sequence": sequence, "implementation": name,
                         "latency_ms": event_ms(captures[name].replay, start, end)})
    return rows, {
        "T": t,
        "inputs": {name: tensor_meta(value) for name, value in zip(("q", "k", "v", "g", "beta", "initial_state"), inputs)},
        "capture_result_ptrs": {name: captured.result_ptrs for name, captured in captures.items()},
        "correctness": correctness,
    }


def boundary_benchmark(t: int, args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    q, k, v, g, beta, h0 = make_inputs(t, 2026071700 + t, "random", True)
    a = qwen_gdn_full_bt64_stage6s_current_asm_stages(q, k, v, g, beta, initial_state=h0)
    b = qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge_stages(q, k, v, g, beta, initial_state=h0)
    calls = {
        "w_fp32_to_bf16": lambda: a["w"].to(torch.bfloat16),
        "u_fp32_to_bf16": lambda: a["u"].to(torch.bfloat16),
        "wu_fp32_to_bf16": lambda: (a["w"].to(torch.bfloat16), a["u"].to(torch.bfloat16)),
        "vnew_bf16_to_fp32": lambda: b["v_new_bf16"].to(torch.float32),
        "all_boundary_casts": lambda: (a["w"].to(torch.bfloat16), a["u"].to(torch.bfloat16), b["v_new_bf16"].to(torch.float32)),
        "asm_v0_recurrence": lambda: qwen_gdn_bt64_gfx942_asm_v0(
            k, a["w"], a["u"], a["g_cumsum"], h0
        ),
        "current_vllm_bridge": lambda: qwen_gdn_bt64_stage6s_recurrence_bridge(k, b["w_bf16"], b["u_bf16"], b["g_cumsum"], h0),
    }
    captures = {name: CapturedGraph.create(name, fn) for name, fn in calls.items()}
    rows: list[dict[str, object]] = []
    for session in range(args.sessions):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for name, captured in captures.items():
            for _ in range(args.warmup):
                captured.replay()
            torch.cuda.synchronize()
            for sequence in range(args.repeat):
                rows.append({"scope": "boundary_body", "T": t, "chunks": t // BT, "session": session,
                             "order": "single_graph_replay", "sequence": sequence, "implementation": name,
                             "latency_ms": event_ms(captured.replay, start, end)})
    return rows, {
        "w_fp32_to_bf16_widened": error(a["w"], b["w_bf16"].float()),
        "u_fp32_to_bf16_widened": error(a["u"], b["u_bf16"].float()),
        "vnew_widened": error(b["v_new"], b["v_new_bf16"].float()),
        "graph_a": {key: tensor_meta(value) for key, value in a.items()},
        "graph_b": {key: tensor_meta(value) for key, value in b.items()},
    }


def dispatch_map() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    contract = stage6s_contract()
    common = ["cumsum", "kkt", "hierarchical_solve_external_hsaco", "w_fp32", "u_fp32"]
    a = {"graph": "A", "dispatches": common + ["asm_v0_recurrence", "chunk_o_fp32", "final_fp32_to_bf16_cast"],
         "count": 8, "all_shared_before_recurrence": True}
    b = {"graph": "B", "dispatches": common + ["w_fp32_to_bf16_cast", "u_fp32_to_bf16_cast", "stage6r_current_vllm_recurrence",
         "vnew_bf16_to_fp32_cast", "chunk_o_fp32", "final_fp32_to_bf16_cast"], "count": 11,
         "recurrence": {key: contract[key] for key in ("hsaco_sha256", "symbol", "grid", "workgroup", "dynamic_lds")}}
    c = {"graph": "C", "dispatches": "captured from native vLLM runtime trace; see rocprof/", "count": None,
         "note": "No inferred vLLM fusion claim is made before trace capture."}
    return a, b, c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--mode", choices=("all", "full", "boundary"), default="all")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    patch_rocm_autotune()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 6S requires a HIP GPU")

    a_map, b_map, c_map = dispatch_map()
    write_json(args.out_dir / "graph_a_dispatch_map.json", a_map)
    write_json(args.out_dir / "graph_b_dispatch_map.json", b_map)
    write_json(args.out_dir / "graph_c_dispatch_map.json", c_map)
    write_json(args.out_dir / "frozen_measurement_contract.json", {
        "stream": "current stream", "graph_replay_only_timing": True,
        "warmup": args.warmup, "repeat": args.repeat, "sessions": args.sessions,
        "orders": ["ABBA", "BCCB", "ACCA"], "implementations": IMPLEMENTATIONS,
        "stage6s_contract": stage6s_contract(),
    })

    full_raw: list[dict[str, object]] = []
    boundary_raw: list[dict[str, object]] = []
    correctness: dict[str, object] = {}
    boundary_contract: dict[str, object] = {}
    for t in args.T:
        if args.mode in ("all", "full"):
            rows, metadata = full_benchmark(t, args)
            full_raw.extend(rows)
            correctness[str(t)] = metadata
        if args.mode in ("all", "boundary"):
            rows, metadata = boundary_benchmark(t, args)
            boundary_raw.extend(rows)
            boundary_contract[str(t)] = metadata

    if full_raw:
        write_csv(args.out_dir / "full_raw.csv", full_raw)
        summary, samples = summarise(full_raw, scope="full")
        write_csv(args.out_dir / "full_summary.csv", summary)
        write_csv(args.out_dir / "full_slopes.csv", fit_lines(summary, "full"))
        gains: list[dict[str, object]] = []
        for t in args.T:
            current = samples[(t, "graph_a_current_asm")]
            candidate = samples[(t, "graph_b_stage6s_bridge")]
            vllm = samples[(t, "graph_c_vllm")]
            mean_gain, lo, hi = bootstrap_delta_us(current, candidate)
            gains.append({"T": t, "chunks": t // BT, "stage6s_minus_current_us": mean_gain,
                          "ci95_low_us": lo, "ci95_high_us": hi,
                          "current_minus_vllm_us": statistics.mean([x - y for x, y in zip(current, vllm)]) * 1000.0,
                          "stage6s_minus_vllm_us": statistics.mean([x - y for x, y in zip(candidate, vllm)]) * 1000.0})
        write_csv(args.out_dir / "full_gain_analysis.csv", gains)
    if boundary_raw:
        write_csv(args.out_dir / "boundary_body_raw.csv", boundary_raw)
        summary, _ = summarise(boundary_raw, scope="boundary_body")
        write_csv(args.out_dir / "boundary_body_summary.csv", summary)
        boundary_series = tuple(sorted({str(row["implementation"]) for row in summary if row["session"] == "aggregate"}))
        slopes = fit_lines(summary, "boundary_body", boundary_series)
        write_csv(args.out_dir / "boundary_body_slopes.csv", slopes)
        write_csv(args.out_dir / "recurrence_body_summary.csv", [row for row in summary if "recurrence" in str(row["implementation"])])
        write_csv(args.out_dir / "recurrence_body_slopes.csv", [row for row in slopes if "recurrence" in str(row["series"])])

    write_json(args.out_dir / "correctness_summary.json", correctness)
    write_json(args.out_dir / "bf16_boundary_contract.json", boundary_contract)


if __name__ == "__main__":
    main()
