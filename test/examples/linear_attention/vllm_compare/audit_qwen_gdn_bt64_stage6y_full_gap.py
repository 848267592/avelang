#!/usr/bin/env python3
"""Stage 6Y Eager full/body audit for the frozen five-dispatch Stage 6X graph.

This is measurement-only.  It changes no kernel and leaves the default
selector untouched.  Full timing is the authority: uncaptured public calls,
one fixed input set per T, current stream, and ABBA pairing.  Body timing is a
diagnostic view of logical stages; it uses their current public/wrapper
boundaries and must not be added together to replace the full result.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_chunko_boundary_stage6w import qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w  # noqa: E402
from qwen_gdn_bt64_bf16_recurrence_full_stage6s import qwen_gdn_bt64_stage6s_recurrence_bridge  # noqa: E402
from qwen_gdn_bt64_bf16_solved_boundary_stage6u import qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u  # noqa: E402
from qwen_gdn_bt64_kkt_solve_handoff_stage6x import qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2  # noqa: E402
from qwen_gdn_bt64_kkt_solve_handoff_stage6x_full import qwen_gdn_full_bt64_stage6x_kkt_solve_eager  # noqa: E402
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import BT, K_DIM  # noqa: E402
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import qwen_gdn_chunk_cumsum_avelang_v6_standalone  # noqa: E402
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_o import chunk_fwd_o  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd  # noqa: E402
from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum  # noqa: E402
from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril  # noqa: E402
from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd  # noqa: E402


NAMES = ("stage6x_x2", "vllm")
STAGES = ("cumsum", "kkt_solve", "wu", "recurrence", "chunk_o")
OUTPUT_ATOL = 1.0 / 128.0
STATE_ATOL = 2.0e-2


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def event_and_wall_ms(fn: Callable[[], Any]) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    start.record()
    value = fn()
    end.record()
    end.synchronize()
    if value is None:
        raise AssertionError("stage call returned None")
    return float(start.elapsed_time(end)), (time.perf_counter_ns() - wall_start) / 1e6


def build_inputs(t: int) -> tuple[torch.Tensor, ...]:
    return make_inputs(t, 2026072900 + t, "random", True)


def full_calls(inputs: tuple[torch.Tensor, ...]) -> dict[str, Callable[[], Any]]:
    q, k, v, g, beta, h0 = inputs
    common = dict(initial_state=h0, output_final_state=True, scale=K_DIM ** -0.5)
    return {
        "stage6x_x2": lambda: qwen_gdn_full_bt64_stage6x_kkt_solve_eager(q, k, v, g, beta, **common),
        "vllm": lambda: vllm_full(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True,
            scale=K_DIM ** -0.5, head_first=False, use_qk_l2norm_in_kernel=False,
        ),
    }


def check_full_contract(calls: dict[str, Callable[[], Any]], t: int) -> dict[str, float]:
    x2_out, x2_state = calls["stage6x_x2"]()
    vllm_out, vllm_state = calls["vllm"]()
    if x2_state is None or vllm_state is None:
        raise AssertionError("full contract requires final state")
    output_max_abs = float((x2_out.float() - vllm_out.float()).abs().max().item())
    state_max_abs = float((x2_state.float() - vllm_state.float()).abs().max().item())
    accepted = output_max_abs <= OUTPUT_ATOL and state_max_abs <= STATE_ATOL
    if not accepted:
        raise AssertionError(
            f"Stage 6Y full correctness failed at T={t}: output={output_max_abs}, state={state_max_abs}"
        )
    return {
        "T": float(t),
        "output_max_abs": output_max_abs,
        "state_max_abs": state_max_abs,
        "output_atol": OUTPUT_ATOL,
        "state_atol": STATE_ATOL,
    }


def stage_values_stage6x(inputs: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    q, k, v, g, beta, h0 = inputs
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    a_solved = qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2(k, g_cumsum, beta)
    w, u = qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u(k, v, g_cumsum, beta, a_solved)
    h, v_new, final_state = qwen_gdn_bt64_stage6s_recurrence_bridge(k, w, u, g_cumsum, h0)
    output = qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w(q, k, v_new, h, g_cumsum, scale=K_DIM ** -0.5)
    return {"g": g_cumsum, "a_solved": a_solved, "w": w, "u": u, "h": h, "v_new": v_new, "final_state": final_state, "output": output}


def stage_values_vllm(inputs: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    q, k, v, g, beta, h0 = inputs
    g_cumsum = chunk_local_cumsum(g, chunk_size=BT)
    a = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=g_cumsum, output_dtype=torch.float32)
    a_solved = solve_tril(A=a, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=a_solved, g_cumsum=g_cumsum, cu_seqlens=None)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g_cumsum, initial_state=h0, output_final_state=True,
        chunk_size=BT, save_new_value=True, cu_seqlens=None,
    )
    output = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g_cumsum, scale=K_DIM ** -0.5, chunk_size=BT)
    return {"g": g_cumsum, "a": a, "a_solved": a_solved, "w": w, "u": u, "h": h, "v_new": v_new, "final_state": final_state, "output": output}


def stage_calls_stage6x(inputs: tuple[torch.Tensor, ...], values: dict[str, Any]) -> dict[str, Callable[[], Any]]:
    q, k, v, g, beta, h0 = inputs
    return {
        "cumsum": lambda: qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT),
        "kkt_solve": lambda: qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2(k, values["g"], beta),
        "wu": lambda: qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u(k, v, values["g"], beta, values["a_solved"]),
        "recurrence": lambda: qwen_gdn_bt64_stage6s_recurrence_bridge(k, values["w"], values["u"], values["g"], h0),
        "chunk_o": lambda: qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w(q, k, values["v_new"], values["h"], values["g"], scale=K_DIM ** -0.5),
    }


def stage_calls_vllm(inputs: tuple[torch.Tensor, ...], values: dict[str, Any]) -> dict[str, Callable[[], Any]]:
    q, k, v, g, beta, h0 = inputs
    return {
        "cumsum": lambda: chunk_local_cumsum(g, chunk_size=BT),
        "kkt_solve": lambda: solve_tril(
            A=chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=values["g"], output_dtype=torch.float32), output_dtype=k.dtype
        ),
        "wu": lambda: recompute_w_u_fwd(k=k, v=v, beta=beta, A=values["a_solved"], g_cumsum=values["g"], cu_seqlens=None),
        "recurrence": lambda: chunk_gated_delta_rule_fwd_h(
            k=k, w=values["w"], u=values["u"], g=values["g"], initial_state=h0, output_final_state=True,
            chunk_size=BT, save_new_value=True, cu_seqlens=None,
        ),
        "chunk_o": lambda: chunk_fwd_o(q=q, k=k, v=values["v_new"], h=values["h"], g=values["g"], scale=K_DIM ** -0.5, chunk_size=BT),
    }


def run_pair(t: int, scope: str, calls: dict[str, Callable[[], Any]], args: argparse.Namespace) -> list[dict[str, object]]:
    for fn in calls.values():
        fn()
    torch.cuda.synchronize()
    rows: list[dict[str, object]] = []
    order = ("stage6x_x2", "vllm", "vllm", "stage6x_x2")
    for session in range(args.sessions):
        for _ in range(args.warmup):
            for name in order:
                calls[name]()
        torch.cuda.synchronize()
        for repeat in range(args.repeat):
            for position, name in enumerate(order):
                event_ms, wall_ms = event_and_wall_ms(calls[name])
                rows.append({
                    "scope": scope, "T": t, "chunks": t // BT, "session": session,
                    "repeat": repeat, "order": "ABBA", "position": position,
                    "implementation": name, "event_ms": event_ms, "wall_ms": wall_ms,
                })
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scope"]), int(row["T"]), int(row["session"]), str(row["implementation"]))].append(float(row["event_ms"]))
    output: list[dict[str, object]] = []
    for (scope, t, session, implementation), values in sorted(grouped.items()):
        output.append({"scope": scope, "T": t, "chunks": t // BT, "session": session, "implementation": implementation, "event_median_ms": statistics.median(values)})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full", "bodies", "all"), default="all")
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 6Y requires the gfx942 HIP runtime")
    if any(t < BT or t % BT for t in args.T):
        raise ValueError("T must be >=64 and divisible by 64")
    patch_rocm_autotune()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    full_rows: list[dict[str, object]] = []
    body_rows: list[dict[str, object]] = []
    correctness: list[dict[str, float]] = []
    for t in args.T:
        inputs = build_inputs(t)
        if args.mode in ("full", "all"):
            calls = full_calls(inputs)
            correctness.append(check_full_contract(calls, t))
            full_rows.extend(run_pair(t, "full_eager_public_api", calls, args))
        if args.mode in ("bodies", "all"):
            x2_values = stage_values_stage6x(inputs)
            vllm_values = stage_values_vllm(inputs)
            x2_calls = stage_calls_stage6x(inputs, x2_values)
            vllm_calls = stage_calls_vllm(inputs, vllm_values)
            for stage in args.stages:
                rows = run_pair(t, f"body_wrapper:{stage}", {"stage6x_x2": x2_calls[stage], "vllm": vllm_calls[stage]}, args)
                body_rows.extend(rows)
        print(f"complete T={t}", flush=True)
    write_csv(args.out_dir / "stage6y_full_raw.csv", full_rows)
    write_csv(args.out_dir / "stage6y_full_summary.csv", summarize(full_rows))
    write_csv(args.out_dir / "stage6y_body_raw.csv", body_rows)
    write_csv(args.out_dir / "stage6y_body_summary.csv", summarize(body_rows))
    (args.out_dir / "stage6y_full_correctness.json").write_text(json.dumps(correctness, indent=2) + "\n")
    (args.out_dir / "stage6y_contract.json").write_text(json.dumps({
        "timing_contract": "eager_public_api",
        "cuda_graph_used": False,
        "full_authority": True,
        "body_scope": "wrapper-level diagnostic; do not sum bodies to replace full timing",
        "stage6x_logical_dispatches": list(STAGES),
        "vllm_kkt_solve_note": "body executes native KKT then native solve as one logical composite for comparison with X2",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
