#!/usr/bin/env python3
"""Process-isolated, clustered Eager confirmation for Stage 6W.

Each independent session runs complete randomized Williams blocks.  A block
contains all six balanced orderings of U1, Stage 6W, and vLLM, so every
implementation occupies every position and directly precedes/follows every
other implementation equally often.  Timing is always a HIP event around one
whole public API call; CUDA graph capture/replay is never used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

NAMES = ("u1", "w1", "vllm")
# First three are a cyclic Latin square; final three are its reversals.
WILLIAMS_ORDERS = (
    ("u1", "w1", "vllm"),
    ("w1", "vllm", "u1"),
    ("vllm", "u1", "w1"),
    ("u1", "vllm", "w1"),
    ("vllm", "w1", "u1"),
    ("w1", "u1", "vllm"),
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * p)]


@lru_cache(maxsize=1)
def runtime_modules() -> tuple[object, ...]:
    """Delay all GPU-runtime imports until a child session has passed parent preflight."""
    import torch
    from qwen_gdn_bt64_bf16_chunko_boundary_stage6w import qwen_gdn_full_bt64_stage6w_bf16_chunko_eager
    from qwen_gdn_bt64_bf16_solved_boundary_stage6u import qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager
    from stage2_runner import make_inputs, patch_rocm_autotune
    from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full

    return torch, qwen_gdn_full_bt64_stage6w_bf16_chunko_eager, qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager, make_inputs, patch_rocm_autotune, vllm_full


def calls_for_t(t: int) -> dict[str, Callable[[], object]]:
    _, stage6w, stage6u, make_inputs, _, vllm_full = runtime_modules()
    q, k, v, g, beta, h0 = make_inputs(t, 2026074700 + t, "random", True)
    common = dict(initial_state=h0, output_final_state=True, scale=128 ** -0.5)
    return {
        "u1": lambda: stage6u(q, k, v, g, beta, **common),
        "w1": lambda: stage6w(q, k, v, g, beta, **common),
        "vllm": lambda: vllm_full(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True,
            scale=128 ** -0.5, head_first=False, use_qk_l2norm_in_kernel=False,
        ),
    }


def measure(fn: Callable[[], object]) -> tuple[float, float]:
    torch, *_ = runtime_modules()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start_ns = time.perf_counter_ns()
    begin.record()
    value = fn()
    end.record()
    end.synchronize()
    if value is None:
        raise AssertionError("public API returned None")
    return float(begin.elapsed_time(end)), (time.perf_counter_ns() - wall_start_ns) / 1_000_000.0


def _run_capture(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
        return {"command": command, "returncode": result.returncode, "output": result.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": None, "output": str(exc)}


def telemetry_snapshot(label: str) -> dict[str, object]:
    """Read-only telemetry; invoked only outside timed blocks."""
    if shutil.which("amd-smi"):
        gpu = _run_capture(["amd-smi", "metric", "-g", "0", "--json"])
    elif shutil.which("rocm-smi"):
        gpu = _run_capture(["rocm-smi", "--showclocks", "--showpower", "--showtemp", "--showuse", "--json"])
    else:
        gpu = {"command": [], "returncode": None, "output": "amd-smi and rocm-smi unavailable"}
    gpu_processes = (
        _run_capture(["amd-smi", "process", "-g", "0", "--json"])
        if shutil.which("amd-smi")
        else {"command": [], "returncode": None, "output": "amd-smi unavailable"}
    )
    processes = _run_capture(["ps", "-eo", "pid,ppid,user,comm,args"])
    return {
        "label": label,
        "utc": utc_now(),
        "time_ns": time.time_ns(),
        "pid": os.getpid(),
        "gpu_metric": gpu,
        "gpu_processes": gpu_processes,
        "process_snapshot": processes,
    }


def gpu_context_summary(snapshot: dict[str, object]) -> dict[str, object]:
    capture = snapshot["gpu_processes"]
    assert isinstance(capture, dict)
    if capture.get("returncode") != 0:
        return {"available": False, "context_count": None, "contexts": [], "reason": capture.get("output")}
    try:
        entries = json.loads(str(capture["output"]))[0].get("process_list", [])
    except (IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"available": False, "context_count": None, "contexts": [], "reason": str(exc)}
    contexts = []
    for entry in entries:
        info = entry.get("process_info", {})
        contexts.append({
            "pid": info.get("pid"),
            "name": info.get("name"),
            "evicted_time": info.get("evicted_time"),
        })
    return {"available": True, "context_count": len(contexts), "contexts": contexts, "reason": ""}


def run_one_session(args: argparse.Namespace) -> dict[str, object]:
    torch, _, _, _, patch_rocm_autotune, _ = runtime_modules()
    patch_rocm_autotune()
    rng = random.Random(args.seed + args.session_index * 104729)
    session_start = telemetry_snapshot("session_start")
    raw: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    warmup_orders: dict[int, list[list[str]]] = {}
    timed_orders: dict[int, list[list[list[str]]]] = {}
    t_order = list(args.T)
    rng.shuffle(t_order)

    for t in t_order:
        calls = calls_for_t(t)
        # Compile, module load, and allocation warmup are outside timing.
        for fn in calls.values():
            fn()
        torch.cuda.synchronize()
        warmup_orders[t] = []
        for _ in range(args.warmup_blocks):
            orders = list(WILLIAMS_ORDERS)
            rng.shuffle(orders)
            warmup_orders[t].append([list(order) for order in orders])
            for order in orders:
                for name in order:
                    calls[name]()
        torch.cuda.synchronize()

        timed_orders[t] = []
        for block in range(args.blocks_per_session):
            orders = list(WILLIAMS_ORDERS)
            rng.shuffle(orders)
            timed_orders[t].append([list(order) for order in orders])
            by_name_event: dict[str, list[float]] = defaultdict(list)
            by_name_wall: dict[str, list[float]] = defaultdict(list)
            block_start_ns = time.time_ns()
            for order_index, order in enumerate(orders):
                for position, name in enumerate(order):
                    event_ms, wall_ms = measure(calls[name])
                    by_name_event[name].append(event_ms)
                    by_name_wall[name].append(wall_ms)
                    raw.append({
                        "timing_contract": "eager_public_api",
                        "cuda_graph_used": False,
                        "session": args.session_index,
                        "T": t,
                        "chunks": t // 64,
                        "block": block,
                        "order_index": order_index,
                        "order": "-".join(order),
                        "position": position,
                        "implementation": name,
                        "event_ms": event_ms,
                        "wall_ms": wall_ms,
                        "time_ns": time.time_ns(),
                    })
            torch.cuda.synchronize()
            event_medians = {name: statistics.median(by_name_event[name]) for name in NAMES}
            wall_medians = {name: statistics.median(by_name_wall[name]) for name in NAMES}
            block_rows.extend([
                {
                    "session": args.session_index,
                    "T": t,
                    "chunks": t // 64,
                    "block": block,
                    "reference": reference,
                    "candidate": "w1",
                    "reference_event_block_median_ms": event_medians[reference],
                    "candidate_event_block_median_ms": event_medians["w1"],
                    "event_gain_us": 1000.0 * (event_medians[reference] - event_medians["w1"]),
                    "reference_wall_block_median_ms": wall_medians[reference],
                    "candidate_wall_block_median_ms": wall_medians["w1"],
                    "wall_gain_us": 1000.0 * (wall_medians[reference] - wall_medians["w1"]),
                    "block_start_ns": block_start_ns,
                    "block_end_ns": time.time_ns(),
                }
                for reference in ("u1", "vllm")
            ])

    session_end = telemetry_snapshot("session_end")
    return {
        "timing_contract": "eager_public_api",
        "cuda_graph_used": False,
        "session": args.session_index,
        "seed": args.seed,
        "T_order": t_order,
        "warmup_blocks": args.warmup_blocks,
        "blocks_per_session": args.blocks_per_session,
        "warmup_orders": warmup_orders,
        "timed_orders": timed_orders,
        "session_start": session_start,
        "session_end": session_end,
        "raw": raw,
        "block_pairs": block_rows,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_block_ci(values: list[float], seed: int, samples: int) -> tuple[float, float]:
    rng = random.Random(seed)
    means = [statistics.mean(values[rng.randrange(len(values))] for _ in values) for _ in range(samples)]
    return percentile(means, 0.025), percentile(means, 0.975)


def nested_cluster_ci(rows: list[dict[str, object]], field: str, seed: int, samples: int) -> tuple[float, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["session"])].append(float(row[field]))
    sessions = sorted(grouped)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        session_means = []
        for _ in sessions:
            selected = grouped[sessions[rng.randrange(len(sessions))]]
            session_means.append(statistics.mean(selected[rng.randrange(len(selected))] for _ in selected))
        estimates.append(statistics.mean(session_means))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def summarize(all_sessions: list[dict[str, object]], bootstrap_samples: int) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    raw = [row for session in all_sessions for row in session["raw"]]
    block_pairs = [row for session in all_sessions for row in session["block_pairs"]]
    session_pairs: list[dict[str, object]] = []
    for t in sorted({int(row["T"]) for row in block_pairs}):
        for reference in ("u1", "vllm"):
            for session in sorted({int(row["session"]) for row in block_pairs if int(row["T"]) == t and row["reference"] == reference}):
                event_values = [float(row["event_gain_us"]) for row in block_pairs if int(row["T"]) == t and row["reference"] == reference and int(row["session"]) == session]
                wall_values = [float(row["wall_gain_us"]) for row in block_pairs if int(row["T"]) == t and row["reference"] == reference and int(row["session"]) == session]
                session_pairs.append({
                    "T": t,
                    "reference": reference,
                    "candidate": "w1",
                    "session": session,
                    "blocks": len(event_values),
                    "event_gain_us_mean": statistics.mean(event_values),
                    "event_gain_us_median": statistics.median(event_values),
                    "event_gain_us_min": min(event_values),
                    "event_gain_us_max": max(event_values),
                    "wall_gain_us_mean": statistics.mean(wall_values),
                    "wall_gain_us_median": statistics.median(wall_values),
                })
    analyses = []
    for t in sorted({int(row["T"]) for row in block_pairs}):
        for reference in ("u1", "vllm"):
            rows = [row for row in block_pairs if int(row["T"]) == t and row["reference"] == reference]
            event_values = [float(row["event_gain_us"]) for row in rows]
            event_session_medians = [float(row["event_gain_us_median"]) for row in session_pairs if int(row["T"]) == t and row["reference"] == reference]
            wall_session_medians = [float(row["wall_gain_us_median"]) for row in session_pairs if int(row["T"]) == t and row["reference"] == reference]
            block_ci = bootstrap_block_ci(event_values, 2026074800 + t + len(reference), bootstrap_samples)
            cluster_ci = nested_cluster_ci(rows, "event_gain_us", 2026074900 + t + len(reference), bootstrap_samples)
            primary_session_ci = bootstrap_block_ci(event_session_medians, 2026075000 + t + len(reference), bootstrap_samples)
            wall_session_ci = bootstrap_block_ci(wall_session_medians, 2026075100 + t + len(reference), bootstrap_samples)
            event_majority_positive = sum(value > 0.0 for value in event_session_medians) > len(event_session_medians) / 2
            wall_majority_positive = sum(value > 0.0 for value in wall_session_medians) > len(wall_session_medians) / 2
            analyses.append({
                "T": t,
                "reference": reference,
                "candidate": "w1",
                "sessions": len(event_session_medians),
                "blocks": len(event_values),
                "block_event_gain_us_mean": statistics.mean(event_values),
                "block_event_gain_us_median": statistics.median(event_values),
                "block_bootstrap_ci_low_us": block_ci[0],
                "block_bootstrap_ci_high_us": block_ci[1],
                "primary_estimator": "session-level paired median HIP-event gain",
                "primary_session_event_gain_us_mean": statistics.mean(event_session_medians),
                "primary_session_event_gain_us_median": statistics.median(event_session_medians),
                "primary_session_bootstrap_ci_low_us": primary_session_ci[0],
                "primary_session_bootstrap_ci_high_us": primary_session_ci[1],
                "wall_session_gain_us_mean": statistics.mean(wall_session_medians),
                "wall_session_gain_us_median": statistics.median(wall_session_medians),
                "wall_session_bootstrap_ci_low_us": wall_session_ci[0],
                "wall_session_bootstrap_ci_high_us": wall_session_ci[1],
                "nested_cluster_ci_low_us": cluster_ci[0],
                "nested_cluster_ci_high_us": cluster_ci[1],
                "event_majority_sessions_positive": event_majority_positive,
                "wall_majority_sessions_positive": wall_majority_positive,
                "primary_gate": primary_session_ci[0] > 0.0 and event_majority_positive and wall_majority_positive and statistics.mean(wall_session_medians) > 0.0,
            })
    decision = {
        "timing_contract": "eager_public_api",
        "cuda_graph_used": False,
        "stage6w_vs_u1_primary_gate": {
            str(row["T"]): row["primary_gate"]
            for row in analyses if row["reference"] == "u1"
        },
        "promotion_gate_without_environment": all(row["primary_gate"] for row in analyses if row["reference"] == "u1"),
        "primary_gate_definition": "session-level paired median HIP-event gain: 95% bootstrap lower bound > 0 us, majority of sessions positive, and wall-clock session-median direction positive",
        "sensitivity_analyses": "block and nested-cluster HIP-event CIs are reported but are not co-primary promotion gates",
    }
    return raw, session_pairs, {"analysis": analyses, "decision": decision}


def run_parent(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preflight = telemetry_snapshot("exclusive_gpu_preflight")
    preflight["gpu_context_summary"] = gpu_context_summary(preflight)
    preflight_valid = bool(preflight["gpu_context_summary"]["available"]) and preflight["gpu_context_summary"]["context_count"] == 0
    (args.out_dir / "exclusive_gpu_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    if not args.allow_shared_gpu and not preflight_valid:
        raise RuntimeError(
            "exclusive-GPU preflight failed: expected zero GPU contexts before the parent launches child sessions; "
            f"observed {preflight['gpu_context_summary']}"
        )
    child_dir = args.out_dir / "sessions"
    child_dir.mkdir(exist_ok=True)
    all_sessions = []
    for session_index in range(args.sessions):
        child_path = child_dir / f"session_{session_index:02d}.json"
        if args.resume and child_path.is_file():
            all_sessions.append(json.loads(child_path.read_text()))
            print(f"reuse completed process-isolated session={session_index}", flush=True)
            continue
        command = [
            sys.executable, str(Path(__file__).resolve()), "--single-session",
            "--session-index", str(session_index), "--seed", str(args.seed),
            "--T", *[str(value) for value in args.T],
            "--warmup-blocks", str(args.warmup_blocks),
            "--blocks-per-session", str(args.blocks_per_session),
            "--session-json", str(child_path),
        ]
        result = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"session {session_index} failed:\n{result.stdout}")
        all_sessions.append(json.loads(child_path.read_text()))
        print(f"complete process-isolated session={session_index}", flush=True)
    raw, session_pairs, summary = summarize(all_sessions, args.bootstrap_samples)
    relative_gate = summary["decision"]["promotion_gate_without_environment"]
    summary["decision"]["exclusive_gpu_preflight_passed"] = preflight_valid
    summary["decision"]["shared_gpu_allowed"] = args.allow_shared_gpu
    summary["decision"]["promotion_scope"] = (
        "paired_shared_environment" if args.allow_shared_gpu else "exclusive_gpu_environment"
    )
    summary["decision"]["promotion_gate"] = relative_gate if args.allow_shared_gpu else preflight_valid and relative_gate
    block_pairs = [row for session in all_sessions for row in session["block_pairs"]]
    telemetry = [
        {"session": session["session"], "when": "start", **session["session_start"]}
        for session in all_sessions
    ] + [
        {"session": session["session"], "when": "end", **session["session_end"]}
        for session in all_sessions
    ]
    write_csv(args.out_dir / "raw_events.csv", raw)
    write_csv(args.out_dir / "block_paired_differences.csv", block_pairs)
    write_csv(args.out_dir / "session_paired_differences.csv", session_pairs)
    (args.out_dir / "telemetry.json").write_text(json.dumps(telemetry, indent=2) + "\n")
    metadata = {
        "timing_contract": "eager_public_api",
        "cuda_graph_used": False,
        "sessions": args.sessions,
        "process_isolated_sessions": True,
        "T": args.T,
        "warmup_blocks": args.warmup_blocks,
        "blocks_per_session": args.blocks_per_session,
        "orders_per_block": len(WILLIAMS_ORDERS),
        "calls_per_block": len(WILLIAMS_ORDERS) * len(NAMES),
        "randomized_order_within_block": True,
        "randomized_T_order_per_session": True,
        "telemetry_outside_timed_blocks": True,
        "exclusive_gpu_preflight_required": not args.allow_shared_gpu,
        "exclusive_gpu_preflight_passed": preflight_valid,
        "shared_gpu_allowed": args.allow_shared_gpu,
        "primary_estimator": "session-level paired median HIP-event gain",
        "bootstrap_samples": args.bootstrap_samples,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.out_dir / "cluster_bootstrap_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--warmup-blocks", type=int, default=2)
    parser.add_argument("--blocks-per-session", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026074700)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--single-session", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-shared-gpu", action="store_true", help="allow a paired relative comparison on a shared GPU; result is explicitly scoped to that environment")
    parser.add_argument("--session-index", type=int, default=0)
    parser.add_argument("--session-json", type=Path)
    args = parser.parse_args()
    if args.single_session:
        if args.session_json is None:
            raise ValueError("--single-session requires --session-json")
        args.session_json.parent.mkdir(parents=True, exist_ok=True)
        args.session_json.write_text(json.dumps(run_one_session(args), indent=2) + "\n")
    else:
        if args.out_dir is None:
            raise ValueError("parent confirmation requires --out-dir")
        run_parent(args)


if __name__ == "__main__":
    main()
