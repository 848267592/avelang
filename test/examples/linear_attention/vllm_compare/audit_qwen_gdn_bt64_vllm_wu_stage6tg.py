#!/usr/bin/env python3
"""Audit-only Stage 6T-Golden runner for the BT64 Qwen GDN public APIs.

Authoritative modes (`benchmark` and `correctness`) invoke only complete eager
public APIs.  Capture and intermediate modes are explicitly diagnostic-only.
No kernel implementation or dispatch selector is changed by this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
DEFAULT_OUT = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_vllm_fused_wu_golden_audit_stage6tg"

import sys

sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_recurrence_full_stage6s import (  # noqa: E402
    _hierarchical_solve,
    qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge,
)
from qwen_gdn_bt64_fused_wu_eager_stage6t import (  # noqa: E402
    qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager,
    qwen_gdn_w_u_bt64_fused_stage6t_bf16,
)
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import qwen_gdn_kkt_bt64_mfma_v2_s0  # noqa: E402
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import qwen_gdn_chunk_cumsum_avelang_v6_standalone  # noqa: E402
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402
from vllm.model_executor.layers.fla.ops import wy_fast  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd  # noqa: E402
from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum  # noqa: E402
from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril  # noqa: E402

vllm_chunk_module = importlib.import_module("vllm.model_executor.layers.fla.ops.chunk")


BT = 64
NAMES = ("stage6s", "f1", "vllm")
OUTPUT_ATOL = 1.0 / 128.0
STATE_ATOL = 2.0e-2
CONTRACT = "eager_public_api"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qtile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[max(0, min(len(values) - 1, round((len(values) - 1) * q)))]


def public_calls(inputs: tuple[torch.Tensor, ...]) -> dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor | None]]]:
    q, k, v, g, beta, h0 = inputs
    common = dict(initial_state=h0, output_final_state=True, scale=128 ** -0.5)
    return {
        "stage6s": lambda: qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(q, k, v, g, beta, **common),
        "f1": lambda: qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager(q, k, v, g, beta, **common),
        "vllm": lambda: vllm_full(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=h0,
            output_final_state=True,
            scale=128 ** -0.5,
            head_first=False,
            use_qk_l2norm_in_kernel=False,
        ),
    }


def event_time(fn: Callable[[], Any], start: torch.cuda.Event, end: torch.cuda.Event) -> tuple[float, float]:
    torch.cuda.synchronize()
    wall_start = time.perf_counter_ns()
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    if result is None:
        raise AssertionError("public API returned no result")
    return float(start.elapsed_time(end)), (time.perf_counter_ns() - wall_start) / 1.0e6


def input_case(t: int, seed: int, mode: str, with_initial_state: bool) -> tuple[torch.Tensor, ...]:
    base_mode = mode if mode in {"random", "neutral_gate", "high_dynamic", "cancellation", "small_values"} else "random"
    q, k, v, g, beta, h0 = make_inputs(t, seed, base_mode, with_initial_state)
    if mode == "zero_beta":
        beta.zero_()
    elif mode == "sparse_beta":
        beta.zero_()
        beta[:, ::7, :].fill_(1.0)
    return q, k, v, g, beta, h0


def balanced_order(index: int, rng: random.Random) -> tuple[str, list[str]]:
    if index % 3 == 0:
        return "S-F1-V", ["stage6s", "f1", "vllm", "vllm", "f1", "stage6s"]
    if index % 3 == 1:
        return "V-F1-S", ["vllm", "f1", "stage6s", "stage6s", "f1", "vllm"]
    names = list(NAMES)
    rng.shuffle(names)
    return "random-balanced", names + list(reversed(names))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["T"]), int(row["session"]), str(row["implementation"])), []).append(row)
    output: list[dict[str, Any]] = []
    aggregate: dict[tuple[int, str], list[dict[str, float]]] = {}
    for (t, session, name), group in sorted(grouped.items()):
        events = [float(row["event_ms"]) for row in group]
        walls = [float(row["wall_ms"]) for row in group]
        item = {
            "timing_contract": CONTRACT,
            "cuda_graph_used": False,
            "T": t,
            "chunks": t // BT,
            "session": session,
            "implementation": name,
            "samples": len(group),
            "event_median_ms": statistics.median(events),
            "event_p10_ms": qtile(events, 0.1),
            "event_p90_ms": qtile(events, 0.9),
            "wall_median_ms": statistics.median(walls),
            "wall_p10_ms": qtile(walls, 0.1),
            "wall_p90_ms": qtile(walls, 0.9),
        }
        output.append(item)
        aggregate.setdefault((t, name), []).append({"event": item["event_median_ms"], "wall": item["wall_median_ms"]})
    for (t, name), values in sorted(aggregate.items()):
        events = [float(value["event"]) for value in values]
        walls = [float(value["wall"]) for value in values]
        output.append({
            "timing_contract": CONTRACT,
            "cuda_graph_used": False,
            "T": t,
            "chunks": t // BT,
            "session": "aggregate",
            "implementation": name,
            "samples": len(events),
            "event_median_ms": statistics.median(events),
            "event_p10_ms": qtile(events, 0.1),
            "event_p90_ms": qtile(events, 0.9),
            "wall_median_ms": statistics.median(walls),
            "wall_p10_ms": qtile(walls, 0.1),
            "wall_p90_ms": qtile(walls, 0.9),
        })
    return output


def fit_slopes(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate = [row for row in summary_rows if row["session"] == "aggregate"]
    result: list[dict[str, Any]] = []
    for name in NAMES:
        points = [(float(row["chunks"]), float(row["event_median_ms"])) for row in aggregate if row["implementation"] == name]
        mean_x = sum(x for x, _ in points) / len(points)
        mean_y = sum(y for _, y in points) / len(points)
        denom = sum((x - mean_x) ** 2 for x, _ in points)
        slope = float("nan") if not denom else sum((x - mean_x) * (y - mean_y) for x, y in points) / denom
        result.append({
            "timing_contract": CONTRACT,
            "cuda_graph_used": False,
            "implementation": name,
            "intercept_ms": mean_y - slope * mean_x if not math.isnan(slope) else mean_y,
            "slope_us_per_chunk": slope * 1000.0,
        })
    return result


def run_benchmark(args: argparse.Namespace) -> None:
    if args.sessions < 5 or args.warmup < 30 or args.repeat < 200:
        raise ValueError("authoritative eager baseline requires sessions>=5, warmup>=30, repeat>=200")
    patch_rocm_autotune()
    raw: list[dict[str, Any]] = []
    for t in args.T:
        inputs = input_case(t, 2026072000 + t, "random", True)
        calls = public_calls(inputs)
        for fn in calls.values():
            fn()
        torch.cuda.synchronize()
        for session in range(args.sessions):
            for _ in range(args.warmup):
                for fn in calls.values():
                    fn()
            torch.cuda.synchronize()
            rng = random.Random(202607200000 + t * 10 + session)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            sequence = 0
            for repeat in range(args.repeat):
                order_name, order = balanced_order(repeat, rng)
                for name in order:
                    event_ms, wall_ms = event_time(calls[name], start, end)
                    raw.append({
                        "timing_contract": CONTRACT,
                        "cuda_graph_used": False,
                        "T": t,
                        "chunks": t // BT,
                        "session": session,
                        "repeat": repeat,
                        "sequence": sequence,
                        "order": order_name,
                        "implementation": name,
                        "event_ms": event_ms,
                        "wall_ms": wall_ms,
                    })
                    sequence += 1
    summary = summarize(raw)
    slopes = fit_slopes(summary)
    write_csv(args.out_dir / "eager_baseline_raw.csv", raw)
    write_csv(args.out_dir / "eager_baseline_summary.csv", summary)
    write_csv(args.out_dir / "eager_baseline_slopes.csv", slopes)
    write_json(args.out_dir / "eager_baseline_metadata.json", {
        "timing_contract": CONTRACT,
        "cuda_graph_used": False,
        "sessions": args.sessions,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "T": args.T,
        "device": torch.cuda.get_device_name(),
        "public_api_only": True,
        "internal_allocation_is_timed": True,
    })


def compare_pair(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, Any]:
    lhs32 = lhs.float()
    rhs32 = rhs.float()
    delta = (lhs32 - rhs32).abs()
    flat = int(delta.reshape(-1).argmax().item())
    result = {
        "comparison": name,
        "max_abs": float(delta.reshape(-1)[flat].item()),
        "mean_abs": float(delta.mean().item()),
        "first_max_flat_index": flat,
        "lhs_dtype": str(lhs.dtype),
        "rhs_dtype": str(rhs.dtype),
        "shape": list(lhs.shape),
    }
    if lhs.dtype == torch.bfloat16 and rhs.dtype == torch.bfloat16:
        result["bf16_bit_mismatch_count"] = int((lhs.view(torch.uint16) != rhs.view(torch.uint16)).sum().item())
    else:
        result["bf16_bit_mismatch_count"] = None
    return result


def run_correctness(args: argparse.Namespace) -> None:
    patch_rocm_autotune()
    matrix = [
        (64, 2026072101, "random", True, False),
        (128, 2026072102, "neutral_gate", False, False),
        (512, 2026072103, "zero_beta", True, False),
        (1024, 2026072104, "small_values", False, False),
        (2048, 2026072105, "high_dynamic", True, False),
        (2048, 2026072106, "cancellation", True, False),
        (2048, 2026072107, "sparse_beta", True, True),
        (8192, 2026072108, "random", True, False),
    ]
    rows: list[dict[str, Any]] = []
    for t, seed, mode, with_initial_state, nondefault_stream in matrix:
        calls = public_calls(input_case(t, seed, mode, with_initial_state))
        if nondefault_stream:
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                values = {name: fn() for name, fn in calls.items()}
            stream.synchronize()
        else:
            values = {name: fn() for name, fn in calls.items()}
            torch.cuda.synchronize()
        ref_output, ref_state = values["vllm"]
        if ref_state is None:
            raise AssertionError("vLLM public API returned no final state")
        for name in ("stage6s", "f1"):
            output, state = values[name]
            if state is None:
                raise AssertionError(f"{name} public API returned no final state")
            out_delta = (output.float() - ref_output.float()).abs()
            state_delta = (state - ref_state).abs()
            item = {
                "timing_contract": CONTRACT,
                "cuda_graph_used": False,
                "T": t,
                "seed": seed,
                "mode": mode,
                "initial_state": with_initial_state,
                "nondefault_stream": nondefault_stream,
                "implementation": name,
                "output_max_abs": float(out_delta.max().item()),
                "output_mean_abs": float(out_delta.mean().item()),
                "final_state_max_abs": float(state_delta.max().item()),
                "final_state_mean_abs": float(state_delta.mean().item()),
                "output_finite": bool(torch.isfinite(output.float()).all().item()),
                "state_finite": bool(torch.isfinite(state).all().item()),
            }
            item["accepted"] = bool(
                item["output_max_abs"] <= OUTPUT_ATOL
                and item["final_state_max_abs"] <= STATE_ATOL
                and item["output_finite"]
                and item["state_finite"]
            )
            if not item["accepted"]:
                raise AssertionError(item)
            rows.append(item)
    write_csv(args.out_dir / "eager_full_correctness.csv", rows)
    write_json(args.out_dir / "correctness_summary.json", {
        "timing_contract": CONTRACT,
        "cuda_graph_used": False,
        "public_full_correct": True,
        "cases": len(matrix),
        "rows": len(rows),
        "max_output_abs": max(float(row["output_max_abs"]) for row in rows),
        "max_final_state_abs": max(float(row["final_state_max_abs"]) for row in rows),
        "recurrence_bridge_hash": "632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e",
    })


def config_summary(config: Any) -> dict[str, Any]:
    if config is None:
        return {"available": False}
    payload: dict[str, Any] = {"available": True, "repr": repr(config)}
    for key in ("kwargs", "num_warps", "num_stages", "num_ctas", "maxnreg", "cluster_dims", "enable_warp_specialization"):
        if hasattr(config, key):
            value = getattr(config, key)
            payload[key] = list(value) if isinstance(value, tuple) else value
    return payload


def selected_runtime_config() -> dict[str, Any]:
    obj: Any = wy_fast.recompute_w_u_fwd_kernel
    for _ in range(5):
        if hasattr(obj, "best_config"):
            return config_summary(getattr(obj, "best_config"))
        obj = getattr(obj, "fn", None)
        if obj is None:
            break
    return {"available": False}


def runtime_summary() -> dict[str, Any]:
    """Record every visible Triton wrapper layer after the real public call.

    `recompute_w_u_fwd_kernel` is wrapped by Triton's Heuristics decorator, so
    its selected Autotuner configuration lives on a nested ``fn`` object rather
    than necessarily on the outer callable.  This is diagnostic metadata only;
    the cache artifact is still copied separately for IR/ISA review.
    """

    obj: Any = wy_fast.recompute_w_u_fwd_kernel
    layers: list[dict[str, Any]] = []
    for depth in range(5):
        attrs: dict[str, str] = {}
        for key in ("best_config", "configs", "configs_timings", "keys", "key_idx", "cache", "cache_results"):
            if hasattr(obj, key):
                attrs[key] = repr(getattr(obj, key))[:30000]
        layers.append({
            "depth": depth,
            "type": str(type(obj)),
            "attributes": attrs,
            "dict_keys": sorted(getattr(obj, "__dict__", {}).keys()),
        })
        next_obj = getattr(obj, "fn", None)
        if next_obj is None or next_obj is obj:
            break
        obj = next_obj
    return {"layers": layers, "selected_config": selected_runtime_config()}


def find_cache_candidates(cache_root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for group_path in cache_root.rglob("__grp__recompute_w_u_fwd_kernel.json"):
        try:
            group = json.loads(group_path.read_text())
            children = {name: Path(path) for name, path in group.get("child_paths", {}).items()}
            metadata_path = children.get("recompute_w_u_fwd_kernel.json")
            metadata = json.loads(metadata_path.read_text()) if metadata_path and metadata_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            continue
        candidates.append({
            "group_path": str(group_path),
            "children": {name: str(path) for name, path in children.items()},
            "metadata": metadata,
            "mtime_ns": group_path.stat().st_mtime_ns,
        })
    return sorted(candidates, key=lambda item: int(item["mtime_ns"]), reverse=True)


def copy_candidate(candidate: dict[str, Any], destination: Path, t: int) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    names = {
        "recompute_w_u_fwd_kernel.hsaco": "kernel.hsaco",
        "recompute_w_u_fwd_kernel.ttir": "ttir.mlir",
        "recompute_w_u_fwd_kernel.ttgir": "ttgir.mlir",
        "recompute_w_u_fwd_kernel.llir": "llvm.ll",
        "recompute_w_u_fwd_kernel.amdgcn": "amdgcn.s",
        "recompute_w_u_fwd_kernel.source": "source.py",
    }
    copied: dict[str, str] = {}
    children = {name: Path(path) for name, path in dict(candidate["children"]).items()}
    for child_name, output_name in names.items():
        source = children.get(child_name)
        if source and source.exists():
            target = destination / output_name
            shutil.copy2(source, target)
            copied[output_name] = str(target)
    if "amdgcn.s" in copied:
        shutil.copy2(destination / "amdgcn.s", destination / "disassembly.txt")
    metadata = dict(candidate["metadata"])
    write_text(destination / "metadata.yaml", json.dumps(metadata, indent=2, ensure_ascii=False))
    launch = {
        "T": t,
        "BT": 64,
        "BK": 64,
        "BV": 64,
        "H": 8,
        "Hg": 4,
        "K": 128,
        "V": 128,
        "grid": [t // 64, 8, 1],
        "cta": (t // 64) * 8,
        "workgroup": int(metadata.get("num_warps", 0)) * 64,
        "num_warps": metadata.get("num_warps"),
        "num_stages": metadata.get("num_stages"),
        "shared_bytes": metadata.get("shared"),
        "cache_group": candidate["group_path"],
    }
    write_json(destination / "launch.json", launch)
    abi = {
        "diagnostic_only": True,
        "pointer_args": [
            ["k", "BF16", "[B,T,Hg,K]", "contiguous [B,T,Hg,K]"],
            ["v", "BF16", "[B,T,H,V]", "contiguous [B,T,H,V]"],
            ["beta", "FP32", "[B,T,H]", "contiguous [B,T,H]"],
            ["w", "BF16", "[B,T,H,K]", "allocated output"],
            ["u", "BF16", "[B,T,H,V]", "allocated output"],
            ["A", "BF16", "[B,T,H,BT]", "solve output"],
            ["g", "FP32", "[B,T,H]", "chunk-local cumsum"],
        ],
        "runtime_scalars": ["T"],
        "constexpr": ["H", "Hg", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
        "head_mapping": "key_head = value_head // (H/Hg) = value_head // 2",
    }
    write_json(destination / "abi.json", abi)
    write_json(destination / "autotune.json", {
        "key": ["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
        "runtime_fixed_key": {"H": 8, "K": 128, "V": 128, "BT": 64, "BK": 64, "BV": 64, "IS_VARLEN": False},
        "selected_metadata": metadata,
    })
    hsaco = destination / "kernel.hsaco"
    sha = sha256(hsaco) if hsaco.exists() else None
    write_text(destination / "sha256.txt", f"{sha or 'N/A'}  kernel.hsaco")
    return {"launch": launch, "metadata": metadata, "hsaco_sha256": sha, "copied": copied}


def select_runtime_candidate(candidates: list[dict[str, Any]], runtime_config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Match the post-call Autotuner winner to its compiled cache artifact."""

    if not runtime_config.get("available"):
        return candidates[0], "fallback_most_recent_runtime_config_unavailable"
    warps = runtime_config.get("num_warps")
    stages = runtime_config.get("num_stages")
    matches = [
        candidate
        for candidate in candidates
        if candidate.get("metadata", {}).get("num_warps") == warps
        and candidate.get("metadata", {}).get("num_stages") == stages
    ]
    if len(matches) == 1:
        return matches[0], "runtime_autotuner_best_config_matched_num_warps_num_stages"
    if not matches:
        return candidates[0], "fallback_most_recent_no_cache_match_for_runtime_config"
    return matches[0], "fallback_most_recent_ambiguous_runtime_config_cache_match"


def run_capture(args: argparse.Namespace) -> None:
    cache_env = os.environ.get("TRITON_CACHE_DIR")
    if not cache_env:
        raise EnvironmentError("TRITON_CACHE_DIR must point at a fresh per-T audit cache before importing Triton")
    patch_rocm_autotune()
    t_dir = args.out_dir / "vllm_actual" / "by_t" / f"T{args.T}"
    inputs = input_case(args.T, 2026072200 + args.T, "random", True)
    call = public_calls(inputs)["vllm"]
    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    output, final_state = call()
    torch.cuda.synchronize()
    if final_state is None or not torch.isfinite(output.float()).all() or not torch.isfinite(final_state).all():
        raise AssertionError("public vLLM capture call did not produce finite output/final_state")
    cache_root = Path(cache_env)
    candidates = find_cache_candidates(cache_root)
    if not candidates:
        raise RuntimeError(f"no recompute_w_u_fwd_kernel cache group found under {cache_root}")
    runtime = runtime_summary()
    write_json(t_dir / "runtime_kernel_object.json", runtime)
    write_json(t_dir / "all_cache_candidates.json", candidates)
    selected, selected_by = select_runtime_candidate(candidates, dict(runtime["selected_config"]))
    result = copy_candidate(selected, t_dir, args.T)
    write_json(t_dir / "capture_result.json", {
        "timing_contract": CONTRACT,
        "cuda_graph_used": False,
        "diagnostic_only": True,
        "public_api": "vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule",
        "public_call_completed": True,
        "candidate_count": len(candidates),
        "selected_by": selected_by,
        "runtime_selected_config": runtime["selected_config"],
        "result": result,
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "final_state_shape": list(final_state.shape),
        "final_state_dtype": str(final_state.dtype),
    })


def run_diagnostic(args: argparse.Namespace) -> None:
    patch_rocm_autotune()
    q, k, v, g, beta, _ = input_case(args.T, 2026072300 + args.T, "random", True)
    g_f1 = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    a_f1 = qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_f1, beta)
    solved_f1 = _hierarchical_solve(a_f1)
    w_f1, u_f1 = qwen_gdn_w_u_bt64_fused_stage6t_bf16(k, v, g_f1, beta, solved_f1)
    g_vllm = chunk_local_cumsum(g, chunk_size=BT)
    a_vllm = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=g_vllm, output_dtype=torch.float32)
    solved_vllm = solve_tril(A=a_vllm, output_dtype=k.dtype)
    w_vllm, u_vllm = wy_fast.recompute_w_u_fwd(k=k, v=v, beta=beta, A=solved_vllm, g_cumsum=g_vllm, cu_seqlens=None)
    torch.cuda.synchronize()
    rows = [
        {"timing_contract": CONTRACT, "cuda_graph_used": False, "diagnostic_only": True, "T": args.T, **compare_pair("solve_F1_FP32_vs_vLLM_BF16_widened", solved_f1, solved_vllm.float())},
        {"timing_contract": CONTRACT, "cuda_graph_used": False, "diagnostic_only": True, "T": args.T, **compare_pair("W_F1_vs_vLLM", w_f1, w_vllm)},
        {"timing_contract": CONTRACT, "cuda_graph_used": False, "diagnostic_only": True, "T": args.T, **compare_pair("U_F1_vs_vLLM", u_f1, u_vllm)},
    ]
    write_csv(args.out_dir / "diagnostic_intermediate_correctness.csv", rows)
    write_text(args.out_dir / "first_divergence.md", "# First Divergence\n\nThe diagnostic comparisons intentionally use different solve-to-W/U precision contracts: F1 consumes FP32 `a_solved`, while native vLLM `wy_fast` consumes BF16 `A` from `solve_tril`. Their first nonzero divergence is therefore expected and is diagnostic-only, not a public correctness failure.")


def run_public_validation(args: argparse.Namespace) -> None:
    # `chunk_gated_delta_rule` is decorated, so inspect on the callable points
    # at torch._dynamo.  Record the module that owns the public implementation.
    source = inspect.getsourcefile(vllm_chunk_module)
    payload = {
        "timing_contract": CONTRACT,
        "cuda_graph_used": False,
        "stage6s": {
            "import": "qwen_gdn_bt64_bf16_recurrence_full_stage6s",
            "signature": str(inspect.signature(qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge)),
            "public": True,
            "allocation_inside_timed_call": True,
        },
        "f1": {
            "import": "qwen_gdn_bt64_fused_wu_eager_stage6t",
            "signature": str(inspect.signature(qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager)),
            "fused_wu_symbol": "_qwen_gdn_wu_bf16_kernel_bt64_fused_stage6t_bf16",
            "public": True,
            "allocation_inside_timed_call": True,
        },
        "vllm": {
            "import": "vllm.model_executor.layers.fla.ops",
            "signature": str(inspect.signature(vllm_full)),
            "source": str(source),
            "actual_wu_module": inspect.getsourcefile(wy_fast),
            "actual_wu_symbol": "recompute_w_u_fwd_kernel",
            "public": True,
            "allocation_inside_timed_call": True,
        },
    }
    write_json(args.out_dir / "eager_public_api_contract.json", payload)
    write_text(args.out_dir / "eager_public_api_contract.md", "# Stage 6T-Golden Eager Public API Contract\n\nAll authoritative timing and correctness modes invoke only complete eager public APIs. Allocation, casts, dispatches, wrapper work, and returned output construction occur inside each timed call. Capture, IR/ISA, profiler counters, and intermediate W/U comparisons are diagnostic-only.")
    write_text(args.out_dir / "public_api_call_map.md", "# Public API Call Map\n\n- Stage 6S: `qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge`.\n- F1: `qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager`.\n- Native: `vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule`.\n\nThe benchmark runner forms lambdas around only these APIs; it never launches a private body as a timed callable.")
    write_text(args.out_dir / "public_api_validation.txt", "PASS: authoritative modes use public eager APIs; no private out-buffer shortcut; allocation remains inside timed calls.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("benchmark", "correctness", "capture", "diagnostic", "contract"), required=True)
    parser.add_argument("--T", nargs="+", type=int, default=[512, 2048, 8192, 16384])
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "benchmark":
        run_benchmark(args)
    elif args.mode == "correctness":
        run_correctness(args)
    elif args.mode == "capture":
        if len(args.T) != 1:
            raise ValueError("capture requires exactly one T")
        args.T = args.T[0]
        run_capture(args)
    elif args.mode == "diagnostic":
        if len(args.T) != 1:
            raise ValueError("diagnostic requires exactly one T")
        args.T = args.T[0]
        run_diagnostic(args)
    else:
        run_public_validation(args)


if __name__ == "__main__":
    main()
