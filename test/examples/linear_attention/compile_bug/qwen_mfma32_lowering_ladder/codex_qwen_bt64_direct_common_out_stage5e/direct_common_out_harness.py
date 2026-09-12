#!/usr/bin/env python3
"""Stage 5E audit harness for true caller-provided common solve output.

The existing solve kernels are launched directly into one preallocated tensor.
There is no solve-output copy, fill, allocation, or production-path change.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch


HERE = Path(__file__).resolve().parent
LADDER = HERE.parent
LINEAR = LADDER.parents[1]
VLLM_COMPARE = LINEAR / "vllm_compare"
STAGE5D = LADDER / "codex_qwen_bt64_downstream_state_coupling_stage5d"
STAGE2 = LADDER / "codex_qwen_bt64_full_pipeline_stage2"
for path in (VLLM_COMPARE, STAGE5D, STAGE2):
    sys.path.insert(0, str(path))

from downstream_state_coupling_harness import (  # noqa: E402
    BT,
    H_V,
    SOLVE_A,
    SOLVE_B,
    SOLVES,
    Stage5DRunner,
    ordered_labels,
    percentile,
    solved_statistics,
    summarize,
    tensor_info,
    write_csv,
)
from qwen_gdn_bt64_solve_direct_out_stage5e_audit import (  # noqa: E402
    qwen_gdn_solve_hierarchical_fp32_v1_direct_out_audit,
    qwen_gdn_solve_v18_bt64_direct_out_audit,
)
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import (  # noqa: E402
    qwen_gdn_solve_avelang_v18_bt64_layout,
)
from qwen_gdn_solve_bt64_hierarchical_fp32_v1 import (  # noqa: E402
    qwen_gdn_solve_bt64_hierarchical_fp32_v1,
)
from stage2_runner import patch_rocm_autotune  # noqa: E402


ORDERS = ("ABAB", "BABA", "RANDOM_ABBA", "ABAB", "BABA")


class Stage5ERunner(Stage5DRunner):
    """Stage 5D frozen graph with one additional fixed solve output tensor."""

    def __init__(self, t: int, seed: int, mode: str = "random", with_state: bool = True) -> None:
        super().__init__(t, seed, mode, with_state)
        self.solved_common = torch.empty_like(self.buffers.a)
        self.allocation_id = f"T{t}-ptr-{int(self.solved_common.data_ptr()):x}"

    def launch_direct_solve(self, solve_impl: str) -> torch.Tensor:
        pointer = int(self.solved_common.data_ptr())
        if solve_impl == SOLVE_A:
            returned = qwen_gdn_solve_v18_bt64_direct_out_audit(self.buffers.a, self.solved_common)
        elif solve_impl == SOLVE_B:
            returned = qwen_gdn_solve_hierarchical_fp32_v1_direct_out_audit(
                self.buffers.a, self.solved_common
            )
        else:
            raise ValueError(f"unknown solve_impl {solve_impl!r}")
        if returned.data_ptr() != pointer or self.solved_common.data_ptr() != pointer:
            raise RuntimeError("direct solve did not preserve the exact common output pointer")
        return self.solved_common

    def launch_full_direct(self, solve_impl: str) -> None:
        self.launch_cumsum()
        self.launch_kkt()
        self.launch_direct_solve(solve_impl)
        self.launch_tail(self.solved_common)

    def common_info(self) -> dict[str, object]:
        result = tensor_info("solved_common", self.solved_common)
        result["allocation_id"] = self.allocation_id
        return result


class EventTimer:
    def __init__(self) -> None:
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def measure(self, fn: Callable[[], None]) -> float:
        self.start.record()
        fn()
        self.end.record()
        self.end.synchronize()
        return float(self.start.elapsed_time(self.end))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float((diff / expected.float().abs().clamp_min(1.0e-6)).max().item()),
    }


def make_lower_input(t: int, pattern: str, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    shape = (1, t, H_V, BT)
    values = torch.zeros(shape, dtype=torch.float32, device="cuda")
    if pattern == "single_subdiagonal":
        for row in range(1, BT):
            values[:, row::BT, :, row - 1] = 0.015625
        return values.contiguous()
    if pattern == "block_boundary":
        for row in (16, 32, 48, 63):
            values[:, row::BT, :, row - 1] = -0.03125
            values[:, row::BT, :, 0] = 0.015625
        return values.contiguous()

    scale = 0.02
    if pattern == "small_values":
        scale = 1.0e-5
    elif pattern == "high_dynamic":
        scale = 0.08
    random_values = torch.randn(shape, generator=generator, dtype=torch.float32, device="cuda") * scale
    rows = torch.arange(BT, device="cuda").view(BT, 1)
    cols = torch.arange(BT, device="cuda").view(1, BT)
    mask = (cols < rows).repeat(t // BT, 1).view(1, t, 1, BT)
    values = random_values * mask
    if pattern == "cancellation":
        signs = torch.where((cols + rows) % 2 == 0, 1.0, -1.0)
        values = values.abs() * signs.repeat(t // BT, 1).view(1, t, 1, BT) * mask
    elif pattern == "sparse_strict_lower":
        sparse = torch.rand(shape, generator=generator, device="cuda") < 0.08
        values = values * sparse
    elif pattern not in ("random", "small_values", "high_dynamic"):
        raise ValueError(f"unknown pattern {pattern}")
    return values.contiguous()


def authority_solve(a: torch.Tensor) -> torch.Tensor:
    chunks = a.shape[1] // BT
    matrices = a[0].view(chunks, BT, H_V, BT).permute(0, 2, 1, 3).float()
    identity = torch.eye(BT, dtype=torch.float32, device=a.device).expand(chunks, H_V, BT, BT)
    solved = torch.linalg.solve_triangular(matrices + identity, identity, upper=False)
    return solved.permute(0, 2, 1, 3).reshape_as(a).contiguous()


def residual_inf(a: torch.Tensor, solved: torch.Tensor) -> float:
    chunks = a.shape[1] // BT
    matrices = a[0].view(chunks, BT, H_V, BT).permute(0, 2, 1, 3).float()
    result = solved[0].view(chunks, BT, H_V, BT).permute(0, 2, 1, 3).float()
    identity = torch.eye(BT, dtype=torch.float32, device=a.device).expand(chunks, H_V, BT, BT)
    return float(((matrices + identity) @ result - identity).abs().max().item())


def direct_solve(a: torch.Tensor, out: torch.Tensor, solve_impl: str) -> torch.Tensor:
    if solve_impl == SOLVE_A:
        return qwen_gdn_solve_v18_bt64_direct_out_audit(a, out)
    if solve_impl == SOLVE_B:
        return qwen_gdn_solve_hierarchical_fp32_v1_direct_out_audit(a, out)
    raise ValueError(solve_impl)


def original_solve(a: torch.Tensor, solve_impl: str) -> torch.Tensor:
    if solve_impl == SOLVE_A:
        return qwen_gdn_solve_avelang_v18_bt64_layout(a)
    if solve_impl == SOLVE_B:
        return qwen_gdn_solve_bt64_hierarchical_fp32_v1(a)
    raise ValueError(solve_impl)


def bootstrap_ci(values: list[float], seed: int, samples: int = 2000) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = random.Random(seed)
    medians = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        medians.append(statistics.median(draw))
    return percentile(medians, 0.025), percentile(medians, 0.975)


def run_contract(args: argparse.Namespace) -> None:
    runner = Stage5ERunner(2048, args.seed)
    runner.prepare_local_matrix()
    common = runner.common_info()
    rows = []
    for ordinal, solve_impl in enumerate(SOLVES):
        before = int(runner.solved_common.data_ptr())
        returned = runner.launch_direct_solve(solve_impl)
        after = int(returned.data_ptr())
        rows.append({
            "allocation_session": 0,
            "solve_impl": solve_impl,
            "data_ptr": after,
            "same_as_common": after == int(runner.solved_common.data_ptr()),
            "pointer_stable": before == after,
            **{key: value for key, value in common.items() if key not in ("name", "data_ptr")},
        })
    torch.cuda.synchronize()
    write_csv(args.out_dir / "direct_out_pointer_log.csv", rows)

    common_dispatches = runner.graph_contract()[SOLVE_A]["dispatches"]
    mode_dir = args.out_dir / "mode_dispatch_maps"
    mode_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("original", "copy_control", "direct_common"):
        for solve_impl in SOLVES:
            dispatches = [dict(item) for item in common_dispatches]
            dispatches[2]["symbol"] = (
                "_qwen_gdn_solve_kernel_v18_parallel"
                if solve_impl == SOLVE_A
                else "_qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1"
            )
            dispatches[2]["workgroup"] = [128 if solve_impl == SOLVE_A else 256, 1, 1]
            if mode == "copy_control":
                dispatches.insert(3, {
                    "ordinal": 3,
                    "stage": "solve-output copy",
                    "symbol": "PyTorch device-to-device copy",
                    "grid": None,
                    "workgroup": None,
                    "dynamic_lds": None,
                })
                for ordinal, dispatch in enumerate(dispatches):
                    dispatch["ordinal"] = ordinal
            payload = {
                "mode": mode,
                "solve_impl": solve_impl,
                "dispatches": dispatches,
                "solve_output_pointer": int(runner.solved_common.data_ptr()) if mode == "direct_common" else "separate",
                "copy_between_solve_and_wu": mode == "copy_control",
                "allocation_between_solve_and_wu": False,
                "fill_between_solve_and_wu": False,
            }
            write_json(mode_dir / f"{mode}_{solve_impl}.json", payload)


def run_standalone_correctness(args: argparse.Namespace) -> None:
    patterns = (
        "random",
        "small_values",
        "high_dynamic",
        "cancellation",
        "sparse_strict_lower",
        "single_subdiagonal",
        "block_boundary",
    )
    rows: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    residuals: list[dict[str, object]] = []
    data_statistics: list[dict[str, object]] = []
    for t in (64, 128, 512, 2048, 8192):
        selected = patterns if t <= 512 else ("random", "high_dynamic", "sparse_strict_lower", "block_boundary")
        for pattern in selected:
            a = make_lower_input(t, pattern, args.seed + t)
            authority = authority_solve(a)
            originals = {solve: original_solve(a, solve) for solve in SOLVES}
            for solve_impl in SOLVES:
                out = torch.full_like(a, float("nan"))
                before = int(out.data_ptr())
                direct = direct_solve(a, out, solve_impl)
                torch.cuda.synchronize()
                stats = error(direct, originals[solve_impl])
                auth = error(direct, authority)
                bitwise = bool(torch.equal(direct.view(torch.int32), originals[solve_impl].view(torch.int32)))
                upper = torch.triu(
                    direct[0].view(t // BT, BT, H_V, BT).permute(0, 2, 1, 3), diagonal=1
                )
                diagonal = direct[0].view(t // BT, BT, H_V, BT).permute(0, 2, 1, 3).diagonal(dim1=-2, dim2=-1)
                rows.append({
                    "T": t,
                    "pattern": pattern,
                    "solve_impl": solve_impl,
                    "original_vs_direct_bitwise": bitwise,
                    "original_max_abs": stats["max_abs"],
                    "original_mean_abs": stats["mean_abs"],
                    "authority_max_abs": auth["max_abs"],
                    "authority_mean_abs": auth["mean_abs"],
                    "nan_count": int(torch.isnan(direct).sum().item()),
                    "inf_count": int(torch.isinf(direct).sum().item()),
                })
                coverage.append({
                    "T": t,
                    "pattern": pattern,
                    "solve_impl": solve_impl,
                    "pointer_stable": before == int(direct.data_ptr()),
                    "nan_after_prefill": int(torch.isnan(direct).sum().item()),
                    "upper_max_abs": float(upper.abs().max().item()),
                    "diagonal_max_abs_from_one": float((diagonal - 1.0).abs().max().item()),
                })
                residuals.append({
                    "T": t,
                    "pattern": pattern,
                    "solve_impl": solve_impl,
                    "residual_inf": residual_inf(a, direct),
                })

            # Reuse one exact allocation across alternating implementations.
            common = torch.full_like(a, 12345.0)
            v18_first = direct_solve(a, common, SOLVE_A).clone()
            direct_solve(a, common, SOLVE_B)
            v18_second = direct_solve(a, common, SOLVE_A)
            torch.cuda.synchronize()
            coverage.append({
                "T": t,
                "pattern": pattern,
                "solve_impl": "v18_after_v1_reuse",
                "pointer_stable": True,
                "nan_after_prefill": int(torch.isnan(v18_second).sum().item()),
                "upper_max_abs": float((v18_first - v18_second).abs().max().item()),
                "diagonal_max_abs_from_one": 0.0,
            })

    # Real Stage 4 KKT output at all required T values.
    for t in (64, 128, 512, 2048, 8192):
        runner = Stage5ERunner(t, args.seed + 100000 + t)
        runner.prepare_local_matrix()
        a = runner.buffers.a
        originals = {solve: original_solve(a, solve) for solve in SOLVES}
        direct_outputs = {}
        for solve_impl in SOLVES:
            out = torch.empty_like(a)
            direct_outputs[solve_impl] = direct_solve(a, out, solve_impl).clone()
            torch.cuda.synchronize()
            stats = error(direct_outputs[solve_impl], originals[solve_impl])
            rows.append({
                "T": t,
                "pattern": "stage4_kkt",
                "solve_impl": solve_impl,
                "original_vs_direct_bitwise": bool(torch.equal(out.view(torch.int32), originals[solve_impl].view(torch.int32))),
                "original_max_abs": stats["max_abs"],
                "original_mean_abs": stats["mean_abs"],
                "authority_max_abs": math.nan,
                "authority_mean_abs": math.nan,
                "nan_count": int(torch.isnan(out).sum().item()),
                "inf_count": int(torch.isinf(out).sum().item()),
            })
        stats = solved_statistics(direct_outputs[SOLVE_A], direct_outputs[SOLVE_B])
        stats.update({"T": t, "pattern": "stage4_kkt"})
        data_statistics.append(stats)

    write_csv(args.out_dir / "standalone_direct_out_correctness.csv", rows)
    write_csv(args.out_dir / "write_coverage_validation.csv", coverage)
    write_csv(args.out_dir / "residual_analysis.csv", residuals)
    write_csv(args.out_dir / "direct_out_data_statistics.csv", data_statistics)


def run_full_correctness(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    worst_output = 0.0
    worst_state = 0.0
    for t in (64, 512, 2048, 8192):
        runner = Stage5ERunner(t, args.seed + 200000 + t)
        runner.launch_full(SOLVE_A)
        original_output, original_state = runner.output_snapshot()
        runner.launch_full_direct(SOLVE_A)
        direct_a_output, direct_a_state = runner.output_snapshot()
        runner.launch_full_direct(SOLVE_B)
        direct_b_output, direct_b_state = runner.output_snapshot()
        a_out = error(direct_a_output, original_output)
        a_state = error(direct_a_state, original_state)
        b_out = error(direct_b_output, direct_a_output)
        b_state = error(direct_b_state, direct_a_state)
        worst_output = max(worst_output, b_out["max_abs"])
        worst_state = max(worst_state, b_state["max_abs"])
        rows.append({
            "T": t,
            "v18_original_direct_output_max_abs": a_out["max_abs"],
            "v18_original_direct_state_max_abs": a_state["max_abs"],
            "v1_vs_v18_output_max_abs": b_out["max_abs"],
            "v1_vs_v18_output_mean_abs": b_out["mean_abs"],
            "v1_vs_v18_state_max_abs": b_state["max_abs"],
            "v1_vs_v18_state_mean_abs": b_state["mean_abs"],
            "common_data_ptr": int(runner.solved_common.data_ptr()),
        })
    write_csv(args.out_dir / "tests/full_direct_correctness.csv", rows)
    write_json(args.out_dir / "correctness_summary.json", {
        "standalone_executed": True,
        "full_executed": True,
        "v18_original_vs_direct_bitwise": all(
            row["original_vs_direct_bitwise"].lower() == "true"
            for row in _read_csv(args.out_dir / "standalone_direct_out_correctness.csv")
            if row["solve_impl"] == SOLVE_A
        ),
        "v1_original_vs_direct_bitwise": all(
            row["original_vs_direct_bitwise"].lower() == "true"
            for row in _read_csv(args.out_dir / "standalone_direct_out_correctness.csv")
            if row["solve_impl"] == SOLVE_B
        ),
        "full_output_max_abs_v1_vs_v18": worst_output,
        "full_final_state_max_abs_v1_vs_v18": worst_state,
        "output_threshold": 0.0078125,
        "final_state_threshold": 0.02,
    })


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def setup_runner(t: int, seed: int) -> Stage5ERunner:
    runner = Stage5ERunner(t, seed)
    runner.prepare_local_matrix()
    return runner


def measure_direct_pair(
    runner: Stage5ERunner,
    operation: str,
    solve_impl: str,
    timer: EventTimer,
    control: str = "none",
    flush_mib: int = 512,
) -> float:
    if operation == "tail":
        runner.launch_direct_solve(solve_impl)
        if control == "warm":
            runner.launch_tail(runner.solved_common)
        elif control == "perturb":
            runner.perturb_cache(flush_mib)
        elif control == "prime":
            runner.prime_downstream_inputs(runner.solved_common)
        elif control != "none":
            raise ValueError(control)
        return timer.measure(lambda: runner.launch_tail(runner.solved_common))
    if operation == "solve_tail":
        return timer.measure(lambda: (runner.launch_direct_solve(solve_impl), runner.launch_tail(runner.solved_common)))
    if operation == "full":
        return timer.measure(lambda: runner.launch_full_direct(solve_impl))
    raise ValueError(operation)


def benchmark_operation(
    *,
    operation: str,
    t_values: tuple[int, ...],
    args: argparse.Namespace,
    output_summary: str,
    output_raw: str,
    control: str = "none",
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    summaries: list[dict[str, object]] = []
    raw: list[dict[str, object]] = []
    pointers: list[dict[str, object]] = []
    for t in t_values:
        session_deltas: list[float] = []
        for session in range(args.sessions):
            runner = setup_runner(t, args.seed + t * 100 + session)
            pointer_row = runner.common_info()
            pointer_row.update({"T": t, "session": session, "operation": operation, "control": control})
            pointers.append(pointer_row)
            timer = EventTimer()
            for _ in range(args.warmup):
                for solve_impl in SOLVES:
                    measure_direct_pair(runner, operation, solve_impl, timer, control, args.flush_mib)
            order = ORDERS[session % len(ORDERS)]
            labels = ordered_labels(order, args.repeat, args.seed + session + t)
            values = {SOLVE_A: [], SOLVE_B: []}
            for sequence, solve_impl in enumerate(labels):
                latency = measure_direct_pair(
                    runner, operation, solve_impl, timer, control, args.flush_mib
                )
                values[solve_impl].append(latency)
                raw.append({
                    "operation": operation,
                    "control": control,
                    "T": t,
                    "session": session,
                    "allocation_id": runner.allocation_id,
                    "common_data_ptr": int(runner.solved_common.data_ptr()),
                    "order": order,
                    "sequence": sequence,
                    "solve_impl": solve_impl,
                    "latency_ms": latency,
                })
            stats = {solve: summarize(values[solve]) for solve in SOLVES}
            delta = stats[SOLVE_B]["median_ms"] - stats[SOLVE_A]["median_ms"]
            session_deltas.append(delta)
            for solve_impl in SOLVES:
                summaries.append({
                    "operation": operation,
                    "control": control,
                    "T": t,
                    "session": session,
                    "allocation_id": runner.allocation_id,
                    "common_data_ptr": int(runner.solved_common.data_ptr()),
                    "order": order,
                    "solve_impl": solve_impl,
                    **stats[solve_impl],
                    "v1_minus_v18_us": delta * 1000.0,
                })
        low, high = bootstrap_ci(session_deltas, args.seed + t)
        summaries.append({
            "operation": operation,
            "control": control,
            "T": t,
            "session": "aggregate",
            "allocation_id": "multiple",
            "common_data_ptr": "multiple",
            "order": "all",
            "solve_impl": "v1_minus_v18",
            "median_ms": statistics.median(session_deltas),
            "p10_ms": min(session_deltas),
            "p90_ms": max(session_deltas),
            "minimum_ms": low,
            "maximum_ms": high,
            "v1_minus_v18_us": statistics.median(session_deltas) * 1000.0,
        })
    write_csv(args.out_dir / output_summary, summaries)
    write_csv(args.out_dir / output_raw, raw)
    return summaries, raw, pointers


def run_benchmarks(args: argparse.Namespace) -> None:
    pointer_rows: list[dict[str, object]] = []
    _, _, pointers = benchmark_operation(
        operation="tail",
        t_values=(512, 2048, 8192, 16384),
        args=args,
        output_summary="direct_common_tail_benchmark.csv",
        output_raw="direct_common_tail_raw.csv",
    )
    pointer_rows.extend(pointers)
    _, _, pointers = benchmark_operation(
        operation="solve_tail",
        t_values=(512, 1024, 2048, 4096, 8192, 16384),
        args=args,
        output_summary="solve_tail_benchmark.csv",
        output_raw="tests/solve_tail_raw.csv",
    )
    pointer_rows.extend(pointers)
    _, _, pointers = benchmark_operation(
        operation="full",
        t_values=(512, 1024, 2048, 4096, 8192, 16384),
        args=args,
        output_summary="full_direct_common_benchmark.csv",
        output_raw="full_direct_common_raw.csv",
    )
    pointer_rows.extend(pointers)
    write_csv(args.out_dir / "tests/cross_allocation_pointer_log.csv", pointer_rows)


def run_controls(args: argparse.Namespace) -> None:
    all_rows: list[dict[str, object]] = []
    all_raw: list[dict[str, object]] = []
    for control in ("none", "warm", "perturb", "prime"):
        rows, raw, _ = benchmark_operation(
            operation="tail",
            t_values=(2048, 8192),
            args=args,
            output_summary=f"tests/cache_{control}_summary.csv",
            output_raw=f"tests/cache_{control}_raw.csv",
            control=control,
        )
        all_rows.extend(rows)
        all_raw.extend(raw)
    write_csv(args.out_dir / "direct_common_cache_control.csv", all_rows)
    write_csv(args.out_dir / "tests/direct_common_cache_raw.csv", all_raw)


def run_profile(args: argparse.Namespace) -> None:
    t = args.T[0]
    runner = setup_runner(t, args.seed)
    abba = (SOLVE_A, SOLVE_B, SOLVE_B, SOLVE_A)
    for iteration in range(args.profile_repeat):
        solve_impl = abba[iteration % len(abba)] if args.profile_sequence == "abba" else args.solve_impl
        if args.profile_operation == "full":
            if args.control != "none":
                raise ValueError("full profile currently supports control=none only")
            runner.launch_full_direct(solve_impl)
        else:
            runner.launch_direct_solve(solve_impl)
            if args.control == "warm":
                runner.launch_tail(runner.solved_common)
            elif args.control == "perturb":
                runner.perturb_cache(args.flush_mib)
            elif args.control == "prime":
                runner.prime_downstream_inputs(runner.solved_common)
            runner.launch_tail(runner.solved_common)
    torch.cuda.synchronize()
    print(json.dumps({
        "T": t,
        "solve_impl": args.solve_impl,
        "control": args.control,
        "profile_operation": args.profile_operation,
        "profile_sequence": args.profile_sequence,
        "common_data_ptr": int(runner.solved_common.data_ptr()),
        "repeat": args.profile_repeat,
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contract", "correctness", "benchmark", "controls", "profile", "all"), default="all")
    parser.add_argument("--T", type=int, nargs="+", default=[2048])
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--flush-mib", type=int, default=512)
    parser.add_argument("--solve-impl", choices=SOLVES, default=SOLVE_A)
    parser.add_argument("--control", choices=("none", "warm", "perturb", "prime"), default="none")
    parser.add_argument("--profile-repeat", type=int, default=20)
    parser.add_argument("--profile-operation", choices=("tail", "full"), default="tail")
    parser.add_argument("--profile-sequence", choices=("single", "abba"), default="single")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    patch_rocm_autotune()
    if args.mode in ("contract", "all"):
        run_contract(args)
    if args.mode in ("correctness", "all"):
        run_standalone_correctness(args)
        run_full_correctness(args)
    if args.mode in ("benchmark", "all"):
        run_benchmarks(args)
    if args.mode in ("controls", "all"):
        run_controls(args)
    if args.mode == "profile":
        run_profile(args)


if __name__ == "__main__":
    main()
