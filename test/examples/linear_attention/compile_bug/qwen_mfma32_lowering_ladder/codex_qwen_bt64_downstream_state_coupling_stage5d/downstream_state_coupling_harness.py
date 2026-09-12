#!/usr/bin/env python3
"""Audit-only Stage 5D harness for BT64 solve/downstream coupling.

This file deliberately launches the existing Stage 4 kernels into fixed,
preallocated buffers.  It does not change any kernel, default dispatch, or
production entry point.  The two frozen graphs differ only in the solve JIT
symbol and its required workgroup size.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Iterable

import torch


HERE = Path(__file__).resolve().parent
LADDER = HERE.parent
LINEAR = LADDER.parents[1]
VLLM_COMPARE = LINEAR / "vllm_compare"
STAGE2 = LADDER / "codex_qwen_bt64_full_pipeline_stage2"
for path in (VLLM_COMPARE, STAGE2):
    sys.path.insert(0, str(path))

from qwen_gdn_bt64_gfx942_asm_v0_experimental import (  # noqa: E402
    contract as asm_contract,
    qwen_gdn_bt64_gfx942_asm_v0_preallocated,
)
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import (  # noqa: E402
    BT,
    H_K,
    H_V,
    K_DIM,
    V_DIM,
    _qwen_gdn_chunk_o_bf16_kernel_bt64_mfma_v2_s0,
    _qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0,
    _qwen_gdn_u_bf16_kernel_bt64_mfma_v2_s0,
    _qwen_gdn_w_bf16_kernel_bt64_mfma_v2_s0,
)
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import (  # noqa: E402
    _qwen_gdn_solve_kernel_v18_parallel,
)
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (  # noqa: E402
    _qwen_gdn_chunk_cumsum_kernel_v6_standalone,
)
from qwen_gdn_solve_bt64_hierarchical_fp32_v1 import (  # noqa: E402
    _qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1,
)
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402


SOLVE_A = "v18"
SOLVE_B = "hierarchical_fp32_v1"
SOLVES = (SOLVE_A, SOLVE_B)
OUTPUT_ATOL = 1.0e-3
STATE_ATOL = 1.0e-3


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "p10_ms": percentile(values, 0.10),
        "p90_ms": percentile(values, 0.90),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_info(name: str, tensor: torch.Tensor) -> dict[str, object]:
    pointer = int(tensor.data_ptr())
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "data_ptr": pointer,
        "storage_offset": int(tensor.storage_offset()),
        "nbytes": int(tensor.numel() * tensor.element_size()),
        "alignment_mod_16": pointer % 16,
        "alignment_mod_64": pointer % 64,
        "alignment_mod_128": pointer % 128,
        "alignment_mod_256": pointer % 256,
        "address_mod_4k": pointer % 4096,
        "address_mod_64k": pointer % 65536,
        "device": str(tensor.device),
    }


@dataclass
class Buffers:
    g_cumsum: torch.Tensor
    a: torch.Tensor
    solved_a: torch.Tensor
    solved_b: torch.Tensor
    canonical: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    h: torch.Tensor
    v_new: torch.Tensor
    final_state: torch.Tensor
    output_fp32: torch.Tensor
    output_bf16: torch.Tensor
    dummy: torch.Tensor

    @classmethod
    def allocate(cls, t: int, device: torch.device) -> "Buffers":
        chunks = t // BT
        solved_shape = (1, t, H_V, BT)
        wu_shape = (1, t, H_V, V_DIM)
        return cls(
            g_cumsum=torch.empty((1, t, H_V), dtype=torch.float32, device=device),
            a=torch.empty(solved_shape, dtype=torch.float32, device=device),
            solved_a=torch.empty(solved_shape, dtype=torch.float32, device=device),
            solved_b=torch.empty(solved_shape, dtype=torch.float32, device=device),
            canonical=torch.empty(solved_shape, dtype=torch.float32, device=device),
            w=torch.empty(wu_shape, dtype=torch.float32, device=device),
            u=torch.empty(wu_shape, dtype=torch.float32, device=device),
            h=torch.empty((1, chunks, H_V, V_DIM, K_DIM), dtype=torch.bfloat16, device=device),
            v_new=torch.empty(wu_shape, dtype=torch.float32, device=device),
            final_state=torch.empty((1, H_V, V_DIM, K_DIM), dtype=torch.float32, device=device),
            output_fp32=torch.empty(wu_shape, dtype=torch.float32, device=device),
            output_bf16=torch.empty(wu_shape, dtype=torch.bfloat16, device=device),
            dummy=torch.empty((1,), dtype=torch.float32, device=device),
        )

    def info(self) -> list[dict[str, object]]:
        return [tensor_info(field.name, getattr(self, field.name)) for field in fields(self)]


class Stage5DRunner:
    """Fixed-buffer launch graph used only by the audit harness."""

    def __init__(self, t: int, seed: int, mode: str = "random", with_state: bool = True) -> None:
        if t <= 0 or t % BT:
            raise ValueError("Stage 5D requires T > 0 and divisible by 64")
        q, k, v, g, beta, h0 = make_inputs(t, seed, mode, with_state)
        if h0 is None:
            h0 = torch.zeros((1, H_V, V_DIM, K_DIM), dtype=torch.float32, device=q.device)
        self.t = t
        self.num_chunks = t // BT
        self.q, self.k, self.v, self.g, self.beta, self.h0 = q, k, v, g, beta, h0
        self.buffers = Buffers.allocate(t, q.device)
        self.scale = K_DIM ** -0.5
        self.flush_buffer: torch.Tensor | None = None

    def launch_cumsum(self) -> None:
        _qwen_gdn_chunk_cumsum_kernel_v6_standalone[
            lambda: ((self.num_chunks * H_V, 1, 1), (1, 1, 1))
        ](self.g, self.buffers.g_cumsum, 1, self.t, H_V, BT, self.num_chunks)

    def launch_kkt(self) -> None:
        grid = self.num_chunks * H_V * 16
        _qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0[lambda: ((grid, 1, 1), (64, 1, 1))](
            self.k, self.buffers.g_cumsum, self.beta, self.buffers.a, self.t, self.num_chunks
        )

    def solved_buffer(self, solve_impl: str) -> torch.Tensor:
        if solve_impl == SOLVE_A:
            return self.buffers.solved_a
        if solve_impl == SOLVE_B:
            return self.buffers.solved_b
        raise ValueError(f"unknown solve_impl {solve_impl!r}")

    def launch_solve(self, solve_impl: str, out: torch.Tensor | None = None) -> torch.Tensor:
        if out is None:
            out = self.solved_buffer(solve_impl)
        grid = self.num_chunks * H_V
        if solve_impl == SOLVE_A:
            _qwen_gdn_solve_kernel_v18_parallel[lambda: ((grid, 1, 1), (128, 1, 1))](
                self.buffers.a, out, self.t, BT, self.num_chunks
            )
        elif solve_impl == SOLVE_B:
            _qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1[
                lambda: ((grid, 1, 1), (256, 1, 1))
            ](self.buffers.a, out, self.t, self.num_chunks)
        else:
            raise ValueError(f"unknown solve_impl {solve_impl!r}")
        return out

    def launch_wu(self, solved: torch.Tensor) -> None:
        grid = self.num_chunks * H_V * 8
        _qwen_gdn_w_bf16_kernel_bt64_mfma_v2_s0[lambda: ((grid, 1, 1), (256, 1, 1))](
            self.k,
            self.buffers.g_cumsum,
            self.beta,
            solved,
            self.buffers.w,
            self.t,
            self.num_chunks,
            False,
            True,
        )
        _qwen_gdn_u_bf16_kernel_bt64_mfma_v2_s0[lambda: ((grid, 1, 1), (256, 1, 1))](
            self.v,
            self.beta,
            solved,
            self.buffers.u,
            self.t,
            self.num_chunks,
            False,
            True,
        )

    def launch_asm(self) -> None:
        qwen_gdn_bt64_gfx942_asm_v0_preallocated(
            self.k,
            self.buffers.w,
            self.buffers.u,
            self.buffers.g_cumsum,
            self.h0,
            self.buffers.h,
            self.buffers.v_new,
            self.buffers.final_state,
        )

    def launch_chunk_o(self) -> None:
        grid = self.num_chunks * H_V * 8
        _qwen_gdn_chunk_o_bf16_kernel_bt64_mfma_v2_s0[lambda: ((grid, 1, 1), (256, 1, 1))](
            self.q,
            self.k,
            self.buffers.v_new,
            self.buffers.h,
            self.buffers.g_cumsum,
            self.buffers.output_fp32,
            float(self.scale),
            self.t,
            self.num_chunks,
        )

    def launch_tail(self, solved: torch.Tensor, include_cast: bool = True) -> None:
        self.launch_wu(solved)
        self.launch_asm()
        self.launch_chunk_o()
        if include_cast:
            self.buffers.output_bf16.copy_(self.buffers.output_fp32)

    def launch_full(self, solve_impl: str) -> None:
        self.launch_cumsum()
        self.launch_kkt()
        solved = self.launch_solve(solve_impl)
        self.launch_tail(solved)

    def prepare_local_matrix(self) -> None:
        self.launch_cumsum()
        self.launch_kkt()
        torch.cuda.synchronize()

    def prepare_canonical(self) -> None:
        self.prepare_local_matrix()
        self.launch_solve(SOLVE_A)
        self.buffers.canonical.copy_(self.buffers.solved_a)
        torch.cuda.synchronize()

    def dummy_predecessor(self) -> None:
        self.buffers.dummy.copy_(self.buffers.canonical.reshape(-1)[:1])

    def ensure_flush_buffer(self, mib: int) -> torch.Tensor:
        elements = mib * 1024 * 1024 // 4
        if self.flush_buffer is None or self.flush_buffer.numel() != elements:
            self.flush_buffer = torch.zeros((elements,), dtype=torch.float32, device=self.q.device)
        return self.flush_buffer

    def perturb_cache(self, mib: int) -> None:
        self.ensure_flush_buffer(mib).add_(1.0)

    def prime_downstream_inputs(self, solved: torch.Tensor) -> None:
        # All reductions are outside the timed tail and identical for A/B.
        checksum = solved.sum()
        checksum = checksum + self.k.sum() + self.v.sum()
        checksum = checksum + self.buffers.g_cumsum.sum() + self.beta.sum() + self.h0.sum()
        self.buffers.dummy.copy_(checksum.reshape(1))

    def output_snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.cuda.synchronize()
        return self.buffers.output_bf16.clone(), self.buffers.final_state.clone()

    def graph_contract(self) -> dict[str, object]:
        asm = asm_contract()
        common = [
            {"ordinal": 0, "stage": "cumsum", "symbol": "_qwen_gdn_chunk_cumsum_kernel_v6_standalone", "grid": [self.num_chunks * H_V, 1, 1], "workgroup": [1, 1, 1], "dynamic_lds": 0},
            {"ordinal": 1, "stage": "KKT", "symbol": "_qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0", "grid": [self.num_chunks * H_V * 16, 1, 1], "workgroup": [64, 1, 1], "dynamic_lds": 0},
            {"ordinal": 2, "stage": "solve", "symbol": None, "grid": [self.num_chunks * H_V, 1, 1], "workgroup": None, "dynamic_lds": 0},
            {"ordinal": 3, "stage": "W", "symbol": "_qwen_gdn_w_bf16_kernel_bt64_mfma_v2_s0", "grid": [self.num_chunks * H_V * 8, 1, 1], "workgroup": [256, 1, 1], "dynamic_lds": 0},
            {"ordinal": 4, "stage": "U", "symbol": "_qwen_gdn_u_bf16_kernel_bt64_mfma_v2_s0", "grid": [self.num_chunks * H_V * 8, 1, 1], "workgroup": [256, 1, 1], "dynamic_lds": 0},
            {"ordinal": 5, "stage": "asm recurrence", "symbol": asm["symbol"], "grid": list(asm["grid"]), "workgroup": [asm["workgroup"], 1, 1], "dynamic_lds": asm["dynamic_lds_bytes"]},
            {"ordinal": 6, "stage": "chunk-o", "symbol": "_qwen_gdn_chunk_o_bf16_kernel_bt64_mfma_v2_s0", "grid": [self.num_chunks * H_V * 8, 1, 1], "workgroup": [256, 1, 1], "dynamic_lds": 0},
            {"ordinal": 7, "stage": "BF16 cast", "symbol": "PyTorch elementwise copy/cast", "grid": None, "workgroup": None, "dynamic_lds": None},
        ]
        graphs: dict[str, object] = {}
        for solve_impl in SOLVES:
            dispatches = [dict(row) for row in common]
            dispatches[2]["symbol"] = (
                "_qwen_gdn_solve_kernel_v18_parallel"
                if solve_impl == SOLVE_A
                else "_qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1"
            )
            dispatches[2]["workgroup"] = [128 if solve_impl == SOLVE_A else 256, 1, 1]
            graphs[solve_impl] = {
                "T": self.t,
                "dispatch_count": len(dispatches),
                "stream": int(torch.cuda.current_stream(self.q.device).cuda_stream),
                "dispatches": dispatches,
                "inputs": [tensor_info(name, getattr(self, name)) for name in ("q", "k", "v", "g", "beta", "h0")],
                "buffers": self.buffers.info(),
            }
        return graphs


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


def ordered_labels(order: str, repeat: int, seed: int) -> list[str]:
    if order == "ABAB":
        return [item for _ in range(repeat) for item in SOLVES]
    if order == "BABA":
        return [item for _ in range(repeat) for item in reversed(SOLVES)]
    if order == "RANDOM_ABBA":
        rng = random.Random(seed)
        labels: list[str] = []
        while labels.count(SOLVE_A) < repeat:
            block = [SOLVE_A, SOLVE_B, SOLVE_B, SOLVE_A]
            if rng.getrandbits(1):
                block.reverse()
            labels.extend(block)
        kept: list[str] = []
        counts = {SOLVE_A: 0, SOLVE_B: 0}
        for label in labels:
            if counts[label] < repeat:
                kept.append(label)
                counts[label] += 1
        return kept
    raise ValueError(f"unknown order {order}")


def warm_pair(call_a: Callable[[], None], call_b: Callable[[], None], warmup: int) -> None:
    for _ in range(warmup):
        call_a()
        call_b()
    torch.cuda.synchronize()


def benchmark_cases(
    cases: dict[str, Callable[[], None]],
    *,
    t: int,
    experiment: str,
    sessions: int,
    warmup: int,
    repeat: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if set(cases) != set(SOLVES):
        raise ValueError("paired benchmark requires v18 and hierarchical_fp32_v1 cases")
    raw: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    orders = ("ABAB", "BABA", "RANDOM_ABBA", "ABAB", "BABA")
    timer = EventTimer()
    for session in range(sessions):
        order = orders[session % len(orders)]
        warm_pair(cases[SOLVE_A], cases[SOLVE_B], warmup)
        labels = ordered_labels(order, repeat, seed + session)
        per_variant: dict[str, list[float]] = {SOLVE_A: [], SOLVE_B: []}
        for sequence, label in enumerate(labels):
            latency = timer.measure(cases[label])
            per_variant[label].append(latency)
            raw.append(
                {
                    "experiment": experiment,
                    "T": t,
                    "session": session,
                    "order": order,
                    "sequence": sequence,
                    "solve_impl": label,
                    "latency_ms": latency,
                }
            )
        stats = {label: summarize(values) for label, values in per_variant.items()}
        for label in SOLVES:
            summary.append(
                {
                    "experiment": experiment,
                    "T": t,
                    "session": session,
                    "order": order,
                    "solve_impl": label,
                    **stats[label],
                }
            )
        summary.append(
            {
                "experiment": experiment,
                "T": t,
                "session": session,
                "order": order,
                "solve_impl": "delta_v18_minus_v1",
                "median_ms": stats[SOLVE_A]["median_ms"] - stats[SOLVE_B]["median_ms"],
                "p10_ms": math.nan,
                "p90_ms": math.nan,
                "minimum_ms": math.nan,
                "maximum_ms": math.nan,
            }
        )
    return summary, raw


def benchmark_tail_after_predecessors(
    runner: Stage5DRunner,
    predecessors: dict[str, Callable[[], torch.Tensor]],
    *,
    t: int,
    experiment: str,
    sessions: int,
    warmup: int,
    repeat: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Time one continuous W/U->asm->chunk-o->cast tail.

    The predecessor and all pointer/cache controls are enqueued before the
    start event.  There are no events inside the downstream tail.
    """
    raw: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    orders = ("ABAB", "BABA", "RANDOM_ABBA", "ABAB", "BABA")
    for session in range(sessions):
        order = orders[session % len(orders)]
        for _ in range(warmup):
            for label in SOLVES:
                solved = predecessors[label]()
                runner.launch_tail(solved)
        torch.cuda.synchronize()
        per_variant: dict[str, list[float]] = {SOLVE_A: [], SOLVE_B: []}
        labels = ordered_labels(order, repeat, seed + session)
        for sequence, label in enumerate(labels):
            solved = predecessors[label]()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            runner.launch_tail(solved)
            end.record()
            end.synchronize()
            latency = float(start.elapsed_time(end))
            per_variant[label].append(latency)
            raw.append(
                {
                    "experiment": experiment,
                    "T": t,
                    "session": session,
                    "order": order,
                    "sequence": sequence,
                    "solve_impl": label,
                    "latency_ms": latency,
                }
            )
        stats = {label: summarize(values) for label, values in per_variant.items()}
        for label in SOLVES:
            summary.append(
                {
                    "experiment": experiment,
                    "T": t,
                    "session": session,
                    "order": order,
                    "solve_impl": label,
                    **stats[label],
                }
            )
        summary.append(
            {
                "experiment": experiment,
                "T": t,
                "session": session,
                "order": order,
                "solve_impl": "delta_v18_minus_v1",
                "median_ms": stats[SOLVE_A]["median_ms"] - stats[SOLVE_B]["median_ms"],
                "p10_ms": math.nan,
                "p90_ms": math.nan,
                "minimum_ms": math.nan,
                "maximum_ms": math.nan,
            }
        )
    return summary, raw


def benchmark_single_tail(
    runner: Stage5DRunner,
    predecessor: Callable[[], torch.Tensor],
    *,
    label: str,
    t: int,
    sessions: int,
    warmup: int,
    repeat: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for session in range(sessions):
        for _ in range(warmup):
            runner.launch_tail(predecessor())
        torch.cuda.synchronize()
        values: list[float] = []
        for sample in range(repeat):
            solved = predecessor()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            runner.launch_tail(solved)
            end.record()
            end.synchronize()
            value = float(start.elapsed_time(end))
            values.append(value)
            raw.append(
                {
                    "experiment": "canonical",
                    "T": t,
                    "session": session,
                    "order": "single",
                    "sequence": sample,
                    "solve_impl": label,
                    "latency_ms": value,
                }
            )
        summaries.append(
            {
                "experiment": "canonical",
                "T": t,
                "session": session,
                "order": "single",
                "solve_impl": label,
                **summarize(values),
            }
        )
    return summaries, raw


def solved_statistics(a: torch.Tensor, b: torch.Tensor) -> dict[str, object]:
    torch.cuda.synchronize()
    diff = (a - b).abs()
    bits_a = a.view(torch.int32)
    bits_b = b.view(torch.int32)
    tiny = torch.finfo(torch.float32).tiny
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "bitwise_mismatch_count": int((bits_a != bits_b).sum().item()),
        "element_count": int(a.numel()),
        "a_nan_count": int(torch.isnan(a).sum().item()),
        "b_nan_count": int(torch.isnan(b).sum().item()),
        "a_inf_count": int(torch.isinf(a).sum().item()),
        "b_inf_count": int(torch.isinf(b).sum().item()),
        "a_subnormal_count": int(((a != 0) & (a.abs() < tiny)).sum().item()),
        "b_subnormal_count": int(((b != 0) & (b.abs() < tiny)).sum().item()),
        "a_min": float(a.min().item()),
        "a_max": float(a.max().item()),
        "a_mean": float(a.mean().item()),
        "b_min": float(b.min().item()),
        "b_max": float(b.max().item()),
        "b_mean": float(b.mean().item()),
    }


def run_contract(args: argparse.Namespace) -> None:
    runner = Stage5DRunner(args.T[0], args.seed)
    runner.launch_full(SOLVE_A)
    runner.launch_full(SOLVE_B)
    torch.cuda.synchronize()
    contract = runner.graph_contract()
    asm_hsaco = LADDER / "codex_qwen_asm_v0_integration/assembly/qwen_gdn_bt64_gfx942_asm_v0.hsaco"
    contract["asm_hsaco"] = {"path": str(asm_hsaco), "sha256": sha256_file(asm_hsaco)}
    (args.out_dir / "graph_a_dispatch_map.json").write_text(json.dumps(contract[SOLVE_A], indent=2) + "\n")
    (args.out_dir / "graph_b_dispatch_map.json").write_text(json.dumps(contract[SOLVE_B], indent=2) + "\n")
    (args.out_dir / "graph_contract_raw.json").write_text(json.dumps(contract, indent=2) + "\n")


def run_full_benchmark(args: argparse.Namespace) -> None:
    all_summary: list[dict[str, object]] = []
    all_raw: list[dict[str, object]] = []
    for t in args.T:
        runner = Stage5DRunner(t, args.seed + t)
        cases = {label: (lambda label=label: runner.launch_full(label)) for label in SOLVES}
        summary, raw = benchmark_cases(
            cases,
            t=t,
            experiment="preallocated_full",
            sessions=args.sessions,
            warmup=args.warmup,
            repeat=args.repeat,
            seed=args.seed + t,
        )
        all_summary.extend(summary)
        all_raw.extend(raw)
    write_csv(args.out_dir / "full_ab_benchmark.csv", all_summary)
    write_csv(args.out_dir / "full_ab_raw_samples.csv", all_raw)


def run_tail_benchmark(args: argparse.Namespace) -> None:
    all_summary: list[dict[str, object]] = []
    all_raw: list[dict[str, object]] = []
    for t in args.T:
        runner = Stage5DRunner(t, args.seed + 1000 + t)
        runner.prepare_local_matrix()

        def case(label: str) -> None:
            solved = runner.launch_solve(label)
            runner.launch_tail(solved)

        cases = {label: (lambda label=label: case(label)) for label in SOLVES}
        summary, raw = benchmark_cases(
            cases,
            t=t,
            experiment="solve_plus_tail_boundary",
            sessions=args.sessions,
            warmup=args.warmup,
            repeat=args.repeat,
            seed=args.seed + t,
        )
        # The event wraps solve+tail here; a second run below measures tail only.
        all_summary.extend(summary)
        all_raw.extend(raw)

        predecessors = {label: (lambda label=label: runner.launch_solve(label)) for label in SOLVES}
        tail_summary, tail_raw = benchmark_tail_after_predecessors(
            runner,
            predecessors,
            t=t,
            experiment="downstream_tail_only",
            sessions=args.sessions,
            warmup=args.warmup,
            repeat=args.repeat,
            seed=args.seed + t,
        )
        all_summary.extend(tail_summary)
        all_raw.extend(tail_raw)
    write_csv(args.out_dir / "downstream_tail_benchmark.csv", all_summary)
    write_csv(args.out_dir / "downstream_tail_raw.csv", all_raw)


def _control_cases(runner: Stage5DRunner, kind: str, flush_mib: int) -> dict[str, Callable[[], torch.Tensor]]:
    canonical = runner.buffers.canonical

    def canonical_after(label: str) -> torch.Tensor:
        runner.launch_solve(label)
        return canonical

    def same_pointer(label: str) -> torch.Tensor:
        solved = runner.launch_solve(label)
        canonical.copy_(solved)
        return canonical

    def cache_control(label: str, mode: str) -> torch.Tensor:
        solved = runner.launch_solve(label)
        canonical.copy_(solved)
        if mode == "cold":
            runner.perturb_cache(flush_mib)
        elif mode == "prime":
            runner.prime_downstream_inputs(canonical)
        elif mode == "warm":
            runner.launch_tail(canonical)
        elif mode != "none":
            raise ValueError(mode)
        return canonical

    if kind == "canonical":
        return {label: (lambda label=label: canonical_after(label)) for label in SOLVES}
    if kind == "same_pointer":
        return {label: (lambda label=label: same_pointer(label)) for label in SOLVES}
    if kind.startswith("cache_"):
        mode = kind.removeprefix("cache_")
        return {label: (lambda label=label, mode=mode: cache_control(label, mode)) for label in SOLVES}
    raise ValueError(kind)


def run_controls(args: argparse.Namespace) -> None:
    canonical_rows: list[dict[str, object]] = []
    same_pointer_rows: list[dict[str, object]] = []
    cache_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    pointer_rows: list[dict[str, object]] = []
    stats_rows: list[dict[str, object]] = []
    copy_rows: list[dict[str, object]] = []
    for t in args.T:
        runner = Stage5DRunner(t, args.seed + 2000 + t)
        runner.prepare_canonical()
        runner.launch_solve(SOLVE_B)
        torch.cuda.synchronize()
        stats_rows.append({"T": t, **solved_statistics(runner.buffers.solved_a, runner.buffers.solved_b)})
        for name in ("solved_a", "solved_b", "canonical"):
            pointer_rows.append({"T": t, **tensor_info(name, getattr(runner.buffers, name))})

        for kind in ("canonical", "same_pointer", "cache_none", "cache_warm", "cache_cold", "cache_prime"):
            cases = _control_cases(runner, kind, args.flush_mib)
            summary, raw = benchmark_tail_after_predecessors(
                runner,
                cases,
                t=t,
                experiment=kind,
                sessions=args.sessions,
                warmup=args.warmup,
                repeat=args.repeat,
                seed=args.seed + t,
            )
            if kind == "canonical":
                canonical_rows.extend(summary)
            elif kind == "same_pointer":
                same_pointer_rows.extend(summary)
            else:
                cache_rows.extend(summary)
            raw_rows.extend(raw)

        no_solve_summary, no_solve_raw = benchmark_single_tail(
            runner,
            lambda: runner.buffers.canonical,
            label="no_solve",
            t=t,
            sessions=args.sessions,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        dummy_summary, dummy_raw = benchmark_single_tail(
            runner,
            lambda: (runner.dummy_predecessor() or runner.buffers.canonical),
            label="dummy_predecessor",
            t=t,
            sessions=args.sessions,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        canonical_rows.extend(no_solve_summary)
        canonical_rows.extend(dummy_summary)
        raw_rows.extend(no_solve_raw)
        raw_rows.extend(dummy_raw)

        timer = EventTimer()
        for label in SOLVES:
            solved = runner.launch_solve(label)
            torch.cuda.synchronize()
            copy_values = [timer.measure(lambda solved=solved: runner.buffers.canonical.copy_(solved)) for _ in range(args.repeat)]
            copy_rows.append({"T": t, "solve_impl": label, **summarize(copy_values)})

    write_csv(args.out_dir / "canonical_data_control.csv", canonical_rows)
    write_csv(args.out_dir / "same_pointer_control.csv", same_pointer_rows)
    write_csv(args.out_dir / "cache_state_control.csv", cache_rows)
    write_csv(args.out_dir / "control_raw_samples.csv", raw_rows)
    write_csv(args.out_dir / "pointer_alignment.csv", pointer_rows)
    write_csv(args.out_dir / "solved_data_statistics.csv", stats_rows)
    write_csv(args.out_dir / "same_pointer_copy_cost.csv", copy_rows)
    memory = {
        "memory_stats": torch.cuda.memory_stats(),
        "memory_summary": torch.cuda.memory_summary(),
    }
    (args.out_dir / "allocator_state.json").write_text(json.dumps(memory["memory_stats"], indent=2) + "\n")
    (args.out_dir / "allocator_state.txt").write_text(memory["memory_summary"])


def profile_case(runner: Stage5DRunner, case: str) -> Callable[[], None]:
    if case == "graph_a":
        return lambda: runner.launch_full(SOLVE_A)
    if case == "graph_b":
        return lambda: runner.launch_full(SOLVE_B)
    if case in ("canonical_a", "canonical_b"):
        label = SOLVE_A if case.endswith("_a") else SOLVE_B

        def run_canonical() -> None:
            runner.launch_solve(label)
            runner.launch_tail(runner.buffers.canonical)

        return run_canonical
    if case in ("same_pointer_a", "same_pointer_b"):
        label = SOLVE_A if case.endswith("_a") else SOLVE_B

        def run_same_pointer() -> None:
            solved = runner.launch_solve(label)
            runner.buffers.canonical.copy_(solved)
            runner.launch_tail(runner.buffers.canonical)

        return run_same_pointer
    for mode in ("cold", "prime"):
        if case in (f"{mode}_a", f"{mode}_b"):
            label = SOLVE_A if case.endswith("_a") else SOLVE_B
            predecessor = _control_cases(runner, f"cache_{mode}", 512)[label]

            def run_cache_case() -> None:
                runner.launch_tail(predecessor())

            return run_cache_case
    raise ValueError(case)


def run_profile(args: argparse.Namespace) -> None:
    if len(args.T) != 1:
        raise ValueError("profile mode takes exactly one T")
    runner = Stage5DRunner(args.T[0], args.seed + 3000 + args.T[0])
    runner.prepare_canonical()
    fn = profile_case(runner, args.profile_case)
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    for _ in range(args.repeat):
        fn()
    torch.cuda.synchronize()
    print(json.dumps({"case": args.profile_case, "T": args.T[0], "checksum": float(runner.buffers.output_fp32[0, 0, 0, 0].item())}))


def run_smoke(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    for t in args.T:
        runner = Stage5DRunner(t, args.seed + 4000 + t)
        runner.launch_full(SOLVE_A)
        out_a, state_a = runner.output_snapshot()
        runner.launch_full(SOLVE_B)
        out_b, state_b = runner.output_snapshot()
        runner.prepare_canonical()
        runner.launch_tail(runner.buffers.canonical)
        out_c, state_c = runner.output_snapshot()
        rows.append(
            {
                "T": t,
                "full_output_max_abs": float((out_a.float() - out_b.float()).abs().max().item()),
                "full_state_max_abs": float((state_a - state_b).abs().max().item()),
                "canonical_output_max_abs_vs_a": float((out_a.float() - out_c.float()).abs().max().item()),
                "canonical_state_max_abs_vs_a": float((state_a - state_c).abs().max().item()),
            }
        )
    write_csv(args.out_dir / "tests/smoke_correctness.csv", rows)
    for row in rows:
        if row["full_output_max_abs"] > OUTPUT_ATOL or row["full_state_max_abs"] > STATE_ATOL:
            raise AssertionError(row)
        if row["canonical_output_max_abs_vs_a"] != 0.0 or row["canonical_state_max_abs_vs_a"] != 0.0:
            raise AssertionError(row)
    print(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contract", "full", "tail", "controls", "profile", "smoke"), required=True)
    parser.add_argument("--T", type=int, nargs="+", default=[2048])
    parser.add_argument("--seed", type=int, default=20260750)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--flush-mib", type=int, default=512)
    parser.add_argument("--profile-case", default="graph_a")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 5D requires a HIP GPU")
    patch_rocm_autotune()
    dispatch = {
        "contract": run_contract,
        "full": run_full_benchmark,
        "tail": run_tail_benchmark,
        "controls": run_controls,
        "profile": run_profile,
        "smoke": run_smoke,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
