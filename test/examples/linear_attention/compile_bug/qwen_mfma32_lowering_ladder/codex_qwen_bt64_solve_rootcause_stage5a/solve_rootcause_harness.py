#!/usr/bin/env python3
"""Stage 5A BT64 solve audit harness.

This is deliberately isolated from the production graph.  It dispatches the
existing v18 JIT kernel and the installed vLLM Triton kernel directly with
preallocated output buffers so allocation and compilation can be reported
separately from device work.  It does not implement, replace, or register a
new solve.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch


BT = 64
HEADS = 8
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
COMPARE = REPO / "test/examples/linear_attention/vllm_compare"
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(COMPARE), str(STAGE2)]

from qwen_gdn_bt64_nonrecurrence_mfma_v2 import qwen_gdn_kkt_bt64_mfma_v2_s0  # noqa: E402
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import (  # noqa: E402
    _qwen_gdn_solve_kernel_v18_parallel,
    qwen_gdn_solve_avelang_v18_bt64_layout,
)
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (  # noqa: E402
    _num_chunks,
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
)
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402


SOLVE_MODULE = importlib.import_module("vllm.model_executor.layers.fla.ops.solve_tril")
VLLM_SOLVE = SOLVE_MODULE.solve_tril
VLLM_MERGE64 = SOLVE_MODULE.merge_16x16_to_64x64_inverse_kernel
VLLM_TMA = SOLVE_MODULE.is_tma_supported
VLLM_PRECISION = SOLVE_MODULE.FLA_TRIL_PRECISION


@dataclass
class Timing:
    implementation: str
    measurement: str
    T: int
    chunks: int
    session: int
    warmup: int
    repeat: int
    median_ms: float
    p10_ms: float
    p90_ms: float
    mean_ms: float
    minimum_ms: float
    maximum_ms: float


def synchronize() -> None:
    torch.cuda.synchronize()


def stage4_a(num_tokens: int, seed: int) -> torch.Tensor:
    """Create the exact FP32 BT64 KKT layout consumed by both solve paths."""
    _, k, _, g, beta, _ = make_inputs(num_tokens, seed, "random", False)
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    a = qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
    assert a.shape == (1, num_tokens, HEADS, BT)
    assert a.dtype == torch.float32 and a.is_contiguous()
    return a


def avelang_direct(a: torch.Tensor, out: torch.Tensor) -> None:
    """The existing v18 kernel with only its wrapper allocation bypassed."""
    num_tokens = a.shape[1]
    num_chunks = _num_chunks(num_tokens, BT)
    _qwen_gdn_solve_kernel_v18_parallel[lambda: ((num_chunks * HEADS, 1, 1), (128, 1, 1))](
        a, out, num_tokens, BT, num_chunks
    )


def vllm_direct(a: torch.Tensor, out: torch.Tensor) -> None:
    """The installed vLLM BT64 Triton kernel with wrapper allocation bypassed."""
    # solve_tril uses zeros_like because the upper triangular output is unwritten.
    # The caller prepares a zeroed output before each timed dispatch.
    if not hasattr(VLLM_MERGE64.fn, "best_config"):
        VLLM_SOLVE(A=a, output_dtype=torch.float32)
        synchronize()
    vllm_direct_kernel(a, out)


def vllm_kernel_at_config(a: torch.Tensor, out: torch.Tensor, num_warps: int, num_stages: int) -> None:
    """Dispatch the installed Triton source at a recorded solve_tril-selected config."""
    batch, num_tokens, heads, _ = a.shape
    VLLM_MERGE64.fn.fn[_num_chunks(num_tokens, BT), batch * heads](
        A=a,
        Ai=out,
        cu_seqlens=None,
        chunk_indices=None,
        T=num_tokens,
        H=heads,
        BT=BT,
        USE_TMA=VLLM_TMA,
        IS_VARLEN=False,
        DOT_PRECISION=VLLM_PRECISION,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def vllm_direct_kernel(a: torch.Tensor, out: torch.Tensor) -> None:
    """Selected vLLM Triton BT64 body; ``out`` must already have a zero upper half."""
    config = VLLM_MERGE64.fn.best_config
    vllm_kernel_at_config(a, out, config.num_warps, config.num_stages)


def samples(fn: Callable[[], None], warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        fn()
    synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values: list[float] = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def samples_with_prepare(
    prepare: Callable[[], None], fn: Callable[[], None], warmup: int, repeat: int
) -> list[float]:
    """Run preparation before every event, excluding it from the device interval."""
    for _ in range(warmup):
        prepare()
        fn()
    synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values: list[float] = []
    for _ in range(repeat):
        prepare()
        synchronize()
        start.record()
        fn()
        end.record()
        synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered),
        "p10_ms": ordered[(len(ordered) - 1) // 10],
        "p90_ms": ordered[(len(ordered) - 1) * 9 // 10],
        "mean_ms": statistics.fmean(ordered),
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    items = list(rows)
    if not items:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)


def direct_outputs(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    avelang_out = torch.empty_like(a)
    vllm_out = torch.empty_like(a)
    avelang_direct(a, avelang_out)
    vllm_out.zero_()
    vllm_direct(a, vllm_out)
    authority = torch_solve_authority(a)
    synchronize()
    return avelang_out, vllm_out, authority


def torch_solve_authority(a: torch.Tensor) -> torch.Tensor:
    """FP32 authority for X=(I+A)^-1, evaluated chunk by chunk."""
    _, num_tokens, heads, width = a.shape
    assert width == BT and num_tokens % BT == 0
    blocks = a.view(1, num_tokens // BT, BT, heads, BT).permute(0, 1, 3, 2, 4).contiguous()
    eye = torch.eye(BT, device=a.device, dtype=torch.float32).view(1, 1, 1, BT, BT)
    result = torch.linalg.solve_triangular(eye + blocks, eye.expand_as(blocks), upper=False)
    return result.permute(0, 1, 3, 2, 4).contiguous().view_as(a)


def residual(a: torch.Tensor, x: torch.Tensor) -> float:
    _, num_tokens, heads, _ = a.shape
    a_blocks = a.view(1, num_tokens // BT, BT, heads, BT).permute(0, 1, 3, 2, 4).contiguous()
    x_blocks = x.view(1, num_tokens // BT, BT, heads, BT).permute(0, 1, 3, 2, 4).contiguous()
    eye = torch.eye(BT, device=a.device, dtype=torch.float32).view(1, 1, 1, BT, BT)
    return float(((eye + a_blocks) @ x_blocks - eye).abs().max().item())


def synthetic_a(kind: str, seed: int, num_tokens: int = BT) -> torch.Tensor:
    """Stable strictly-lower FP32 matrices for contract and residual coverage."""
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    a = torch.zeros((1, num_tokens, HEADS, BT), device="cuda", dtype=torch.float32)
    coords = torch.tril_indices(BT, BT, offset=-1, device="cuda")
    scale = {
        "random": 0.025,
        "small": 1.0e-5,
        "high_dynamic": 0.12,
        "near_singular": 0.35,
        "sparse": 0.02,
        "single_subdiagonal": 0.04,
        "cancellation": 0.03,
        "diagonal_boundary": 0.0,
    }[kind]
    for chunk in range(num_tokens // BT):
        base = torch.randn((HEADS, coords.shape[1]), device="cuda", dtype=torch.float32, generator=generator) * scale
        if kind == "high_dynamic":
            base[:, ::7] *= 8.0
            base[:, 1::11] *= 0.015625
        elif kind == "near_singular":
            base[:, :] = 0.0
            lower = coords[0] - coords[1]
            base[:, lower == 1] = -0.82
            base[:, lower == 2] = 0.19
        elif kind == "sparse":
            base[:, (coords[0] + 3 * coords[1]) % 7 != 0] = 0.0
        elif kind == "single_subdiagonal":
            base[:, coords[0] - coords[1] != 1] = 0.0
        elif kind == "cancellation":
            signs = torch.where((coords[0] + coords[1]) % 2 == 0, 1.0, -1.0)
            base[:] = signs * scale
        a[0, chunk * BT : (chunk + 1) * BT, :, coords[1]] = 0.0
        a[0, chunk * BT + coords[0], :, coords[1]] = base.transpose(0, 1)
    return a.contiguous()


def tensor_digest(tensor: torch.Tensor) -> dict[str, object]:
    host = tensor.detach().cpu().contiguous().view(torch.uint8)
    import hashlib

    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "contiguous": bool(tensor.is_contiguous()),
        "sha256": hashlib.sha256(host.numpy().tobytes()).hexdigest(),
    }


def compare_case(case_id: int, kind: str, a: torch.Tensor) -> dict[str, object]:
    avelang_out, vllm_out, authority = direct_outputs(a)
    avelang_diff = (avelang_out - authority).abs()
    vllm_diff = (vllm_out - authority).abs()
    cross_diff = (avelang_out - vllm_out).abs()
    max_index = tuple(int(v) for v in torch.nonzero(cross_diff == cross_diff.max(), as_tuple=False)[0].cpu().tolist())
    return {
        "case_id": case_id,
        "kind": kind,
        "T": a.shape[1],
        "input_sha256": tensor_digest(a)["sha256"],
        "avelang_max_abs": float(avelang_diff.max().item()),
        "avelang_mean_abs": float(avelang_diff.mean().item()),
        "avelang_max_rel": float((avelang_diff / authority.abs().clamp_min(1e-7)).max().item()),
        "vllm_max_abs": float(vllm_diff.max().item()),
        "vllm_mean_abs": float(vllm_diff.mean().item()),
        "vllm_max_rel": float((vllm_diff / authority.abs().clamp_min(1e-7)).max().item()),
        "cross_max_abs": float(cross_diff.max().item()),
        "first_max_cross_index": str(max_index),
        "avelang_residual_inf": residual(a, avelang_out),
        "vllm_residual_inf": residual(a, vllm_out),
        "authority_residual_inf": residual(a, authority),
        "passed": bool(torch.allclose(avelang_out, authority, atol=1e-5, rtol=1e-5)
                       and torch.allclose(vllm_out, authority, atol=1e-5, rtol=1e-5)),
    }


def run_correctness(out_dir: Path) -> None:
    patch_rocm_autotune()
    cases: list[dict[str, object]] = []
    kinds = ["random", "small", "high_dynamic", "near_singular", "diagonal_boundary", "sparse", "single_subdiagonal", "cancellation"]
    case_id = 0
    # 48 controlled synthetic cases plus six actual Stage-4 KKT cases.
    for repeat in range(6):
        for kind in kinds:
            cases.append(compare_case(case_id, kind, synthetic_a(kind, 5000 + case_id)))
            case_id += 1
    for tokens in [64, 128, 512, 1024, 2048, 4096]:
        cases.append(compare_case(case_id, "stage4_kkt", stage4_a(tokens, 9000 + tokens)))
        case_id += 1
    write_csv(out_dir / "correctness.csv", cases)
    residual_rows = [{key: row[key] for key in row if key.endswith("residual_inf") or key in {"case_id", "kind", "T", "passed"}} for row in cases]
    write_csv(out_dir / "residual_analysis.csv", residual_rows)
    contract = tensor_digest(stage4_a(64, 9064))
    contract.update({
        "same_input_object_for_both": True,
        "avelang_output_dtype": "torch.float32",
        "vllm_output_dtype": "torch.float32",
        "contract": "X=(I+A)^-1 for a chunk-local strictly lower A",
        "all_cases_passed": all(bool(row["passed"]) for row in cases),
        "case_count": len(cases),
    })
    (out_dir / "input_contract_validation.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "all_passed": contract["all_cases_passed"]}, indent=2))


def append_raw_samples(path: Path, implementation: str, measurement: str, num_tokens: int, session: int, values: list[float]) -> None:
    rows = [{"implementation": implementation, "measurement": measurement, "T": num_tokens, "chunks": num_tokens // BT,
             "session": session, "sample": index, "elapsed_ms": value} for index, value in enumerate(values)]
    write_csv(path, rows)


def run_benchmark(out_dir: Path, session: int, warmup: int, repeat: int) -> None:
    patch_rocm_autotune()
    rows: list[dict[str, object]] = []
    raw_dir = out_dir / "raw_sessions"
    for num_tokens in [64, 128, 512, 1024, 2048, 4096, 8192, 16384]:
        a = stage4_a(num_tokens, 10000 + session * 100 + num_tokens)
        avelang_out = torch.empty_like(a)
        vllm_out = torch.empty_like(a)

        # Warm the exact wrapper/configuration first; compile time is excluded.
        qwen_gdn_solve_avelang_v18_bt64_layout(a)
        VLLM_SOLVE(A=a, output_dtype=torch.float32)
        synchronize()
        benchmarks: list[tuple[str, str, Callable[[], None] | None, Callable[[], None]]] = [
            ("avelang_v18", "body_preallocated_kernel", None, lambda: avelang_direct(a, avelang_out)),
            ("vllm", "body_preallocated_triton_kernel", lambda: vllm_out.zero_(), lambda: vllm_direct(a, vllm_out)),
            ("avelang_v18", "public_wrapper_including_allocation", None, lambda: qwen_gdn_solve_avelang_v18_bt64_layout(a)),
            ("vllm", "public_wrapper_including_zeros_like", None, lambda: VLLM_SOLVE(A=a, output_dtype=torch.float32)),
        ]
        for implementation, measurement, prepare, fn in benchmarks:
            values = samples(fn, warmup, repeat) if prepare is None else samples_with_prepare(prepare, fn, warmup, repeat)
            append_raw_samples(raw_dir / f"session_{session}_{implementation}_{measurement}_T{num_tokens}.csv", implementation, measurement, num_tokens, session, values)
            rows.append(asdict(Timing(implementation=implementation, measurement=measurement, T=num_tokens, chunks=num_tokens // BT,
                                     session=session, warmup=warmup, repeat=repeat, **summarize(values))))
            print(rows[-1], flush=True)
    write_csv(out_dir / f"standalone_benchmark_session_{session}.csv", rows)


def fit_line(points: list[tuple[int, float]]) -> dict[str, float]:
    n = len(points)
    mean_x = statistics.fmean(point[0] for point in points)
    mean_y = statistics.fmean(point[1] for point in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    intercept = mean_y - slope * mean_x
    fitted = [intercept + slope * x for x, _ in points]
    ss_total = sum((y - mean_y) ** 2 for _, y in points)
    ss_residual = sum((y - fit) ** 2 for (_, y), fit in zip(points, fitted))
    return {"intercept_us": intercept * 1000.0, "slope_us_per_chunk": slope * 1000.0,
            "r_squared": 1.0 - ss_residual / ss_total if ss_total else 1.0}


def run_summarize(out_dir: Path) -> None:
    rows: list[dict[str, str]] = []
    for path in sorted(out_dir.glob("standalone_benchmark_session_*.csv")):
        with path.open() as stream:
            rows.extend(csv.DictReader(stream))
    groups: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        groups.setdefault((row["implementation"], row["measurement"], int(row["T"])), []).append(float(row["median_ms"]))
    summary_rows: list[dict[str, object]] = []
    for (implementation, measurement, num_tokens), medians in sorted(groups.items()):
        summary_rows.append({"implementation": implementation, "measurement": measurement, "T": num_tokens,
                             "chunks": num_tokens // BT, "sessions": len(medians), "session_median_ms": statistics.median(medians),
                             "session_p10_ms": min(medians), "session_p90_ms": max(medians)})
    write_csv(out_dir / "standalone_benchmark.csv", summary_rows)
    body_rows = [row for row in summary_rows if row["measurement"].startswith("body_preallocated_")]
    write_csv(out_dir / "chunk_count_sweep.csv", body_rows)
    fits = {}
    for implementation in ["avelang_v18", "vllm"]:
        points = [(int(row["chunks"]), float(row["session_median_ms"])) for row in body_rows if row["implementation"] == implementation]
        fits[implementation] = fit_line(points)
    (out_dir / "latency_fit.json").write_text(json.dumps(fits, indent=2) + "\n")
    print(json.dumps(fits, indent=2))


def run_ablations(out_dir: Path, warmup: int, repeat: int) -> None:
    """Only audit diagnostics; none is imported by a solve or full pipeline."""
    from solve_rootcause_diagnostics import (
        avelang_launch_floor,
        avelang_load_store_only,
        avelang_no_store_checksum,
    )

    patch_rocm_autotune()
    rows: list[dict[str, object]] = []
    for num_tokens, input_kind in [(64, "random"), (2048, "random"), (2048, "diagonal_boundary")]:
        a = stage4_a(num_tokens, 12000 + num_tokens) if input_kind == "random" else synthetic_a(input_kind, 13000 + num_tokens, num_tokens)
        out = torch.empty_like(a)
        checksum = torch.empty((num_tokens // BT * HEADS,), device="cuda", dtype=torch.float32)
        variants: list[tuple[str, bool, Callable[[], None]]] = [
            ("actual_v18_direct", True, lambda: avelang_direct(a, out)),
            ("launch_floor", False, lambda: avelang_launch_floor(out, num_tokens)),
            ("load_store_only", False, lambda: avelang_load_store_only(a, out, num_tokens)),
            ("no_store_checksum", False, lambda: avelang_no_store_checksum(a, checksum, num_tokens)),
        ]
        for variant, mathematically_correct, fn in variants:
            values = samples(fn, warmup, repeat)
            row = {"variant": variant, "T": num_tokens, "chunks": num_tokens // BT, "input_kind": input_kind,
                   "mathematically_correct": mathematically_correct, "DCE_guard": "full output" if variant != "no_store_checksum" else "checksum[0]",
                   **summarize(values)}
            rows.append(row)
            print(row, flush=True)
    write_csv(out_dir / "ablations" / "ablation_results.csv", rows)


def run_profile(implementation: str, num_tokens: int, warmup: int, repeat: int) -> None:
    """Dispatch a single already-compiled solve family for rocprof collection."""
    patch_rocm_autotune()
    a = stage4_a(num_tokens, 14000 + num_tokens)
    out = torch.empty_like(a)
    if implementation == "avelang_v18":
        fn = lambda: avelang_direct(a, out)
    elif implementation == "vllm":
        config_path = HERE / "vllm_selected_config.json"
        if not config_path.exists():
            raise RuntimeError("run --mode capture-vllm-config before a vLLM rocprof profile")
        config = json.loads(config_path.read_text())
        out.zero_()
        synchronize()
        fn = lambda: vllm_kernel_at_config(a, out, int(config["num_warps"]), int(config["num_stages"]))
    else:
        raise ValueError(implementation)
    for _ in range(warmup):
        fn()
    synchronize()
    for _ in range(repeat):
        fn()
    synchronize()
    print(json.dumps({"profiled": implementation, "T": num_tokens, "repeat": repeat,
                      "vllm_tma": bool(VLLM_TMA), "vllm_precision": str(VLLM_PRECISION)}))


def capture_vllm_config(out_dir: Path) -> None:
    """Ask the installed solve_tril wrapper for its real autotuned choice once."""
    patch_rocm_autotune()
    a = stage4_a(2048, 14999)
    VLLM_SOLVE(A=a, output_dtype=torch.float32)
    synchronize()
    config = VLLM_MERGE64.fn.best_config
    payload = {
        "kernel": "merge_16x16_to_64x64_inverse_kernel",
        "key": "H=8,BT=64,IS_VARLEN=False,input=output=torch.float32",
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
        "num_ctas": config.num_ctas,
        "precision": str(VLLM_PRECISION),
        "use_tma": bool(VLLM_TMA),
    }
    (out_dir / "vllm_selected_config.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["correctness", "benchmark", "summarize", "ablation", "profile", "capture-vllm-config"])
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--implementation", choices=["avelang_v18", "vllm"])
    parser.add_argument("--T", type=int, default=2048)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 5A requires a CUDA/HIP GPU.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "correctness":
        run_correctness(args.out_dir)
    elif args.mode == "benchmark":
        run_benchmark(args.out_dir, args.session, args.warmup, args.repeat)
    elif args.mode == "summarize":
        run_summarize(args.out_dir)
    elif args.mode == "ablation":
        run_ablations(args.out_dir, args.warmup, args.repeat)
    elif args.mode == "profile":
        if args.implementation is None:
            parser.error("--implementation is required for --mode profile")
        run_profile(args.implementation, args.T, args.warmup, args.repeat)
    else:
        capture_vllm_config(args.out_dir)


if __name__ == "__main__":
    main()
