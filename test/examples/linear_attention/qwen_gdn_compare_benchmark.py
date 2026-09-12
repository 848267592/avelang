#!/usr/bin/env python3
"""Compare Qwen GDN forward baselines on the same ROCm GPU.

This script intentionally benchmarks baselines only. It does not optimize or
modify any Avelang kernels.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from qwen_gdn_chunked_avelang_v6_standalone import qwen_gdn_chunked_avelang_v6_standalone
from qwen_gdn_ref import qwen_gdn_forward_ref


OUTPUT_NAMES = ("g_cumsum", "output", "A_solved", "chunk_states", "final_state")
REPORT_PATH = Path(__file__).with_name("qwen_gdn_compare_results.md")
QWEN_OFFICIAL_CANDIDATES = (
    Path("/workspace/workspace/qwen_GDN"),
    Path("/workspace/workspace/qwen_GDN/FlashQLA"),
    Path("/home/jiandongliu/workspace/qwen_GDN"),
    Path("/home/jiandongliu/workspace/qwen_GDN/FlashQLA"),
)
FP32_ATOL = 1e-4
FP32_RTOL = 1e-4
BF16_ATOL = 5e-3
BF16_RTOL = 5e-3


@dataclass(frozen=True)
class BenchShape:
    name: str
    batch_size: int
    num_tokens: int
    num_k_heads: int
    num_v_heads: int
    head_dim_k: int
    head_dim_v: int
    chunk_size: int


@dataclass(frozen=True)
class DTypePolicy:
    name: str
    qkv_dtype: torch.dtype
    reference_uses_fp32_cast: bool


@dataclass
class BenchResult:
    implementation: str
    gpu: str
    dtype_policy: str
    shape: BenchShape
    initial_state: str
    latency_ms: float | None
    speedup_vs_pytorch_eager: float | None
    speedup_vs_torch_compile: float | None
    correctness_status: str
    notes: str


@dataclass(frozen=True)
class OfficialStatus:
    implementation: str
    exists: bool
    runnable: bool
    path: str
    revision: str
    reason: str


SHAPES = (
    BenchShape("small", 1, 16, 1, 2, 4, 4, 4),
    BenchShape("medium", 1, 64, 2, 4, 8, 8, 8),
    BenchShape("large_v", 1, 128, 2, 4, 16, 64, 8),
    BenchShape("larger_tv", 1, 256, 4, 8, 32, 64, 16),
)

D_TYPE_POLICIES = (
    DTypePolicy("fp32", torch.float32, False),
    DTypePolicy("bf16_qkv_fp32_ref", torch.bfloat16, True),
)

ROCPROFV3_V6_BOTTLENECK_ROWS = (
    ("larger_debug", "bf16", "_qwen_gdn_chunk_gdr_bf16_kernel_v6_standalone", 111.438, 62.54),
    ("larger_debug", "bf16", "_qwen_gdn_chunk_o_bf16_kernel_v6_standalone", 34.087, 19.13),
    ("larger_debug", "bf16", "_qwen_gdn_w_u_bf16_kernel_v6_standalone", 13.507, 7.58),
    ("larger_debug", "fp32", "_qwen_gdn_chunk_gdr_fp32_kernel_v6_standalone", 94.837, 60.21),
    ("larger_debug", "fp32", "_qwen_gdn_chunk_o_fp32_opt_kernel_v6_standalone", 33.240, 21.10),
    ("larger_debug", "fp32", "_qwen_gdn_w_u_fp32_opt_kernel_v6_standalone", 10.909, 6.93),
)


def _ensure_rocm_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda is not available.")
    if getattr(torch.version, "hip", None) is None:
        raise RuntimeError("This benchmark is intended for ROCm/HIP. torch.version.hip is None.")


def _git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=path,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _package_version(path: Path) -> str:
    init_py = path / "flash_qla" / "__init__.py"
    if not init_py.exists():
        return "unknown"
    try:
        tree = ast.parse(init_py.read_text())
    except Exception:
        return "unknown"
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    return "unknown"


def _source_version(path: Path) -> str:
    revision = _git_revision(path)
    if revision != "unknown":
        return revision
    package_version = _package_version(path)
    if package_version != "unknown":
        return f"{package_version} (package version; git revision unavailable)"
    return "unknown"


def _exception_text(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return (x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)).to(x.dtype)


def _make_inputs(
    shape: BenchShape,
    dtype: torch.dtype,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q_fp32 = _l2norm(
        torch.randn(
            shape.batch_size,
            shape.num_tokens,
            shape.num_k_heads,
            shape.head_dim_k,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
    ).contiguous()
    k_fp32 = _l2norm(
        torch.randn(
            shape.batch_size,
            shape.num_tokens,
            shape.num_k_heads,
            shape.head_dim_k,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
    ).contiguous()
    v_fp32 = torch.randn(
        shape.batch_size,
        shape.num_tokens,
        shape.num_v_heads,
        shape.head_dim_v,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).contiguous()
    g = torch.nn.functional.logsigmoid(
        torch.randn(
            shape.batch_size,
            shape.num_tokens,
            shape.num_v_heads,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
    )
    g = (g / 16).contiguous()
    beta = torch.sigmoid(
        torch.randn(
            shape.batch_size,
            shape.num_tokens,
            shape.num_v_heads,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
    ).contiguous()
    if dtype == torch.bfloat16:
        return (
            q_fp32.to(torch.bfloat16).contiguous(),
            k_fp32.to(torch.bfloat16).contiguous(),
            v_fp32.to(torch.bfloat16).contiguous(),
            g,
            beta,
        )
    return q_fp32, k_fp32, v_fp32, g, beta


def _reference_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    policy: DTypePolicy,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if policy.reference_uses_fp32_cast:
        return q.float().contiguous(), k.float().contiguous(), v.float().contiguous()
    return q, k, v


def _forward_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    shape: BenchShape,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return qwen_gdn_forward_ref(q, k, v, g, beta, scale=scale, chunk_size=shape.chunk_size)


def _time_ms(fn: Callable[[], object], *, warmup: int, repeat: int) -> float:
    result = None
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(repeat):
        result = fn()
    end_evt.record()
    torch.cuda.synchronize()
    del result
    return start_evt.elapsed_time(end_evt) / repeat


def _compare_outputs(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    *,
    atol: float,
    rtol: float,
) -> str:
    max_abs = 0.0
    max_rel = 0.0
    worst_name = "none"
    for name, actual_tensor, expected_tensor in zip(OUTPUT_NAMES, actual, expected, strict=True):
        diff = (actual_tensor - expected_tensor).abs()
        tensor_max_abs = float(diff.max().item())
        tensor_max_rel = float((diff / expected_tensor.abs().clamp_min(1e-12)).max().item())
        if tensor_max_abs > max_abs:
            max_abs = tensor_max_abs
            worst_name = name
        max_rel = max(max_rel, tensor_max_rel)
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=atol, rtol=rtol)
    return f"pass max_abs={max_abs:.3e} max_rel={max_rel:.3e} worst={worst_name}"


def _detect_qwen_official(qwen_root: str | None) -> OfficialStatus:
    candidates: list[Path] = []
    if qwen_root:
        candidates.append(Path(qwen_root))
    candidates.extend(QWEN_OFFICIAL_CANDIDATES)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        flash_root = candidate if (candidate / "flash_qla").exists() else candidate / "FlashQLA"
        chunk_init = flash_root / "flash_qla" / "ops" / "gated_delta_rule" / "chunk" / "__init__.py"
        if not chunk_init.exists():
            continue
        source = chunk_init.read_text()
        revision = _source_version(flash_root)
        if "chunk_gated_delta_rule_fwd" not in source:
            return OfficialStatus(
                "qwen_official_flashqla_forward",
                True,
                False,
                str(flash_root),
                revision,
                "FlashQLA source found, but chunk_gated_delta_rule_fwd was not found.",
            )
        if "FlashQLA now support sm90 only" in source and "tilelang.contrib.nvcc" in source:
            return OfficialStatus(
                "qwen_official_flashqla_forward",
                True,
                False,
                str(flash_root),
                revision,
                "Local FlashQLA forward exists but has an import-time NVIDIA Hopper sm90-only guard.",
            )
        return OfficialStatus(
            "qwen_official_flashqla_forward",
            True,
            True,
            str(flash_root),
            revision,
            "Local FlashQLA forward exists and no sm90-only guard was detected by source scan.",
        )

    return OfficialStatus(
        "qwen_official_flashqla_forward",
        False,
        False,
        "not found",
        "unknown",
        "No local FlashQLA chunk_gated_delta_rule_fwd source was found in known candidate paths.",
    )


def _append_official_rows(
    results: list[BenchResult],
    *,
    official_status: OfficialStatus,
    gpu: str,
    policy: DTypePolicy,
    shape: BenchShape,
) -> None:
    reason = official_status.reason
    if official_status.exists:
        reason = f"{reason} path={official_status.path} rev={official_status.revision}"
    results.append(
        BenchResult(
            implementation=official_status.implementation,
            gpu=gpu,
            dtype_policy=policy.name,
            shape=shape,
            initial_state="none",
            latency_ms=None,
            speedup_vs_pytorch_eager=None,
            speedup_vs_torch_compile=None,
            correctness_status="skipped",
            notes=reason,
        )
    )


def _benchmark_case(
    *,
    shape: BenchShape,
    policy: DTypePolicy,
    gpu: str,
    official_status: OfficialStatus,
    warmup: int,
    repeat: int,
    seed: int,
) -> list[BenchResult]:
    q, k, v, g, beta = _make_inputs(shape, policy.qkv_dtype, seed=seed)
    q_ref, k_ref, v_ref = _reference_inputs(q, k, v, policy)
    scale = shape.head_dim_k**-0.5
    atol = BF16_ATOL if policy.reference_uses_fp32_cast else FP32_ATOL
    rtol = BF16_RTOL if policy.reference_uses_fp32_cast else FP32_RTOL
    results: list[BenchResult] = []

    expected = _forward_ref(q_ref, k_ref, v_ref, g, beta, shape=shape, scale=scale)
    torch.cuda.synchronize()

    eager_latency: float | None = None
    compile_latency: float | None = None

    if policy.qkv_dtype == torch.float32:
        eager_latency = _time_ms(
            lambda: _forward_ref(q, k, v, g, beta, shape=shape, scale=scale),
            warmup=warmup,
            repeat=repeat,
        )
        results.append(
            BenchResult(
                implementation="pytorch_eager_qwen_gdn_forward_ref",
                gpu=gpu,
                dtype_policy=policy.name,
                shape=shape,
                initial_state="none",
                latency_ms=eager_latency,
                speedup_vs_pytorch_eager=1.0,
                speedup_vs_torch_compile=None,
                correctness_status="reference",
                notes="PyTorch eager qwen_gdn_forward_ref.",
            )
        )
    else:
        try:
            _forward_ref(q, k, v, g, beta, shape=shape, scale=scale)
            torch.cuda.synchronize()
            native_note = "Native BF16 PyTorch reference unexpectedly ran; timing still skipped by policy."
        except Exception as exc:
            native_note = (
                "Native BF16 qwen_gdn_forward_ref is not a fair performance baseline: "
                f"{_exception_text(exc)}"
            )
        results.append(
            BenchResult(
                implementation="pytorch_eager_qwen_gdn_forward_ref",
                gpu=gpu,
                dtype_policy=policy.name,
                shape=shape,
                initial_state="none",
                latency_ms=None,
                speedup_vs_pytorch_eager=None,
                speedup_vs_torch_compile=None,
                correctness_status="reference uses q/k/v .float() for BF16 correctness",
                notes=native_note,
            )
        )

    if policy.qkv_dtype == torch.float32:
        try:
            torch._dynamo.reset()

            def compiled_wrapper(
                q_arg: torch.Tensor,
                k_arg: torch.Tensor,
                v_arg: torch.Tensor,
                g_arg: torch.Tensor,
                beta_arg: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                return _forward_ref(q_arg, k_arg, v_arg, g_arg, beta_arg, shape=shape, scale=scale)

            compiled_fn = torch.compile(compiled_wrapper)
            compiled_actual = compiled_fn(q, k, v, g, beta)
            torch.cuda.synchronize()
            correctness = _compare_outputs(compiled_actual, expected, atol=atol, rtol=rtol)
            compile_latency = _time_ms(lambda: compiled_fn(q, k, v, g, beta), warmup=warmup, repeat=repeat)
            results.append(
                BenchResult(
                    implementation="torch_compile_qwen_gdn_forward_ref",
                    gpu=gpu,
                    dtype_policy=policy.name,
                    shape=shape,
                    initial_state="none",
                    latency_ms=compile_latency,
                    speedup_vs_pytorch_eager=eager_latency / compile_latency if eager_latency else None,
                    speedup_vs_torch_compile=1.0,
                    correctness_status=correctness,
                    notes="torch.compile wrapper around qwen_gdn_forward_ref.",
                )
            )
        except Exception as exc:
            results.append(
                BenchResult(
                    implementation="torch_compile_qwen_gdn_forward_ref",
                    gpu=gpu,
                    dtype_policy=policy.name,
                    shape=shape,
                    initial_state="none",
                    latency_ms=None,
                    speedup_vs_pytorch_eager=None,
                    speedup_vs_torch_compile=None,
                    correctness_status="skipped",
                    notes=f"torch.compile failed: {_exception_text(exc)}",
                )
            )
    else:
        results.append(
            BenchResult(
                implementation="torch_compile_qwen_gdn_forward_ref",
                gpu=gpu,
                dtype_policy=policy.name,
                shape=shape,
                initial_state="none",
                latency_ms=None,
                speedup_vs_pytorch_eager=None,
                speedup_vs_torch_compile=None,
                correctness_status="skipped",
                notes="Skipped: BF16 PyTorch reference is not a fair native performance baseline.",
            )
        )

    try:
        v6_actual = qwen_gdn_chunked_avelang_v6_standalone(q, k, v, g, beta, scale=scale, chunk_size=shape.chunk_size)
        torch.cuda.synchronize()
        correctness = _compare_outputs(v6_actual, expected, atol=atol, rtol=rtol)
        v6_latency = _time_ms(
            lambda: qwen_gdn_chunked_avelang_v6_standalone(q, k, v, g, beta, scale=scale, chunk_size=shape.chunk_size),
            warmup=warmup,
            repeat=repeat,
        )
        results.append(
            BenchResult(
                implementation="avelang_dsl_standalone_v6",
                gpu=gpu,
                dtype_policy=policy.name,
                shape=shape,
                initial_state="none",
                latency_ms=v6_latency,
                speedup_vs_pytorch_eager=eager_latency / v6_latency if eager_latency else None,
                speedup_vs_torch_compile=compile_latency / v6_latency if compile_latency else None,
                correctness_status=correctness,
                notes="Standalone v6 from qwen_gdn_chunked_avelang_v6_standalone.py.",
            )
        )
    except Exception as exc:
        results.append(
            BenchResult(
                implementation="avelang_dsl_standalone_v6",
                gpu=gpu,
                dtype_policy=policy.name,
                shape=shape,
                initial_state="none",
                latency_ms=None,
                speedup_vs_pytorch_eager=None,
                speedup_vs_torch_compile=None,
                correctness_status="failed",
                notes=_exception_text(exc),
            )
        )

    _append_official_rows(results, official_status=official_status, gpu=gpu, policy=policy, shape=shape)
    return results


def _fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _result_csv_row(result: BenchResult) -> str:
    shape = result.shape
    values = (
        result.implementation,
        result.gpu,
        result.dtype_policy,
        str(shape.batch_size),
        str(shape.num_tokens),
        str(shape.num_k_heads),
        str(shape.num_v_heads),
        str(shape.head_dim_k),
        str(shape.head_dim_v),
        str(shape.chunk_size),
        result.initial_state,
        _fmt_float(result.latency_ms),
        _fmt_float(result.speedup_vs_pytorch_eager),
        _fmt_float(result.speedup_vs_torch_compile),
        result.correctness_status,
        result.notes.replace("\n", " "),
    )
    return ",".join(values)


def _markdown_table(results: list[BenchResult]) -> list[str]:
    header = (
        "| implementation | GPU | dtype_policy | B | T | Hk | Hv | K | V | chunk_size | initial_state | "
        "latency_ms | speedup_vs_pytorch_eager | speedup_vs_torch_compile | correctness_status | notes |"
    )
    sep = (
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|"
    )
    lines = [header, sep]
    for result in results:
        shape = result.shape
        lines.append(
            "| "
            + " | ".join(
                (
                    result.implementation,
                    result.gpu,
                    result.dtype_policy,
                    str(shape.batch_size),
                    str(shape.num_tokens),
                    str(shape.num_k_heads),
                    str(shape.num_v_heads),
                    str(shape.head_dim_k),
                    str(shape.head_dim_v),
                    str(shape.chunk_size),
                    result.initial_state,
                    _fmt_float(result.latency_ms),
                    _fmt_float(result.speedup_vs_pytorch_eager),
                    _fmt_float(result.speedup_vs_torch_compile),
                    result.correctness_status.replace("|", "\\|"),
                    result.notes.replace("|", "\\|").replace("\n", " "),
                )
            )
            + " |"
        )
    return lines


def _summarize_v6_speedups(results: list[BenchResult]) -> list[str]:
    lines = []
    for result in results:
        if result.implementation != "avelang_dsl_standalone_v6":
            continue
        if result.speedup_vs_pytorch_eager is None and result.speedup_vs_torch_compile is None:
            continue
        parts = [f"{result.shape.name}/{result.dtype_policy}"]
        if result.speedup_vs_pytorch_eager is not None:
            parts.append(f"vs eager {_fmt_float(result.speedup_vs_pytorch_eager)}x")
        if result.speedup_vs_torch_compile is not None:
            parts.append(f"vs torch.compile {_fmt_float(result.speedup_vs_torch_compile)}x")
        lines.append("- " + ": ".join((parts[0], ", ".join(parts[1:]))))
    if not lines:
        return ["- No v6 speedup rows were available."]
    return lines


def _write_report(
    *,
    results: list[BenchResult],
    official_status: OfficialStatus,
    warmup: int,
    repeat: int,
    seed: int,
    report_path: Path,
) -> None:
    gpu = torch.cuda.get_device_name(torch.cuda.current_device())
    repo_root = Path(__file__).resolve().parents[3]
    lines = [
        "# Qwen GDN Baseline Compare Results",
        "",
        f"- Date: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Avelang git revision: {_git_revision(repo_root)}",
        f"- Qwen official FlashQLA revision: {official_status.revision}",
        f"- PyTorch: {torch.__version__}",
        f"- ROCm/HIP: {getattr(torch.version, 'hip', None)}",
        f"- GPU: {gpu}",
        f"- HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES', 'not set')}",
        f"- warmup: {warmup}",
        f"- repeat: {repeat}",
        f"- seed: {seed}",
        "",
        "## Method",
        "",
        "- All measured implementations use the same tensors for a given shape and dtype policy.",
        "- Timing uses `torch.cuda.Event(enable_timing=True)` and `torch.cuda.synchronize()`.",
        "- First-call Avelang JIT and `torch.compile` compile cost are excluded by correctness/warmup calls before timing.",
        "- FP32 correctness is checked against `qwen_gdn_forward_ref` on FP32 inputs.",
        "- BF16 v6 correctness is checked against `qwen_gdn_forward_ref(q_bf16.float(), k_bf16.float(), v_bf16.float())`.",
        "- Native BF16 PyTorch eager/compile reference latency is marked N/A when direct BF16 execution fails.",
        "",
        "## Qwen Official Optimized Forward",
        "",
        f"- exists: {official_status.exists}",
        f"- runnable_on_mi210: {official_status.runnable}",
        f"- path: `{official_status.path}`",
        f"- reason: {official_status.reason}",
        "",
        "The local official implementation is FlashQLA. Its forward entry is `chunk_gated_delta_rule_fwd`; "
        "the scanned source has an import-time `sm90 only` guard when running outside NVIDIA Hopper, "
        "so it is not a runnable AMD MI210 baseline in this environment.",
        "",
        "## Results",
        "",
        *_markdown_table(results),
        "",
        "## v6 Speedup Summary",
        "",
        *_summarize_v6_speedups(results),
        "",
        "## Existing rocprofv3 v6 Bottleneck Data",
        "",
        "These rows come from the previous rocprofv3 run on 2026-06-02 in the same ROCm container on AMD MI210. "
        "They are kernel-time percentages, not end-to-end wall-clock percentages.",
        "",
        "| shape | dtype | kernel | avg_us | kernel_time_pct |",
        "|---|---|---|---:|---:|",
    ]
    for shape_name, dtype_name, kernel_name, avg_us, pct in ROCPROFV3_V6_BOTTLENECK_ROWS:
        lines.append(f"| {shape_name} | {dtype_name} | `{kernel_name}` | {avg_us:.3f} | {pct:.2f}% |")
    lines.extend(
        [
            "",
            "## Fairness Conclusion",
            "",
            "- The fairest runnable baseline on AMD MI210 is PyTorch eager `qwen_gdn_forward_ref` FP32 vs "
            "Avelang standalone v6 FP32, plus `torch.compile` FP32 when it succeeds.",
            "- BF16 v6 can be correctness-checked against an FP32-cast reference, but native BF16 PyTorch reference "
            "latency is not a fair baseline if direct BF16 eager fails.",
            "- Qwen official FlashQLA exists locally, but is Hopper sm90-only by source guard and is not a runnable "
            "MI210 baseline.",
            "- For the next kernel optimization, rocprofv3 points first at `chunk_gdr`, then `chunk_o`.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Qwen GDN baselines on ROCm.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--qwen-root", default=os.environ.get("QWEN_GDN_ROOT"))
    parser.add_argument("--report-path", default=str(REPORT_PATH))
    args = parser.parse_args()

    _ensure_rocm_available()
    torch.cuda.set_device(0)
    gpu = torch.cuda.get_device_name(torch.cuda.current_device())
    official_status = _detect_qwen_official(args.qwen_root)

    all_results: list[BenchResult] = []
    print(
        "implementation,GPU,dtype_policy,B,T,Hk,Hv,K,V,chunk_size,initial_state,latency_ms,"
        "speedup_vs_pytorch_eager,speedup_vs_torch_compile,correctness_status,notes"
    )
    for shape in SHAPES:
        for policy in D_TYPE_POLICIES:
            case_results = _benchmark_case(
                shape=shape,
                policy=policy,
                gpu=gpu,
                official_status=official_status,
                warmup=args.warmup,
                repeat=args.repeat,
                seed=args.seed,
            )
            all_results.extend(case_results)
            for result in case_results:
                print(_result_csv_row(result))

    report_path = Path(args.report_path)
    _write_report(
        results=all_results,
        official_status=official_status,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        report_path=report_path,
    )
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
