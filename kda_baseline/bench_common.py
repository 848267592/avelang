#!/usr/bin/env python3
"""Shared contract, reference, timing, and result helpers for KDA benchmarks.

The benchmark inputs intentionally model only ordinary forward KDA.  The
contract excludes causal convolution and RMSNorm.  Inputs are BF16 and the
recurrent state is FP32, matching the two reference operator paths.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import torch


DTYPE = torch.bfloat16
STATE_DTYPE = torch.float32
HEAD_DIM = 128
LOWER_BOUND = -5.0
L2NORM_EPS = 1e-6
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class BenchResult:
    median_us: float
    min_us: float
    p90_us: float


def sync() -> None:
    torch.cuda.synchronize()


def require_rocm_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA/HIP device is available for this benchmark")
    torch.cuda.set_device(0)
    return torch.device("cuda")


@torch.inference_mode()
def bench_cuda(
    fn: Callable[[], object],
    *,
    prepare: Callable[[], None] | None = None,
    warmup: int = 50,
    rep: int = 300,
) -> BenchResult:
    """Measure GPU time only for ``fn``.

    ``prepare`` resets mutable state and input buffers before each launch.  It
    is deliberately recorded before the start event, so state/input copies are
    not included in the reported KDA latency.  Warmup similarly includes all
    Triton compilation/autotune before the timed samples begin.
    """
    if warmup < 0 or rep <= 0:
        raise ValueError(f"invalid warmup/rep: {warmup=}, {rep=}")

    for _ in range(warmup):
        if prepare is not None:
            prepare()
        fn()
    sync()

    samples_us: list[float] = []
    for _ in range(rep):
        if prepare is not None:
            prepare()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)

    samples_us.sort()
    return BenchResult(
        median_us=float(statistics.median(samples_us)),
        min_us=float(samples_us[0]),
        p90_us=float(samples_us[int(0.90 * (len(samples_us) - 1))]),
    )


def make_tensor(
    shape: tuple[int, ...],
    *,
    scale: float = 0.5,
    device: torch.device,
    dtype: torch.dtype = DTYPE,
) -> torch.Tensor:
    """Create a moderate-valued BF16 tensor via deterministic FP32 sampling."""
    return (torch.randn(shape, device=device, dtype=torch.float32) * scale).to(dtype)


def make_kda_prefill_inputs(
    *,
    T: int,
    H: int = 12,
    K: int = HEAD_DIM,
    V: int = HEAD_DIM,
    seed: int = 42,
) -> dict[str, torch.Tensor | float | int]:
    """Create a single variable-length sequence and a two-row state pool."""
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if K != HEAD_DIM or V != HEAD_DIM:
        raise ValueError(f"only K=V={HEAD_DIM} is in the benchmark contract")
    device = require_rocm_cuda()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    q = make_tensor((1, T, H, K), device=device)
    k = make_tensor((1, T, H, K), device=device)
    v = make_tensor((1, T, H, V), device=device)
    raw_g = make_tensor((1, T, H, K), scale=1.0, device=device)
    raw_beta = make_tensor((1, T, H), scale=1.0, device=device)
    A_log = torch.randn((H,), device=device, dtype=torch.float32) * 0.5
    dt_bias = torch.randn((H * K,), device=device, dtype=torch.float32) * 0.5

    # Row zero remains unused so cache-index semantics match decode.
    state_pool = torch.zeros((2, H, V, K), device=device, dtype=STATE_DTYPE)
    state_index = torch.tensor([1], device=device, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)
    return {
        "q": q,
        "k": k,
        "v": v,
        "raw_g": raw_g,
        "raw_beta": raw_beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "state_pool": state_pool,
        "state_index": state_index,
        "cu_seqlens": cu_seqlens,
        "scale": K**-0.5,
        "T": T,
        "H": H,
        "K": K,
        "V": V,
    }


def make_kda_decode_inputs(
    *,
    B: int,
    H: int = 12,
    K: int = HEAD_DIM,
    V: int = HEAD_DIM,
    seed: int = 42,
) -> dict[str, torch.Tensor | float | int]:
    """Create post-conv packed QKV inputs and a cache-style state pool."""
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}")
    if K != HEAD_DIM or V != HEAD_DIM:
        raise ValueError(f"only K=V={HEAD_DIM} is in the benchmark contract")
    device = require_rocm_cuda()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    qkv_dim = H * (2 * K + V)
    mixed_qkv = make_tensor((B, qkv_dim), device=device)
    raw_g = make_tensor((B, H * K), scale=1.0, device=device)
    raw_beta = make_tensor((B, H), scale=1.0, device=device)
    A_log = torch.randn((H,), device=device, dtype=torch.float32) * 0.5
    dt_bias = torch.randn((H * K,), device=device, dtype=torch.float32) * 0.5

    # Index zero is deliberately unused.  This also satisfies vLLM AMD
    # recurrent paths that reserve non-positive state rows as invalid.
    state_pool = torch.zeros((B + 1, H, V, K), device=device, dtype=STATE_DTYPE)
    state_indices = torch.arange(1, B + 1, device=device, dtype=torch.int32)
    return {
        "mixed_qkv": mixed_qkv,
        "raw_g": raw_g,
        "raw_beta": raw_beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "state_pool": state_pool,
        "state_indices": state_indices,
        "scale": K**-0.5,
        "B": B,
        "H": H,
        "K": K,
        "V": V,
    }


def l2_normalize_for_reference(
    value: torch.Tensor,
    *,
    round_to_bf16: bool,
) -> torch.Tensor:
    """Reproduce the KDA L2 norm's FP32 accumulation and epsilon.

    Prefill calls a separate Triton L2Norm kernel which stores a BF16 result;
    packed decode normalizes in registers, with no BF16 round trip.  The caller
    selects the corresponding behavior.
    """
    out = value.float() * torch.rsqrt(
        torch.sum(value.float().square(), dim=-1, keepdim=True) + L2NORM_EPS
    )
    return out.to(DTYPE).float() if round_to_bf16 else out


@torch.inference_mode()
def torch_kda_reference(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    lower_bound: float | None = LOWER_BOUND,
    l2norm_round_to_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slow independent FP32 recurrence for ordinary KDA forward semantics."""
    qf = l2_normalize_for_reference(q, round_to_bf16=l2norm_round_to_bf16)
    kf = l2_normalize_for_reference(k, round_to_bf16=l2norm_round_to_bf16)
    vf = v.float()
    B, T, H, K = qf.shape

    gate_input = raw_g.float() + dt_bias.float().view(1, 1, H, K)
    decay_scale = torch.exp(A_log.float()).view(1, 1, H, 1)
    if lower_bound is None:
        gate = -decay_scale * torch.nn.functional.softplus(gate_input)
    else:
        gate = lower_bound * torch.sigmoid(decay_scale * gate_input)
    beta = torch.sigmoid(raw_beta.float())

    state = initial_state.float().clone()
    outputs: list[torch.Tensor] = []
    scale = K**-0.5
    for t in range(T):
        qt = qf[:, t] * scale
        kt = kf[:, t]
        vt = vf[:, t]
        state = state * torch.exp(gate[:, t]).unsqueeze(-2)
        prediction = torch.sum(state * kt.unsqueeze(-2), dim=-1)
        delta = (vt - prediction) * beta[:, t].unsqueeze(-1)
        state = state + delta.unsqueeze(-1) * kt.unsqueeze(-2)
        outputs.append(torch.sum(state * qt.unsqueeze(-2), dim=-1))

    return torch.stack(outputs, dim=1).to(DTYPE), state


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-6)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float((diff / denom).max().item()),
    }


def assert_close(
    *,
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    metrics = error_metrics(actual, expected)
    try:
        torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise AssertionError(
            f"{name} failed: max_abs={metrics['max_abs']:.6e}, "
            f"mean_abs={metrics['mean_abs']:.6e}, "
            f"max_rel={metrics['max_rel']:.6e}, {exc}"
        ) from exc
    return metrics


def result_row(**items: object) -> dict[str, object]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        **items,
    }


def save_result(row: dict[str, object], *, result_dir: str | Path, stem: str) -> Path:
    directory = Path(result_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True))
    return path


def result_fields(result: BenchResult) -> dict[str, float]:
    return asdict(result)


def effective_state_tbps(*, B: int, H: int, K: int, V: int, median_us: float) -> float:
    # Each recurrent step reads and writes FP32 [V,K] state once.
    state_bytes = 2 * B * H * K * V * torch.empty((), dtype=STATE_DTYPE).element_size()
    return float(state_bytes / (median_us * 1e-6) / 1e12)
