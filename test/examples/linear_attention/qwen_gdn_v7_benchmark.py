#!/usr/bin/env python3
"""Qwen GDN v7 benchmark：比较 standalone v6 与 v7。"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from qwen_gdn_chunked_avelang_v6_standalone import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_chunk_gdr_avelang_v6_standalone,
    qwen_gdn_chunk_o_avelang_v6_standalone,
    qwen_gdn_chunked_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v7 import qwen_gdn_chunk_gdr_avelang_v7, qwen_gdn_chunked_avelang_v7


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
class PreparedStages:
    g_cumsum: torch.Tensor
    a: torch.Tensor
    a_solved: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    h_v6: torch.Tensor
    vn_v6: torch.Tensor
    h_v7: torch.Tensor
    vn_v7: torch.Tensor


SMALL_SHAPE = BenchShape("small", 1, 16, 1, 2, 4, 4, 4)
MEDIUM_SHAPE = BenchShape("medium", 1, 64, 2, 4, 8, 8, 8)
LARGER_DEBUG_SHAPE = BenchShape("larger_debug", 2, 128, 2, 4, 16, 16, 8)
OPTIONAL_SHAPE = BenchShape("optional_256", 1, 256, 4, 8, 32, 32, 16)


def _ensure_rocm_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not hasattr(torch.version, "hip") or torch.version.hip is None:
        raise RuntimeError("HIP is not available; Qwen GDN v7 benchmark requires ROCm.")


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


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "fp32"
    if dtype == torch.bfloat16:
        return "bf16"
    return str(dtype)


def _time_ms(fn: Callable[[], object], *, warmup: int, repeat: int) -> float:
    # warmup 包含第一次 JIT 编译，计时前强制同步，避免把编译成本算进结果。
    result = None
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()
    try:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(repeat):
            result = fn()
        end_evt.record()
        torch.cuda.synchronize()
        elapsed_ms = start_evt.elapsed_time(end_evt) / repeat
    except RuntimeError:
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeat):
            result = fn()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeat
    del result
    return elapsed_ms


def _prepare_stages(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    shape: BenchShape,
    scale: float,
) -> PreparedStages:
    # 预先生成下游 stage 输入，分 stage 计时时不重复计算上游。
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=shape.chunk_size)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=shape.chunk_size)
    a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=shape.chunk_size)
    w, u = qwen_gdn_w_u_avelang_v6_standalone(k, v, g_cumsum, beta, a_solved, chunk_size=shape.chunk_size)
    h_v6, vn_v6, _ = qwen_gdn_chunk_gdr_avelang_v6_standalone(k, w, u, g_cumsum, chunk_size=shape.chunk_size)
    h_v7, vn_v7, _ = qwen_gdn_chunk_gdr_avelang_v7(k, w, u, g_cumsum, chunk_size=shape.chunk_size)
    qwen_gdn_chunk_o_avelang_v6_standalone(
        q,
        k,
        vn_v6,
        h_v6,
        g_cumsum,
        scale=scale,
        chunk_size=shape.chunk_size,
    )
    qwen_gdn_chunk_o_avelang_v6_standalone(
        q,
        k,
        vn_v7,
        h_v7,
        g_cumsum,
        scale=scale,
        chunk_size=shape.chunk_size,
    )
    torch.cuda.synchronize()
    return PreparedStages(g_cumsum, a, a_solved, w, u, h_v6, vn_v6, h_v7, vn_v7)


def _stage_functions(
    impl: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    prepared: PreparedStages,
    *,
    shape: BenchShape,
    scale: float,
) -> list[tuple[str, Callable[[], object]]]:
    if impl == "v6":
        h = prepared.h_v6
        vn = prepared.vn_v6

        def chunk_gdr() -> object:
            return qwen_gdn_chunk_gdr_avelang_v6_standalone(
                k,
                prepared.w,
                prepared.u,
                prepared.g_cumsum,
                chunk_size=shape.chunk_size,
            )

        def end_to_end() -> object:
            return qwen_gdn_chunked_avelang_v6_standalone(
                q,
                k,
                v,
                g,
                beta,
                scale=scale,
                chunk_size=shape.chunk_size,
            )

    else:
        h = prepared.h_v7
        vn = prepared.vn_v7

        def chunk_gdr() -> object:
            return qwen_gdn_chunk_gdr_avelang_v7(
                k,
                prepared.w,
                prepared.u,
                prepared.g_cumsum,
                chunk_size=shape.chunk_size,
            )

        def end_to_end() -> object:
            return qwen_gdn_chunked_avelang_v7(
                q,
                k,
                v,
                g,
                beta,
                scale=scale,
                chunk_size=shape.chunk_size,
            )

    return [
        ("cumsum", lambda: qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=shape.chunk_size)),
        ("KKT", lambda: qwen_gdn_kkt_avelang_v6_standalone(k, prepared.g_cumsum, beta, chunk_size=shape.chunk_size)),
        ("solve", lambda: qwen_gdn_solve_avelang_v6_standalone(prepared.a, chunk_size=shape.chunk_size)),
        (
            "w/u",
            lambda: qwen_gdn_w_u_avelang_v6_standalone(
                k,
                v,
                prepared.g_cumsum,
                beta,
                prepared.a_solved,
                chunk_size=shape.chunk_size,
            ),
        ),
        ("chunk_gdr", chunk_gdr),
        (
            "chunk_o",
            lambda: qwen_gdn_chunk_o_avelang_v6_standalone(
                q,
                k,
                vn,
                h,
                prepared.g_cumsum,
                scale=scale,
                chunk_size=shape.chunk_size,
            ),
        ),
        ("end-to-end", end_to_end),
    ]


def _run_shape_dtype(shape: BenchShape, dtype: torch.dtype, *, warmup: int, repeat: int, seed: int) -> None:
    q, k, v, g, beta = _make_inputs(shape, dtype, seed=seed)
    scale = shape.head_dim_k**-0.5
    prepared = _prepare_stages(q, k, v, g, beta, shape=shape, scale=scale)

    timings: dict[tuple[str, str], float] = {}
    for impl in ("v6", "v7"):
        for stage_name, stage_fn in _stage_functions(impl, q, k, v, g, beta, prepared, shape=shape, scale=scale):
            elapsed_ms = _time_ms(stage_fn, warmup=warmup, repeat=repeat)
            timings[(impl, stage_name)] = elapsed_ms
            print(f"{shape.name},{_dtype_name(dtype)},{impl},{stage_name},{elapsed_ms:.6f}")

    v6_e2e = timings[("v6", "end-to-end")]
    v7_e2e = timings[("v7", "end-to-end")]
    speedup = v6_e2e / v7_e2e if v7_e2e > 0 else float("nan")
    v6_gdr = timings[("v6", "chunk_gdr")]
    v7_gdr = timings[("v7", "chunk_gdr")]
    gdr_speedup = v6_gdr / v7_gdr if v7_gdr > 0 else float("nan")
    print(f"{shape.name},{_dtype_name(dtype)},speedup,end-to-end,{speedup:.6f}")
    print(f"{shape.name},{_dtype_name(dtype)},speedup,chunk_gdr,{gdr_speedup:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen GDN v7 stage benchmark")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "both"), default="both")
    parser.add_argument("--include-optional", action="store_true", default=False)
    args = parser.parse_args()

    _ensure_rocm_available()
    dtypes = [torch.float32, torch.bfloat16] if args.dtype == "both" else [torch.float32 if args.dtype == "fp32" else torch.bfloat16]
    shapes = [SMALL_SHAPE, MEDIUM_SHAPE, LARGER_DEBUG_SHAPE]
    if args.include_optional:
        shapes.append(OPTIONAL_SHAPE)

    print("shape,dtype,impl,stage,time_ms")
    for shape in shapes:
        for dtype in dtypes:
            _run_shape_dtype(shape, dtype, warmup=args.warmup, repeat=args.repeat, seed=args.seed)


if __name__ == "__main__":
    main()
