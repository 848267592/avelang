#!/usr/bin/env python3
"""Qwen GDN v6 standalone 分 stage benchmark。"""

from __future__ import annotations

import argparse
from collections.abc import Callable

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


StageResult = tuple[str, float]


def _ensure_rocm_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not hasattr(torch.version, "hip") or torch.version.hip is None:
        raise RuntimeError("HIP is not available; Qwen GDN benchmark requires ROCm.")


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return (x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)).to(x.dtype)


def _parse_bool_selector(value: str) -> list[bool]:
    if value == "both":
        return [True, False]
    if value == "true":
        return [True]
    if value == "false":
        return [False]
    raise ValueError(f"prefer_optimized must be one of true/false/both, got {value}.")


def _parse_dtype_selector(value: str) -> list[torch.dtype]:
    if value == "both":
        return [torch.float32, torch.bfloat16]
    if value == "fp32":
        return [torch.float32]
    if value == "bf16":
        return [torch.bfloat16]
    raise ValueError(f"dtype must be one of fp32/bf16/both, got {value}.")


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "fp32"
    if dtype == torch.bfloat16:
        return "bf16"
    return str(dtype)


def _make_inputs(
    *,
    batch_size: int,
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    dtype: torch.dtype,
    use_initial_state: bool,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q_fp32 = _l2norm(
        torch.randn(
            batch_size,
            num_tokens,
            num_k_heads,
            head_dim_k,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
    ).contiguous()
    k_fp32 = _l2norm(
        torch.randn(
            batch_size,
            num_tokens,
            num_k_heads,
            head_dim_k,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
    ).contiguous()
    v_fp32 = torch.randn(
        batch_size,
        num_tokens,
        num_v_heads,
        head_dim_v,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).contiguous()
    g = torch.nn.functional.logsigmoid(
        torch.randn(
            batch_size,
            num_tokens,
            num_v_heads,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
    )
    g = (g / 16).contiguous()
    beta = torch.sigmoid(
        torch.randn(
            batch_size,
            num_tokens,
            num_v_heads,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
    ).contiguous()
    initial_state = None
    if use_initial_state:
        initial_state = torch.randn(
            batch_size,
            num_v_heads,
            head_dim_k,
            head_dim_v,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        ).contiguous()

    if dtype == torch.bfloat16:
        q = q_fp32.to(torch.bfloat16).contiguous()
        k = k_fp32.to(torch.bfloat16).contiguous()
        v = v_fp32.to(torch.bfloat16).contiguous()
    else:
        q, k, v = q_fp32, k_fp32, v_fp32
    return q, k, v, g, beta, initial_state


def _time_cuda_ms(fn: Callable[[], object], *, warmup: int, repeat: int) -> float:
    # warmup 会同时触发 Avelang JIT 编译，正式计时只看后续 repeat 次执行。
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


def _prepare_stage_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    scale: float,
    chunk_size: int,
    prefer_optimized: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    # 分 stage benchmark 固定上游输出，避免下游 stage 计时重复计算上游。
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk_size)
    a = qwen_gdn_kkt_avelang_v6_standalone(
        k,
        g_cumsum,
        beta,
        chunk_size=chunk_size,
        prefer_optimized=prefer_optimized,
    )
    a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk_size)
    w, u = qwen_gdn_w_u_avelang_v6_standalone(
        k,
        v,
        g_cumsum,
        beta,
        a_solved,
        chunk_size=chunk_size,
        prefer_optimized=prefer_optimized,
    )
    h, vn, _ = qwen_gdn_chunk_gdr_avelang_v6_standalone(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk_size,
        prefer_optimized=prefer_optimized,
    )
    qwen_gdn_chunk_o_avelang_v6_standalone(
        q,
        k,
        vn,
        h,
        g_cumsum,
        scale=scale,
        chunk_size=chunk_size,
        prefer_optimized=prefer_optimized,
    )
    torch.cuda.synchronize()
    return g_cumsum, a, a_solved, w, u, h, vn


def run_one_config(args: argparse.Namespace, dtype: torch.dtype, prefer_optimized: bool) -> None:
    q, k, v, g, beta, initial_state = _make_inputs(
        batch_size=args.batch_size,
        num_tokens=args.num_tokens,
        num_k_heads=args.num_k_heads,
        num_v_heads=args.num_v_heads,
        head_dim_k=args.head_dim_k,
        head_dim_v=args.head_dim_v,
        dtype=dtype,
        use_initial_state=args.initial_state,
        seed=args.seed,
    )
    scale = args.head_dim_k**-0.5 if args.scale is None else args.scale
    g_cumsum, a, a_solved, w, u, h, vn = _prepare_stage_inputs(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=args.chunk_size,
        prefer_optimized=prefer_optimized,
    )

    stage_fns: list[tuple[str, Callable[[], object]]] = [
        ("cumsum", lambda: qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=args.chunk_size)),
        (
            "KKT",
            lambda: qwen_gdn_kkt_avelang_v6_standalone(
                k,
                g_cumsum,
                beta,
                chunk_size=args.chunk_size,
                prefer_optimized=prefer_optimized,
            ),
        ),
        ("solve", lambda: qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=args.chunk_size)),
        (
            "w/u",
            lambda: qwen_gdn_w_u_avelang_v6_standalone(
                k,
                v,
                g_cumsum,
                beta,
                a_solved,
                chunk_size=args.chunk_size,
                prefer_optimized=prefer_optimized,
            ),
        ),
        (
            "chunk_gdr",
            lambda: qwen_gdn_chunk_gdr_avelang_v6_standalone(
                k,
                w,
                u,
                g_cumsum,
                initial_state=initial_state,
                chunk_size=args.chunk_size,
                prefer_optimized=prefer_optimized,
            ),
        ),
        (
            "chunk_o",
            lambda: qwen_gdn_chunk_o_avelang_v6_standalone(
                q,
                k,
                vn,
                h,
                g_cumsum,
                scale=scale,
                chunk_size=args.chunk_size,
                prefer_optimized=prefer_optimized,
            ),
        ),
        (
            "end-to-end",
            lambda: qwen_gdn_chunked_avelang_v6_standalone(
                q,
                k,
                v,
                g,
                beta,
                initial_state=initial_state,
                scale=scale,
                chunk_size=args.chunk_size,
                prefer_optimized=prefer_optimized,
            ),
        ),
    ]

    results: list[StageResult] = []
    for stage_name, stage_fn in stage_fns:
        elapsed_ms = _time_cuda_ms(stage_fn, warmup=args.warmup, repeat=args.repeat)
        results.append((stage_name, elapsed_ms))

    shape = (
        f"B={args.batch_size} T={args.num_tokens} Hk={args.num_k_heads} "
        f"Hv={args.num_v_heads} K={args.head_dim_k} V={args.head_dim_v} C={args.chunk_size}"
    )
    print(f"\nconfig dtype={_dtype_name(dtype)} prefer_optimized={prefer_optimized} {shape}")
    print("stage,time_ms")
    for stage_name, elapsed_ms in results:
        print(f"{stage_name},{elapsed_ms:.6f}")

    stage_only = [item for item in results if item[0] != "end-to-end"]
    bottlenecks = sorted(stage_only, key=lambda item: item[1], reverse=True)[: args.topk]
    summary = ", ".join(f"{name}={time_ms:.6f}ms" for name, time_ms in bottlenecks)
    print(f"bottleneck_top{args.topk},{summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen GDN v6 standalone stage benchmark")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--num-k-heads", type=int, default=2)
    parser.add_argument("--num-v-heads", type=int, default=4)
    parser.add_argument("--head-dim-k", type=int, default=8)
    parser.add_argument("--head-dim-v", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "both"), default="bf16")
    parser.add_argument("--prefer-optimized", choices=("true", "false", "both"), default="true")
    parser.add_argument("--initial-state", action="store_true", default=False)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--topk", type=int, default=2)
    args = parser.parse_args()

    _ensure_rocm_available()
    for dtype in _parse_dtype_selector(args.dtype):
        for prefer_optimized in _parse_bool_selector(args.prefer_optimized):
            run_one_config(args, dtype, prefer_optimized)


if __name__ == "__main__":
    main()
