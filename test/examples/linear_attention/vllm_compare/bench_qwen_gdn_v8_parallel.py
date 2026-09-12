#!/usr/bin/env python3
"""Stage and full-forward benchmark for v8 parallel chunk_gdr."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_chunk_gdr_avelang_v6_standalone,
    qwen_gdn_chunk_o_avelang_v6_standalone,
    qwen_gdn_chunked_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v7_vllm_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v7_vllm_layout,
)
from qwen_gdn_chunked_avelang_v8_vllm_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v8_vllm_layout,
    qwen_gdn_chunked_avelang_v8_vllm_layout,
)


def sync() -> None:
    torch.cuda.synchronize()


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_f = x.float()
    return (x_f * torch.rsqrt((x_f * x_f).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def make_inputs(
    batch_size: int,
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    *,
    seed: int,
    with_initial_state: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = l2norm(
        torch.randn(
            batch_size,
            num_tokens,
            num_k_heads,
            head_dim_k,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
    )
    k = l2norm(
        torch.randn(
            batch_size,
            num_tokens,
            num_k_heads,
            head_dim_k,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
    )
    v = torch.randn(
        batch_size,
        num_tokens,
        num_v_heads,
        head_dim_v,
        device="cuda",
        dtype=torch.bfloat16,
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
    g = (g / 16.0).contiguous()
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
    if with_initial_state:
        initial_state = (
            torch.randn(
                batch_size,
                num_v_heads,
                head_dim_v,
                head_dim_k,
                device="cuda",
                dtype=torch.float32,
                generator=generator,
            )
            * 0.01
        ).contiguous()
    return q, k, v, g, beta, initial_state


def time_ms(fn: Callable[[], object], *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        start.record()
        result = fn()
        end.record()
        sync()
        times.append(start.elapsed_time(end))
        del result
    return torch.tensor(times).median().item()


def max_abs_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def maybe_import_vllm():
    try:
        from vllm.model_executor.layers.fla.ops import chunk_delta_h
        from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_chunk_gdn
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"vllm_import_error,{type(exc).__name__},{exc}")
        return None

    if getattr(torch.version, "hip", None) is not None:
        kernel = chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
        autotuner = getattr(kernel, "fn", None)
        configs = getattr(autotuner, "configs", None)
        if configs:
            filtered = [config for config in configs if getattr(config, "num_stages", None) != 4]
            if len(filtered) != len(configs):
                autotuner.configs = filtered
                autotuner.cache.clear()
    return vllm_chunk_gdn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--T", type=int, default=512)
    parser.add_argument("--Hk", type=int, default=4)
    parser.add_argument("--Hv", type=int, default=8)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--V", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--no-initial-state", action="store_true")
    parser.add_argument("--vk-block-v", type=int, default=4)
    parser.add_argument("--vk-block-k", type=int, default=64)
    parser.add_argument("--vblock-block-v", type=int, default=32)
    parser.add_argument("--include-vllm", action="store_true")
    parser.add_argument("--sweep-vk", action="store_true")
    parser.add_argument("--stage-breakdown", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is not available.")

    q, k, v, g, beta, initial_state = make_inputs(
        args.B,
        args.T,
        args.Hk,
        args.Hv,
        args.K,
        args.V,
        seed=args.seed,
        with_initial_state=not args.no_initial_state,
    )
    scale = args.K**-0.5

    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=args.chunk)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=args.chunk, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=args.chunk)
    w, u = qwen_gdn_w_u_avelang_v6_standalone(
        k,
        v,
        g_cumsum,
        beta,
        a_solved,
        chunk_size=args.chunk,
        prefer_optimized=True,
    )
    sync()

    h6, vn6, final6 = qwen_gdn_chunk_gdr_avelang_v6_standalone(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=args.chunk,
        prefer_optimized=True,
    )
    h7, vn7, final7 = qwen_gdn_chunk_gdr_avelang_v7_vllm_layout(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=args.chunk,
        prefer_optimized=True,
        value_tile=1,
    )
    h8_vblock, vn8_vblock, final8_vblock = qwen_gdn_chunk_gdr_avelang_v8_vllm_layout(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=args.chunk,
        use_parallel_chunk_gdr=True,
        parallel_mode="vblock",
        block_v=args.vblock_block_v,
    )
    h8_vk, vn8_vk, final8_vk = qwen_gdn_chunk_gdr_avelang_v8_vllm_layout(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=args.chunk,
        use_parallel_chunk_gdr=True,
        parallel_mode="vk",
        block_v=args.vk_block_v,
        block_k=args.vk_block_k,
    )
    sync()

    print(
        "shape,"
        f"B={args.B},T={args.T},Hk={args.Hk},Hv={args.Hv},K={args.K},V={args.V},chunk={args.chunk},"
        f"initial_state={initial_state is not None}"
    )
    for name, h, vn, final in (
        ("v7", h7, vn7, final7),
        ("v8_vblock", h8_vblock, vn8_vblock, final8_vblock),
        ("v8_vk", h8_vk, vn8_vk, final8_vk),
    ):
        print(
            f"stage_error,{name},"
            f"h={max_abs_err(h, h6):.9g},"
            f"vn={max_abs_err(vn, vn6):.9g},"
            f"final_state={max_abs_err(final, final6):.9g}"
        )

    stage_fns: list[tuple[str, Callable[[], object]]] = [
        (
            "v6_chunk_gdr",
            lambda: qwen_gdn_chunk_gdr_avelang_v6_standalone(
                k,
                w,
                u,
                g_cumsum,
                initial_state=initial_state,
                chunk_size=args.chunk,
                prefer_optimized=True,
            ),
        ),
        (
            "v7_chunk_gdr",
            lambda: qwen_gdn_chunk_gdr_avelang_v7_vllm_layout(
                k,
                w,
                u,
                g_cumsum,
                initial_state=initial_state,
                chunk_size=args.chunk,
                prefer_optimized=True,
                value_tile=1,
            ),
        ),
        (
            "v8_vblock_chunk_gdr",
            lambda: qwen_gdn_chunk_gdr_avelang_v8_vllm_layout(
                k,
                w,
                u,
                g_cumsum,
                initial_state=initial_state,
                chunk_size=args.chunk,
                use_parallel_chunk_gdr=True,
                parallel_mode="vblock",
                block_v=args.vblock_block_v,
            ),
        ),
        (
            "v8_vk_chunk_gdr",
            lambda: qwen_gdn_chunk_gdr_avelang_v8_vllm_layout(
                k,
                w,
                u,
                g_cumsum,
                initial_state=initial_state,
                chunk_size=args.chunk,
                use_parallel_chunk_gdr=True,
                parallel_mode="vk",
                block_v=args.vk_block_v,
                block_k=args.vk_block_k,
            ),
        ),
    ]
    medians: dict[str, float] = {}
    for name, fn in stage_fns:
        medians[name] = time_ms(fn, warmup=args.warmup, repeat=args.repeat)
        print(f"stage_median_ms,{name},{medians[name]:.9g}")
    print(f"stage_speedup_vs_v6,v8_vk,{medians['v6_chunk_gdr'] / medians['v8_vk_chunk_gdr']:.9g}")
    print(f"stage_speedup_vs_v7,v8_vk,{medians['v7_chunk_gdr'] / medians['v8_vk_chunk_gdr']:.9g}")
    print(f"stage_speedup_vs_v8_vblock,v8_vk,{medians['v8_vblock_chunk_gdr'] / medians['v8_vk_chunk_gdr']:.9g}")

    if args.stage_breakdown:
        breakdown_fns: list[tuple[str, Callable[[], object]]] = [
            ("cumsum", lambda: qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=args.chunk)),
            (
                "KKT",
                lambda: qwen_gdn_kkt_avelang_v6_standalone(
                    k,
                    g_cumsum,
                    beta,
                    chunk_size=args.chunk,
                    prefer_optimized=True,
                ),
            ),
            ("solve", lambda: qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=args.chunk)),
            (
                "w_u",
                lambda: qwen_gdn_w_u_avelang_v6_standalone(
                    k,
                    v,
                    g_cumsum,
                    beta,
                    a_solved,
                    chunk_size=args.chunk,
                    prefer_optimized=True,
                ),
            ),
            (
                "chunk_gdr_v8_vk",
                lambda: qwen_gdn_chunk_gdr_avelang_v8_vllm_layout(
                    k,
                    w,
                    u,
                    g_cumsum,
                    initial_state=initial_state,
                    chunk_size=args.chunk,
                    use_parallel_chunk_gdr=True,
                    parallel_mode="vk",
                    block_v=args.vk_block_v,
                    block_k=args.vk_block_k,
                ),
            ),
            (
                "chunk_o",
                lambda: qwen_gdn_chunk_o_avelang_v6_standalone(
                    q,
                    k,
                    vn8_vk,
                    h8_vk,
                    g_cumsum,
                    scale=scale,
                    chunk_size=args.chunk,
                    prefer_optimized=True,
                ),
            ),
        ]
        for name, fn in breakdown_fns:
            elapsed = time_ms(fn, warmup=args.warmup, repeat=args.repeat)
            print(f"breakdown_median_ms,{name},{elapsed:.9g}")

    sweep_had_error = False
    if args.sweep_vk:
        for sweep_block_v in (1, 2, 4, 8, 16, 32, 64):
            for sweep_block_k in (4, 8, 16, 32, 64, 128):
                if sweep_block_v * sweep_block_k > 1024:
                    continue
                sweep_name = f"v8_vk_bv{sweep_block_v}_bk{sweep_block_k}"
                try:
                    elapsed = time_ms(
                        lambda bv=sweep_block_v, bk=sweep_block_k: qwen_gdn_chunk_gdr_avelang_v8_vllm_layout(
                            k,
                            w,
                            u,
                            g_cumsum,
                            initial_state=initial_state,
                            chunk_size=args.chunk,
                            use_parallel_chunk_gdr=True,
                            parallel_mode="vk",
                            block_v=bv,
                            block_k=bk,
                        ),
                        warmup=max(3, args.warmup // 2),
                        repeat=max(10, args.repeat // 2),
                    )
                    print(f"sweep_stage_median_ms,{sweep_name},{elapsed:.9g}")
                except Exception as exc:
                    sweep_had_error = True
                    print(f"sweep_stage_error,{sweep_name},{type(exc).__name__},{exc}")
        if sweep_had_error:
            print("sweep_had_error,skip_full_after_sweep,rerun_without_sweep_for_full_benchmark")
            return

    full8 = qwen_gdn_chunked_avelang_v8_vllm_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=args.chunk,
        use_parallel_chunk_gdr=True,
        parallel_mode="vk",
        block_v=args.vk_block_v,
        block_k=args.vk_block_k,
    )
    full6 = qwen_gdn_chunked_avelang_v6_standalone(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=args.chunk,
        prefer_optimized=True,
    )
    sync()
    print(f"full_error_vs_v6,output={max_abs_err(full8[0], full6[1]):.9g},final_state={max_abs_err(full8[1], full6[4]):.9g}")
    full8_ms = time_ms(
        lambda: qwen_gdn_chunked_avelang_v8_vllm_layout(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            scale=scale,
            chunk_size=args.chunk,
            use_parallel_chunk_gdr=True,
            parallel_mode="vk",
            block_v=args.vk_block_v,
            block_k=args.vk_block_k,
        ),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    print(f"full_median_ms,avelang_v8_vk,{full8_ms:.9g}")

    if args.include_vllm:
        vllm_chunk_gdn = maybe_import_vllm()
        if vllm_chunk_gdn is not None:
            o_vllm, s_vllm = vllm_chunk_gdn(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=None,
                head_first=False,
                use_qk_l2norm_in_kernel=False,
            )
            sync()
            print(f"full_error_vs_vllm,output={max_abs_err(full8[0], o_vllm):.9g},final_state={max_abs_err(full8[1], s_vllm):.9g}")
            vllm_ms = time_ms(
                lambda: vllm_chunk_gdn(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    scale=scale,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=None,
                    head_first=False,
                    use_qk_l2norm_in_kernel=False,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
            print(f"full_median_ms,vllm,{vllm_ms:.9g}")
            print(f"full_speedup_vllm_over_avelang_v8,{vllm_ms / full8_ms:.9g}")


if __name__ == "__main__":
    main()
