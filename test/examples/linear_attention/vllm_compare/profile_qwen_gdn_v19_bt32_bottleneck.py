#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

AVELANG_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, AVELANG_DIR)

from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v17_mfma_layout_fixed import (
    _qwen_gdn_u_bf16_kernel_v14_mfma,
    _qwen_gdn_w_bf16_kernel_v14_mfma,
    qwen_gdn_chunk_gdr_avelang_v17_4wave_predecay_mfma_layout,
    qwen_gdn_chunk_o_avelang_v14_mfma_layout,
    qwen_gdn_chunked_avelang_v17_predecay_mfma_layout,
    qwen_gdn_gdr_decay_avelang_v17,
    qwen_gdn_w_u_avelang_v14_mfma_layout,
)
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_layout
from qwen_gdn_chunked_avelang_v19_bt32_mfma_layout_fixed import (
    _qwen_gdn_u_bf16_kernel_v19_bt32_mfma,
    _qwen_gdn_w_bf16_kernel_v19_bt32_mfma,
    qwen_gdn_chunk_gdr_avelang_v19_bt32_mfma_layout,
    qwen_gdn_chunked_avelang_v19_bt32_mfma_layout,
    qwen_gdn_w_u_avelang_v19_bt32_mfma_layout,
)
from qwen_gdn_chunked_avelang_v13_mfma_layout_fixed import qwen_gdn_chunk_o_avelang_v13_mfma_layout


def sync() -> None:
    torch.cuda.synchronize()


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def make_inputs(t: int, seed: int = 19700, with_initial_state: bool = True):
    torch.manual_seed(seed + t)
    b, hk, hv, kdim, vdim = 1, 4, 8, 128, 128
    q = l2norm(torch.randn(b, t, hk, kdim, device="cuda", dtype=torch.bfloat16))
    k = l2norm(torch.randn(b, t, hk, kdim, device="cuda", dtype=torch.bfloat16))
    v = torch.randn(b, t, hv, vdim, device="cuda", dtype=torch.bfloat16).contiguous()
    g = (torch.nn.functional.logsigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)) / 16.0).contiguous()
    beta = torch.sigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)).contiguous()
    initial_state = None
    if with_initial_state:
        initial_state = (torch.randn(b, hv, vdim, kdim, device="cuda", dtype=torch.float32) * 0.01).contiguous()
    return q, k, v, g, beta, initial_state


def time_fn(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    sync()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        sync()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def build_common(q, k, v, g, beta, initial_state, version: str):
    if version == "v17":
        chunk = 16
        g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk)
        a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk, prefer_optimized=True)
        a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk)
        w, u = qwen_gdn_w_u_avelang_v14_mfma_layout(k, v, g_cumsum, beta, a_solved, chunk_size=chunk)
        gdr_decay, gdr_g_last_exp = qwen_gdn_gdr_decay_avelang_v17(g_cumsum, chunk_size=chunk)
        h, vn, final_state = qwen_gdn_chunk_gdr_avelang_v17_4wave_predecay_mfma_layout(
            k,
            w,
            u,
            g_cumsum,
            gdr_decay,
            gdr_g_last_exp,
            initial_state=initial_state,
            chunk_size=chunk,
        )
        return chunk, g_cumsum, a, a_solved, w, u, h, vn, final_state, gdr_decay, gdr_g_last_exp
    if version == "v19":
        chunk = 32
        g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk)
        a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk, prefer_optimized=True)
        a_solved = qwen_gdn_solve_avelang_v18_layout(a, chunk_size=chunk)
        w, u = qwen_gdn_w_u_avelang_v19_bt32_mfma_layout(k, v, g_cumsum, beta, a_solved, chunk_size=chunk)
        h, vn, final_state = qwen_gdn_chunk_gdr_avelang_v19_bt32_mfma_layout(
            k,
            w,
            u,
            g_cumsum,
            initial_state=initial_state,
            chunk_size=chunk,
        )
        return chunk, g_cumsum, a, a_solved, w, u, h, vn, final_state, None, None
    raise ValueError(version)


def full_fn(q, k, v, g, beta, initial_state, version: str):
    scale = 128 ** -0.5
    if version == "v17":
        return lambda: qwen_gdn_chunked_avelang_v17_predecay_mfma_layout(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            scale=scale,
            chunk_size=16,
        )
    if version == "v19":
        return lambda: qwen_gdn_chunked_avelang_v19_bt32_mfma_layout(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            scale=scale,
            chunk_size=32,
        )
    raise ValueError(version)


def stage_times(q, k, v, g, beta, initial_state, version: str, warmup: int, repeat: int):
    scale = 128 ** -0.5
    if version == "v17":
        chunk = 16
        cumsum = lambda: qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk)
        g_cumsum = cumsum()
        kkt = lambda: qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk, prefer_optimized=True)
        a = kkt()
        solve = lambda: qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk)
        a_solved = solve()
        wu = lambda: qwen_gdn_w_u_avelang_v14_mfma_layout(k, v, g_cumsum, beta, a_solved, chunk_size=chunk)
        w, u = wu()
        decay = lambda: qwen_gdn_gdr_decay_avelang_v17(g_cumsum, chunk_size=chunk)
        gdr_decay, gdr_g_last_exp = decay()
        gdr = lambda: qwen_gdn_chunk_gdr_avelang_v17_4wave_predecay_mfma_layout(
            k,
            w,
            u,
            g_cumsum,
            gdr_decay,
            gdr_g_last_exp,
            initial_state=initial_state,
            chunk_size=chunk,
        )
        h, vn, _ = gdr()
        chunk_o = lambda: qwen_gdn_chunk_o_avelang_v14_mfma_layout(q, k, vn, h, g_cumsum, scale=scale, chunk_size=chunk)
    else:
        chunk = 32
        cumsum = lambda: qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk)
        g_cumsum = cumsum()
        kkt = lambda: qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk, prefer_optimized=True)
        a = kkt()
        solve = lambda: qwen_gdn_solve_avelang_v18_layout(a, chunk_size=chunk)
        a_solved = solve()
        wu = lambda: qwen_gdn_w_u_avelang_v19_bt32_mfma_layout(k, v, g_cumsum, beta, a_solved, chunk_size=chunk)
        w, u = wu()
        decay = None
        gdr = lambda: qwen_gdn_chunk_gdr_avelang_v19_bt32_mfma_layout(k, w, u, g_cumsum, initial_state=initial_state, chunk_size=chunk)
        h, vn, _ = gdr()
        chunk_o = lambda: qwen_gdn_chunk_o_avelang_v13_mfma_layout(q, k, vn, h, g_cumsum, scale=scale, chunk_size=chunk)

    result = {
        "full": time_fn(full_fn(q, k, v, g, beta, initial_state, version), warmup, repeat),
        "cumsum": time_fn(cumsum, warmup, repeat),
        "KKT": time_fn(kkt, warmup, repeat),
        "solve": time_fn(solve, warmup, repeat),
        "w_u": time_fn(wu, warmup, repeat),
        "gdr_decay": 0.0 if decay is None else time_fn(decay, warmup, repeat),
        "chunk_gdr": time_fn(gdr, warmup, repeat),
        "chunk_o": time_fn(chunk_o, warmup, repeat),
    }
    return result


def run_benchmark(args) -> None:
    for t in args.T:
        q, k, v, g, beta, initial_state = make_inputs(t)
        v17 = stage_times(q, k, v, g, beta, initial_state, "v17", args.warmup, args.repeat)
        v19 = stage_times(q, k, v, g, beta, initial_state, "v19", args.warmup, args.repeat)
        full_delta = v19["full"] - v17["full"]
        print(f"benchmark,T={t},full_delta={full_delta:.6f}")
        for stage in ("full", "cumsum", "KKT", "solve", "w_u", "gdr_decay", "chunk_gdr", "chunk_o"):
            delta = v19[stage] - v17[stage]
            contribution = 0.0 if full_delta == 0.0 else delta / full_delta * 100.0
            print(
                "stage_delta,"
                f"T={t},stage={stage},"
                f"v17={v17[stage]:.6f},"
                f"v19={v19[stage]:.6f},"
                f"delta={delta:.6f},"
                f"full_regression_contribution_pct={contribution:.2f}"
            )


def run_profile_stage(args) -> None:
    q, k, v, g, beta, initial_state = make_inputs(args.T[0])
    chunk, g_cumsum, a, a_solved, w, u, h, vn, _, gdr_decay, gdr_g_last_exp = build_common(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        args.version,
    )
    num_tokens = k.shape[1]
    num_chunks = num_tokens // chunk
    scale = 128 ** -0.5

    if args.stage == "KKT":
        fn = lambda: qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk, prefer_optimized=True)
    elif args.stage == "w_u_w":
        if args.version == "v17":
            out = torch.empty((1, num_tokens, 8, 128), dtype=torch.float32, device=k.device)
            grid_size = num_chunks * 8 * 8
            fn = lambda: _qwen_gdn_w_bf16_kernel_v14_mfma[lambda: ((grid_size, 1, 1), (64, 1, 1))](
                k,
                g_cumsum,
                beta,
                a_solved,
                out,
                num_tokens,
                num_chunks,
            )
        else:
            out = torch.empty((1, num_tokens, 8, 128), dtype=torch.float32, device=k.device)
            grid_size = num_chunks * 8 * 8 * 2
            fn = lambda: _qwen_gdn_w_bf16_kernel_v19_bt32_mfma[lambda: ((grid_size, 1, 1), (64, 1, 1))](
                k,
                g_cumsum,
                beta,
                a_solved,
                out,
                num_tokens,
                num_chunks,
            )
    elif args.stage == "w_u_u":
        if args.version == "v17":
            out = torch.empty((1, num_tokens, 8, 128), dtype=torch.float32, device=k.device)
            grid_size = num_chunks * 8 * 8
            fn = lambda: _qwen_gdn_u_bf16_kernel_v14_mfma[lambda: ((grid_size, 1, 1), (64, 1, 1))](
                v,
                beta,
                a_solved,
                out,
                num_tokens,
                num_chunks,
            )
        else:
            out = torch.empty((1, num_tokens, 8, 128), dtype=torch.float32, device=k.device)
            grid_size = num_chunks * 8 * 8 * 2
            fn = lambda: _qwen_gdn_u_bf16_kernel_v19_bt32_mfma[lambda: ((grid_size, 1, 1), (64, 1, 1))](
                v,
                beta,
                a_solved,
                out,
                num_tokens,
                num_chunks,
            )
    elif args.stage == "chunk_gdr":
        if args.version == "v17":
            fn = lambda: qwen_gdn_chunk_gdr_avelang_v17_4wave_predecay_mfma_layout(
                k,
                w,
                u,
                g_cumsum,
                gdr_decay,
                gdr_g_last_exp,
                initial_state=initial_state,
                chunk_size=chunk,
            )
        else:
            fn = lambda: qwen_gdn_chunk_gdr_avelang_v19_bt32_mfma_layout(
                k,
                w,
                u,
                g_cumsum,
                initial_state=initial_state,
                chunk_size=chunk,
            )
    elif args.stage == "chunk_o":
        if args.version == "v17":
            fn = lambda: qwen_gdn_chunk_o_avelang_v14_mfma_layout(q, k, vn, h, g_cumsum, scale=scale, chunk_size=chunk)
        else:
            fn = lambda: qwen_gdn_chunk_o_avelang_v13_mfma_layout(q, k, vn, h, g_cumsum, scale=scale, chunk_size=chunk)
    else:
        raise ValueError(args.stage)

    for _ in range(args.warmup):
        fn()
    sync()
    for _ in range(args.repeat):
        fn()
    sync()
    print(f"profile_done,version={args.version},stage={args.stage},T={args.T[0]},repeat={args.repeat}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["benchmark", "profile"], default="benchmark")
    parser.add_argument("--version", choices=["v17", "v19"], default="v19")
    parser.add_argument("--stage", choices=["KKT", "w_u_w", "w_u_u", "chunk_gdr", "chunk_o"], default="KKT")
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP GPU is required")
    if args.mode == "benchmark":
        run_benchmark(args)
    else:
        run_profile_stage(args)


if __name__ == "__main__":
    main()
