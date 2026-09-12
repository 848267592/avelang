#!/usr/bin/env python3
"""Benchmark v9 chunk_o vk on Qwen3Next TP4 per-rank shape."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import torch

from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_chunk_o_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v9_vllm_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v9_vllm_layout,
    qwen_gdn_chunk_o_avelang_v9_vllm_layout,
    qwen_gdn_chunked_avelang_v9_vllm_layout,
)


def sync() -> None:
    torch.cuda.synchronize()


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_f = x.float()
    return (x_f * torch.rsqrt((x_f * x_f).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def make_inputs(args: argparse.Namespace):
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    q = l2norm(torch.randn(args.B, args.T, args.Hk, args.K, device="cuda", dtype=torch.bfloat16, generator=generator))
    k = l2norm(torch.randn(args.B, args.T, args.Hk, args.K, device="cuda", dtype=torch.bfloat16, generator=generator))
    v = torch.randn(args.B, args.T, args.Hv, args.V, device="cuda", dtype=torch.bfloat16, generator=generator).contiguous()
    g = torch.nn.functional.logsigmoid(
        torch.randn(args.B, args.T, args.Hv, device="cuda", dtype=torch.float32, generator=generator)
    )
    g = (g / 16.0).contiguous()
    beta = torch.sigmoid(
        torch.randn(args.B, args.T, args.Hv, device="cuda", dtype=torch.float32, generator=generator)
    ).contiguous()
    initial_state = (
        torch.randn(args.B, args.Hv, args.V, args.K, device="cuda", dtype=torch.float32, generator=generator) * 0.01
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


def max_rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = expected.float().abs().clamp_min(1e-6)
    return ((actual.float() - expected.float()).abs() / denom).max().item()


def maybe_import_vllm():
    try:
        from vllm.model_executor.layers.fla.ops import chunk_delta_h
        from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_chunk_gdn
    except Exception as exc:  # pragma: no cover
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
                print(f"patched_vllm_rocm_autotune,{len(configs)},{len(filtered)}")
    return vllm_chunk_gdn


def build_stages(args: argparse.Namespace):
    q, k, v, g, beta, initial_state = make_inputs(args)
    scale = args.K**-0.5
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=args.chunk)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=args.chunk, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=args.chunk)
    w, u = qwen_gdn_w_u_avelang_v6_standalone(k, v, g_cumsum, beta, a_solved, chunk_size=args.chunk, prefer_optimized=True)
    h, vn, final_state = qwen_gdn_chunk_gdr_avelang_v9_vllm_layout(
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
    return q, k, v, g, beta, initial_state, scale, g_cumsum, a, a_solved, w, u, h, vn, final_state


def run_one(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is not available")
    q, k, v, g, beta, initial_state, scale, g_cumsum, a, a_solved, w, u, h, vn, final_state = build_stages(args)
    print(
        "shape,"
        f"q={tuple(q.shape)},k={tuple(k.shape)},v={tuple(v.shape)},vn={tuple(vn.shape)},"
        f"h={tuple(h.shape)},out={(args.B, args.T, args.Hv, args.V)},chunk={args.chunk}"
    )
    print(
        "config,"
        f"chunk_gdr_block_v={args.vk_block_v},chunk_gdr_block_k={args.vk_block_k},"
        f"chunk_o_block_v={args.chunk_o_block_v},chunk_o_block_k={args.chunk_o_block_k},"
        f"chunk_o_workgroup={args.chunk_o_block_v * args.chunk_o_block_k}"
    )

    out_v6 = qwen_gdn_chunk_o_avelang_v6_standalone(q, k, vn, h, g_cumsum, scale=scale, chunk_size=args.chunk, prefer_optimized=True)
    out_v9 = qwen_gdn_chunk_o_avelang_v9_vllm_layout(
        q,
        k,
        vn,
        h,
        g_cumsum,
        scale=scale,
        chunk_size=args.chunk,
        use_parallel_chunk_o=True,
        chunk_o_parallel_mode=args.chunk_o_parallel_mode,
        chunk_o_block_v=args.chunk_o_block_v,
        chunk_o_block_k=args.chunk_o_block_k,
    )
    sync()
    out_err = max_abs_err(out_v9, out_v6)
    out_rel = max_rel_err(out_v9, out_v6)
    status = "ok" if out_err <= args.err_atol else "wrong_result"
    print(f"stage_error_vs_v6,output_max_abs={out_err:.9g},output_max_rel={out_rel:.9g},status={status}")

    breakdown = [
        ("cumsum", lambda: qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=args.chunk)),
        ("KKT", lambda: qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=args.chunk, prefer_optimized=True)),
        ("solve", lambda: qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=args.chunk)),
        ("w_u", lambda: qwen_gdn_w_u_avelang_v6_standalone(k, v, g_cumsum, beta, a_solved, chunk_size=args.chunk, prefer_optimized=True)),
        (
            "chunk_gdr",
            lambda: qwen_gdn_chunk_gdr_avelang_v9_vllm_layout(
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
            "chunk_o_v6",
            lambda: qwen_gdn_chunk_o_avelang_v6_standalone(q, k, vn, h, g_cumsum, scale=scale, chunk_size=args.chunk, prefer_optimized=True),
        ),
        (
            "chunk_o_v9_vk",
            lambda: qwen_gdn_chunk_o_avelang_v9_vllm_layout(
                q,
                k,
                vn,
                h,
                g_cumsum,
                scale=scale,
                chunk_size=args.chunk,
                use_parallel_chunk_o=True,
                chunk_o_parallel_mode=args.chunk_o_parallel_mode,
                chunk_o_block_v=args.chunk_o_block_v,
                chunk_o_block_k=args.chunk_o_block_k,
            ),
        ),
    ]
    for name, fn in breakdown:
        elapsed = time_ms(fn, warmup=args.warmup, repeat=args.repeat)
        print(f"breakdown_median_ms,{name},{elapsed:.9g}")

    full_prev_ms = time_ms(
        lambda: qwen_gdn_chunked_avelang_v9_vllm_layout(
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
            use_parallel_chunk_o=False,
        ),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    full_v9_ms = time_ms(
        lambda: qwen_gdn_chunked_avelang_v9_vllm_layout(
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
            use_parallel_chunk_o=True,
            chunk_o_parallel_mode=args.chunk_o_parallel_mode,
            chunk_o_block_v=args.chunk_o_block_v,
            chunk_o_block_k=args.chunk_o_block_k,
        ),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    print(f"full_median_ms,previous_v8_gdr_v6_chunk_o,{full_prev_ms:.9g}")
    print(f"full_median_ms,avelang_v9,{full_v9_ms:.9g}")
    print(f"full_speedup_v9_over_previous,{full_prev_ms / full_v9_ms:.9g}")

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
            full_v9 = qwen_gdn_chunked_avelang_v9_vllm_layout(
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
                use_parallel_chunk_o=True,
                chunk_o_parallel_mode=args.chunk_o_parallel_mode,
                chunk_o_block_v=args.chunk_o_block_v,
                chunk_o_block_k=args.chunk_o_block_k,
            )
            sync()
            print(f"full_error_vs_vllm,output={max_abs_err(full_v9[0], o_vllm):.9g},final_state={max_abs_err(full_v9[1], s_vllm):.9g}")
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
            print(f"full_speedup_vllm_over_avelang_v9,{vllm_ms / full_v9_ms:.9g}")


def run_sweep(args: argparse.Namespace) -> None:
    cases = [(bv, bk) for bv in (2, 4, 8, 16) for bk in (16, 32, 64) if bv * bk <= 256]
    cases.extend([(8, 64), (16, 32)])
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    for bv, bk in cases:
        cmd = [
            sys.executable,
            __file__,
            "--B",
            str(args.B),
            "--T",
            str(args.T),
            "--Hk",
            str(args.Hk),
            "--Hv",
            str(args.Hv),
            "--K",
            str(args.K),
            "--V",
            str(args.V),
            "--chunk",
            str(args.chunk),
            "--vk-block-v",
            str(args.vk_block_v),
            "--vk-block-k",
            str(args.vk_block_k),
            "--chunk-o-block-v",
            str(bv),
            "--chunk-o-block-k",
            str(bk),
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            "--seed",
            str(args.seed),
        ]
        if args.include_vllm:
            cmd.append("--include-vllm")
        print("=" * 100)
        print(f"sweep_case,chunk_o_block_v={bv},chunk_o_block_k={bk},workgroup={bv * bk}")
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        log_path = log_dir / f"chunk_o_bv{bv}_bk{bk}.log"
        log_path.write_text(result.stdout)
        print(result.stdout)
        print(f"sweep_status,chunk_o_block_v={bv},chunk_o_block_k={bk},returncode={result.returncode},log={log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--T", type=int, default=512)
    parser.add_argument("--Hk", type=int, default=4)
    parser.add_argument("--Hv", type=int, default=8)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--V", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--vk-block-v", type=int, default=4)
    parser.add_argument("--vk-block-k", type=int, default=64)
    parser.add_argument("--chunk-o-parallel-mode", default="vk")
    parser.add_argument("--chunk-o-block-v", type=int, default=4)
    parser.add_argument("--chunk-o-block-k", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--err-atol", type=float, default=1e-3)
    parser.add_argument("--include-vllm", action="store_true")
    parser.add_argument("--sweep-chunk-o", action="store_true")
    parser.add_argument("--log-dir", default="qwen_gdn_v9_chunk_o_vk_logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sweep_chunk_o:
        run_sweep(args)
    else:
        run_one(args)


if __name__ == "__main__":
    main()
