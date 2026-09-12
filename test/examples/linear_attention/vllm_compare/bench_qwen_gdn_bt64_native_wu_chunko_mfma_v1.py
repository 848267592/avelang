#!/usr/bin/env python3
"""HIP-event benchmarks for opt-in BT64 native MFMA W/U and chunk-o."""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch

from qwen_gdn_bt64_native_wu_chunko_mfma_v1 import (
    qwen_gdn_chunk_o_bt64_from_v24_mfma_v1,
    qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1,
    qwen_gdn_w_u_bt64_from_v24_mfma_v1,
)
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_bt64_layout
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_chunk_o_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)
from qwen_gdn_full_bt64_gfx942_asm_v0_experimental import qwen_gdn_full_bt64_gfx942_asm_v0


def _inputs(t: int, seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed + t)
    q = (torch.randn((1, t, 4, 128), device="cuda") * 0.05).to(torch.bfloat16).contiguous()
    k = (torch.randn((1, t, 4, 128), device="cuda") * 0.05).to(torch.bfloat16).contiguous()
    v = (torch.randn((1, t, 8, 128), device="cuda") * 0.05).to(torch.bfloat16).contiguous()
    g = (torch.randn((1, t, 8), device="cuda") * 0.01).float().contiguous()
    beta = (0.5 + torch.rand((1, t, 8), device="cuda")).float().contiguous()
    h0 = (torch.randn((1, 8, 128, 128), device="cuda") * 0.01).float().contiguous()
    return q, k, v, g, beta, h0


def _time_ms(fn: Callable[[], object], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--full", action="store_true", help="also time the opt-in full Stage 3 graph")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires a ROCm/CUDA GPU")

    print("T,stage,fallback_ms,native_ms,speedup")
    for t in args.T:
        q, k, v, g, beta, h0 = _inputs(t, 20260713)
        g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=64)
        a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=64, prefer_optimized=True)
        a_solved = qwen_gdn_solve_avelang_v18_bt64_layout(a)
        w, u = qwen_gdn_w_u_bt64_from_v24_mfma_v1(k, v, g_cumsum, beta, a_solved)
        # The chunk-o microbenchmark uses a real native W/U preparation but its
        # state/v_new operands are independent, as required for a standalone gate.
        h_bf16 = (torch.randn((1, t // 64, 8, 128, 128), device="cuda") * 0.01).to(torch.bfloat16).contiguous()
        v_new = u
        wu_fallback = _time_ms(
            lambda: qwen_gdn_w_u_avelang_v6_standalone(k, v, g_cumsum, beta, a_solved, chunk_size=64),
            args.warmup,
            args.repeat,
        )
        wu_native = _time_ms(
            lambda: qwen_gdn_w_u_bt64_from_v24_mfma_v1(k, v, g_cumsum, beta, a_solved), args.warmup, args.repeat
        )
        o_fallback = _time_ms(
            lambda: qwen_gdn_chunk_o_avelang_v6_standalone(q, k, v_new, h_bf16.float().contiguous(), g_cumsum, chunk_size=64),
            args.warmup,
            args.repeat,
        )
        o_native = _time_ms(
            lambda: qwen_gdn_chunk_o_bt64_from_v24_mfma_v1(q, k, v_new, h_bf16, g_cumsum), args.warmup, args.repeat
        )
        print(f"{t},w_u,{wu_fallback:.6f},{wu_native:.6f},{wu_fallback / wu_native:.4f}")
        print(f"{t},chunk_o,{o_fallback:.6f},{o_native:.6f},{o_fallback / o_native:.4f}")
        if args.full:
            stage2 = _time_ms(
                lambda: qwen_gdn_full_bt64_gfx942_asm_v0(q, k, v, g, beta, initial_state=h0, output_final_state=True),
                args.warmup,
                args.repeat,
            )
            stage3 = _time_ms(
                lambda: qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1(q, k, v, g, beta, initial_state=h0, output_final_state=True),
                args.warmup,
                args.repeat,
            )
            print(f"{t},full_stage3,{stage2:.6f},{stage3:.6f},{stage2 / stage3:.4f}")


if __name__ == "__main__":
    main()
