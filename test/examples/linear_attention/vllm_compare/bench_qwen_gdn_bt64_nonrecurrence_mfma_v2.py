#!/usr/bin/env python3
"""HIP-event microbenchmarks for Stage 4 BT64 non-recurrence kernels."""

from __future__ import annotations

import argparse
import statistics

import torch

from qwen_gdn_bt64_native_wu_chunko_mfma_v1 import (
    qwen_gdn_chunk_o_bt64_from_v24_mfma_v1,
    qwen_gdn_w_u_bt64_from_v24_mfma_v1,
)
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import (
    qwen_gdn_chunk_o_bt64_mfma_v2_s0,
    qwen_gdn_kkt_bt64_mfma_v2_s0,
    qwen_gdn_w_u_bt64_mfma_v2_s0,
    qwen_gdn_w_u_bt64_mfma_v2_s1,
)
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_bt64_layout
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
)


def _time_ms(fn, warmup: int, repeat: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    samples.sort()
    return statistics.median(samples), samples[(len(samples) - 1) // 10], samples[(len(samples) - 1) * 9 // 10]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", nargs="+", type=int, default=[512, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()
    print("T,stage,stage3_ms,stage4_s0_ms,speedup,stage3_p10,stage3_p90,stage4_p10,stage4_p90")
    for t in args.T:
        torch.manual_seed(20260715 + t)
        q = (torch.randn((1, t, 4, 128), device="cuda") * 0.05).to(torch.bfloat16).contiguous()
        k = (torch.randn((1, t, 4, 128), device="cuda") * 0.05).to(torch.bfloat16).contiguous()
        v = (torch.randn((1, t, 8, 128), device="cuda") * 0.05).to(torch.bfloat16).contiguous()
        g = (torch.randn((1, t, 8), device="cuda") * 0.01).float().contiguous()
        beta = (0.5 + torch.rand((1, t, 8), device="cuda")).float().contiguous()
        g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=64)
        baseline = _time_ms(
            lambda: qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=64),
            args.warmup,
            args.repeat,
        )
        candidate = _time_ms(
            lambda: qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta), args.warmup, args.repeat
        )
        print(
            f"{t},KKT,{baseline[0]:.6f},{candidate[0]:.6f},{baseline[0] / candidate[0]:.4f},"
            f"{baseline[1]:.6f},{baseline[2]:.6f},{candidate[1]:.6f},{candidate[2]:.6f}"
        )
        a = qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
        a_solved = qwen_gdn_solve_avelang_v18_bt64_layout(a)
        baseline = _time_ms(
            lambda: qwen_gdn_w_u_bt64_from_v24_mfma_v1(k, v, g_cumsum, beta, a_solved),
            args.warmup,
            args.repeat,
        )
        candidate = _time_ms(
            lambda: qwen_gdn_w_u_bt64_mfma_v2_s0(k, v, g_cumsum, beta, a_solved),
            args.warmup,
            args.repeat,
        )
        print(
            f"{t},W_U,{baseline[0]:.6f},{candidate[0]:.6f},{baseline[0] / candidate[0]:.4f},"
            f"{baseline[1]:.6f},{baseline[2]:.6f},{candidate[1]:.6f},{candidate[2]:.6f}"
        )
        candidate_s1 = _time_ms(
            lambda: qwen_gdn_w_u_bt64_mfma_v2_s1(k, v, g_cumsum, beta, a_solved),
            args.warmup,
            args.repeat,
        )
        print(
            f"{t},W_U_S1,{baseline[0]:.6f},{candidate_s1[0]:.6f},{baseline[0] / candidate_s1[0]:.4f},"
            f"{baseline[1]:.6f},{baseline[2]:.6f},{candidate_s1[1]:.6f},{candidate_s1[2]:.6f}"
        )
        h = (torch.randn((1, t // 64, 8, 128, 128), device="cuda") * 0.01).to(torch.bfloat16).contiguous()
        v_new = qwen_gdn_w_u_bt64_mfma_v2_s1(k, v, g_cumsum, beta, a_solved)[1]
        baseline = _time_ms(
            lambda: qwen_gdn_chunk_o_bt64_from_v24_mfma_v1(q, k, v_new, h, g_cumsum),
            args.warmup,
            args.repeat,
        )
        candidate = _time_ms(
            lambda: qwen_gdn_chunk_o_bt64_mfma_v2_s0(q, k, v_new, h, g_cumsum),
            args.warmup,
            args.repeat,
        )
        print(
            f"{t},chunk_o,{baseline[0]:.6f},{candidate[0]:.6f},{baseline[0] / candidate[0]:.4f},"
            f"{baseline[1]:.6f},{baseline[2]:.6f},{candidate[1]:.6f},{candidate[2]:.6f}"
        )


if __name__ == "__main__":
    main()
