from __future__ import annotations

import argparse
import statistics

import torch

from qwen_gdn_chunked_avelang_v30_bt64_bv32_hierarchical import (
    _make_inputs,
    qwen_gdn_fused_chunk_gdr_full_avelang_v30_bt64_bv32_hierarchical,
)


def median_ms(fn, warmup: int, repeat: int) -> float:
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
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    print("T,d1_chunk_gdr_ms,d1_us_per_bt64_chunk,valid_for_comparison")
    for num_tokens in args.T:
        data = _make_inputs(num_tokens, seed=20260711)
        latency = median_ms(
            lambda: qwen_gdn_fused_chunk_gdr_full_avelang_v30_bt64_bv32_hierarchical(*data),
            args.warmup,
            args.repeat,
        )
        print(f"{num_tokens},{latency:.6f},{latency * 1000 / (num_tokens // 64):.6f},false")


if __name__ == "__main__":
    main()
