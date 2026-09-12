from __future__ import annotations

import argparse
import statistics

import torch

from qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full import (
    _make_inputs,
    qwen_gdn_fused_chunk_gdr_full_avelang_v29_mfma32 as original,
)
from qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_kfrag_rewrite_exp import (
    qwen_gdn_fused_chunk_gdr_full_avelang_v29_mfma32 as rewrite,
)
from qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_kfrag_pred_streaming_exp import (
    qwen_gdn_fused_chunk_gdr_full_kfrag_pred_streaming_avelang_v29_mfma32 as streaming,
)


def median_ms(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)))
    return statistics.median(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    print("T,original_ms,rewrite_ms,rewrite_streaming_ms,stream_vs_rewrite,h_max_abs,state_max_abs")
    for num_tokens in args.T:
        data = _make_inputs(num_tokens, seed=20260711)
        a = lambda: original(*data)
        b = lambda: rewrite(*data)
        c = lambda: streaming(*data)
        h_b, state_b = b()
        h_c, state_c = c()
        torch.cuda.synchronize()
        a_ms = median_ms(a, args.warmup, args.repeat)
        b_ms = median_ms(b, args.warmup, args.repeat)
        c_ms = median_ms(c, args.warmup, args.repeat)
        print(
            f"{num_tokens},{a_ms:.6f},{b_ms:.6f},{c_ms:.6f},{b_ms / c_ms:.4f},"
            f"{(h_b - h_c).abs().max().item():.8e},{(state_b - state_c).abs().max().item():.8e}"
        )


if __name__ == "__main__":
    main()
