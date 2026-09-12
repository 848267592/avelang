from __future__ import annotations

import argparse
import statistics

import torch

from qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full import (
    _make_inputs,
    qwen_gdn_fused_chunk_gdr_full_avelang_v29_mfma32,
)
from qwen_gdn_chunked_avelang_v29_mfma32_pred_epilogue_streaming_exp import (
    qwen_gdn_fused_chunk_gdr_full_avelang_v29_mfma32_pred_epilogue_streaming,
)


def median_ms(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    print("T,original_ms,streaming_ms,speedup,h_max_abs,state_max_abs")
    for num_tokens in args.T:
        k, w, u, decay, g_last, initial_state = _make_inputs(num_tokens, seed=args.seed)
        original = lambda: qwen_gdn_fused_chunk_gdr_full_avelang_v29_mfma32(
            k, w, u, decay, g_last, initial_state
        )
        streaming = lambda: qwen_gdn_fused_chunk_gdr_full_avelang_v29_mfma32_pred_epilogue_streaming(
            k, w, u, decay, g_last, initial_state
        )
        h_original, state_original = original()
        h_stream, state_stream = streaming()
        torch.cuda.synchronize()
        original_ms = median_ms(original, args.warmup, args.repeat)
        streaming_ms = median_ms(streaming, args.warmup, args.repeat)
        print(
            f"{num_tokens},{original_ms:.6f},{streaming_ms:.6f},"
            f"{original_ms / streaming_ms:.4f},"
            f"{(h_stream - h_original).abs().max().item():.8e},"
            f"{(state_stream - state_original).abs().max().item():.8e}"
        )


if __name__ == "__main__":
    main()
