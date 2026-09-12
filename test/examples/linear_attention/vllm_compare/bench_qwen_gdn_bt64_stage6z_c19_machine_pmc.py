#!/usr/bin/env python3
"""Preallocated C19 worker for diagnostic rocprof collection."""

from __future__ import annotations

import argparse
import json

import torch

from qwen_gdn_bt64_native_chunko_stage6z_c19_full_physical_region import (
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c19_full_physical_region_launch_into,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if args.T < 64 or args.T % 64:
        raise ValueError("T must be divisible by 64")
    torch.manual_seed(1919 + args.T)
    chunks = args.T // 64
    q = torch.randn((1, args.T, 4, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    vn = torch.randn((1, args.T, 8, 128), device="cuda", dtype=torch.bfloat16)
    h = torch.randn((1, chunks, 8, 128, 128), device="cuda", dtype=torch.bfloat16)
    g = (0.01 * torch.randn((1, args.T, 8), device="cuda", dtype=torch.float32)).contiguous()
    output = torch.empty_like(vn)

    def launch() -> None:
        qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c19_full_physical_region_launch_into(
            q, k, vn, h, g, output,
            lowering="specialized", planner="bdv2_p1_affine",
            preservation="p2_first_class",
        )

    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    for _ in range(args.repeat):
        launch()
    torch.cuda.synchronize()
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("C19 produced non-finite output")
    print(json.dumps({
        "arm": "c19_full_physical_region",
        "T": args.T,
        "chunks": chunks,
        "workgroup": 256,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "cuda_graph_used": False,
        "caller_owned_output": True,
        "finite": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
