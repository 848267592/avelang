#!/usr/bin/env python3
"""Run standalone v6 repeatedly as a rocprofv3 profiling target."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from qwen_gdn_chunked_avelang_v6_standalone import qwen_gdn_chunked_avelang_v6_standalone
from qwen_gdn_v7_benchmark import LARGER_DEBUG_SHAPE, MEDIUM_SHAPE, SMALL_SHAPE, _make_inputs


@dataclass(frozen=True)
class ProfileShape:
    name: str
    batch_size: int
    num_tokens: int
    num_k_heads: int
    num_v_heads: int
    head_dim_k: int
    head_dim_v: int
    chunk_size: int


LARGE_V_SHAPE = ProfileShape("large_v", 1, 128, 2, 4, 16, 64, 8)
LARGER_TV_SHAPE = ProfileShape("larger_tv", 1, 256, 4, 8, 32, 64, 16)

SHAPES = {
    "small": SMALL_SHAPE,
    "medium": MEDIUM_SHAPE,
    "larger_debug": LARGER_DEBUG_SHAPE,
    "large_v": LARGE_V_SHAPE,
    "larger_tv": LARGER_TV_SHAPE,
}


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone v6 as a rocprofv3 target.")
    parser.add_argument("--shape", choices=tuple(SHAPES), default="larger_debug")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    shape = SHAPES[args.shape]
    dtype = _dtype_from_name(args.dtype)
    q, k, v, g, beta = _make_inputs(shape, dtype, seed=args.seed)
    scale = shape.head_dim_k**-0.5

    for _ in range(args.warmup):
        qwen_gdn_chunked_avelang_v6_standalone(q, k, v, g, beta, scale=scale, chunk_size=shape.chunk_size)
    torch.cuda.synchronize()

    for _ in range(args.repeat):
        qwen_gdn_chunked_avelang_v6_standalone(q, k, v, g, beta, scale=scale, chunk_size=shape.chunk_size)
    torch.cuda.synchronize()

    print(
        "PROFILED standalone v6 "
        f"shape={args.shape} dtype={args.dtype} warmup={args.warmup} repeat={args.repeat}"
    )


if __name__ == "__main__":
    main()
