#!/usr/bin/env python3
"""Single-arm T=2048 kernel launcher for C21 dynamic PMC collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

from bench_qwen_gfx942_c21_selected_native_t2048 import (  # noqa: E402
    ARMS,
    FROZEN,
    T,
    _inputs,
    _launch_frozen,
    _launch_native_selected,
    _load_bridge,
    _load_native_bridge,
)
from qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--bridge",
        type=Path,
        default=REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_gfx942_c20_formal_performance/libc20_avelang_hsaco_bridge.so",
    )
    parser.add_argument("--native-bridge", type=Path, default=HERE / "libc21_selected_native_hsaco_bridge.so")
    args = parser.parse_args()
    tensors = _inputs(2026082500)
    output = torch.empty_like(tensors[2])
    bridge = _load_bridge(args.bridge)
    native_bridge = _load_native_bridge(args.native_bridge)

    def launch() -> None:
        if args.arm == "z5b":
            qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into(*tensors, output)
        elif args.arm == "native_selected":
            _launch_native_selected(native_bridge, tensors, output)
        else:
            _launch_frozen(bridge, args.arm, tensors, output)

    for _ in range(args.warmup):
        launch()
    for _ in range(args.repeat):
        launch()
    torch.cuda.synchronize()
    print(json.dumps({
        "arm": args.arm,
        "T": T,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "finite": bool(torch.isfinite(output).all().item()),
        "frozen_identity": FROZEN.get(args.arm),
    }, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
