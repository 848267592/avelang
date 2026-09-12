#!/usr/bin/env python3
"""Rocprof driver: prepare once, then dispatch exactly one BT64 candidate stage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[5] / "test/examples/linear_attention/vllm_compare"))
from qwen_gdn_bt64_gfx942_asm_v0_experimental import qwen_gdn_bt64_gfx942_asm_v0_preallocated
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_bt64_layout
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_chunk_o_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)
from qwen_gdn_full_bt64_gfx942_asm_v0_experimental import qwen_gdn_full_bt64_gfx942_asm_v0_stages
from stage2_runner import make_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("cumsum", "kkt", "solve", "wu", "asm", "chunk_o"), required=True)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    q, k, v, g, beta, h0 = make_inputs(args.T, 20261100 + args.T, "random", True)
    st = qwen_gdn_full_bt64_gfx942_asm_v0_stages(q, k, v, g, beta, initial_state=h0)
    h = torch.empty_like(st["h_bf16"]); vn = torch.empty_like(st["u"]); ht = torch.empty_like(st["final_state"])
    funcs = {
        "cumsum": lambda: qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=64),
        "kkt": lambda: qwen_gdn_kkt_avelang_v6_standalone(k, st["g_cumsum"], beta, chunk_size=64),
        "solve": lambda: qwen_gdn_solve_avelang_v18_bt64_layout(st["a"], chunk_size=64),
        "wu": lambda: qwen_gdn_w_u_avelang_v6_standalone(k, v, st["g_cumsum"], beta, st["a_solved"], chunk_size=64),
        "asm": lambda: qwen_gdn_bt64_gfx942_asm_v0_preallocated(k, st["w"], st["u"], st["g_cumsum"], st["initial_state"], h, vn, ht),
        "chunk_o": lambda: qwen_gdn_chunk_o_avelang_v6_standalone(q, k, st["v_new"], st["h_bf16"].float().contiguous(), st["g_cumsum"], chunk_size=64),
    }
    fn = funcs[args.stage]
    for _ in range(args.warmup): fn()
    torch.cuda.synchronize()
    for _ in range(args.repeat): fn()
    torch.cuda.synchronize()
    print(f"profiled stage={args.stage} T={args.T}")


if __name__ == "__main__":
    main()
