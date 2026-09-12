#!/usr/bin/env python3
"""C17 full-region correctness gate.

The C17 wrapper uses the existing BDV2 logical source contract and changes
only the compiler physical-plan gate.  This driver compares its BF16 output
with the frozen Z5B output and records the checks as JSON for the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_native_chunko_stage6z_c17_full_physical_plan import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c17_full_physical_plan_launch_into,
)
from qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache,
)
from stage2_runner import make_inputs  # noqa: E402


BT = 64


def make_inputs_c17(t: int, seed: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(t, seed, "random", True)
    torch.manual_seed(seed + 43)
    vn = (
        torch.randn((1, t, 8, 128), device=q.device, dtype=torch.float32) * 0.02
    ).to(torch.bfloat16)
    h = (
        torch.randn((1, t // BT, 8, 128, 128), device=q.device, dtype=torch.float32)
        * 0.01
    ).to(torch.bfloat16)
    return q.contiguous(), k.contiguous(), vn.contiguous(), h.contiguous(), g.contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--zero-v", action="store_true")
    parser.add_argument("--nan-prefill", action="store_true")
    args = parser.parse_args()
    if args.T < BT or args.T % BT:
        raise ValueError("T must be a positive multiple of 64")

    q, k, vn, h, g = make_inputs_c17(args.T, 2026081700 + args.T)
    if args.zero_v:
        vn.zero_()
    reference = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache(
        q, k, vn, h, g
    )
    output = torch.full_like(vn, float("nan")) if args.nan_prefill else torch.empty_like(vn)
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c17_full_physical_plan_launch_into(
        q, k, vn, h, g, output
    )
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(output).all().item())
    byte_exact = bool(torch.equal(output, reference))
    result = {
        "candidate": "C17_FULL",
        "T": args.T,
        "zero_v_new": args.zero_v,
        "nan_prefilled_caller_output": args.nan_prefill,
        "finite": finite,
        "byte_exact_vs_Z5B": byte_exact,
        "max_abs_vs_Z5B": float((output.float() - reference.float()).abs().max().item()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not finite or not byte_exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
