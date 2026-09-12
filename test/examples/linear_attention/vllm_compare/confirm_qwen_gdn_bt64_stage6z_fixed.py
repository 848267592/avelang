#!/usr/bin/env python3
"""Fresh-process correctness driver for the repaired Stage 6Z Z2/Z3 bodies.

This is deliberately a test-only driver.  One invocation runs exactly one
kernel arm and one length.  The outer confirmation can therefore prove
fresh-process coverage without compiling/running four candidates in one GPU
process.  Outputs can be saved for an offline cross-arm comparison.
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

from qwen_gdn_bt64_bf16_chunko_boundary_stage6w import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w,
)
from qwen_gdn_bt64_native_chunko_stage6z import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z1,
)
from qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2_launch_into,
)
from qwen_gdn_bt64_native_chunko_stage6z_z3_wg128 import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z3,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z3_launch_into,
)
from stage2_runner import make_inputs  # noqa: E402


BT = 64
ATOL = 1.0 / 128.0


def make_case(t: int, seed: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(t, seed, "random", True)
    torch.manual_seed(seed + 43)
    vn = (torch.randn((1, t, 8, 128), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h = (torch.randn((1, t // BT, 8, 128, 128), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return q, k, vn, h, g


def _max_abs(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs.float() - rhs.float()).abs().max().item())


def run_arm(arm: str, tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    q, k, vn, h, g = tensors
    if arm == "z1":
        output = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z1(q, k, vn, h, g)
    elif arm == "z2":
        output = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2(q, k, vn, h, g)
    elif arm == "z3":
        output = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z3(q, k, vn, h, g)
    else:
        output = qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w(q, k, vn, h, g)
    torch.cuda.synchronize()
    return output


def check_arm(
    arm: str,
    tensors: tuple[torch.Tensor, ...],
    reference: Path | None,
    save_output: Path | None,
) -> dict[str, object]:
    candidate = run_arm(arm, tensors)
    result = {
        "arm": arm,
        "finite": bool(torch.isfinite(candidate).all().item()),
        "output_path": None,
    }
    if save_output is not None:
        save_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(candidate.cpu(), save_output)
        result["output_path"] = str(save_output)
    if reference is not None:
        expected = torch.load(reference, map_location="cpu", weights_only=True)
        actual = candidate.cpu()
        result.update(
            {
                "reference": str(reference),
                "byte_exact_to_reference": bool(torch.equal(actual, expected)),
                "max_abs_to_reference": _max_abs(actual, expected),
            }
        )
        required = result["byte_exact_to_reference"]
    else:
        required = result["finite"]
    if not required:
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


def check_zero_v(arm: str, tensors: tuple[torch.Tensor, ...]) -> dict[str, object]:
    q, k, vn, h, g = tensors
    vn.zero_()
    output = torch.full_like(vn, float("nan"))
    if arm == "z2":
        qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2_launch_into(q, k, vn, h, g, output)
        expected = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2(q, k, vn, h, g)
    else:
        qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z3_launch_into(q, k, vn, h, g, output)
        expected = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z3(q, k, vn, h, g)
    torch.cuda.synchronize()
    result = {
        "arm": arm,
        "zero_v": True,
        "finite": bool(torch.isfinite(output).all().item()),
        "no_nan_output_reuse": not bool(torch.isnan(output).any().item()),
        "output_byte_exact_to_fresh": bool(torch.equal(output, expected)),
    }
    if not all(result[key] for key in ("finite", "no_nan_output_reuse", "output_byte_exact_to_fresh")):
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("z1", "z2", "z3", "stage6w"), required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026080800)
    parser.add_argument("--zero-v", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--save-output", type=Path)
    args = parser.parse_args()
    if args.T < BT or args.T % BT:
        raise ValueError("T must be a positive multiple of 64")
    tensors = make_case(args.T, args.seed + args.T)
    if args.zero_v:
        if args.arm not in ("z2", "z3"):
            raise ValueError("--zero-v is only defined for z2/z3")
        result = check_zero_v(args.arm, tensors)
    else:
        result = check_arm(args.arm, tensors, args.reference, args.save_output)
    result.update({"T": args.T, "fresh_process": True})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
