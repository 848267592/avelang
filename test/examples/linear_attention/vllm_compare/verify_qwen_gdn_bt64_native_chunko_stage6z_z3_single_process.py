"""One-arm-per-process Z2/Z3 correctness check for the WG128 candidate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware import qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2  # noqa: E402
from qwen_gdn_bt64_native_chunko_stage6z_z3_wg128 import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z3,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z3_launch_into,
)
from stage2_runner import make_inputs  # noqa: E402


def make_case(t: int, seed: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(t, seed, "random", True)
    torch.manual_seed(seed + 43)
    vn = (torch.randn((1, t, 8, 128), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h = (torch.randn((1, t // 64, 8, 128, 128), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return q, k, vn, h, g


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("reference", "candidate"), required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026080800)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--zero-v", action="store_true", help="Run caller-owned zero-V-new finite/output reuse check.")
    args = parser.parse_args()
    tensors = make_case(args.T, args.seed + args.T)
    if args.mode == "reference":
        output = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2(*tensors)
        torch.cuda.synchronize()
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output.cpu(), args.reference)
        print(f"saved Z2 reference: {args.reference}")
        return
    if args.zero_v:
        q, k, vn, h, g = tensors
        vn.zero_()
        output = torch.full_like(vn, float("nan"))
        qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z3_launch_into(q, k, vn, h, g, output)
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(output).all().item())
        no_nan = not bool(torch.isnan(output).any().item())
        print({"T": args.T, "zero_v": True, "finite": finite, "no_nan_output_reuse": no_nan})
        if not (finite and no_nan):
            raise SystemExit(1)
        return
    output = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z3(*tensors)
    torch.cuda.synchronize()
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    candidate = output.cpu()
    equal = bool(torch.equal(candidate, reference))
    max_abs = float((candidate.float() - reference.float()).abs().max().item())
    print({"T": args.T, "byte_exact": equal, "max_abs": max_abs})
    if not equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
