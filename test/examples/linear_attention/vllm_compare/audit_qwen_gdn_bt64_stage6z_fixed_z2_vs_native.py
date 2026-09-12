#!/usr/bin/env python3
"""Fresh-process T=2048 correctness audit for fixed Z2 and native chunk-o.

This is an audit harness only.  It does not change either kernel source.  One
process runs one arm, so compilation, Triton selection, and candidate output
are isolated from the other arm.
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
from qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2_launch_into,
)
from stage2_runner import make_inputs  # noqa: E402


BT = 64
T = 2048
ATOL = 1.0 / 128.0


def make_case() -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(T, 2026080600 + T, "random", True)
    torch.manual_seed(2026080700 + T)
    vn = (torch.randn((1, T, 8, 128), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h = (
        torch.randn((1, T // BT, 8, 128, 128), device=q.device, dtype=torch.float32) * 0.01
    ).to(torch.bfloat16)
    return q.contiguous(), k.contiguous(), vn.contiguous(), h.contiguous(), g.contiguous()


def native_direct(
    tensors: tuple[torch.Tensor, ...], output: torch.Tensor
) -> dict[str, object]:
    q, k, vn, h, g = tensors
    from vllm.model_executor.layers.fla.ops import chunk_o

    # This public chunk-o call is selection/compile setup only and is outside
    # the correctness result and all timing intervals.
    selected = chunk_o.chunk_fwd_o(q=q, k=k, v=vn, h=h, g=g, scale=128 ** -0.5, chunk_size=BT)
    torch.cuda.synchronize()
    tuner = chunk_o.chunk_fwd_kernel_o.fn
    keys = list(tuner.cache)
    if len(keys) != 1:
        raise RuntimeError(f"expected one native cache key, got {keys!r}")
    config = next(
        config
        for config in tuner.configs
        if dict(getattr(config, "kwargs", {})) == {"BK": 32, "BV": 64}
        and int(getattr(config, "num_warps", -1)) == 4
        and int(getattr(config, "num_stages", -1)) == 2
    )
    tuner.cache[keys[0]] = config
    kernel = chunk_o.chunk_fwd_kernel_o
    kernel[
        lambda meta: ((128 + meta["BV"] - 1) // meta["BV"], T // BT, 8)
    ](q, k, vn, h, g, output, None, None, 128 ** -0.5, T=T, H=8, Hg=4, K=128, V=128, BT=BT)
    torch.cuda.synchronize()
    summary = {
        "selected_setup_finite": bool(torch.isfinite(selected).all().item()),
        "selected_config": {
            "kwargs": dict(getattr(config, "kwargs", {})),
            "num_warps": int(getattr(config, "num_warps", -1)),
            "num_stages": int(getattr(config, "num_stages", -1)),
            "num_ctas": int(getattr(config, "num_ctas", -1)),
        },
    }
    return summary


def max_abs(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs.float() - rhs.float()).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("z2", "native"), required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path, required=True)
    parser.add_argument("--reference-out", type=Path, required=True)
    args = parser.parse_args()

    tensors = make_case()
    q, k, vn, h, g = tensors
    candidate = torch.empty_like(vn)
    native_setup: dict[str, object] = {}
    if args.arm == "z2":
        qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2_launch_into(q, k, vn, h, g, candidate)
    else:
        native_setup = native_direct(tensors, candidate)
    torch.cuda.synchronize()
    reference = qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w(q, k, vn, h, g)
    torch.cuda.synchronize()

    result = {
        "arm": args.arm,
        "T": T,
        "fresh_process": True,
        "cuda_graph_used": False,
        "current_stream": True,
        "dtype_contract": "q/k/v_new/h=BF16, g=FP32, output=BF16",
        "candidate_finite": bool(torch.isfinite(candidate).all().item()),
        "reference_finite": bool(torch.isfinite(reference).all().item()),
        "byte_exact_to_stage6w": bool(torch.equal(candidate, reference)),
        "max_abs_to_stage6w": max_abs(candidate, reference),
        "atol": ATOL,
        "within_atol": max_abs(candidate, reference) <= ATOL,
        "native_setup": native_setup,
        "candidate_out": str(args.candidate_out),
        "reference_out": str(args.reference_out),
    }
    args.candidate_out.parent.mkdir(parents=True, exist_ok=True)
    args.reference_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(candidate.cpu(), args.candidate_out)
    torch.save(reference.cpu(), args.reference_out)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["candidate_finite"] or not result["reference_finite"] or not result["within_atol"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
