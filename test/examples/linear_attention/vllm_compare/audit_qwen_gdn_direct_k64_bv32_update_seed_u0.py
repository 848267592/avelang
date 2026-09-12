#!/usr/bin/env python3
"""U0: characterize the P1 update-boundary rounding contract.

This is a host-side audit of one existing P1 launch.  It distinguishes
``round_bf16((U-pred)*decay)`` from the frozen ABI contract
``round_bf16(round_bf16(U-pred)*decay)``.  The same script records both the
archived pre-fix mismatch and the repaired source's post-fix behavior.  No
timing is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p0 as p0
import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p1 as p1


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    diff = (actual.float() - expected.float()).abs()
    flat = int(diff.flatten().argmax().item())
    index = tuple(int(item.item()) for item in torch.unravel_index(torch.tensor(flat, device=diff.device), diff.shape))
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "argmax": index,
        "actual_at_argmax": float(actual[index].float().item()),
        "expected_at_argmax": float(expected[index].float().item()),
    }


def _delta(v_decay: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    result = torch.empty((1, p0.H_V, p0.KDIM, p0.KDIM), device="cuda", dtype=torch.float32)
    for head in range(p0.H_V):
        result[0, head] = v_decay[0, :, head].float().t() @ k[0, :, head // 2].float()
    return result


def run_u0(seed: int) -> dict[str, Any]:
    p1._set_c0_lowering()
    case = p0._machine_cases(seed)[-1]
    k = p1._make_k(seed + 1005)
    g = p1._case_g(case, seed + 2005)
    actual = p1._run_kernel(case, k, g)
    torch.cuda.synchronize()

    g_last = g[:, p0.BT - 1 : p0.BT]
    decay = torch.exp(g_last - g)
    corrected = case.u.float() - actual["pred_f32"]
    v_new_contract = corrected.to(torch.bfloat16)
    vdecay_unrounded = (corrected * decay[..., None]).to(torch.bfloat16)
    vdecay_contract = (v_new_contract.float() * decay[..., None]).to(torch.bfloat16)
    delta_unrounded = _delta(vdecay_unrounded, k)
    delta_contract = _delta(vdecay_contract, k)
    scale = torch.empty_like(delta_contract)
    for head in range(p0.H_V):
        scale[0, head] = case.initial_state[0, head] * torch.exp(g[0, p0.BT - 1, head])

    result = {
        "schema": "qwen-direct-k64-bv32-update-seed-u0-v1",
        "formulae": {
            "unrounded": "round_bf16((U - pred_f32) * exp(g_last - g))",
            "frozen_bf16_abi": "round_bf16(round_bf16(U - pred_f32) * exp(g_last - g))",
        },
        "v_new_kernel_vs_contract": _metrics(actual["v_new"], v_new_contract),
        "vdecay_kernel_vs_unrounded_formula": _metrics(actual["v_decay"], vdecay_unrounded),
        "vdecay_kernel_vs_bf16_boundary_contract": _metrics(actual["v_decay"], vdecay_contract),
        "delta_kernel_vs_unrounded_vdecay_reference": _metrics(actual["delta"], delta_unrounded),
        "delta_kernel_vs_bf16_boundary_reference": _metrics(actual["delta"], delta_contract),
        "state_after_vs_scale_plus_kernel_delta": _metrics(actual["state_after"], scale + actual["delta"]),
        "state_after_vs_scale_plus_contract_delta": _metrics(actual["state_after"], scale + delta_contract),
        "finite": all(bool(torch.isfinite(value).all()) for value in actual.values()),
    }
    result["verdict"] = {
        "v_new_matches_contract": result["v_new_kernel_vs_contract"]["max_abs"] == 0.0,
        "kernel_uses_unrounded_corrected_for_vdecay": result["vdecay_kernel_vs_unrounded_formula"]["max_abs"] == 0.0,
        "kernel_uses_bf16_vnew_boundary_for_vdecay": result["vdecay_kernel_vs_bf16_boundary_contract"]["max_abs"] == 0.0,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "rocprof_outputs/qwen_direct_k64_bv32_update_seed_u0",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_u0(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "u0_update_seed_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps({"verdict": result["verdict"], "vdecay_contract_error": result["vdecay_kernel_vs_bf16_boundary_contract"]["max_abs"]}, sort_keys=True))
    if not (result["finite"] and result["verdict"]["v_new_matches_contract"]):
        raise SystemExit("U0 could not produce a finite V-new boundary audit.")


if __name__ == "__main__":
    main()
