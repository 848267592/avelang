#!/usr/bin/env python3
"""A68 Phase-2 public-production correctness gate.

This deliberately uses the established A65/A56 production tolerance.  The
separate A68 T32 trace remains the strict 1e-5 semantic gate.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
PREFILL = ROOT / "test/examples/kda/avelang_impl/prefill"
TESTS = ROOT / "test/examples/kda/tests"
for item in (str(PREFILL), str(TESTS), str(ROOT / "kda_baseline")):
    if item not in sys.path:
        sys.path.insert(0, item)

from A_20260925_68_BW16_PRODUCTION_CLOSURE_PROFILE import (  # noqa: E402
    GRID_X,
    HEADS,
    WG,
    a68_bw16_k2,
)
from bench_common import make_kda_prefill_inputs  # noqa: E402
from kda_prefill_v0 import (  # noqa: E402
    CHUNK,
    HEAD_DIM,
    LOWER_BOUND,
    capture_aiter_prepare_workspace,
)

ATOL = 3.0e-2
RTOL = 3.0e-2
CASES = (32, 64, 127, 128, 256, 512, 1024, 2048, 4096)
REPORT = ROOT / "test/examples/kda/reports/A_20260925_68_production_closure/A_20260925_68_PRODUCTION_CORRECTNESS.json"


def _public(inp, beta, state):
    from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn

    out, final_state = chunk_kimi_delta_attn(
        q=inp["q"], k=inp["k"], v=inp["v"], g=inp["raw_g"],
        beta=beta, A_log=inp["A_log"], dt_bias=inp["dt_bias"],
        scale=HEAD_DIM ** -0.5, initial_state=state,
        output_final_state=True, use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
        safe_gate=True, lower_bound=LOWER_BOUND, state_v_first=False,
        chunk_size=CHUNK, cu_seqlens=inp["cu_seqlens"],
    )
    if final_state is None:
        raise AssertionError("public AITER returned no final_state")
    return out, final_state


def _slice(inp, start: int, end: int):
    result = dict(inp)
    for key in ("q", "k", "v", "raw_g", "raw_beta"):
        result[key] = inp[key][:, start:end].contiguous()
    result["cu_seqlens"] = torch.tensor(
        [0, end - start], device=inp["q"].device, dtype=torch.int32
    )
    return result


def _metrics(actual, expected, name: str):
    actual_f = actual.float()
    expected_f = expected.float()
    if not torch.isfinite(actual_f).all() or not torch.isfinite(expected_f).all():
        raise AssertionError(f"non-finite tensor at {name}")
    diff = (actual_f - expected_f).abs()
    max_abs = float(diff.max().item())
    denom = expected_f.abs().clamp_min(1.0e-5)
    max_rel = float((diff / denom).max().item())
    passed = bool(torch.allclose(actual_f, expected_f, atol=ATOL, rtol=RTOL))
    return {"max_abs": max_abs, "max_rel": max_rel, "passed": passed}


def _initial_state(T: int, device):
    state = torch.zeros((1, HEADS, HEAD_DIM, HEAD_DIM), device=device, dtype=torch.float32)
    if T == 127:
        state = torch.linspace(
            -0.03125, 0.03125, state.numel(), device=device, dtype=torch.float32
        ).reshape_as(state)
    return state


def _run_case(T: int, seed: int):
    device = torch.device("cuda")
    inp = make_kda_prefill_inputs(T=T, H=HEADS, seed=seed)
    beta = inp["raw_beta"].float().contiguous()
    state = _initial_state(T, device)
    workspace = capture_aiter_prepare_workspace(
        inp["q"], inp["k"], inp["v"], inp["raw_g"], beta,
        inp["A_log"], inp["dt_bias"], initial_state=state,
    )
    actual_out, actual_state, _, _, _ = a68_bw16_k2(
        workspace=workspace, v=inp["v"], beta=beta,
        initial_state=state, debug=False,
    )
    public_out, public_state = _public(inp, beta, state)
    result = {
        "T": T,
        "NT": (T + CHUNK - 1) // CHUNK,
        "seed": seed,
        "initial_state_nonzero": bool(state.abs().max().item() > 0),
        "launch": {
            "grid": [GRID_X, HEADS, 1], "block": [WG, 1, 1],
            "exact_v_ownership": GRID_X * 16 == 128,
            "exact_head_ownership": HEADS == 12,
        },
        "output": _metrics(actual_out, public_out, f"T{T} output"),
        "final_state": _metrics(actual_state, public_state, f"T{T} final_state"),
    }
    if T == 64:
        first_inp = _slice(inp, 0, 32)
        second_inp = _slice(inp, 32, 64)
        first_beta = beta[:, :32].contiguous()
        second_beta = beta[:, 32:64].contiguous()
        _, first_public_state = _public(first_inp, first_beta, state)
        _, second_public_state = _public(second_inp, second_beta, first_public_state)
        _, _, snapshots, _, _ = a68_bw16_k2(
            workspace=workspace, v=inp["v"], beta=beta,
            initial_state=state, debug=True,
        )
        if snapshots is None or len(snapshots) != 2:
            raise AssertionError("T64 did not return two state snapshots")
        result["recurrent_backedge"] = {
            "chunk0_H1": _metrics(
                snapshots[0][:, :, :64], first_public_state[:, :, :64],
                "T64 chunk0 H1 backedge",
            ),
            "chunk0_H2": _metrics(
                snapshots[0][:, :, 64:], first_public_state[:, :, 64:],
                "T64 chunk0 H2 backedge",
            ),
            "chunk1_H1": _metrics(
                snapshots[1][:, :, :64], second_public_state[:, :, :64],
                "T64 chunk1 H1 backedge",
            ),
            "chunk1_H2": _metrics(
                snapshots[1][:, :, 64:], second_public_state[:, :, 64:],
                "T64 chunk1 H2 backedge",
            ),
        }
    return result


def main():
    report = {
        "phase": "A68_PHASE2_FULL_CORRECTNESS",
        "tolerance": {"atol": ATOL, "rtol": RTOL, "provenance": "A65/A56 production gate"},
        "cases": {},
        "requirements": {
            "T": list(CASES), "T127_nonzero_initial_state": True,
            "T64_complete_H1_H2_backedge": True,
            "finite": True, "exact_launch_grid_ownership": True,
        },
        "passed": True,
    }
    for index, T in enumerate(CASES):
        try:
            result = _run_case(T, 74 + index)
        except Exception as exc:  # noqa: BLE001
            result = {"T": T, "passed": False, "error": repr(exc)}
        report["cases"][str(T)] = result
        if result.get("error") or not result.get("output", {}).get("passed", False) or not result.get("final_state", {}).get("passed", False):
            report["passed"] = False
        if "recurrent_backedge" in result:
            if not all(item["passed"] for item in result["recurrent_backedge"].values()):
                report["passed"] = False
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
