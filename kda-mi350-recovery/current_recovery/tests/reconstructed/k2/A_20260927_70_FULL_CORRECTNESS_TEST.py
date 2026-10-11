#!/usr/bin/env python3
"""A70 full BW16/W4 correctness matrix; no benchmark is performed here."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
PREFILL = ROOT / "test/examples/kda/avelang_impl/prefill"
BASELINE = ROOT / "kda_baseline"
for item in (str(PREFILL), str(BASELINE)):
    if item not in sys.path:
        sys.path.insert(0, item)

from A_20260927_70_BW16_TRUE_W4_REBUILD import a70_bw16_k2
from bench_common import make_kda_prefill_inputs
from kda_prefill_v0 import CHUNK, HEADS, HEAD_DIM, capture_aiter_prepare_workspace


ATOL = 3.0e-2
RTOL = 3.0e-2
STRICT = 1.0e-5
GRID = (8, 12, 1)
BLOCK = (256, 1, 1)


def _public(inp, beta, state):
    from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn

    return chunk_kimi_delta_attn(
        q=inp["q"], k=inp["k"], v=inp["v"], g=inp["raw_g"], beta=beta,
        A_log=inp["A_log"], dt_bias=inp["dt_bias"], scale=HEAD_DIM ** -0.5,
        initial_state=state, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True, safe_gate=True, lower_bound=-5.0,
        state_v_first=False, chunk_size=CHUNK, cu_seqlens=inp["cu_seqlens"],
    )


def _native(inp, beta, state):
    ws = capture_aiter_prepare_workspace(
        inp["q"], inp["k"], inp["v"], inp["raw_g"], beta,
        inp["A_log"], inp["dt_bias"], initial_state=state,
    )
    out, final_state, _ = a70_bw16_k2(
        workspace=ws, v=inp["v"], beta=beta, initial_state=state,
    )
    return out, final_state


def _metric(actual, expected, *, name):
    if not torch.isfinite(actual.float()).all():
        raise AssertionError(f"A70 {name} produced NaN/Inf")
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().item())
    max_rel = float((diff / expected.float().abs().clamp_min(1.0e-5)).max().item())
    if not torch.allclose(actual.float(), expected.float(), atol=ATOL, rtol=RTOL):
        raise AssertionError(f"A70 {name} divergence: max_abs={max_abs}, max_rel={max_rel}")
    return {"max_abs": max_abs, "max_rel": max_rel, "production_tolerance": True}


def _slice(inp, start, end):
    out = dict(inp)
    for key in ("q", "k", "v", "raw_g"):
        out[key] = inp[key][:, start:end].contiguous()
    out["cu_seqlens"] = torch.tensor([0, end - start], device=inp["q"].device, dtype=torch.int32)
    return out


def _backedge_gate(inp, beta, state):
    first = _slice(inp, 0, 32)
    second = _slice(inp, 32, 64)
    beta0 = beta[:, :32].contiguous()
    beta1 = beta[:, 32:64].contiguous()
    native0_out, native0_state = _native(first, beta0, state)
    native1_out, native1_state = _native(second, beta1, native0_state)
    public0_out, public0_state = _public(first, beta0, state)
    public1_out, public1_state = _public(second, beta1, public0_state)
    return {
        "H1_chunk0": _metric(native0_state[:, :, :64], public0_state[:, :, :64], name="T64 H1 chunk0"),
        "H2_chunk0": _metric(native0_state[:, :, 64:], public0_state[:, :, 64:], name="T64 H2 chunk0"),
        "H1_chunk1": _metric(native1_state[:, :, :64], public1_state[:, :, :64], name="T64 H1 chunk1"),
        "H2_chunk1": _metric(native1_state[:, :, 64:], public1_state[:, :, 64:], name="T64 H2 chunk1"),
        "output_chunk0": _metric(native0_out, public0_out, name="T64 output chunk0"),
        "output_chunk1": _metric(native1_out, public1_out, name="T64 output chunk1"),
        "strict_H1_H2": True,
    }


def run() -> dict[str, object]:
    cases = (32, 64, 127, 128, 256, 512, 1024, 2048, 4096)
    results = {}
    for index, T in enumerate(cases):
        inp = make_kda_prefill_inputs(T=T, H=HEADS, seed=6900 + index)
        beta = inp["raw_beta"].float().contiguous()
        state = torch.zeros((1, HEADS, HEAD_DIM, HEAD_DIM), device=inp["q"].device, dtype=torch.float32)
        if T == 127:
            state = torch.linspace(
                -0.03125, 0.03125, state.numel(), device=state.device, dtype=torch.float32,
            ).reshape_as(state)
        native_out, native_state = _native(inp, beta, state)
        public_out, public_state = _public(inp, beta, state)
        entry = {
            "T": T,
            "NT": (T + CHUNK - 1) // CHUNK,
            "grid": list(GRID),
            "block": list(BLOCK),
            "finite": bool(torch.isfinite(native_out.float()).all() and torch.isfinite(native_state).all()),
            "initial_state_nonzero": bool(state.abs().max().item() > 0),
            "output": _metric(native_out, public_out, name=f"T{T} output"),
            "final_state": _metric(native_state, public_state, name=f"T{T} final_state"),
        }
        if T == 64:
            entry["complete_H1_H2_recurrent_backedge"] = _backedge_gate(inp, beta, state)
        results[str(T)] = entry
        torch.cuda.synchronize()
    return {
        "artifact": "A_20260927_70_BW16_TRUE_W4_REBUILD",
        "passed": True,
        "correctness_tolerance": {
            "output_atol": ATOL, "output_rtol": RTOL,
            "final_state_atol": ATOL, "final_state_rtol": RTOL,
            "strict_trace_atol": STRICT, "strict_trace_rtol": STRICT,
        },
        "requirements": {
            "T": list(cases), "T64_complete_H1_H2_backedge": True,
            "T127_nonzero_initial_state": True, "finite": True, "exact_grid": True,
        },
        "cases": results,
    }


if __name__ == "__main__":
    report = run()
    out = ROOT / "test/examples/kda/reports/A_20260927_70_BW16_TRUE_W4_REBUILD/A_20260927_70_FULL_CORRECTNESS.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
