#!/usr/bin/env python3
"""A70 T64/NT2 complete H1/H2 recurrent-backedge gate."""

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


def _slice(inp, start, end):
    out = dict(inp)
    for key in ("q", "k", "v", "raw_g"):
        out[key] = inp[key][:, start:end].contiguous()
    out["cu_seqlens"] = torch.tensor([0, end - start], device=inp["q"].device, dtype=torch.int32)
    return out


def _metric(actual, expected):
    if not torch.isfinite(actual.float()).all():
        raise AssertionError("A70 T64 produced NaN/Inf")
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().item())
    max_rel = float((diff / expected.float().abs().clamp_min(1.0e-5)).max().item())
    passed = bool(torch.allclose(actual.float(), expected.float(), atol=STRICT, rtol=STRICT))
    if not passed:
        raise AssertionError(f"A70 T64 backedge divergence: max_abs={max_abs}, max_rel={max_rel}")
    return {"max_abs": max_abs, "max_rel": max_rel, "strict_1e-5": True}


def run() -> dict[str, object]:
    inp = make_kda_prefill_inputs(T=64, H=HEADS, seed=6901)
    beta = inp["raw_beta"].float().contiguous()
    state = torch.zeros((1, HEADS, HEAD_DIM, HEAD_DIM), device=inp["q"].device, dtype=torch.float32)
    first = _slice(inp, 0, 32)
    second = _slice(inp, 32, 64)
    beta0 = beta[:, :32].contiguous()
    beta1 = beta[:, 32:64].contiguous()

    native0_out, native0_state = _native(first, beta0, state)
    native1_out, native1_state = _native(second, beta1, native0_state)
    public0_out, public0_state = _public(first, beta0, state)
    public1_out, public1_state = _public(second, beta1, public0_state)

    report = {
        "artifact": "A_20260927_70_BW16_TRUE_W4_REBUILD",
        "T": 64,
        "NT": 2,
        "grid": list(GRID),
        "block": list(BLOCK),
        "complete_H1_H2_recurrent_backedge": {
            "H1_chunk0": _metric(native0_state[:, :, :64], public0_state[:, :, :64]),
            "H2_chunk0": _metric(native0_state[:, :, 64:], public0_state[:, :, 64:]),
            "H1_chunk1": _metric(native1_state[:, :, :64], public1_state[:, :, :64]),
            "H2_chunk1": _metric(native1_state[:, :, 64:], public1_state[:, :, 64:]),
            "output_chunk0": _metric(native0_out, public0_out),
            "output_chunk1": _metric(native1_out, public1_out),
        },
        "finite": bool(torch.isfinite(native1_out.float()).all() and torch.isfinite(native1_state).all()),
    }
    torch.cuda.synchronize()
    return report


if __name__ == "__main__":
    report = run()
    out = ROOT / "test/examples/kda/reports/A_20260927_70_BW16_TRUE_W4_REBUILD/A_20260927_70_T64_BACKEDGE.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
