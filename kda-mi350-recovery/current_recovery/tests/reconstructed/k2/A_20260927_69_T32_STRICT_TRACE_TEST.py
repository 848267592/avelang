#!/usr/bin/env python3
"""A69 T32 strict semantic trace through K2's first chunk."""

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

from A_20260927_69_BW16_PERSISTENT_W4_REBUILD import a69_bw16_k2
from bench_common import make_kda_prefill_inputs
from kda_prefill_v0 import CHUNK, HEADS, HEAD_DIM, capture_aiter_prepare_workspace


ATOL = 1.0e-5
RTOL = 1.0e-5


def run() -> dict[str, object]:
    T = 32
    inp = make_kda_prefill_inputs(T=T, H=HEADS, seed=6900)
    beta = inp["raw_beta"].float().contiguous()
    state = torch.zeros((1, HEADS, HEAD_DIM, HEAD_DIM), device=inp["q"].device, dtype=torch.float32)
    ws = capture_aiter_prepare_workspace(
        inp["q"], inp["k"], inp["v"], inp["raw_g"], beta,
        inp["A_log"], inp["dt_bias"], initial_state=state,
    )
    actual_out, actual_state, debug = a69_bw16_k2(
        workspace=ws, v=inp["v"], beta=beta, initial_state=state, debug=True,
    )
    if debug is None:
        raise AssertionError("A69 did not return strict trace data")
    debug = debug.reshape(HEADS, 8, HEAD_DIM, 128)

    h1 = state[0, 0, :64, :16].to(torch.bfloat16).float()
    h2 = state[0, 0, 64:, :16].to(torch.bfloat16).float()
    kd = ws.kd[0]
    pred = kd[:, :64].float() @ h1 + kd[:, 64:].float() @ h2
    u_pre = (inp["v"][0, :, 0, :16].float() - pred) * beta[0, :, 0].sigmoid().view(32, 1)
    U = ws.inv_mqk[0, 0].float() @ u_pre.to(torch.float16).float()
    qd = ws.qd[0]
    qd_h = qd[:, :64].float() @ h1 + qd[:, 64:].float() @ h2
    mqk = ws.inv_mqk[0, 1].float() @ U.to(torch.float16).float()
    output_acc = qd_h + mqk
    u_bf = U.to(torch.bfloat16).float()
    h1_new = ws.kr[0, :, :64].float().T @ u_bf
    h2_new = ws.kr[0, :, 64:].float().T @ u_bf
    expected = (pred, u_pre, U, qd_h, mqk, output_acc, h1_new, h2_new)
    phases = (0, 1, 2, 3, 4, 5, 6, 7)
    names = ("Kd/pred", "u_pre", "INV/U", "Qd", "Mqk", "output_acc", "H1", "H2")
    gates = {}
    for name, phase, ref in zip(names, phases, expected):
        actual = debug[0, phase, : ref.shape[0], :16]
        diff = (actual.float() - ref.float()).abs()
        max_abs = float(diff.max().item())
        max_rel = float((diff / ref.float().abs().clamp_min(1.0e-5)).max().item())
        if not torch.allclose(actual.float(), ref.float(), atol=ATOL, rtol=RTOL):
            raise AssertionError(f"first strict divergence at {name}: {max_abs} / {max_rel}")
        gates[name] = {"max_abs": max_abs, "max_rel": max_rel, "pass": True}

    if not torch.isfinite(actual_out.float()).all() or not torch.isfinite(actual_state).all():
        raise AssertionError("A69 T32 produced NaN/Inf")
    return {
        "artifact": "A_20260927_69_BW16_PERSISTENT_W4_REBUILD",
        "T": 32,
        "strict_atol": ATOL,
        "strict_rtol": RTOL,
        "first_divergent_boundary": None,
        "gates": gates,
        "one_launch_grid": [8, 12, 1],
        "block": [256, 1, 1],
        "persistent_nt_loop": True,
    }


if __name__ == "__main__":
    report = run()
    out = ROOT / "test/examples/kda/reports/A_20260927_69_BW16_PERSISTENT_W4_REBUILD/A_20260927_69_T32_STRICT_TRACE.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
