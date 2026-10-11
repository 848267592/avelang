#!/usr/bin/env python3
"""A70 diagnostic for the first T64 chunk, stopped before the backedge."""

from __future__ import annotations

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


def _slice(inp, start, end):
    out = dict(inp)
    for key in ("q", "k", "v", "raw_g"):
        out[key] = inp[key][:, start:end].contiguous()
    out["cu_seqlens"] = torch.tensor([0, end - start], device=inp["q"].device, dtype=torch.int32)
    return out


def run() -> dict[str, object]:
    inp = _slice(make_kda_prefill_inputs(T=64, H=HEADS, seed=6900), 0, 32)
    beta = inp["raw_beta"][:, :32].float().contiguous()
    state = torch.zeros((1, HEADS, HEAD_DIM, HEAD_DIM), device=inp["q"].device, dtype=torch.float32)
    ws = capture_aiter_prepare_workspace(
        inp["q"], inp["k"], inp["v"], inp["raw_g"], beta,
        inp["A_log"], inp["dt_bias"], initial_state=state,
    )
    _, _, debug = a70_bw16_k2(
        workspace=ws, v=inp["v"], beta=beta, initial_state=state, debug=True,
    )
    debug = debug.reshape(HEADS, 12, HEAD_DIM, 128)

    h1 = state[0, 0, :64, :16].to(torch.bfloat16).float()
    h2 = state[0, 0, 64:, :16].to(torch.bfloat16).float()
    kd = ws.kd[0]
    pred = kd[:, :64].float() @ h1 + kd[:, 64:].float() @ h2
    u_pre = (inp["v"][0, :, 0, :16].float() - pred) * beta[0, :, 0].sigmoid().view(32, 1)
    U = ws.inv_mqk[0, 0].float() @ u_pre.to(torch.float16).float()
    qd_h = ws.qd[0, :, :64].float() @ h1 + ws.qd[0, :, 64:].float() @ h2
    mqk = ws.inv_mqk[0, 1].float() @ U.to(torch.float16).float()
    output_acc = qd_h + mqk
    u_bf = U.to(torch.bfloat16).float()
    h1_new = ws.kr[0, :, :64].float().T @ u_bf
    h2_new = ws.kr[0, :, 64:].float().T @ u_bf
    refs = (pred, u_pre, U, qd_h, mqk, output_acc, h1_new, h2_new)
    names = ("Kd/pred", "u_pre", "INV/U", "Qd", "Mqk", "output_acc", "H1", "H2")
    result = {}
    for name, phase, ref in zip(names, range(8), refs):
        actual = debug[0, phase, : ref.shape[0], :16]
        diff = (actual.float() - ref.float()).abs()
        result[name] = {
            "max_abs": float(diff.max().item()),
            "max_rel": float((diff / ref.float().abs().clamp_min(1.0e-5)).max().item()),
            "strict_1e-5": bool(torch.allclose(actual.float(), ref.float(), atol=1.0e-5, rtol=1.0e-5)),
        }
    return {
        "artifact": "A_20260927_70_BW16_TRUE_W4_REBUILD",
        "source_shape": [1, 64, HEADS, HEAD_DIM],
        "traced_chunk": [0, 32],
        "chunk_size": CHUNK,
        "stages": result,
    }


if __name__ == "__main__":
    import json

    report = run()
    out = ROOT / "test/examples/kda/reports/A_20260927_70_BW16_TRUE_W4_REBUILD/A_20260927_70_T64_FIRST_CHUNK_TRACE.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
