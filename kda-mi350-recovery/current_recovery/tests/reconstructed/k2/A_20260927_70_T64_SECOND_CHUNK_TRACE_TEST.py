#!/usr/bin/env python3
"""A70 diagnostic for the second T64 chunk with the carried H1/H2 state."""

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


def _slice(inp, start, end):
    out = dict(inp)
    for key in ("q", "k", "v", "raw_g"):
        out[key] = inp[key][:, start:end].contiguous()
    out["cu_seqlens"] = torch.tensor([0, end - start], device=inp["q"].device, dtype=torch.int32)
    return out


def run() -> dict[str, object]:
    full = make_kda_prefill_inputs(T=64, H=HEADS, seed=6901)
    first = _slice(full, 0, 32)
    second = _slice(full, 32, 64)
    beta = full["raw_beta"].float().contiguous()
    beta0 = beta[:, :32].contiguous()
    beta1 = beta[:, 32:64].contiguous()
    state0 = torch.zeros((1, HEADS, HEAD_DIM, HEAD_DIM), device=full["q"].device, dtype=torch.float32)

    ws0 = capture_aiter_prepare_workspace(
        first["q"], first["k"], first["v"], first["raw_g"], beta0,
        first["A_log"], first["dt_bias"], initial_state=state0,
    )
    _, carried, _ = a70_bw16_k2(
        workspace=ws0, v=first["v"], beta=beta0, initial_state=state0,
    )
    ws = capture_aiter_prepare_workspace(
        second["q"], second["k"], second["v"], second["raw_g"], beta1,
        second["A_log"], second["dt_bias"], initial_state=carried,
    )
    _, _, debug = a70_bw16_k2(
        workspace=ws, v=second["v"], beta=beta1, initial_state=carried, debug=True,
    )
    debug = debug.reshape(HEADS, 12, HEAD_DIM, 128)

    h1 = carried[0, 0, :64, :16].to(torch.bfloat16).float()
    h2 = carried[0, 0, 64:, :16].to(torch.bfloat16).float()
    kd = ws.kd[0]
    pred = kd[:, :64].float() @ h1 + kd[:, 64:].float() @ h2
    u_pre = (second["v"][0, :, 0, :16].float() - pred) * beta1[0, :, 0].sigmoid().view(32, 1)
    U = ws.inv_mqk[0, 0].float() @ u_pre.to(torch.float16).float()
    qd_h = ws.qd[0, :, :64].float() @ h1 + ws.qd[0, :, 64:].float() @ h2
    mqk = ws.inv_mqk[0, 1].float() @ U.to(torch.float16).float()
    output_acc = qd_h + mqk
    u_bf = U.to(torch.bfloat16).float()
    h1_new = ws.kr[0, :, :64].float().T @ u_bf
    h2_new = ws.kr[0, :, 64:].float().T @ u_bf
    refs = (pred, u_pre, U, qd_h, mqk, output_acc, h1_new, h2_new)
    names = ("Kd/pred", "u_pre", "INV/U", "Qd", "Mqk", "output_acc", "H1", "H2")
    stages = {}
    for name, phase, ref in zip(names, range(8), refs):
        actual = debug[0, phase, : ref.shape[0], :16]
        diff = (actual.float() - ref.float()).abs()
        stages[name] = {
            "max_abs": float(diff.max().item()),
            "max_rel": float((diff / ref.float().abs().clamp_min(1.0e-5)).max().item()),
            "strict_1e-5": bool(torch.allclose(actual.float(), ref.float(), atol=1.0e-5, rtol=1.0e-5)),
        }
    qd1_s = torch.zeros((2048,), dtype=torch.bfloat16)
    h_s = torch.zeros((1024,), dtype=torch.bfloat16)
    qd_cpu = ws.qd[0].cpu()
    h_cpu = carried[0, 0].cpu()
    for tid in range(256):
        wave, lane = divmod(tid, 64)
        qd_row = wave * 8 + lane // 8
        qd_k_base = (lane & 7) * 8
        s = (tid << 4) ^ (tid & 120)
        qd1_s[s // 2:s // 2 + 4] = qd_cpu[qd_row, qd_k_base:qd_k_base + 4]
        qd1_s[(s ^ 8) // 2:(s ^ 8) // 2 + 4] = qd_cpu[qd_row, qd_k_base + 4:qd_k_base + 8]
        h_row = wave * 16 + (lane & 15)
        h_col = (lane >> 4) * 4
        h_write = (wave << 5) ^ (((lane & 48) << 5) | (lane << 1))
        vals = h_cpu[h_row, h_col:h_col + 4].to(torch.bfloat16)
        h_s[h_write // 2] = vals[0]
        h_s[(h_write ^ 136) // 2] = vals[1]
        h_s[(h_write ^ 272) // 2] = vals[2]
        h_s[(h_write ^ 408) // 2] = vals[3]
    exp_a = torch.empty((1024,), dtype=torch.float32)
    exp_b = torch.empty((1024,), dtype=torch.float32)
    for tid in range(256):
        wave, lane = divmod(tid, 64)
        r = lane & 15
        b = lane & 48
        a_shared = ((((64 * wave) & 128) << 4) | (r << 7) | ((r << 3) ^ (b >> 1)))
        h_read = (r << 7) | ((r << 3) ^ (b >> 1))
        exp_a[tid * 4:tid * 4 + 4] = qd1_s[a_shared // 2:a_shared // 2 + 4].float()
        exp_b[tid * 4:tid * 4 + 4] = h_s[h_read // 2:h_read // 2 + 4].float()
    got_a = debug[0, 8].flatten()[:1024].cpu()
    got_b = debug[0, 9].flatten()[:1024].cpu()
    stages["Qd_first_operand_payloads"] = {
        "A_max_abs": float((got_a - exp_a).abs().max().item()),
        "B_max_abs": float((got_b - exp_b).abs().max().item()),
        "A_strict": bool(torch.allclose(got_a, exp_a, atol=1.0e-5, rtol=1.0e-5)),
        "B_strict": bool(torch.allclose(got_b, exp_b, atol=1.0e-5, rtol=1.0e-5)),
        "B_wave_max_abs": [
            float((got_b[w * 256:(w + 1) * 256] - exp_b[w * 256:(w + 1) * 256]).abs().max().item())
            for w in range(4)
        ],
        "B_first_bad": int((got_b - exp_b).abs().argmax().item()),
    }
    qd_cpu_f32 = ws.qd[0].float().cpu()
    h1_cpu_f32 = h1.cpu()
    exp_qd_partial = torch.empty((1024,), dtype=torch.float32)
    exp_qd_full = torch.empty((1024,), dtype=torch.float32)
    for tid in range(256):
        wave, lane = divmod(tid, 64)
        row = 16 * (wave >> 1) + (lane & 15)
        col = (lane >> 4) * 4
        exp_qd_partial[tid * 4:tid * 4 + 4] = qd_cpu_f32[row, :16] @ h1_cpu_f32[:16, col:col + 4]
        exp_qd_full[tid * 4:tid * 4 + 4] = qd_cpu_f32[row, :64] @ h1_cpu_f32[:64, col:col + 4]
    got_qd_partial = debug[0, 10].flatten()[:1024].cpu()
    got_qd_full = debug[0, 11].flatten()[:1024].cpu()
    stages["Qd_first_partial"] = {
        "max_abs": float((got_qd_partial - exp_qd_partial).abs().max().item()),
        "max_rel": float(((got_qd_partial - exp_qd_partial).abs() / exp_qd_partial.abs().clamp_min(1.0e-5)).max().item()),
        "strict_1e-5": bool(torch.allclose(got_qd_partial, exp_qd_partial, atol=1.0e-5, rtol=1.0e-5)),
    }
    stages["Qd_first_accumulated"] = {
        "max_abs": float((got_qd_full - exp_qd_full).abs().max().item()),
        "max_rel": float(((got_qd_full - exp_qd_full).abs() / exp_qd_full.abs().clamp_min(1.0e-5)).max().item()),
        "strict_1e-5": bool(torch.allclose(got_qd_full, exp_qd_full, atol=1.0e-5, rtol=1.0e-5)),
    }
    return {
        "artifact": "A_20260927_70_BW16_TRUE_W4_REBUILD",
        "T": 64,
        "traced_chunk": [32, 64],
        "chunk_size": CHUNK,
        "stages": stages,
    }


if __name__ == "__main__":
    report = run()
    out = ROOT / "test/examples/kda/reports/A_20260927_70_BW16_TRUE_W4_REBUILD/A_20260927_70_T64_SECOND_CHUNK_TRACE.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
