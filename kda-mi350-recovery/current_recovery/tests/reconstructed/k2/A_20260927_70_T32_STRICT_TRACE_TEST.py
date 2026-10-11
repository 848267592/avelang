#!/usr/bin/env python3
"""A70 T32 strict semantic trace through K2's first chunk."""

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
    actual_out, actual_state, debug = a70_bw16_k2(
        workspace=ws, v=inp["v"], beta=beta, initial_state=state, debug=True,
    )
    if debug is None:
        raise AssertionError("A70 did not return strict trace data")
    debug = debug.reshape(HEADS, 12, HEAD_DIM, 128)

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
            print("A70_FIRST_DIVERGENCE", name, "actual0", actual.flatten()[:16].float().tolist(), "expected0", ref.flatten()[:16].float().tolist())
            print("A70_INV_TRACE", "A", debug[0, 8].flatten()[:64].float().tolist(), "B", debug[0, 9].flatten()[:64].float().tolist(), "P1", debug[0, 10].flatten()[:64].float().tolist(), "P2", debug[0, 11].flatten()[:64].float().tolist())
            inv_cpu = ws.inv_mqk[0, 0].float().cpu()
            u_cpu = u_pre.float().cpu()
            inv_s = torch.empty((1024,), dtype=torch.float32)
            u_s = torch.zeros((1024,), dtype=torch.float32)
            for tt in range(256):
                ww, ll = divmod(tt, 64)
                rr = 8 * ww + ll // 8
                cc = (ll & 7) * 4
                iw = ((tt << 3) ^ ((tt >> 1) & 56)) >> 1
                inv_s[iw:iw + 4] = inv_cpu[rr, cc:cc + 4]
                if (ww & 1) == 0:
                    ur = 16 * (ww >> 1) + (ll & 15)
                    uc = (ll >> 4) * 4
                    u0 = ((((ww << 6) & 128) >> 2) | ((ll & 48) << 4) | ((ll & 15) << 1)) ^ (ll & 48)
                    u2 = ((((tt << 2) & 56) ^ ((ll & 48) >> 1)) | ((ll & 15) << 6))
                    u_s[u0 >> 1] = u_cpu[ur, uc + 0]
                    u_s[(u0 + 64) >> 1] = u_cpu[ur, uc + 1]
                    u_s[(u0 ^ 136) >> 1] = u_cpu[ur, uc + 2]
                    u_s[((u0 ^ 136) + 64) >> 1] = u_cpu[ur, uc + 3]
            tt = 0
            ww, ll = 0, 0
            aa = (((64 * ww) & 128) << 3) | ((ll & 15) << 6) | ((((tt << 2) & 56) ^ ((ll & 48) >> 1)))
            uu = ((((tt << 2) & 56) ^ ((ll & 48) >> 1)) | ((ll & 15) << 6))
            print("A70_INV_EXPECTED_PAYLOAD", "A", inv_s[aa // 2:aa // 2 + 4].tolist(), "B", u_s[uu // 2:uu // 2 + 4].tolist())
            act_a = debug[0, 8].flatten()[:1024].cpu()
            act_b = debug[0, 9].flatten()[:1024].cpu()
            exp_a = torch.empty((1024,), dtype=torch.float32)
            exp_b = torch.empty((1024,), dtype=torch.float32)
            exp_p = torch.empty((1024,), dtype=torch.float32)
            for tt in range(256):
                ww, ll = divmod(tt, 64)
                aa = (((64 * ww) & 128) << 3) | ((ll & 15) << 6) | ((((tt << 2) & 56) ^ ((ll & 48) >> 1)))
                uu = ((((tt << 2) & 56) ^ ((ll & 48) >> 1)) | ((ll & 15) << 6))
                exp_a[tt * 4:tt * 4 + 4] = inv_s[aa // 2:aa // 2 + 4]
                exp_b[tt * 4:tt * 4 + 4] = u_s[uu // 2:uu // 2 + 4]
                row = 16 * (ww >> 1) + (ll & 15)
                col = (ll >> 4) * 4
                exp_p[tt * 4:tt * 4 + 4] = U[row, col:col + 4].float().cpu()
            print("A70_INV_MAX", "A", float((act_a - exp_a).abs().max()), "B", float((act_b - exp_b).abs().max()), "P1", float((debug[0, 10].flatten()[:1024].cpu() - exp_p).abs().max()), "P2", float((debug[0, 11].flatten()[:1024].cpu() - exp_p).abs().max()))
            p1_cpu = debug[0, 10].flatten()[:1024].cpu()
            p2_cpu = debug[0, 11].flatten()[:1024].cpu()
            print("A70_INV_WAVE_MAX", [(w, float((p1_cpu[w*256:(w+1)*256] - exp_p[w*256:(w+1)*256]).abs().max()), float((p2_cpu[w*256:(w+1)*256] - exp_p[w*256:(w+1)*256]).abs().max())) for w in range(4)])
            print("A70_INV_WAVE2", "actual", p1_cpu[512:528].tolist(), "expected", exp_p[512:528].tolist())
            bad_b = int((act_b - exp_b).abs().argmax().item())
            bad_p = int((debug[0, 10].flatten()[:1024].cpu() - exp_p).abs().argmax().item())
            bad_p2 = int((debug[0, 11].flatten()[:1024].cpu() - exp_p).abs().argmax().item())
            p2_cpu = debug[0, 11].flatten()[:1024].cpu()
            print("A70_INV_BAD", "B", bad_b, act_b[bad_b].item(), exp_b[bad_b].item(), "P1", bad_p, debug[0, 10].flatten()[:1024].cpu()[bad_p].item(), exp_p[bad_p].item(), "P2", bad_p2, p2_cpu[bad_p2].item(), exp_p[bad_p2].item())
            raise AssertionError(f"first strict divergence at {name}: {max_abs} / {max_rel}")
        gates[name] = {"max_abs": max_abs, "max_rel": max_rel, "pass": True}

    if not torch.isfinite(actual_out.float()).all() or not torch.isfinite(actual_state).all():
        raise AssertionError("A70 T32 produced NaN/Inf")
    return {
        "artifact": "A_20260927_70_BW16_PERSISTENT_W4_REBUILD",
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
    out = ROOT / "test/examples/kda/reports/A_20260927_70_BW16_PERSISTENT_W4_REBUILD/A_20260927_70_T32_STRICT_TRACE.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))

