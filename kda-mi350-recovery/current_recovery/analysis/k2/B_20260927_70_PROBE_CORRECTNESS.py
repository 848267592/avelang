#!/usr/bin/env python3
"""B70-only logical matrix reconstruction and four-wave coverage gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "test/examples/kda/tools"))
from B_20260927_70_LAYOUT_PROBES import _b70_mma1_inv, _b70_mma1_kd, _b70_mma_kr  # noqa: E402


OUT_DIR = ROOT / "test/examples/kda/reports/B_20260927_70_BW16_W4_MMA1_LAYOUT_RECOVERY"
OUT = OUT_DIR / "B_20260927_70_PROBE_CORRECTNESS.json"


def _matrix_case(name, kernel, shape_a, shape_b, shape_c):
    count_a = shape_a[0] * shape_a[1]
    count_b = shape_b[0] * shape_b[1]
    a = torch.linspace(-0.5, 0.5, count_a, device="cuda", dtype=torch.float32).reshape(shape_a).to(torch.bfloat16)
    b = torch.linspace(0.25, -0.25, count_b, device="cuda", dtype=torch.float32).reshape(shape_b).to(torch.bfloat16)
    out = torch.empty(shape_c, device="cuda", dtype=torch.float32)
    kernel[(1,)](a, b, out, num_warps=4, num_stages=2)
    torch.cuda.synchronize()
    ref = a.float() @ b.float()
    diff = (out - ref).abs()
    max_abs = float(diff.max().item())
    max_rel = float((diff / ref.abs().clamp_min(1.0e-5)).max().item())
    return {
        "name": name,
        "shape": [list(shape_a), list(shape_b), list(shape_c)],
        "finite": bool(torch.isfinite(out).all()),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "allclose_atol_1e-2_rtol_1e-2": bool(torch.allclose(out, ref, atol=1.0e-2, rtol=1.0e-2)),
    }


def _coverage():
    packet = {}
    packet["blocked1_32x64"] = sorted({
        (8 * w + lane // 8, 8 * (lane % 8) + q)
        for w in range(4) for lane in range(64) for q in range(8)
    })
    packet["blocked4_32x32"] = sorted({
        (8 * w + lane // 8, 4 * (lane % 8) + q)
        for w in range(4) for lane in range(64) for q in range(4)
    })
    packet["blocked3_32x16"] = sorted({
        (8 * w + lane // 8, 2 * (lane % 8) + q)
        for w in range(4) for lane in range(64) for q in range(2)
    })
    packet["blocked_64x32"] = sorted({
        (8 * (lane % 8) + q, 8 * w + lane // 8)
        for w in range(4) for lane in range(64) for q in range(8)
    })
    expected = {
        "blocked1_32x64": 32 * 64,
        "blocked4_32x32": 32 * 32,
        "blocked3_32x16": 32 * 16,
        "blocked_64x32": 64 * 32,
    }
    result = {name: {"elements": len(values), "expected": expected[name], "exact": len(values) == expected[name]} for name, values in packet.items()}
    mma1_all = {(16 * (w // 2) + (lane & 15), 4 * (lane >> 4) + f) for w in range(4) for lane in range(64) for f in range(4)}
    mma1_stored = {(16 * (w // 2) + (lane & 15), 4 * (lane >> 4) + f) for w in (0, 2) for lane in range(64) for f in range(4)}
    mma = {(16 * w + (lane & 15), 4 * (lane >> 4) + f) for w in range(4) for lane in range(64) for f in range(4)}
    result["mma1_result_all_waves"] = {"elements": len(mma1_all), "expected": 32 * 16, "exact": len(mma1_all) == 32 * 16}
    result["mma1_result_store_active_even_waves"] = {"elements": len(mma1_stored), "expected": 32 * 16, "exact": len(mma1_stored) == 32 * 16}
    result["mma_result"] = {"elements": len(mma), "expected": 64 * 16, "exact": len(mma) == 64 * 16}
    return result


def main():
    cases = [
        _matrix_case("Kd_Qd_mma1_shape", _b70_mma1_kd, (32, 64), (64, 16), (32, 16)),
        _matrix_case("INV_Mqk_mma1_shape", _b70_mma1_inv, (32, 32), (32, 16), (32, 16)),
        _matrix_case("Kr_mma_shape", _b70_mma_kr, (64, 32), (32, 16), (64, 16)),
    ]
    coverage = _coverage()
    report = {
        "artifact": "B_20260927_70_BW16_W4_MMA1_LAYOUT_RECOVERY",
        "read_only": True,
        "matrix_cases": cases,
        "coverage": coverage,
        "matrix_reconstruction_pass": all(x["finite"] and x["allclose_atol_1e-2_rtol_1e-2"] for x in cases),
        "four_wave_coverage_pass": all(x["exact"] for x in coverage.values()),
        "accepted": all(x["finite"] and x["allclose_atol_1e-2_rtol_1e-2"] for x in cases) and all(x["exact"] for x in coverage.values()),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
