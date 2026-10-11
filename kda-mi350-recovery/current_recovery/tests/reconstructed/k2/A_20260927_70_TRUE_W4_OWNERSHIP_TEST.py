#!/usr/bin/env python3
"""Non-timing A70 four-wave ownership gate for the persistent NT=128 path."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
PREFILL = ROOT / "test/examples/kda/avelang_impl/prefill"
BASELINE = ROOT / "kda_baseline"
REPORT_DIR = ROOT / "test/examples/kda/reports/A_20260927_70_BW16_TRUE_W4_REBUILD"
ISA = REPORT_DIR / "A_20260927_70_T4096_final.isa"
for item in (str(PREFILL), str(BASELINE)):
    if item not in sys.path:
        sys.path.insert(0, item)

from A_20260927_70_BW16_TRUE_W4_REBUILD import HEADS, HD, TRACE_PHASES, a70_bw16_k2  # noqa: E402
from bench_common import make_kda_prefill_inputs  # noqa: E402
from kda_prefill_v0 import capture_aiter_prepare_workspace  # noqa: E402


def _isa_counts() -> dict[str, int]:
    text = ISA.read_text()
    return {
        "mfma": len(re.findall(r"v_mfma_f32_16x16x16_(?:bf16|f16)\b", text)),
        "mfma_bf16": len(re.findall(r"v_mfma_f32_16x16x16_bf16\b", text)),
        "mfma_f16": len(re.findall(r"v_mfma_f32_16x16x16_f16\b", text)),
    }


def run() -> dict[str, object]:
    T = 4096
    inp = make_kda_prefill_inputs(T=T, H=HEADS, seed=7004096)
    beta = inp["raw_beta"].float().contiguous()
    state = torch.zeros((1, HEADS, HD, HD), device=inp["q"].device, dtype=torch.float32)
    ws = capture_aiter_prepare_workspace(
        inp["q"], inp["k"], inp["v"], inp["raw_g"], beta,
        inp["A_log"], inp["dt_bias"], initial_state=state,
    )
    out, final_state, debug = a70_bw16_k2(
        workspace=ws, v=inp["v"], beta=beta, initial_state=state, debug=True,
    )
    if debug is None:
        raise AssertionError("A70 ownership probe did not return debug markers")
    debug = debug.reshape(HEADS, TRACE_PHASES, HD, 128)
    # Phase 8/9 are written by the first INV MFMA of every wave; phase 11 is
    # written after INV's continuation/drain by every wave.  The addresses are
    # lane-raw, so no logical paired-wave projection can hide wave 2/3.
    inv_a = debug[0, 8].flatten()[:1024].cpu()
    inv_b = debug[0, 9].flatten()[:1024].cpu()
    inv_done = debug[0, 11].flatten()[:1024].cpu()
    stage_names = ("Kd", "INV", "Mqk", "Qd")
    stage_counts = {}
    for stage, name in enumerate(stage_names):
        raw = debug[0, stage].flatten().cpu()[8192:8192 + 1024]
        stage_counts[name] = {
            "per_wave_unique": [
                sorted(set(float(x) for x in raw[w * 256:(w + 1) * 256:4].tolist()))
                for w in range(4)
            ],
            "per_wave_count": [int(round(float(raw[w * 256].item()))) for w in range(4)],
            "all_lanes_finite": bool(torch.isfinite(raw).all()),
        }
    waves = []
    for wave in range(4):
        sl = slice(wave * 256, (wave + 1) * 256)
        waves.append({
            "wave": wave,
            "inv_a_finite": bool(torch.isfinite(inv_a[sl]).all()),
            "inv_b_finite": bool(torch.isfinite(inv_b[sl]).all()),
            "inv_a_nonzero": int((inv_a[sl].abs() > 0).sum().item()),
            "inv_b_nonzero": int((inv_b[sl].abs() > 0).sum().item()),
            "inv_done_finite": bool(torch.isfinite(inv_done[sl]).all()),
            "inv_done_nonzero": int((inv_done[sl].abs() > 0).sum().item()),
        })
    counts = _isa_counts()
    report = {
        "artifact": "A_20260927_70_BW16_TRUE_W4_REBUILD",
        "T": T,
        "NT": 128,
        "debug_only": True,
        "production_object_untouched": True,
        "dynamic_wave_markers": waves,
        "dynamic_waves_seen": [entry["wave"] for entry in waves if entry["inv_a_nonzero"] or entry["inv_b_nonzero"] or entry["inv_done_nonzero"]],
        "dynamic_stage_counts": stage_counts,
        "isa_mfma_per_chunk_body": counts,
        "expected_mfma_per_wave_per_chunk": 24,
        "expected_mfma_per_wave_over_NT128": 24 * 128,
        "finite_output_and_state": bool(torch.isfinite(out.float()).all() and torch.isfinite(final_state).all()),
        "true_four_wave_ownership_pass": (
            [entry["wave"] for entry in waves] == [0, 1, 2, 3]
            and all(entry["inv_a_finite"] and entry["inv_b_finite"] and entry["inv_done_finite"] for entry in waves)
            and all(
                stage_counts[name]["per_wave_count"] == [128, 128, 128, 128]
                and all(len(values) == 1 and values[0] == 128.0 for values in stage_counts[name]["per_wave_unique"])
                for name in stage_names
            )
            and counts == {"mfma": 24, "mfma_bf16": 20, "mfma_f16": 4}
        ),
    }
    return report


if __name__ == "__main__":
    report = run()
    out = REPORT_DIR / "A_20260927_70_TRUE_W4_OWNERSHIP.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
