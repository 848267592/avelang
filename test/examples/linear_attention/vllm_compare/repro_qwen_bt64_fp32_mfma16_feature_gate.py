"""Live Stage 5B source-level FP32 MFMA16 availability gate.

This is intentionally minimal: one 64-thread wave loads one FP32 fragment per
lane and invokes the canonical high-level FP32 16x16x4 MFMA name.  The source
gate is paired with the ISA audit in ``compile_bug/qwen_mfma32_lowering_ladder``;
S0 still needs its own correctness and performance validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch

import avelang
import avelang.language as al


@avelang.jit
def _probe_mfma_16x16x4_f32_f32(
    src_ptr: al.Pointer(al.f32), out_ptr: al.Pointer(al.f32)
):
    values = al.make_tensor(src_ptr, al.f32, al.make_layout((64,), (1,)))
    lane = al.thread_id(0)
    values_fragment = al.view(values, al.f32, al.make_layout((64, 1), (1, 1)))
    a_fragment = values_fragment[lane]
    b_fragment = values_fragment[(lane + 1) & 63]
    acc = al.full((4,), 0.0, al.f32)
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((64, 4), (4, 1)))
    acc = al.amdgpu.mfma_16x16x4_f32_f32(a_fragment, b_fragment, acc)
    for i in al.range(4):
        out[lane, i] = acc[i]


PROBES: tuple[tuple[str, Callable], ...] = (
    ("mfma_16x16x4_f32_f32", _probe_mfma_16x16x4_f32_f32),
)


def run() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"gpu_available": False, "available": False, "attempts": []}

    src = torch.arange(64, device="cuda", dtype=torch.float32)
    out = torch.empty((64, 4), device="cuda", dtype=torch.float32)
    attempts: list[dict[str, object]] = []
    for name, probe in PROBES:
        try:
            probe[lambda: ((1, 1, 1), (64, 1, 1))](src, out)
            torch.cuda.synchronize()
        except Exception as exc:  # The expected result when no source entry exists.
            attempts.append({"name": name, "compiled": False, "error": str(exc)})
        else:
            attempts.append(
                {
                    "name": name,
                    "compiled": True,
                    "finite": bool(torch.isfinite(out).all().item()),
                }
            )
    return {
        "gpu_available": True,
        "available": any(bool(item["compiled"]) for item in attempts),
        "attempts": attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = run()
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n")


if __name__ == "__main__":
    main()
