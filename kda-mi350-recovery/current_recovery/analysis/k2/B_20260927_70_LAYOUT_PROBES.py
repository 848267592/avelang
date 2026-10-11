#!/usr/bin/env python3
"""B70 read-only Triton layout probes; never imports or edits A69."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import torch
import triton
import triton.language as tl


ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "test/examples/kda/reports/B_20260927_70_BW16_W4_MMA1_LAYOUT_RECOVERY"


@triton.jit
def _b70_mma1_kd(a_ptr, b_ptr, out_ptr):
    m = tl.arange(0, 32)
    k = tl.arange(0, 64)
    n = tl.arange(0, 16)
    a = tl.load(a_ptr + m[:, None] * 64 + k[None, :])
    b = tl.load(b_ptr + k[:, None] * 16 + n[None, :])
    c = tl.dot(a, b, tl.zeros((32, 16), dtype=tl.float32), input_precision="tf32")
    tl.store(out_ptr + m[:, None] * 16 + n[None, :], c)


@triton.jit
def _b70_mma1_inv(a_ptr, b_ptr, out_ptr):
    m = tl.arange(0, 32)
    k = tl.arange(0, 32)
    n = tl.arange(0, 16)
    a = tl.load(a_ptr + m[:, None] * 32 + k[None, :])
    b = tl.load(b_ptr + k[:, None] * 16 + n[None, :])
    c = tl.dot(a, b, tl.zeros((32, 16), dtype=tl.float32), input_precision="tf32")
    tl.store(out_ptr + m[:, None] * 16 + n[None, :], c)


@triton.jit
def _b70_mma_kr(a_ptr, b_ptr, out_ptr):
    m = tl.arange(0, 64)
    k = tl.arange(0, 32)
    n = tl.arange(0, 16)
    a = tl.load(a_ptr + m[:, None] * 32 + k[None, :])
    b = tl.load(b_ptr + k[:, None] * 16 + n[None, :])
    c = tl.dot(a, b, tl.zeros((64, 16), dtype=tl.float32), input_precision="tf32")
    tl.store(out_ptr + m[:, None] * 16 + n[None, :], c)


def _cache_files(name: str) -> list[Path]:
    cache = Path("/root/.triton/cache")
    found = []
    for path in cache.rglob(f"_{name}.*"):
        if path.is_file() and path.suffix in {".source", ".ttir", ".ttgir", ".llir", ".amdgcn", ".hsaco", ".json"}:
            found.append(path)
    return sorted(found, key=lambda p: p.stat().st_mtime_ns)


def _run(name: str, kernel, shape_a: tuple[int, int], shape_b: tuple[int, int], shape_c: tuple[int, int]) -> dict[str, object]:
    a = torch.arange(shape_a[0] * shape_a[1], device="cuda", dtype=torch.float32).reshape(shape_a).to(torch.bfloat16)
    b = torch.arange(shape_b[0] * shape_b[1], device="cuda", dtype=torch.float32).reshape(shape_b).to(torch.bfloat16)
    c = torch.empty(shape_c, device="cuda", dtype=torch.float32)
    kernel[(1,)](a, b, c, num_warps=4, num_stages=2)
    torch.cuda.synchronize()
    result: dict[str, object] = {
        "kernel": name,
        "shape_a": list(shape_a),
        "shape_b": list(shape_b),
        "shape_c": list(shape_c),
        "finite": bool(torch.isfinite(c).all()),
        "checksum": float(c.double().sum().item()),
        "cache_files": [],
    }
    for src in _cache_files(name):
        dst = OUT_DIR / f"B_20260927_70_{name}{src.suffix}"
        shutil.copyfile(src, dst)
        result["cache_files"].append({
            "path": str(dst.relative_to(ROOT)),
            "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
        })
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "artifact": "B_20260927_70_BW16_W4_MMA1_LAYOUT_RECOVERY",
        "read_only": True,
        "target": "gfx942 BW16/W4/S2, 4 waves, num_stages=2",
        "probes": [
            _run("b70_mma1_kd", _b70_mma1_kd, (32, 64), (64, 16), (32, 16)),
            _run("b70_mma1_inv", _b70_mma1_inv, (32, 32), (32, 16), (32, 16)),
            _run("b70_mma_kr", _b70_mma_kr, (64, 32), (32, 16), (64, 16)),
        ],
    }
    (OUT_DIR / "B_20260927_70_LAYOUT_PROBES.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
