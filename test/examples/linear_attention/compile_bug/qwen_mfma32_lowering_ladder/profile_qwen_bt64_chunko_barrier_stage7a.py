#!/usr/bin/env python3
"""Run Stage 7A's three minimal barrier provenance experiments.

The driver captures one HSACO per constexpr mode, checks paired semantic
equivalence, and records static synchronization/resource evidence.  It does
not launch a full Qwen graph and never imports a production selector.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from avelang.backends.amdgpu import compiler as amdgpu_compiler


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
import repro_qwen_bt64_chunko_barrier_stage7a as repro  # noqa: E402


DEFAULT_OUT = HERE / "codex_qwen_bt64_chunko_barrier_stage7a"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def tool(name: str) -> str:
    for candidate in (f"/opt/rocm/llvm/bin/{name}", f"/opt/rocm/bin/{name}", shutil.which(name)):
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise RuntimeError(f"could not locate {name}")


def capture_mode_hsaco(mode: str, path: Path, seed: int) -> None:
    lhs, rhs = repro._inputs(seed)
    out = torch.zeros((repro.WORKGROUP,), device="cuda", dtype=torch.float32)
    original = amdgpu_compiler.AmdgpuCompiler.compile
    captured = False

    def wrapped(self: Any, src: Any, target: Any, options: Any = None) -> bytes:
        nonlocal captured
        binary = original(self, src, target, options)
        if src.fn.fn.__name__ == repro._qwen_bt64_chunko_barrier_stage7a_kernel.fn.__name__ and not captured:
            path.write_bytes(binary)
            captured = True
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped
    try:
        repro.launch(mode, lhs, rhs, out)
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original
    if not captured:
        raise RuntimeError(f"did not capture Stage 7A HSACO for {mode}")


def static_isa(hsaco: Path, out_dir: Path) -> dict[str, Any]:
    objdump = tool("llvm-objdump")
    readobj = tool("llvm-readobj")
    isa = subprocess.run([objdump, "-d", str(hsaco)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout
    isa_path = out_dir / f"{hsaco.stem}.isa"
    isa_path.write_text(isa)
    notes = subprocess.run([readobj, "--notes", str(hsaco)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout
    notes_path = out_dir / f"{hsaco.stem}.readobj.txt"
    notes_path.write_text(notes)
    count = lambda pattern: len(re.findall(pattern, isa))
    return {
        "hsaco": str(hsaco),
        "isa": str(isa_path),
        "readobj": str(notes_path),
        "s_barrier": count(r"\bs_barrier\b"),
        "mfma32": count(r"v_mfma_f32_32x32x8_bf16"),
        "mfma16": count(r"v_mfma_f32_16x16x16_bf16"),
        "ds_read": count(r"\bds_read"),
        "ds_write": count(r"\bds_write"),
        "buffer_load": count(r"\bbuffer_load"),
        "buffer_store": count(r"\bbuffer_store"),
        "private_segment_fixed_size": _note_int(notes, "private_segment_fixed_size"),
        "vgpr_spill_count": _note_int(notes, "vgpr_spill_count"),
        "sgpr_spill_count": _note_int(notes, "sgpr_spill_count"),
    }


def _note_int(text: str, name: str) -> int | None:
    match = re.search(rf"\.{re.escape(name)}:\s*(\d+)", text)
    return int(match.group(1)) if match else None


def write_markdown(out: Path, rows: list[dict[str, Any]], checks: list[dict[str, Any]]) -> None:
    lines = ["# Stage 7A Minimal Barrier Repros", "", "| mode | barriers | MFMA32 | ds read/write | private | spills | median ms | finite |", "|:---|---:|---:|:---|---:|:---|---:|:---:|"]
    by_mode = {row["mode"]: row for row in rows}
    for mode in repro.MODES:
        row = by_mode[mode]
        isa = row["isa"]
        timing = row["timing"]
        lines.append(
            f"| {mode} | {isa['s_barrier']} | {isa['mfma32']} | {isa['ds_read']}/{isa['ds_write']} | "
            f"{isa['private_segment_fixed_size']} | {isa['vgpr_spill_count']}/{isa['sgpr_spill_count']} | "
            f"{timing['hip_ms_median']:.6f} | {timing['finite']} |"
        )
    lines.extend(["", "## Paired Semantics", ""])
    for check in checks:
        lines.append(
            f"- `{check['first']}` vs `{check['second']}`: bit_exact={check['bit_exact']}, "
            f"max_abs={check['max_abs']:.9g}."
        )
    (out / "minimal_repros.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=2026072207)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 7A requires the gfx942 HIP runtime")
    out = args.out_dir
    hsaco_dir = out / "minimal_hsaco"
    hsaco_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for mode in repro.MODES:
        hsaco = hsaco_dir / f"{mode}.hsaco"
        capture_mode_hsaco(mode, hsaco, args.seed)
        timing = repro.run_mode(mode, seed=args.seed, warmup=args.warmup, repeat=args.repeat)
        rows.append({"mode": mode, "timing": timing, "isa": static_isa(hsaco, hsaco_dir)})
    checks = [
        repro._equal_modes("A_direct_shared_to_mfma", "A_fragment_shared_to_mfma", args.seed),
        repro._equal_modes("B_score_reuse_split_barriers", "B_score_reuse_merged_barrier", args.seed),
    ]
    if not all(bool(row["timing"]["finite"]) for row in rows) or not all(bool(check["bit_exact"]) for check in checks):
        raise AssertionError(f"Stage 7A minimal repro gate failed: checks={checks}")
    payload = {"rows": rows, "paired_checks": checks}
    write_json(out / "minimal_repros.json", payload)
    write_markdown(out, rows, checks)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
