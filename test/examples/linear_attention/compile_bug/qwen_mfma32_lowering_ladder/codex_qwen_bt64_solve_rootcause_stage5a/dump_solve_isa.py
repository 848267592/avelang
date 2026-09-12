#!/usr/bin/env python3
"""Capture only existing v18/vLLM solve code objects for Stage5A ISA audit."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
COMPARE = REPO / "test/examples/linear_attention/vllm_compare"
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(COMPARE), str(STAGE2), str(HERE)]

from solve_rootcause_harness import (  # noqa: E402
    VLLM_SOLVE,
    VLLM_MERGE64,
    VLLM_PRECISION,
    avelang_direct,
    stage4_a,
    synchronize,
    vllm_kernel_at_config,
)


def disassemble(hsaco: Path, destination: Path) -> None:
    result = subprocess.run(
        ["/opt/rocm/llvm/bin/llvm-objdump", "-d", "--no-show-raw-insn", str(hsaco)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    destination.write_text(result.stdout)


def capture_avelang(destination: Path) -> None:
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    destination.mkdir(parents=True, exist_ok=True)
    captured: list[Path] = []
    original = amdgpu_compiler.AmdgpuCompiler.compile

    def wrapped(self, src, target, options=None):
        binary = original(self, src, target, options)
        name = src.fn.fn.__name__
        if name == "_qwen_gdn_solve_kernel_v18_parallel" and not captured:
            path = destination / "qwen_v18_bt64_solve.hsaco"
            path.write_bytes(binary)
            captured.append(path)
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped
    try:
        a = stage4_a(2048, 15111)
        avelang_direct(a, torch.empty_like(a))
        synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original
    if not captured:
        raise RuntimeError("failed to intercept the existing v18 solve HSACO")
    disassemble(captured[0], destination / "qwen_v18_bt64_solve.isa")


def capture_vllm(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    config_path = HERE / "vllm_selected_config.json"
    config = json.loads(config_path.read_text())
    a = stage4_a(2048, 15112)
    VLLM_SOLVE(A=a, output_dtype=torch.float32)
    synchronize()
    out = torch.zeros_like(a)
    vllm_kernel_at_config(a, out, int(config["num_warps"]), int(config["num_stages"]))
    synchronize()
    candidates = []
    for metadata in Path("/root/.triton/cache").rglob("merge_16x16_to_64x64_inverse_kernel.json"):
        record = json.loads(metadata.read_text())
        if record.get("num_warps") == config["num_warps"] and record.get("num_stages") == config["num_stages"]:
            hsaco = metadata.with_suffix(".hsaco")
            if hsaco.exists():
                candidates.append((metadata.stat().st_mtime_ns, metadata, hsaco, record))
    if not candidates:
        raise RuntimeError("unable to find a matching selected Triton solve HSACO")
    _, metadata, hsaco, record = max(candidates)
    copied = destination / "vllm_merge_16x16_to_64x64_selected.hsaco"
    shutil.copy2(hsaco, copied)
    shutil.copy2(metadata, destination / "vllm_merge_16x16_to_64x64_selected.json")
    disassemble(copied, destination / "vllm_merge_16x16_to_64x64_selected.isa")
    (destination / "selection.json").write_text(json.dumps({"selected_config": config, "cache_metadata": record,
                                                               "cache_source": str(metadata), "precision": str(VLLM_PRECISION)}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=["avelang", "vllm", "all"], default="all")
    args = parser.parse_args()
    if args.implementation in ("avelang", "all"):
        capture_avelang(HERE / "isa" / "avelang")
    if args.implementation in ("vllm", "all"):
        capture_vllm(HERE / "isa" / "vllm")


if __name__ == "__main__":
    main()
