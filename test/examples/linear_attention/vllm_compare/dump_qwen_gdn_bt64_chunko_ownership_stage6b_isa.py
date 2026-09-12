#!/usr/bin/env python3
"""Capture and disassemble the three Stage 6B chunk-o code objects.

This is deliberately a profiling helper.  It does not modify a kernel or a
compiler pass: each function is compiled through the ordinary Avelang JIT,
the returned HSACO is copied, and the existing link-debug hook preserves its
pre-link bitcode and replayable linker arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_chunko_ownership_stage6b import qwen_gdn_chunk_o_bt64_ownership_o0  # noqa: E402
from qwen_gdn_bt64_chunko_ownership_stage6b_o1 import qwen_gdn_chunk_o_bt64_ownership_o1  # noqa: E402
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import qwen_gdn_chunk_o_bt64_mfma_v2_s0  # noqa: E402
from stage2_runner import make_inputs  # noqa: E402


KERNELS = {
    "current": "_qwen_gdn_chunk_o_bf16_kernel_bt64_mfma_v2_s0",
    "o0": "_qwen_gdn_chunk_o_bf16_kernel_bt64_ownership_o0",
    "o1": "_qwen_gdn_chunk_o_bf16_kernel_bt64_ownership_o1",
}


def _inputs(t: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(t, 20260716 + t, "random", True)
    torch.manual_seed(20260717 + t)
    v_new = torch.randn((1, t, 8, 128), device="cuda", dtype=torch.float32).contiguous()
    h = torch.randn((1, t // 64, 8, 128, 128), device="cuda", dtype=torch.bfloat16).contiguous()
    return q, k, v_new, h, g


def _call(variant: str, tensors: tuple[torch.Tensor, ...]) -> Callable[[], torch.Tensor]:
    fn = {
        "current": qwen_gdn_chunk_o_bt64_mfma_v2_s0,
        "o0": qwen_gdn_chunk_o_bt64_ownership_o0,
        "o1": qwen_gdn_chunk_o_bt64_ownership_o1,
    }[variant]
    return lambda: fn(*tensors)


def _objdump() -> str:
    for candidate in ("/opt/rocm/llvm/bin/llvm-objdump", "/opt/rocm/bin/llvm-objdump", shutil.which("llvm-objdump")):
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("llvm-objdump was not found")


def _llvm_dis() -> str | None:
    for candidate in ("/opt/rocm/llvm/bin/llvm-dis", "/opt/rocm/bin/llvm-dis", shutil.which("llvm-dis")):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _capture(variant: str, fn: Callable[[], torch.Tensor], out_dir: Path) -> dict[str, object]:
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    ir_dir = out_dir / "ir" / variant
    isa_dir = out_dir / "isa" / variant
    ir_dir.mkdir(parents=True, exist_ok=True)
    isa_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AVELANG_AMDGPU_LINK_DEBUG_DIR"] = str(ir_dir / "link")
    original = amdgpu_compiler.AmdgpuCompiler.compile
    hsaco: Path | None = None

    def wrapped(self, src, target, options=None):
        nonlocal hsaco
        binary = original(self, src, target, options)
        if hsaco is None and KERNELS[variant] in src.fn.fn.__name__:
            hsaco = isa_dir / f"{src.fn.fn.__name__}.hsaco"
            hsaco.write_bytes(binary)
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped
    try:
        output = fn()
        torch.cuda.synchronize()
        if not torch.isfinite(output).all().item():
            raise RuntimeError(f"{variant} generated non-finite output")
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original
        os.environ.pop("AVELANG_AMDGPU_LINK_DEBUG_DIR", None)
    if hsaco is None:
        raise RuntimeError(f"failed to capture {KERNELS[variant]}")

    isa = hsaco.with_suffix(".isa")
    isa.write_text(
        subprocess.run(
            [_objdump(), "-d", "--no-show-raw-insn", str(hsaco)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout
    )
    bitcode = next((ir_dir / "link").glob("*.prelink.bc"), None)
    llvm_ir = None
    if bitcode and _llvm_dis():
        llvm_ir = bitcode.with_suffix(".ll")
        subprocess.run([_llvm_dis(), str(bitcode), "-o", str(llvm_ir)], check=True)
    text = isa.read_text()
    patterns = {
        "mfma": "v_mfma",
        "ds_read": "ds_read",
        "ds_write": "ds_write",
        "buffer_load": "buffer_load",
        "buffer_store": "buffer_store",
        "flat_load": "flat_load",
        "flat_store": "flat_store",
        "global_load": "global_load",
        "global_store": "global_store",
        "barrier": "s_barrier",
        "waitcnt": "s_waitcnt",
    }
    return {
        "variant": variant,
        "kernel": KERNELS[variant],
        "hsaco": str(hsaco),
        "isa": str(isa),
        "prelink_bc": None if bitcode is None else str(bitcode),
        "llvm_ir": None if llvm_ir is None else str(llvm_ir),
        "static_counts": {name: text.count(pattern) for name, pattern in patterns.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires a ROCm/CUDA GPU")
    tensors = _inputs(args.T)
    result = [_capture(variant, _call(variant, tensors), args.out_dir) for variant in KERNELS]
    path = args.out_dir / "isa" / "static_counts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
