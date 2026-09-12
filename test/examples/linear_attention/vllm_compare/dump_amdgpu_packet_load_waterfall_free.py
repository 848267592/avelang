#!/usr/bin/env python3
"""Emit LLVM, pre-isel MIR and final ISA for the packet-load microrepro."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

import _avelang_bindings as _C
from avelang.compiler import code_generator as cg
from avelang.runtime.driver import driver


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
COMPILE_BUG = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
sys.path[:0] = [str(HERE), str(COMPILE_BUG)]

from dump_l6_mir_regalloc_artifacts import run_llc_variants  # noqa: E402
from repro_amdgpu_packet_load_waterfall_free import (  # noqa: E402
    PACKETS,
    _packet_load_x4_copy_kernel,
)


def _build_generator(src) -> _C.MLIRGenerator:
    constexprs_json = cg._serialize_constexprs(src)
    dependencies = cg._collect_jit_dependencies(src.fn)
    imports = cg._build_import_module([src.fn, *dependencies])
    generator = _C.MLIRGenerator()
    generator.generate_from_python_ast(imports)
    for dependency in dependencies:
        generator.add_jit_dependency(dependency.parse())
    for dependency in dependencies:
        globals_ = {}
        collect = getattr(dependency, "_collect_global_constexprs", None)
        if callable(collect):
            globals_ = collect()
        generator.visit_function_def(
            cg._get_function_def(dependency.parse()),
            cg._serialize_global_constexprs(globals_),
            "jit",
        )
    generator.visit_function_def(cg._get_function_def(src.fn.parse()), constexprs_json, "kernel")
    return generator


def _make_source():
    device = driver.active.get_current_device()
    src = torch.empty((PACKETS * 4,), device="cuda", dtype=torch.int32)
    dst = torch.empty_like(src)
    _cache, _key, target, backend, binder = _packet_load_x4_copy_kernel.device_caches[device]
    bound, specialization, options = binder(src, dst, PACKETS, num_warps=1)
    options, signature, constexprs, globals_, attrs = _packet_load_x4_copy_kernel._pack_args(
        backend, {"num_warps": 1}, bound, specialization, options
    )
    return _packet_load_x4_copy_kernel.ASTSource(
        _packet_load_x4_copy_kernel, signature, constexprs, attrs, globals_
    ), target, options


def _tool(name: str) -> str:
    for candidate in (f"/opt/rocm/llvm/bin/{name}", f"/opt/rocm/bin/{name}", name):
        if Path(candidate).exists():
            return candidate
    raise RuntimeError(f"tool not found: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("current_raw", "waterfall_free"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["AVELANG_STAGE6Z_PACKET_LOAD_LOWERING"] = args.mode
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source, target, options = _make_source()
    generator = _build_generator(source)
    llvm = generator.get_llvm_ir(target.tuple, target.chip, 1, options.num_warps)
    llvm_path = out / "lowered_llvm.ll"
    llvm_path.write_text(llvm)
    binary = _build_generator(source).compile_to_binary_bytes(
        target.tuple, target.chip, 1, options.num_warps
    )
    hsaco = out / "packet_load.hsaco"
    hsaco.write_bytes(binary)
    isa_path = out / "final_isa.s"
    subprocess.run(
        [_tool("llvm-objdump"), "-d", "--no-show-raw-insn", str(hsaco)],
        check=True,
        text=True,
        stdout=isa_path.open("w"),
    )
    llc_dir = out / "llc_mir"
    llc_dir.mkdir(exist_ok=True)
    llc = run_llc_variants(llvm_path, llc_dir, target.chip)
    isa = isa_path.read_text(errors="replace")
    summary = {
        "mode": args.mode,
        "kernel": _packet_load_x4_copy_kernel.fn.__name__,
        "hsaco_sha256": hashlib.sha256(binary).hexdigest(),
        "llvm_raw_load_calls": len(re.findall(r"raw\.buffer\.load\.v4i32", llvm)),
        "wide_packet_loads": len(re.findall(r"buffer_load_dwordx4", isa)),
        "packet_waterfall_readfirstlane": len(re.findall(r"v_readfirstlane", isa)),
        "packet_waterfall_saveexec": len(re.findall(r"s_and_saveexec", isa)),
        "packet_waterfall_backedge": len(re.findall(r"s_cbranch_execnz", isa)),
        "llc": llc,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
