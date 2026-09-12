#!/usr/bin/env python3
"""Capture source/lowered/LTO machine artifacts for the Z7B A/B arms."""

from __future__ import annotations

import argparse
import hashlib
import inspect
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
sys.path.insert(0, str(COMPILE_BUG))
sys.path.insert(0, str(HERE))

from dump_l6_mir_regalloc_artifacts import run_llc_variants  # noqa: E402
from qwen_gdn_bt64_native_chunko_stage6z_z7b_dot_v2 import (  # noqa: E402
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z7b_dot_v2,
    set_block_dot_lowering,
)


VARIANTS = ("generic", "specialized")


def build_generator(src) -> _C.MLIRGenerator:
    constexprs_json = cg._serialize_constexprs(src)
    jit_deps = cg._collect_jit_dependencies(src.fn)
    imports = cg._build_import_module([src.fn, *jit_deps])
    generator = _C.MLIRGenerator()
    generator.generate_from_python_ast(imports)
    for dep in jit_deps:
        generator.add_jit_dependency(dep.parse())
    for dep in jit_deps:
        dep_func = cg._get_function_def(dep.parse())
        globals_ = {}
        collect = getattr(dep, "_collect_global_constexprs", None)
        if callable(collect):
            globals_ = collect()
        generator.visit_function_def(dep_func, cg._serialize_global_constexprs(globals_), "jit")
    generator.visit_function_def(cg._get_function_def(src.fn.parse()), constexprs_json, "kernel")
    return generator


def make_ast_source(t: int, num_warps: int):
    if t < 64 or t % 64:
        raise ValueError("T must be a positive multiple of 64")
    chunks = t // 64
    device = driver.active.get_current_device()
    q = torch.empty((1, t, 4, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.empty_like(q)
    vn = torch.empty((1, t, 8, 128), device="cuda", dtype=torch.bfloat16)
    h = torch.empty((1, chunks, 8, 128, 128), device="cuda", dtype=torch.bfloat16)
    g = torch.empty((1, t, 8), device="cuda", dtype=torch.float32)
    out = torch.empty_like(vn)
    kernel = _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z7b_dot_v2
    _cache, _key, target, backend, binder = kernel.device_caches[device]
    bound, specialization, options = binder(
        q, k, vn, h, g, out, float(128 ** -0.5), t, chunks, num_warps=num_warps
    )
    options, signature, constexprs, globals_, attrs = kernel._pack_args(
        backend, {"num_warps": num_warps}, bound, specialization, options
    )
    src = kernel.ASTSource(kernel, signature, constexprs, attrs, globals_)
    return src, target, options


def tool(name: str) -> str:
    for candidate in (f"/opt/rocm/llvm/bin/{name}", f"/opt/rocm/bin/{name}", name):
        if Path(candidate).exists():
            return candidate
    raise RuntimeError(f"tool not found: {name}")


def run_tool(cmd: list[str], output: Path) -> int:
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output.write_text(result.stdout)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--skip-initial-mlir", action="store_true")
    args = parser.parse_args()
    set_block_dot_lowering(args.variant)
    os.environ["AVELANG_AMDGPU_LINK_DEBUG_DIR"] = str(args.out_dir / "link_debug")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "link_debug").mkdir(parents=True, exist_ok=True)
    src, target, options = make_ast_source(args.T, 4)
    summary: dict[str, object] = {
        "variant": args.variant,
        "T": args.T,
        "workgroup": 256,
        "num_warps": 4,
        "chip": target.chip,
        "target": target.tuple,
        "launch_executed": False,
        "rocprof_executed": False,
    }
    source_text = inspect.getsource(_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z7b_dot_v2.fn)
    summary["same_source_function_sha256"] = hashlib.sha256(source_text.encode()).hexdigest()
    summary["source_file"] = str(Path(inspect.getsourcefile(_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z7b_dot_v2.fn)).resolve())
    summary["source_file_sha256"] = hashlib.sha256(Path(summary["source_file"]).read_bytes()).hexdigest()
    if args.skip_initial_mlir:
        summary["initial_mlir"] = "skipped by command line"
    else:
        try:
            initial = args.out_dir / "initial_mlir.mlir"
            initial.write_text(build_generator(src).get_mlir())
            summary["initial_mlir"] = str(initial)
            summary["initial_mlir_sha256"] = hashlib.sha256(initial.read_bytes()).hexdigest()
        except Exception as exc:
            status = args.out_dir / "initial_mlir.status.txt"
            status.write_text(f"unavailable: {type(exc).__name__}: {exc}\n")
            summary["initial_mlir"] = "unavailable; see initial_mlir.status.txt"

    llvm = args.out_dir / "lowered_llvm.ll"
    llvm.write_text(build_generator(src).get_llvm_ir(target.tuple, target.chip, 4, options.num_warps))
    summary["llvm"] = str(llvm)
    pre_lto = args.out_dir / "pre_lto_amdgcn.s"
    pre_lto.write_text(build_generator(src).get_assembly(target.tuple, target.chip, 4, options.num_warps))
    summary["pre_lto_assembly"] = str(pre_lto)
    binary = build_generator(src).compile_to_binary_bytes(target.tuple, target.chip, 4, options.num_warps)
    hsaco = args.out_dir / f"z7b_{args.variant}.hsaco"
    hsaco.write_bytes(binary)
    summary["hsaco"] = str(hsaco)
    summary["hsaco_sha256"] = hashlib.sha256(binary).hexdigest()

    objdump = tool("llvm-objdump")
    readelf = tool("llvm-readelf")
    isa = args.out_dir / "final_isa.s"
    notes = args.out_dir / "code_object_notes.txt"
    run_tool([objdump, "-d", "--no-show-raw-insn", str(hsaco)], isa)
    run_tool([readelf, "--notes", str(hsaco)], notes)
    summary["final_isa"] = str(isa)
    summary["code_object_notes"] = str(notes)
    isa_text = isa.read_text(errors="replace")
    notes_text = notes.read_text(errors="replace")
    summary["final_isa_counts"] = {
        "mfma32": len(re.findall(r"\bv_mfma_f32_32x32x8_bf16\b", isa_text)),
        "s_barrier": len(re.findall(r"\bs_barrier\b", isa_text)),
        "global_load": len(re.findall(r"\bglobal_load", isa_text)),
        "global_store": len(re.findall(r"\bglobal_store", isa_text)),
        "ds_read": len(re.findall(r"\bds_read", isa_text)),
        "ds_write": len(re.findall(r"\bds_write", isa_text)),
        "ds_bpermute": len(re.findall(r"\bds_bpermute", isa_text)),
    }
    for field in (
        "agpr_count", "vgpr_count", "sgpr_count", "group_segment_fixed_size",
        "private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count",
    ):
        match = re.search(rf"\.{field}:\s+(\d+)", notes_text)
        if match:
            summary[field] = int(match.group(1))

    exact_lto = args.out_dir / "exact_lto"
    argv_files = sorted((args.out_dir / "link_debug").glob("*.argv.txt"))
    replay = COMPILE_BUG / "replay_qwen_v29_lto_mir.py"
    if argv_files:
        exact_lto.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [sys.executable, str(replay), "--argv-file", str(argv_files[0]), "--out-dir", str(exact_lto), "--kernel", _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z7b_dot_v2.fn.__name__],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        (args.out_dir / "lto_replay.stdout.txt").write_text(result.stdout)
        summary["lto_replay_returncode"] = result.returncode
        summary["exact_lto"] = str(exact_lto)
    else:
        summary["lto_replay_returncode"] = None
        summary["exact_lto"] = "not available: linker argv was not captured"

    llc_dir = args.out_dir / "llc_mir"
    llc_dir.mkdir(parents=True, exist_ok=True)
    llc = run_llc_variants(llvm, llc_dir, target.chip)
    (args.out_dir / "llc_summary.json").write_text(json.dumps(llc, indent=2, sort_keys=True) + "\n")
    summary["llc_mir"] = str(llc_dir)
    summary["initial_mlir_sha256"] = summary.get("initial_mlir_sha256", "unavailable")
    (args.out_dir / "machine_evidence.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
