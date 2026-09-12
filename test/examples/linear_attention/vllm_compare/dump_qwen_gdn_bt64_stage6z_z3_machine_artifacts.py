#!/usr/bin/env python3
"""Compile-only machine-artifact capture for the fixed Stage 6Z Z3 source.

This utility reconstructs the exact ASTSource used by the Z3 JIT without
launching the kernel.  It records initial MLIR, lowered LLVM, pre-LTO AMDGCN
assembly, the linked HSACO, and the replayable full-LTO linker input.  The
existing replay driver is then used to extract pre/post-RA MIR.  No benchmark,
rocprof collection, selector, or X2 graph is executed here.
"""

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
sys.path.insert(0, str(COMPILE_BUG))
sys.path.insert(0, str(HERE))

from dump_l6_mir_regalloc_artifacts import run_llc_variants  # noqa: E402
from qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware import (  # noqa: E402
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z2,
)
from qwen_gdn_bt64_native_chunko_stage6z_z3_wg128 import (  # noqa: E402
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z3,
)


VARIANTS = {
    "z2": {
        "kernel": _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z2,
        "num_warps": 4,
        "workgroup": 256,
    },
    "z3": {
        "kernel": _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z3,
        "num_warps": 2,
        "workgroup": 128,
    },
}


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


def make_ast_source(t: int, kernel, num_warps: int):
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
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="z3")
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--skip-initial-mlir", action="store_true")
    parser.add_argument("--skip-pre-lto-assembly", action="store_true")
    args = parser.parse_args()
    variant = VARIANTS[args.variant]
    kernel = variant["kernel"]
    kernel_name = kernel.fn.__name__
    if args.out_dir is None:
        default_dir = f"codex_qwen_bt64_stage6z_{args.variant}_debug"
        args.out_dir = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder" / default_dir / "machine"
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    link_debug = out / "link_debug"
    exact_lto = out / "exact_lto"
    link_debug.mkdir(parents=True, exist_ok=True)
    os.environ["AVELANG_AMDGPU_LINK_DEBUG_DIR"] = str(link_debug)

    print(f"[{args.variant}-machine] reconstruct ASTSource T={args.T}", flush=True)
    src, target, options = make_ast_source(args.T, kernel, variant["num_warps"])
    summary: dict[str, object] = {
        "variant": args.variant,
        "kernel": kernel_name,
        "T": args.T,
        "workgroup": variant["workgroup"],
        "target": target.tuple,
        "chip": target.chip,
        "num_warps": getattr(options, "num_warps", -1),
        "launch_executed": False,
        "rocprof_executed": False,
    }

    if args.skip_initial_mlir:
        summary["initial_mlir"] = "skipped by command line"
    else:
        print(f"[{args.variant}-machine] dumping initial MLIR", flush=True)
        try:
            (out / "initial_mlir.mlir").write_text(build_generator(src).get_mlir())
            summary["initial_mlir"] = str(out / "initial_mlir.mlir")
        except Exception as exc:  # Some ROCm builds assert while printing this module.
            (out / "initial_mlir.status.txt").write_text(f"unavailable: {type(exc).__name__}: {exc}\n")
            summary["initial_mlir"] = "unavailable; see initial_mlir.status.txt"

    print(f"[{args.variant}-machine] lowering LLVM IR", flush=True)
    llvm_path = out / "lowered_llvm.ll"
    llvm_path.write_text(build_generator(src).get_llvm_ir(target.tuple, target.chip, variant["num_warps"], options.num_warps))
    summary["llvm"] = str(llvm_path)

    if args.skip_pre_lto_assembly:
        summary["pre_lto_assembly"] = "skipped by command line"
    else:
        print(f"[{args.variant}-machine] generating pre-LTO AMDGCN assembly", flush=True)
        assembly_path = out / "pre_lto_amdgcn.s"
        assembly_path.write_text(build_generator(src).get_assembly(target.tuple, target.chip, variant["num_warps"], options.num_warps))
        summary["pre_lto_assembly"] = str(assembly_path)

    print(f"[{args.variant}-machine] linking HSACO", flush=True)
    binary = build_generator(src).compile_to_binary_bytes(
        target.tuple, target.chip, variant["num_warps"], options.num_warps
    )
    hsaco = out / f"{args.variant}_fixed.hsaco"
    hsaco.write_bytes(binary)
    summary["hsaco"] = str(hsaco)
    summary["hsaco_sha256"] = hashlib.sha256(binary).hexdigest()

    objdump = tool("llvm-objdump")
    readelf = tool("llvm-readelf")
    run_tool([objdump, "-d", "--no-show-raw-insn", str(hsaco)], out / "final_isa.s")
    run_tool([readelf, "--notes", str(hsaco)], out / "code_object_notes.txt")
    summary["final_isa"] = str(out / "final_isa.s")
    summary["code_object_notes"] = str(out / "code_object_notes.txt")
    isa_text = (out / "final_isa.s").read_text(errors="replace")
    notes_text = (out / "code_object_notes.txt").read_text(errors="replace")
    summary["final_isa_counts"] = {
        "mfma32": len(re.findall(r"\bv_mfma_f32_32x32x8_bf16\b", isa_text)),
        "s_barrier": len(re.findall(r"\bs_barrier\b", isa_text)),
        "global_load": len(re.findall(r"\bglobal_load", isa_text)),
        "global_store": len(re.findall(r"\bglobal_store", isa_text)),
        "ds_read": len(re.findall(r"\bds_read", isa_text)),
        "ds_write": len(re.findall(r"\bds_write", isa_text)),
        "v_lshl_add": len(re.findall(r"\bv_lshl_add", isa_text)),
        "v_add": len(re.findall(r"\bv_add", isa_text)),
    }
    for field in (
        "agpr_count", "vgpr_count", "sgpr_count", "group_segment_fixed_size",
        "private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count",
    ):
        match = re.search(rf"\.{field}:\s+(\d+)", notes_text)
        if match:
            summary[field] = int(match.group(1))
    (out / "machine_evidence.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("[z3-machine] replaying exact LTO", flush=True)
    argv_files = sorted(link_debug.glob("*.argv.txt"))
    summary["link_argv_files"] = [str(path) for path in argv_files]
    replay = COMPILE_BUG / "replay_qwen_v29_lto_mir.py"
    if argv_files:
        exact_lto.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                sys.executable,
                str(replay),
                "--argv-file",
                str(argv_files[0]),
                "--out-dir",
                str(exact_lto),
                "--kernel",
                kernel_name,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (out / "lto_replay.stdout.txt").write_text(result.stdout)
        summary["lto_replay_returncode"] = result.returncode
        summary["exact_lto"] = str(exact_lto)
    else:
        summary["lto_replay_returncode"] = None
        summary["exact_lto"] = "not available: linker argv was not captured"

    print(f"[{args.variant}-machine] dumping llc stop-point MIR", flush=True)
    llc_dir = out / "llc_mir"
    llc_dir.mkdir(parents=True, exist_ok=True)
    llc = run_llc_variants(llvm_path, llc_dir, target.chip)
    (out / "llc_summary.json").write_text(json.dumps(llc, indent=2, sort_keys=True) + "\n")
    summary["llc_mir"] = str(llc_dir)
    (out / "machine_evidence.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "capture_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
