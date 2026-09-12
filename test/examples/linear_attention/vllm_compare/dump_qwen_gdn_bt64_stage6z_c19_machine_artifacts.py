#!/usr/bin/env python3
"""Capture the C19 FullPhysicalRegion machine graph without benchmarking.

This is intentionally a compile-only helper.  It records the same source,
LLVM, pre-LTO AMDGCN, exact-LTO MIR and final code-object evidence used by the
C18 audit, but selects the C19 compiler-owned full physical-region plan.
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
COMPILE_BUG = REPO / (
    "test/examples/linear_attention/compile_bug/"
    "qwen_mfma32_lowering_ladder"
)
sys.path.insert(0, str(COMPILE_BUG))

from dump_l6_mir_regalloc_artifacts import run_llc_variants  # noqa: E402
from qwen_gdn_bt64_native_chunko_stage6z_c19_full_physical_region import (  # noqa: E402
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_c19_full_physical_region,
)


def build_generator(src):
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
        generator.visit_function_def(
            dep_func, cg._serialize_global_constexprs(globals_), "jit"
        )
    generator.visit_function_def(
        cg._get_function_def(src.fn.parse()), constexprs_json, "kernel"
    )
    return generator


def make_ast_source(t: int):
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
    kernel = _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_c19_full_physical_region
    _cache, _key, target, backend, binder = kernel.device_caches[device]
    bound, specialization, options = binder(
        q, k, vn, h, g, out, float(128 ** -0.5), t, chunks, num_warps=4
    )
    options, signature, constexprs, globals_, attrs = kernel._pack_args(
        backend, {"num_warps": 4}, bound, specialization, options
    )
    src = kernel.ASTSource(kernel, signature, constexprs, attrs, globals_)
    return src, target, options


def tool(name: str) -> str:
    for candidate in (f"/opt/rocm/llvm/bin/{name}", f"/opt/rocm/bin/{name}", name):
        if Path(candidate).exists():
            return candidate
    raise RuntimeError(f"tool not found: {name}")


def run_tool(command: list[str], output: Path) -> None:
    result = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    output.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-initial-mlir", action="store_true")
    parser.add_argument("--skip-pre-lto-assembly", action="store_true")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = COMPILE_BUG / "codex_qwen_gfx942_c19_full_physical_region_t2048" / "machine"
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    link_debug = out / "link_debug"
    exact_lto = out / "exact_lto"
    link_debug.mkdir(parents=True, exist_ok=True)

    os.environ.update(
        {
            "AVELANG_STAGE6Z_FULL_PHYSICAL_REGION": "c19",
            "AVELANG_BLOCK_DOT_LOWERING": "specialized",
            "AVELANG_BLOCK_DOT_LAYOUT_PLANNER": "bdv2_p1_affine",
            "AVELANG_BLOCK_DOT_OPERAND_PRESERVATION": "p2_first_class",
            "AVELANG_AMDGPU_LINK_DEBUG_DIR": str(link_debug),
        }
    )

    src, target, options = make_ast_source(args.T)
    kernel_name = src.fn.__name__
    summary: dict[str, object] = {
        "experiment": "C19-FPRO",
        "kernel": kernel_name,
        "T": args.T,
        "workgroup": 256,
        "num_warps": 4,
        "target": target.tuple,
        "chip": target.chip,
        "launch_executed": False,
        "rocprof_executed": False,
    }

    if args.skip_initial_mlir:
        summary["initial_mlir"] = "skipped by command line"
    else:
        initial = out / "initial_mlir.mlir"
        initial.write_text(build_generator(src).get_mlir())
        summary["initial_mlir"] = str(initial)

    llvm = out / "lowered_llvm.ll"
    llvm.write_text(
        build_generator(src).get_llvm_ir(
            target.tuple, target.chip, 4, options.num_warps
        )
    )
    summary["llvm"] = str(llvm)

    if args.skip_pre_lto_assembly:
        summary["pre_lto_assembly"] = "skipped by command line"
    else:
        pre_lto = out / "pre_lto_amdgcn.s"
        pre_lto.write_text(
            build_generator(src).get_assembly(
                target.tuple, target.chip, 4, options.num_warps
            )
        )
        summary["pre_lto_assembly"] = str(pre_lto)

    binary = build_generator(src).compile_to_binary_bytes(
        target.tuple, target.chip, 4, options.num_warps
    )
    hsaco = out / "c19_full_physical_region.hsaco"
    hsaco.write_bytes(binary)
    summary["hsaco"] = str(hsaco)
    summary["hsaco_sha256"] = hashlib.sha256(binary).hexdigest()

    objdump = tool("llvm-objdump")
    readobj = tool("llvm-readobj")
    isa = out / "final_isa.s"
    notes = out / "code_object_notes.txt"
    run_tool([objdump, "-d", "--no-show-raw-insn", str(hsaco)], isa)
    run_tool([readobj, "--notes", "--sections", "--symbols", str(hsaco)], notes)
    summary["final_isa"] = str(isa)
    summary["code_object_notes"] = str(notes)

    isa_text = isa.read_text(errors="replace")
    notes_text = notes.read_text(errors="replace")
    summary["static_isa"] = {
        "mfma32": len(re.findall(r"\bv_mfma_f32_32x32x8_bf16\b", isa_text)),
        "s_barrier": len(re.findall(r"\bs_barrier\b", isa_text)),
        "s_waitcnt": len(re.findall(r"\bs_waitcnt\b", isa_text)),
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

    (out / "machine_evidence.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    argv_files = sorted(link_debug.glob("*.argv.txt"))
    summary["link_argv_files"] = [str(path) for path in argv_files]
    replay = COMPILE_BUG / "replay_qwen_v29_lto_mir.py"
    if argv_files:
        exact_lto.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                sys.executable, str(replay), "--argv-file", str(argv_files[0]),
                "--out-dir", str(exact_lto), "--kernel", kernel_name,
            ],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        (out / "lto_replay.stdout.txt").write_text(result.stdout)
        summary["lto_replay_returncode"] = result.returncode
        summary["exact_lto"] = str(exact_lto)
    else:
        summary["lto_replay_returncode"] = None
        summary["exact_lto"] = "not available: linker argv was not captured"

    llc_dir = out / "llc_mir"
    llc_dir.mkdir(parents=True, exist_ok=True)
    summary["llc_mir"] = str(llc_dir)
    (out / "llc_summary.json").write_text(
        json.dumps(run_llc_variants(llvm, llc_dir, target.chip), indent=2, sort_keys=True)
        + "\n"
    )
    (out / "machine_evidence.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
