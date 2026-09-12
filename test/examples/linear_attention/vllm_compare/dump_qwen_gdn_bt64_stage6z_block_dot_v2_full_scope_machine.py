#!/usr/bin/env python3
"""Capture BDV2 source, lowered LLVM, pre-LTO ISA and HSACO identity."""

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
sys.path.insert(0, str(HERE))

from qwen_gdn_bt64_native_chunko_stage6z_block_dot_v2_full_scope import (  # noqa: E402
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_bdv2_full_scope,
    set_block_dot_lowering,
    set_block_dot_planner,
    set_block_dot_operand_preservation,
)


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
    device = driver.active.get_current_device()
    q = torch.empty((1, t, 4, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.empty_like(q)
    vn = torch.empty((1, t, 8, 128), device="cuda", dtype=torch.bfloat16)
    h = torch.empty((1, t // 64, 8, 128, 128), device="cuda", dtype=torch.bfloat16)
    g = torch.empty((1, t, 8), device="cuda", dtype=torch.float32)
    out = torch.empty_like(vn)
    kernel = _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_bdv2_full_scope
    _cache, _key, target, backend, binder = kernel.device_caches[device]
    bound, specialization, options = binder(
        q, k, vn, h, g, out, float(128 ** -0.5), t, t // 64,
        num_warps=num_warps,
    )
    options, signature, constexprs, globals_, attrs = kernel._pack_args(
        backend, {"num_warps": num_warps}, bound, specialization, options
    )
    src = kernel.ASTSource(kernel, signature, constexprs, attrs, globals_)
    return src, target, options


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("generic", "specialized"), required=True)
    parser.add_argument("--planner", choices=("legacy", "bdv2_p1_affine"), default="legacy")
    parser.add_argument(
        "--preservation",
        choices=("none", "p2_first_class", "p3_packed_reuse", "p4_accumulator_reuse"),
        default="none",
    )
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--skip-initial-mlir", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_block_dot_lowering(args.variant)
    set_block_dot_planner(args.planner)
    set_block_dot_operand_preservation(args.preservation)
    # Preserve the semantic-lifetime checkpoints for the P2 arm.  The
    # compiler hooks are inert unless these are explicitly enabled, so the
    # ordinary benchmark path remains unchanged.
    ir_dump = args.out_dir / "ir"
    ir_dump.mkdir(parents=True, exist_ok=True)
    os.environ["AVELANG_QWEN_KFRAG_AB_DUMP_DIR"] = str(ir_dump)
    os.environ["AVELANG_QWEN_KFRAG_CONVERGENCE_AUDIT"] = "1"
    link_debug = args.out_dir / "link_debug"
    link_debug.mkdir(parents=True, exist_ok=True)
    os.environ["AVELANG_AMDGPU_LINK_DEBUG_DIR"] = str(link_debug)
    src, target, options = make_ast_source(args.T, 4)
    kernel = _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_bdv2_full_scope
    source = inspect.getsource(kernel.fn)
    summary = {
        "variant": args.variant,
        "planner": args.planner,
        "preservation": args.preservation,
        "T": args.T,
        "workgroup": 256,
        "num_warps": 4,
        "chip": target.chip,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_file": str(Path(inspect.getsourcefile(kernel.fn)).resolve()),
    }
    (args.out_dir / "source.py.txt").write_text(source)
    generator = build_generator(src)
    if args.skip_initial_mlir:
        summary["initial_mlir"] = "skipped by command line"
    else:
        try:
            initial = generator.get_mlir()
            (args.out_dir / "initial_mlir.mlir").write_text(initial)
            summary["initial_mlir_sha256"] = hashlib.sha256(initial.encode()).hexdigest()
        except Exception as exc:
            (args.out_dir / "initial_mlir.status.txt").write_text(
                f"unavailable: {type(exc).__name__}: {exc}\n"
            )
            summary["initial_mlir"] = "unavailable"
    llvm = generator.get_llvm_ir(target.tuple, target.chip, 4, options.num_warps)
    (args.out_dir / "lowered_llvm.ll").write_text(llvm)
    try:
        assembly_generator = build_generator(src)
        assembly = assembly_generator.get_assembly(
            target.tuple, target.chip, 4, options.num_warps
        )
        (args.out_dir / "pre_lto_amdgcn.s").write_text(assembly)
    except Exception as exc:
        (args.out_dir / "pre_lto_amdgcn.status.txt").write_text(
            f"unavailable: {type(exc).__name__}: {exc}\n"
        )
        summary["pre_lto_assembly"] = "unavailable"
    binary_generator = build_generator(src)
    binary = binary_generator.compile_to_binary_bytes(
        target.tuple, target.chip, 4, options.num_warps
    )
    hsaco = args.out_dir / f"bdv2_{args.variant}.hsaco"
    hsaco.write_bytes(binary)
    summary["hsaco_sha256"] = hashlib.sha256(binary).hexdigest()
    llvm_objdump = "/opt/rocm/llvm/bin/llvm-objdump"
    llvm_readelf = "/opt/rocm/llvm/bin/llvm-readelf"
    with (args.out_dir / "final_isa.s").open("w") as output:
        subprocess.run(
            [llvm_objdump, "-d", "--no-show-raw-insn", str(hsaco)],
            check=True,
            stdout=output,
        )
    with (args.out_dir / "code_object_notes.txt").open("w") as output:
        subprocess.run(
            [llvm_readelf, "--notes", str(hsaco)], check=True, stdout=output
        )
    isa = (args.out_dir / "final_isa.s").read_text(errors="replace")
    notes = (args.out_dir / "code_object_notes.txt").read_text(errors="replace")
    summary["isa_counts"] = {
        "mfma32": len(re.findall(r"\bv_mfma_f32_32x32x8_bf16\b", isa)),
        "s_barrier": len(re.findall(r"\bs_barrier\b", isa)),
        "global_load": len(re.findall(r"\bglobal_load", isa)),
        "global_store": len(re.findall(r"\bglobal_store", isa)),
        "ds_read": len(re.findall(r"\bds_read", isa)),
        "ds_write": len(re.findall(r"\bds_write", isa)),
    }
    for field in (
        "agpr_count", "vgpr_count", "sgpr_count", "group_segment_fixed_size",
        "private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count",
    ):
        match = re.search(rf"\.{field}:\s+(\d+)", notes)
        if match:
            summary[field] = int(match.group(1))
    argv_files = sorted(link_debug.glob("*.argv.txt"))
    if argv_files:
        exact_lto = args.out_dir / "exact_lto"
        exact_lto.mkdir(parents=True, exist_ok=True)
        replay = COMPILE_BUG / "replay_qwen_v29_lto_mir.py"
        replay_result = subprocess.run(
            [
                sys.executable,
                str(replay),
                "--argv-file",
                str(argv_files[0]),
                "--out-dir",
                str(exact_lto),
                "--kernel",
                kernel.fn.__name__,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (args.out_dir / "lto_replay.stdout.txt").write_text(replay_result.stdout)
        summary["exact_lto"] = str(exact_lto)
        summary["lto_replay_returncode"] = replay_result.returncode
    else:
        summary["exact_lto"] = "unavailable: linker argv not captured"
        summary["lto_replay_returncode"] = None
    (args.out_dir / "machine_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
