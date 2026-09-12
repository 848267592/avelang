#!/usr/bin/env python3
"""Capture C21 selected-native pipeline compiler artifacts at one T value."""

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
LADDER = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
sys.path.insert(0, str(LADDER))

from dump_l6_mir_regalloc_artifacts import run_llc_variants  # noqa: E402
from qwen_gdn_bt64_native_chunko_stage6z_c21_selected_native_pipeline import (  # noqa: E402
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_c21_selected_native_pipeline,
)


def _generator(source):
    constexprs_json = cg._serialize_constexprs(source)
    deps = cg._collect_jit_dependencies(source.fn)
    imports = cg._build_import_module([source.fn, *deps])
    generator = _C.MLIRGenerator()
    generator.generate_from_python_ast(imports)
    for dep in deps:
        generator.add_jit_dependency(dep.parse())
    for dep in deps:
        globals_ = {}
        collect = getattr(dep, "_collect_global_constexprs", None)
        if callable(collect):
            globals_ = collect()
        generator.visit_function_def(
            cg._get_function_def(dep.parse()),
            cg._serialize_global_constexprs(globals_),
            "jit",
        )
    generator.visit_function_def(
        cg._get_function_def(source.fn.parse()), constexprs_json, "kernel"
    )
    return generator


def _source(t: int):
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
    kernel = _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_c21_selected_native_pipeline
    _cache, _key, target, backend, binder = kernel.device_caches[device]
    bound, specialization, options = binder(
        q, k, vn, h, g, out, float(128 ** -0.5), t, chunks, num_warps=4
    )
    options, signature, constexprs, globals_, attrs = kernel._pack_args(
        backend, {"num_warps": 4}, bound, specialization, options
    )
    return kernel.ASTSource(kernel, signature, constexprs, attrs, globals_), target, options


def _tool(name: str) -> str:
    for candidate in (f"/opt/rocm/llvm/bin/{name}", f"/opt/rocm/bin/{name}", name):
        if Path(candidate).exists():
            return candidate
    raise RuntimeError(f"tool not found: {name}")


def _run(command: list[str], output: Path) -> None:
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    output.write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-initial-mlir", action="store_true")
    args = parser.parse_args()
    out = args.out_dir or LADDER / "codex_qwen_gfx942_c21_selected_native_pipeline" / "machine"
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    link = out / "link_debug"
    link.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "AVELANG_STAGE6Z_FULL_PHYSICAL_REGION": "c21",
            "AVELANG_BLOCK_DOT_LOWERING": "specialized",
            "AVELANG_BLOCK_DOT_LAYOUT_PLANNER": "bdv2_p1_affine",
            "AVELANG_BLOCK_DOT_OPERAND_PRESERVATION": "p2_first_class",
            "AVELANG_AMDGPU_LINK_DEBUG_DIR": str(link),
        }
    )
    source, target, options = _source(args.T)
    kernel_name = source.fn.__name__
    initial = out / "initial_mlir.mlir"
    if args.skip_initial_mlir:
        initial.write_text("skipped: initial MLIR printer is unstable for full physical-region sources\n")
    else:
        initial.write_text(_generator(source).get_mlir())
    llvm = out / "lowered_llvm.ll"
    llvm.write_text(_generator(source).get_llvm_ir(target.tuple, target.chip, 4, options.num_warps))
    pre_lto = out / "pre_lto_amdgcn.s"
    pre_lto.write_text(_generator(source).get_assembly(target.tuple, target.chip, 4, options.num_warps))
    binary = _generator(source).compile_to_binary_bytes(target.tuple, target.chip, 4, options.num_warps)
    hsaco = out / "c21_selected_native_pipeline.hsaco"
    hsaco.write_bytes(binary)
    isa = out / "final_isa.s"
    notes = out / "code_object_notes.txt"
    _run([_tool("llvm-objdump"), "-d", "--no-show-raw-insn", str(hsaco)], isa)
    _run([_tool("llvm-readobj"), "--notes", "--sections", "--symbols", str(hsaco)], notes)
    isa_text = isa.read_text(errors="replace")
    notes_text = notes.read_text(errors="replace")
    summary: dict[str, object] = {
        "schema": "qwen.gfx942.stage6z.c21.machine_capture.v1",
        "experiment": "C21-NSM",
        "kernel": kernel_name,
        "T": args.T,
        "target": target.tuple,
        "chip": target.chip,
        "launch": {"workgroup": 256, "num_warps": 4, "grid_x": (args.T // 64) * 8 * 2},
        "pipeline_mode": "gfx942_bt64_bv64_wg256_stage2",
        "artifacts": {"initial_mlir": str(initial), "llvm": str(llvm), "pre_lto": str(pre_lto), "isa": str(isa), "hsaco": str(hsaco), "notes": str(notes)},
        "hashes": {"mlir": _sha(initial), "llvm": _sha(llvm), "isa": _sha(isa), "hsaco": _sha(hsaco)},
        "static_isa": {
            "mfma32": len(re.findall(r"\bv_mfma_f32_32x32x8_bf16\b", isa_text)),
            "s_barrier": len(re.findall(r"\bs_barrier\b", isa_text)),
            "s_waitcnt": len(re.findall(r"\bs_waitcnt\b", isa_text)),
            "global_load": len(re.findall(r"\b(?:global|buffer)_load", isa_text)),
            "global_store": len(re.findall(r"\b(?:global|buffer)_store", isa_text)),
            "ds_read": len(re.findall(r"\bds_read", isa_text)),
            "ds_write": len(re.findall(r"\bds_write", isa_text)),
        },
    }
    for field in ("agpr_count", "vgpr_count", "sgpr_count", "group_segment_fixed_size", "private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count"):
        found = re.search(rf"\.{field}:\s+(\d+)", notes_text)
        if found:
            summary[field] = int(found.group(1))
    argv_files = sorted(link.glob("*.argv.txt"))
    summary["link_argv_files"] = [str(path) for path in argv_files]
    if argv_files:
        replay_out = out / "exact_lto"
        replay = LADDER / "replay_qwen_v29_lto_mir.py"
        completed = subprocess.run(
            [sys.executable, str(replay), "--argv-file", str(argv_files[0]), "--out-dir", str(replay_out), "--kernel", kernel_name],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        (out / "lto_replay.stdout.txt").write_text(completed.stdout)
        summary["exact_lto"] = str(replay_out)
        summary["lto_replay_returncode"] = completed.returncode
    llc = out / "llc_mir"
    llc.mkdir(exist_ok=True)
    (out / "llc_summary.json").write_text(json.dumps(run_llc_variants(llvm, llc, target.chip), indent=2, sort_keys=True) + "\n")
    (out / "machine_evidence.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
