#!/usr/bin/env python3
"""Dump late LLVM/MIR/regalloc artifacts for L6 baseline/subtile.

This script does not add source variants.  It intercepts the existing L6
K-stage/update repro compile path, dumps AveLang MLIR, LLVM IR, final assembly,
and then asks llc for machine IR at a few late codegen stop points.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import _avelang_bindings as _C

from avelang.compiler import code_generator as cg
from avelang.runtime.driver import driver


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[4]
DEFAULT_OUT_DIR = PROJECT_ROOT / "test/examples/linear_attention/rocprof_outputs/qwen_l6_mir_regalloc_audit"
REPRO = SCRIPT_DIR / "repro_qwen_mfma32_l6_kstage_update_variants.py"

VARIANTS = [
    "L6_baseline_current_update",
    "L6_fixed_kfrag_producer_consumer_rewrite",
    "L6_subtile16_stage_full_update_like",
]

STOP_AFTER_CANDIDATES = [
    "amdgpu-isel",
    "finalize-isel",
    "greedy",
    "virtregrewriter",
    "prologepilog",
    "postrapseudos",
    "post-RA-sched",
]


def load_repro():
    import importlib.util

    spec = importlib.util.spec_from_file_location("repro_qwen_mfma32_l6_kstage_update_variants", REPRO)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {REPRO}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_generator(src) -> _C.MLIRGenerator:
    constexprs_json = cg._serialize_constexprs(src)
    jit_deps = cg._collect_jit_dependencies(src.fn)
    import_module = cg._build_import_module([src.fn, *jit_deps])

    generator = _C.MLIRGenerator()
    generator.generate_from_python_ast(import_module)

    for dep in jit_deps:
        generator.add_jit_dependency(dep.parse())

    for dep in jit_deps:
        dep_func = cg._get_function_def(dep.parse())
        dep_globals = {}
        collect_globals = getattr(dep, "_collect_global_constexprs", None)
        if callable(collect_globals):
            dep_globals = collect_globals()
        generator.visit_function_def(dep_func, cg._serialize_global_constexprs(dep_globals), "jit")

    kernel_func = cg._get_function_def(src.fn.parse())
    generator.visit_function_def(kernel_func, constexprs_json, "kernel")
    return generator


def variant_from_src(src, repro) -> str:
    inverse = {v: k for k, v in repro.VARIANTS.items()}
    for info in getattr(src, "constants", {}).values():
        if info.get("name") == "variant":
            return inverse.get(info.get("value"), f"variant_{info.get('value')}")
    return "unknown"


def find_tool(name: str) -> str:
    candidates = [
        os.environ.get(name.upper().replace("-", "_")),
        f"/opt/rocm/llvm/bin/{name}",
        f"/opt/rocm/bin/{name}",
        shutil.which(name),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    raise RuntimeError(f"{name} not found")


def run_cmd(cmd: list[str], *, cwd: Path = PROJECT_ROOT, check: bool = False) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def run_llc_variants(ll_path: Path, out_dir: Path, chip: str) -> dict[str, dict[str, object]]:
    llc = find_tool("llc")
    out: dict[str, dict[str, object]] = {}

    # Full pass structure is useful even when individual stop points differ
    # across LLVM versions.
    structure = run_cmd(
        [
            llc,
            "-mtriple=amdgcn-amd-amdhsa",
            f"-mcpu={chip}",
            "-filetype=asm",
            "-o",
            str(out_dir / "llc_structure.s"),
            "-debug-pass=Structure",
            str(ll_path),
        ]
    )
    (out_dir / "llc_debug_pass_structure.txt").write_text(structure.stdout)
    out["debug_pass_structure"] = {"returncode": structure.returncode}

    for stop in STOP_AFTER_CANDIDATES:
        mir_path = out_dir / f"stop_after_{stop}.mir"
        result = run_cmd(
            [
                llc,
                "-mtriple=amdgcn-amd-amdhsa",
                f"-mcpu={chip}",
                "-verify-machineinstrs",
                f"-stop-after={stop}",
                "-o",
                str(mir_path),
                str(ll_path),
            ]
        )
        out[stop] = {
            "returncode": result.returncode,
            "stdout_tail": "\n".join(result.stdout.splitlines()[-20:]),
            "mir": str(mir_path) if mir_path.exists() else "",
            "bytes": mir_path.stat().st_size if mir_path.exists() else 0,
        }

    # print-after often captures useful machine IR even when stop-after names
    # are not accepted.  Dump stderr/stdout verbatim.
    for pass_name in ["greedy", "virtregrewriter"]:
        result = run_cmd(
            [
                llc,
                "-mtriple=amdgcn-amd-amdhsa",
                f"-mcpu={chip}",
                "-filetype=asm",
                f"-print-after={pass_name}",
                "-o",
                str(out_dir / f"print_after_{pass_name}.s"),
                str(ll_path),
            ]
        )
        txt = out_dir / f"print_after_{pass_name}.txt"
        txt.write_text(result.stdout)
        out[f"print_after_{pass_name}"] = {
            "returncode": result.returncode,
            "txt": str(txt),
            "bytes": txt.stat().st_size,
        }
    return out


AGPR_RE = re.compile(r"\b(?:%|\\$)?a(?:cc)?(?:vgpr)?(?:_)?([0-9]+)\b|\b(?:AGPR|AReg)_(\d+)\b", re.IGNORECASE)
VREG_RE = re.compile(r"%([0-9]+):|%([0-9]+)\\b")


def summarize_mir(path: Path) -> dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    text = path.read_text(errors="replace")
    agprs: list[int] = []
    for m in AGPR_RE.finditer(text):
        for g in m.groups():
            if g is not None:
                agprs.append(int(g))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "lines": len(text.splitlines()),
        "contains_v_accvgpr_write": "V_ACCVGPR_WRITE" in text or "v_accvgpr_write" in text,
        "contains_v_mfma": "V_MFMA" in text or "v_mfma" in text,
        "max_agpr_best_effort": max(agprs) if agprs else None,
    }


def make_ast_source(repro, variant: str, tensors):
    fn = repro._qwen_mfma32_l6_kstage_update_variants_kernel
    device = driver.active.get_current_device()
    # create_binder side effect stores ASTSource on the JITFunction instance.
    _kernel_cache, _key_cache, target, backend, binder = fn.device_caches[device]
    bound_args, specialization, options = binder(*tensors, repro.VARIANTS[variant], num_warps=2)
    options, signature, constexprs, global_constexprs, attrs = fn._pack_args(
        backend,
        {"num_warps": 2},
        bound_args,
        specialization,
        options,
    )
    return fn.ASTSource(fn, signature, constexprs, attrs, global_constexprs), target, options


def dump_for_variants(args: argparse.Namespace) -> dict[str, object]:
    repro = load_repro()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dumped: dict[str, object] = {}
    tensors = repro.make_inputs(args.seed)
    for variant in VARIANTS:
        src, target, options = make_ast_source(repro, variant, tensors)
        variant_dir = args.out_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        num_warps = getattr(options, "num_warps", -1)
        generator = build_generator(src)
        llvm_ir = generator.get_llvm_ir(target.tuple, target.chip, args.opt_level, num_warps)
        (variant_dir / "avelang_initial_mlir_status.txt").write_text(
            "Skipped: MLIRGenerator.get_mlir() currently trips an MLIR printer "
            "assertion on this generated module in the ROCm Docker build. "
            "LLVM IR and AMDGPU assembly were dumped from the same generator.\n"
        )
        ll_path = variant_dir / "lowered_optimized.ll"
        ll_path.write_text(llvm_ir)
        llc_summary = run_llc_variants(ll_path, variant_dir, target.chip)
        mir_summary = {}
        for stop in STOP_AFTER_CANDIDATES:
            p = Path(llc_summary.get(stop, {}).get("mir", ""))
            if p:
                mir_summary[stop] = summarize_mir(p)
        dumped[variant] = {
            "target": target.tuple,
            "chip": target.chip,
            "num_warps": num_warps,
            "mlir": str(variant_dir / "avelang_initial_mlir_status.txt"),
            "llvm_ir": str(ll_path),
            "assembly": str(variant_dir / "llc_structure.s"),
            "llc": llc_summary,
            "mir": mir_summary,
        }

    (args.out_dir / "dump_summary.json").write_text(json.dumps(dumped, indent=2, sort_keys=True))
    return dumped


def write_status_report(args: argparse.Namespace, dumped: dict[str, object]) -> Path:
    report = args.out_dir / "mir_dump_status.md"
    lines = ["# L6 MIR/Register-Allocation Dump Status\n"]
    lines.append("| variant | LLVM IR | assembly | MIR stop points with files | print-after dumps |")
    lines.append("|:---|:---|:---|:---|:---|")
    for variant, row in dumped.items():
        mir_ok = []
        for stop, info in row.get("mir", {}).items():
            if info.get("bytes", 0):
                mir_ok.append(stop)
        pa = []
        for key, info in row.get("llc", {}).items():
            if key.startswith("print_after") and info.get("bytes", 0):
                pa.append(key)
        lines.append(
            f"| {variant} | `{row.get('llvm_ir')}` | `{row.get('assembly')}` | "
            f"{', '.join(mir_ok) or 'none'} | {', '.join(pa) or 'none'} |"
        )
    lines.append("\nThe Python/JIT API provides initial AveLang MLIR, optimized LLVM IR, and final AMDGPU assembly.")
    lines.append("MIR availability depends on which `llc -stop-after` names are accepted by the installed AMD LLVM.")
    report.write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--opt-level", type=int, default=2)
    args = parser.parse_args()
    dumped = dump_for_variants(args)
    report = write_status_report(args, dumped)
    print(f"out_dir={args.out_dir}")
    print(f"summary={args.out_dir / 'dump_summary.json'}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
