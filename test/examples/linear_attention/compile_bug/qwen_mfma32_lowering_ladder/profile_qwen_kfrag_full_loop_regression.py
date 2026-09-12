#!/usr/bin/env python3
"""Profile the five-step full-loop K-fragment regression ladder."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

from profile_qwen_kfrag_helper_lowering import (
    PMCS,
    analyze_high_agpr,
    analyze_isa,
    disassemble,
    parse_rocprof_dir,
    run_cmd,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[4]
REPRO = SCRIPT_DIR / "repro_qwen_kfrag_full_loop_regression.py"
OUT_DIR = PROJECT_ROOT / "test/examples/linear_attention/rocprof_outputs/qwen_full_kfrag_rewrite_regression_audit"
KERNEL = "_qwen_kfrag_full_loop_regression_kernel"


def load_repro():
    spec = importlib.util.spec_from_file_location("qwen_full_loop_repro", REPRO)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {REPRO}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_json(text: str):
    start = text.find("[")
    if start < 0:
        raise RuntimeError(f"missing JSON output:\n{text}")
    return json.loads(text[start:])


def run_smoke(args, variants):
    rows = []
    debug_output = []
    for variant in variants:
        result = run_cmd([
            sys.executable, str(REPRO), "--variant", variant,
            "--seed", str(args.seed), "--warmup", str(args.warmup),
            "--repeat", str(args.repeat), "--json",
        ], check=False)
        if result.returncode:
            raise RuntimeError(result.stdout)
        row = parse_json(result.stdout)[0]
        row["rewrite_debug"] = bool(
            re.search(r"\[qwen-kfrag\].*rewritten=1", result.stdout)
        )
        rows.append(row)
        debug_output.append(result.stdout)
    return rows, "\n".join(debug_output)


def run_rocprof(args, variant: str):
    out_dir = args.out_dir / f"rocprof_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_cmd([
        "/opt/rocm/bin/rocprofv3", "--kernel-trace", "--pmc", *PMCS,
        "--kernel-include-regex", KERNEL, "-d", str(out_dir), "-o", variant,
        "-f", "csv", "--", sys.executable, str(REPRO), "--variant", variant,
        "--seed", str(args.seed), "--warmup", str(args.rocprof_warmup),
        "--repeat", str(args.rocprof_repeat),
    ], check=False)
    row = parse_rocprof_dir(out_dir, KERNEL)
    row["returncode"] = result.returncode
    row["debug_tail"] = "\n".join(
        line for line in result.stdout.splitlines() if "[qwen-kfrag]" in line
    )
    return row


def dump_hsaco(args, variant: str):
    import torch
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    repro = load_repro()
    args.hsaco_dir.mkdir(parents=True, exist_ok=True)
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped = []

    def wrapped_compile(self, src, target, options=None):
        binary = original_compile(self, src, target, options)
        if KERNEL in src.fn.fn.__name__ and not dumped:
            path = args.hsaco_dir / f"{variant}.{src.fn.fn.__name__}.hsaco"
            path.write_bytes(binary)
            dumped.append(path)
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        repro.launch_variant(variant, repro.make_inputs(args.seed))
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile
    if not dumped:
        raise RuntimeError(f"no hsaco dumped for {variant}")
    return dumped[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--hsaco-dir", type=Path, default=OUT_DIR / "hsaco")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--rocprof-warmup", type=int, default=2)
    parser.add_argument("--rocprof-repeat", type=int, default=5)
    parser.add_argument("--variants", nargs="*", choices=[
        "R0_isolated_l6_rewrite",
        "R1_rewrite_plus_bt64_window_loop",
        "R2_rewrite_plus_pred_vdecay_live",
        "R3_rewrite_plus_state_update_writeback",
        "R4_full_loop_skeleton_rewrite",
    ])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd([sys.executable, "-m", "py_compile", str(REPRO)])
    variants = args.variants or [
        "R0_isolated_l6_rewrite",
        "R1_rewrite_plus_bt64_window_loop",
        "R2_rewrite_plus_pred_vdecay_live",
        "R3_rewrite_plus_state_update_writeback",
        "R4_full_loop_skeleton_rewrite",
    ]
    smoke, debug_output = run_smoke(args, variants)
    (args.out_dir / "smoke.json").write_text(json.dumps(smoke, indent=2))
    (args.out_dir / "rewrite_debug.log").write_text(debug_output)
    variants = [row["variant"] for row in smoke]
    rocprof = {variant: run_rocprof(args, variant) for variant in variants}
    (args.out_dir / "rocprof_summary.json").write_text(json.dumps(rocprof, indent=2))
    isa = {}
    for variant in variants:
        hsaco = dump_hsaco(args, variant)
        isa_path = disassemble(hsaco)
        row = analyze_isa(isa_path)
        row.update(analyze_high_agpr(isa_path))
        row["hsaco"] = str(hsaco)
        row["isa"] = str(isa_path)
        isa[variant] = row
    (args.out_dir / "isa_summary.json").write_text(json.dumps(isa, indent=2))
    print(json.dumps({"smoke": smoke, "rocprof": rocprof, "isa": isa}, indent=2))


if __name__ == "__main__":
    main()
