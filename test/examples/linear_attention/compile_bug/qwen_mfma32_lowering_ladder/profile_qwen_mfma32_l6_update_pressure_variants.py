#!/usr/bin/env python3
"""Profile focused Qwen MFMA32 L6 update-pressure variants."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from profile_qwen_mfma32_lowering_ladder import (
    PMCS,
    analyze_isa,
    disassemble,
    fmt,
    parse_rocprof_dir,
    run_cmd,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[4]
REPRO = SCRIPT_DIR / "repro_qwen_mfma32_l6_update_pressure_variants.py"
OUT_DIR = PROJECT_ROOT / "test/examples/linear_attention/rocprof_outputs/qwen_mfma32_l6_update_pressure_variants"
HSACO_DIR = PROJECT_ROOT / "test/examples/linear_attention/rocprof_outputs/qwen_mfma32_l6_update_pressure_variants_hsaco"
KERNEL = "_qwen_mfma32_l6_update_pressure_variants_kernel"

VARIANT_ORDER = [
    "baseline_L5_no_update",
    "L6_one_update_mfma_only",
    "L6_one_ktile_update",
    "L6_two_ktile_update",
    "L6_full_update_like_current",
    "L6_update_acc_scope_split_source",
    "L6_update_acc_reinit_variant",
]


def load_repro():
    spec = importlib.util.spec_from_file_location("repro_qwen_mfma32_l6_update_pressure_variants", REPRO)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {REPRO}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_smoke(args: argparse.Namespace) -> list[dict[str, object]]:
    result = run_cmd(
        [
            sys.executable,
            str(REPRO),
            "--variant",
            "all",
            "--seed",
            str(args.seed),
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            "--json",
        ]
    )
    text = result.stdout
    return json.loads(text[text.find("[") :])


def run_rocprof(args: argparse.Namespace, variant: str) -> dict[str, object]:
    out_dir = Path(args.out_dir) / f"rocprof_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "/opt/rocm/bin/rocprofv3",
        "--kernel-trace",
        "--pmc",
        *PMCS,
        "--kernel-include-regex",
        KERNEL,
        "-d",
        str(out_dir),
        "-o",
        f"{variant}_counters",
        "-f",
        "csv",
        "--",
        sys.executable,
        str(REPRO),
        "--variant",
        variant,
        "--seed",
        str(args.seed),
        "--warmup",
        str(args.rocprof_warmup),
        "--repeat",
        str(args.rocprof_repeat),
    ]
    result = run_cmd(cmd, check=False)
    summary = parse_rocprof_dir(out_dir, KERNEL)
    summary["returncode"] = result.returncode
    summary["stdout_tail"] = "\n".join(result.stdout.splitlines()[-12:])
    return summary


def dump_hsaco(args: argparse.Namespace, variant: str) -> Path:
    import torch
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    repro = load_repro()
    out_dir = Path(args.hsaco_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped: list[Path] = []

    def wrapped_compile(self, src, target, options=None):
        binary = original_compile(self, src, target, options)
        kernel_name = src.fn.fn.__name__
        if KERNEL in kernel_name and not dumped:
            path = out_dir / f"{variant}.{kernel_name}.hsaco"
            path.write_bytes(binary)
            dumped.append(path)
            print(f"dumped_hsaco={path}")
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        tensors = repro.make_inputs(args.seed)
        repro.launch_variant(variant, tensors)
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile
    if not dumped:
        raise RuntimeError(f"no hsaco dumped for {variant}")
    return dumped[0]


def _to_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def write_report(smoke: list[dict[str, object]], rocprof: dict[str, dict[str, object]], isa: dict[str, dict[str, object]]) -> Path:
    report = SCRIPT_DIR / "qwen_mfma32_l6_update_pressure_variants_report.md"
    lines = ["# Qwen MFMA32 L6 Update-Pressure Variants Report\n"]
    lines.append("## Summary\n")
    base = rocprof.get("baseline_L5_no_update", {})
    one = rocprof.get("L6_one_update_mfma_only", {})
    ktile = rocprof.get("L6_one_ktile_update", {})
    full = rocprof.get("L6_full_update_like_current", {})
    lines.append("This report isolates the L5->L6 transition by keeping the K staging source shape fixed and varying only update MFMA count/scope.\n")
    if base and ktile:
        b_acc = _to_float(base.get("Accum_VGPR_Count"))
        k_acc = _to_float(ktile.get("Accum_VGPR_Count"))
        lines.append(f"One complete Ktile update changes AccVGPR by `{fmt(None if b_acc is None or k_acc is None else k_acc - b_acc)}` versus no update.\n")
    if one:
        lines.append(f"`L6_one_update_mfma_only` AccVGPR is `{fmt(one.get('Accum_VGPR_Count'))}`, useful for separating one intrinsic from a full Ktile branch set.\n")
    if full:
        lines.append(f"`L6_full_update_like_current` trace is `{fmt(full.get('trace_median_us'))} us`.\n")
    lines.append("## Variants\n")
    for name in VARIANT_ORDER:
        lines.append(f"- `{name}`")
    lines.append("\n## Smoke/Checksum\n")
    lines.append("| variant | latency_ms | finite | checksum |")
    lines.append("|:---|---:|:---:|---:|")
    for row in smoke:
        lines.append(f"| {row['variant']} | {row['latency_ms']:.6f} | {row['sink_finite']} | {row['sink_checksum_abs']:.6g} |")
    cols = ["trace_median_us", "VGPR_Count", "Accum_VGPR_Count", "SGPR_Count", "Scratch_Size", "LDS_Block_Size", "SQ_INSTS_MFMA", "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_VMEM", "SQ_INSTS_LDS", "OccupancyPercent"]
    lines.append("\n## Rocprof Counters\n")
    lines.append("| variant | " + " | ".join(cols) + " |")
    lines.append("|:---|" + "|".join(["---:"] * len(cols)) + "|")
    for name in VARIANT_ORDER:
        row = rocprof.get(name, {})
        lines.append(f"| {name} | " + " | ".join(fmt(row.get(c)) for c in cols) + " |")
    lines.append("\n## AccVGPR Trend\n")
    lines.append("| transition | AccVGPR delta | trace delta us | MFMA delta |")
    lines.append("|:---|---:|---:|---:|")
    prev = "baseline_L5_no_update"
    for name in VARIANT_ORDER[1:]:
        a = rocprof.get(prev, {})
        b = rocprof.get(name, {})
        lines.append(
            f"| {prev} -> {name} | "
            f"{fmt(None if _to_float(a.get('Accum_VGPR_Count')) is None or _to_float(b.get('Accum_VGPR_Count')) is None else _to_float(b.get('Accum_VGPR_Count')) - _to_float(a.get('Accum_VGPR_Count')))} | "
            f"{fmt(None if _to_float(a.get('trace_median_us')) is None or _to_float(b.get('trace_median_us')) is None else _to_float(b.get('trace_median_us')) - _to_float(a.get('trace_median_us')))} | "
            f"{fmt(None if _to_float(a.get('SQ_INSTS_MFMA')) is None or _to_float(b.get('SQ_INSTS_MFMA')) is None else _to_float(b.get('SQ_INSTS_MFMA')) - _to_float(a.get('SQ_INSTS_MFMA')))} |"
        )
    isa_cols = ["v_mfma_f32_32x32x8_bf16", "v_mfma_f32_16x16x16_bf16", "global_load", "global_store", "buffer_load", "buffer_store", "ds_read", "ds_write", "s_barrier", "s_waitcnt", "v_add", "v_add3", "v_lshl", "v_lshl_add", "v_or", "v_bfe", "max_vgpr_index_static_best_effort", "max_acc_index_static_best_effort"]
    lines.append("\n## Static ISA Counts\n")
    lines.append("| variant | " + " | ".join(isa_cols) + " |")
    lines.append("|:---|" + "|".join(["---:"] * len(isa_cols)) + "|")
    for name in VARIANT_ORDER:
        row = isa.get(name, {})
        lines.append(f"| {name} | " + " | ".join(fmt(row.get(c)) for c in isa_cols) + " |")
    lines.append("\n## Interpretation\n")
    lines.append("- If AccVGPR grows with update tile count, the pressure is largely expected accumulator footprint.")
    lines.append("- If one MFMA or one Ktile already jumps to the full pressure level, lowering is conservative around mixed MFMA32 pred plus MFMA16 update.")
    lines.append("- If scope split is materially lower than full update, it is a source workaround candidate; otherwise no source-level scope split was found here.")
    report.write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--hsaco-dir", type=Path, default=HSACO_DIR)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--rocprof-warmup", type=int, default=2)
    parser.add_argument("--rocprof-repeat", type=int, default=5)
    parser.add_argument("--skip-rocprof", action="store_true")
    parser.add_argument("--skip-hsaco", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.hsaco_dir.mkdir(parents=True, exist_ok=True)

    run_cmd([sys.executable, "-m", "py_compile", str(REPRO)])
    smoke = run_smoke(args)
    (args.out_dir / "smoke.json").write_text(json.dumps(smoke, indent=2, sort_keys=True))

    rocprof: dict[str, dict[str, object]] = {}
    if not args.skip_rocprof:
        for variant in VARIANT_ORDER:
            rocprof[variant] = run_rocprof(args, variant)
        (args.out_dir / "rocprof_summary.json").write_text(json.dumps(rocprof, indent=2, sort_keys=True))

    isa: dict[str, dict[str, object]] = {}
    if not args.skip_hsaco:
        for variant in VARIANT_ORDER:
            hsaco = dump_hsaco(args, variant)
            isa_path = disassemble(hsaco)
            row = analyze_isa(isa_path)
            row["hsaco"] = str(hsaco)
            isa[variant] = row
        (args.out_dir / "isa_summary.json").write_text(json.dumps(isa, indent=2, sort_keys=True))

    report = write_report(smoke, rocprof, isa)
    print(f"report={report}")


if __name__ == "__main__":
    main()
