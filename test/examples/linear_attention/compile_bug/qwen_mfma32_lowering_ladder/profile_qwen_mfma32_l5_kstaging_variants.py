#!/usr/bin/env python3
"""Profile focused Qwen MFMA32 L5 K-staging variants."""

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
REPRO = SCRIPT_DIR / "repro_qwen_mfma32_l5_kstaging_variants.py"
OUT_DIR = PROJECT_ROOT / "test/examples/linear_attention/rocprof_outputs/qwen_mfma32_l5_kstaging_variants"
HSACO_DIR = PROJECT_ROOT / "test/examples/linear_attention/rocprof_outputs/qwen_mfma32_l5_kstaging_variants_hsaco"
KERNEL = "_qwen_mfma32_l5_kstaging_variants_kernel"

VARIANT_ORDER = [
    "baseline_L5",
    "baseline_L6",
    "L5_token_major_no_kall_vec",
    "L5_transposed_prepacked_input",
    "L5_subtile_k_stage_16token",
    "L5_khalf_stage_64",
    "L5_direct_global_k_update_probe",
    "L5_packed_i32_contiguous_load",
    "L5_alt_token_major_k_stage_existing",
]


def load_repro():
    spec = importlib.util.spec_from_file_location("repro_qwen_mfma32_l5_kstaging_variants", REPRO)
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
    report = SCRIPT_DIR / "qwen_mfma32_l5_kstaging_variants_report.md"
    lines = ["# Qwen MFMA32 L5 K-Staging Variants Report\n"]
    lines.append("## Summary\n")
    base = rocprof.get("baseline_L5", {})
    best_name = None
    best_trace = None
    for name, row in rocprof.items():
        trace = _to_float(row.get("trace_median_us"))
        if trace is not None and (best_trace is None or trace < best_trace):
            best_name = name
            best_trace = trace
    base_trace = _to_float(base.get("trace_median_us"))
    if best_name and best_trace is not None:
        delta = None if base_trace is None else best_trace - base_trace
        lines.append(f"Best measured variant: `{best_name}` with trace `{best_trace:.3f} us` (delta vs baseline_L5 `{fmt(delta)}`).\n")
    lines.append("These variants isolate the L4->L5 K staging/shared-view jump. They are diagnostic and do not change v23/v24/full Qwen.\n")
    lines.append("## Variants\n")
    lines.append("- `baseline_L5`: current ladder L5, transposed `k_all_t[128,BT]` plus `kall_vec`.")
    lines.append("- `baseline_L6`: current ladder L6, same K staging plus one dependent update MFMA.")
    lines.append("- `L5_token_major_no_kall_vec`: stage `[BT,128]`, avoid `kall_vec`.")
    lines.append("- `L5_transposed_prepacked_input`: read prepacked `[head,k,token]` input, skip runtime transposed shared staging.")
    lines.append("- `L5_subtile_k_stage_16token`: stage only a 16-token transposed subtile.")
    lines.append("- `L5_khalf_stage_64`: stage only K half `[64,BT]`.")
    lines.append("- `L5_direct_global_k_update_probe`: avoid full K shared staging; direct global diagnostic loads.")
    lines.append("- `L5_packed_i32_contiguous_load`: token-major shared plus packed i32 view loads.")
    lines.append("- `L5_alt_token_major_k_stage_existing`: preserved existing token-major workaround.\n")
    lines.append("## Smoke/Checksum\n")
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
    isa_cols = ["v_mfma_f32_32x32x8_bf16", "v_mfma_f32_16x16x16_bf16", "global_load", "global_store", "buffer_load", "buffer_store", "ds_read", "ds_write", "s_barrier", "s_waitcnt", "v_add", "v_add3", "v_lshl", "v_lshl_add", "v_or", "v_bfe", "max_vgpr_index_static_best_effort", "max_acc_index_static_best_effort"]
    lines.append("\n## Static ISA Counts\n")
    lines.append("| variant | " + " | ".join(isa_cols) + " |")
    lines.append("|:---|" + "|".join(["---:"] * len(isa_cols)) + "|")
    for name in VARIANT_ORDER:
        row = isa.get(name, {})
        lines.append(f"| {name} | " + " | ".join(fmt(row.get(c)) for c in isa_cols) + " |")
    lines.append("\n## Decision\n")
    if base_trace is not None and best_name and best_trace is not None and best_name != "baseline_L5":
        speed = base_trace / best_trace if best_trace > 0 else 0.0
        lines.append(f"`{best_name}` is `{speed:.3f}x` vs baseline_L5 on trace. The gate for returning to full Qwen is `>=1.10x` without correctness risk.")
        if speed >= 1.10:
            lines.append("This passes the isolated ladder trace gate; applying the same source pattern to an experimental full Qwen copy is allowed.")
        else:
            lines.append("This does not pass the isolated ladder trace gate; do not return to full Qwen from this result.")
    else:
        lines.append("No K-staging variant beat baseline_L5; do not return to full Qwen from this result.")
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
