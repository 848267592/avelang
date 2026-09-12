#!/usr/bin/env python3
"""Profile Qwen-shaped L6 lifetime-boundary prototype."""

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
REPRO = SCRIPT_DIR / "repro_qwen_mfma32_l6_lifetime_boundary.py"
OUT_DIR = PROJECT_ROOT / "test/examples/linear_attention/rocprof_outputs/qwen_mfma32_l6_lifetime_boundary"
HSACO_DIR = PROJECT_ROOT / "test/examples/linear_attention/rocprof_outputs/qwen_mfma32_l6_lifetime_boundary_hsaco"
KERNEL = "_qwen_mfma32_l6_kstage_update_variants_kernel"

VARIANT_ORDER = [
    "L6_baseline_no_lifetime",
    "L6_with_end_lifetime_after_vdecay",
    "L6_memref_only_end_lifetime",
    "L6_subtile_no_lifetime",
    "L6_subtile_with_end_lifetime_after_vdecay",
    "L6_subtile_memref_only_end_lifetime",
]


def load_repro():
    spec = importlib.util.spec_from_file_location("repro_qwen_mfma32_l6_lifetime_boundary", REPRO)
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
    summary["stdout_tail"] = "\n".join(result.stdout.splitlines()[-16:])
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


def _delta(row_a: dict[str, object], row_b: dict[str, object], key: str) -> float | None:
    a = _to_float(row_a.get(key))
    b = _to_float(row_b.get(key))
    if a is None or b is None:
        return None
    return b - a


def write_report(smoke: list[dict[str, object]], rocprof: dict[str, dict[str, object]], isa: dict[str, dict[str, object]]) -> Path:
    report = SCRIPT_DIR / "l6_lifetime_boundary_profile_report.md"
    smoke_by_name = {str(row["variant"]): row for row in smoke}
    base = rocprof.get("L6_baseline_no_lifetime", {})
    base_lifetime = rocprof.get("L6_with_end_lifetime_after_vdecay", {})
    base_memref_lifetime = rocprof.get("L6_memref_only_end_lifetime", {})
    subtile = rocprof.get("L6_subtile_no_lifetime", {})
    subtile_lifetime = rocprof.get("L6_subtile_with_end_lifetime_after_vdecay", {})
    subtile_memref_lifetime = rocprof.get("L6_subtile_memref_only_end_lifetime", {})

    cols = [
        "trace_median_us",
        "VGPR_Count",
        "Accum_VGPR_Count",
        "SGPR_Count",
        "Scratch_Size",
        "LDS_Block_Size",
        "SQ_INSTS_MFMA",
        "SQ_INSTS_VALU",
        "SQ_INSTS_SALU",
        "SQ_INSTS_VMEM",
        "SQ_INSTS_LDS",
        "OccupancyPercent",
    ]
    isa_cols = [
        "v_mfma_f32_32x32x8_bf16",
        "v_mfma_f32_16x16x16_bf16",
        "global_load",
        "global_store",
        "buffer_load",
        "buffer_store",
        "ds_read",
        "ds_write",
        "s_barrier",
        "s_waitcnt",
        "v_add",
        "v_add3",
        "v_lshl",
        "v_lshl_add",
        "v_or",
        "v_bfe",
        "max_vgpr_index_static_best_effort",
        "max_acc_index_static_best_effort",
    ]

    lines = ["# Qwen MFMA32 L6 Lifetime Boundary Profile Report\n"]
    lines.append("## Summary\n")
    lines.append("This report profiles the minimal `al.end_lifetime(...)` marker placed after `v_decay_t` staging and before update MFMA.\n")
    if base and base_lifetime:
        lines.append(
            "- Broad-K baseline lifetime-marker delta: "
            f"trace `{fmt(_delta(base, base_lifetime, 'trace_median_us'))} us`, "
            f"AccVGPR `{fmt(_delta(base, base_lifetime, 'Accum_VGPR_Count'))}`, "
            f"Scratch `{fmt(_delta(base, base_lifetime, 'Scratch_Size'))}`.\n"
        )
    if base and base_memref_lifetime:
        lines.append(
            "- Broad-K memref-only lifetime-marker delta: "
            f"trace `{fmt(_delta(base, base_memref_lifetime, 'trace_median_us'))} us`, "
            f"AccVGPR `{fmt(_delta(base, base_memref_lifetime, 'Accum_VGPR_Count'))}`, "
            f"Scratch `{fmt(_delta(base, base_memref_lifetime, 'Scratch_Size'))}`.\n"
        )
    if subtile and subtile_lifetime:
        lines.append(
            "- K-subtile lifetime-marker delta: "
            f"trace `{fmt(_delta(subtile, subtile_lifetime, 'trace_median_us'))} us`, "
            f"AccVGPR `{fmt(_delta(subtile, subtile_lifetime, 'Accum_VGPR_Count'))}`, "
            f"Scratch `{fmt(_delta(subtile, subtile_lifetime, 'Scratch_Size'))}`.\n"
        )
    if subtile and subtile_memref_lifetime:
        lines.append(
            "- K-subtile memref-only lifetime-marker delta: "
            f"trace `{fmt(_delta(subtile, subtile_memref_lifetime, 'trace_median_us'))} us`, "
            f"AccVGPR `{fmt(_delta(subtile, subtile_memref_lifetime, 'Accum_VGPR_Count'))}`, "
            f"Scratch `{fmt(_delta(subtile, subtile_memref_lifetime, 'Scratch_Size'))}`.\n"
        )

    lines.append("\n## Marker Placement\n")
    lines.append("```python")
    lines.append("# after v_decay_t has been written and synchronized")
    lines.append("al.end_lifetime(pred_acc, pred_partial, state_bf16, w_bf16, state_vec, w_vec)")
    lines.append("al.syncthreads()")
    lines.append("# update MFMA begins here")
    lines.append("```")
    lines.append("The marker does not end the lifetime of `v_decay_t`, K staging buffers, or any state needed by update.")

    lines.append("\n## Smoke/Finite Checks\n")
    lines.append("| variant | latency_ms | finite | checksum | conclusion |")
    lines.append("|:---|---:|:---:|---:|:---|")
    for name in VARIANT_ORDER:
        row = smoke_by_name.get(name, {})
        finite = row.get("sink_finite")
        checksum = _to_float(row.get("sink_checksum_abs"))
        conclusion = "valid" if finite and checksum and checksum != 0.0 else "invalid/profiling-only"
        lines.append(f"| {name} | {fmt(row.get('latency_ms'))} | {finite} | {fmt(row.get('sink_checksum_abs'))} | {conclusion} |")

    lines.append("\n## Rocprof Counters\n")
    lines.append("| variant | " + " | ".join(cols) + " |")
    lines.append("|:---|" + "|".join(["---:"] * len(cols)) + "|")
    for name in VARIANT_ORDER:
        row = rocprof.get(name, {})
        lines.append(f"| {name} | " + " | ".join(fmt(row.get(c)) for c in cols) + " |")

    lines.append("\n## Deltas\n")
    lines.append("| comparison | trace_delta_us | AccVGPR_delta | VGPR_delta | Scratch_delta | MFMA_delta | VALU_delta | VMEM_delta | LDS_delta |")
    lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|---:|")
    comparisons = [
        ("baseline + marker vs baseline", base, base_lifetime),
        ("baseline + memref-only marker vs baseline", base, base_memref_lifetime),
        ("subtile + marker vs subtile", subtile, subtile_lifetime),
        ("subtile + memref-only marker vs subtile", subtile, subtile_memref_lifetime),
    ]
    for label, before, after in comparisons:
        lines.append(
            f"| {label} | {fmt(_delta(before, after, 'trace_median_us'))} | {fmt(_delta(before, after, 'Accum_VGPR_Count'))} | "
            f"{fmt(_delta(before, after, 'VGPR_Count'))} | {fmt(_delta(before, after, 'Scratch_Size'))} | "
            f"{fmt(_delta(before, after, 'SQ_INSTS_MFMA'))} | {fmt(_delta(before, after, 'SQ_INSTS_VALU'))} | "
            f"{fmt(_delta(before, after, 'SQ_INSTS_VMEM'))} | {fmt(_delta(before, after, 'SQ_INSTS_LDS'))} |"
        )

    lines.append("\n## Static ISA Counts\n")
    lines.append("| variant | " + " | ".join(isa_cols) + " |")
    lines.append("|:---|" + "|".join(["---:"] * len(isa_cols)) + "|")
    for name in VARIANT_ORDER:
        row = isa.get(name, {})
        lines.append(f"| {name} | " + " | ".join(fmt(row.get(c)) for c in isa_cols) + " |")

    lines.append("\n## Conclusion\n")
    lines.append("Interpretation belongs in `compiler_lifetime_boundary_patch_report.md`, which combines these counters with the compiler patch behavior.")

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
