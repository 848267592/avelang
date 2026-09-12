#!/usr/bin/env python3
"""Dump comparable L6 lowering artifacts for Qwen-shaped MFMA audit.

This script intentionally reuses the existing L6 K-stage/update repro.  It does
not add new kernels; it collects smoke results, rocprof counters, HSACO, ISA,
and MFMA-local ISA snippets for the four variants needed by the lowering audit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

from profile_qwen_mfma32_l6_kstage_update_variants import KERNEL, dump_hsaco, run_rocprof
from profile_qwen_mfma32_lowering_ladder import (
    ACC_RE,
    PMCS,
    VGPR_RE,
    analyze_isa,
    disassemble,
    max_index,
    run_cmd,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[4]
REPRO = SCRIPT_DIR / "repro_qwen_mfma32_l6_kstage_update_variants.py"
DEFAULT_OUT_DIR = PROJECT_ROOT / "test/examples/linear_attention/rocprof_outputs/qwen_l6_lowering_audit"

KEY_VARIANTS = [
    "L6_baseline_current_update",
    "L6_subtile16_stage_full_update_like",
    "L6_update_mfma_no_pred_dependency",
    "L6_update_mfma_minimal_frag",
]

ALIASES = {
    "L6_baseline_current_update": "L6_baseline_no_lifetime",
    "L6_subtile16_stage_full_update_like": "L6_subtile_no_lifetime",
    "L6_update_mfma_no_pred_dependency": "L6_update_mfma_no_pred_dependency",
    "L6_update_mfma_minimal_frag": "L6_update_mfma_minimal_frag",
}

ISA_METRICS = [
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
    "total_instructions",
]


def load_repro():
    spec = importlib.util.spec_from_file_location("repro_qwen_mfma32_l6_kstage_update_variants", REPRO)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {REPRO}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_smoke(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in KEY_VARIANTS:
        result = run_cmd(
            [
                sys.executable,
                str(REPRO),
                "--variant",
                variant,
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
        parsed = json.loads(text[text.find("[") :])
        row = parsed[0]
        row["audit_alias"] = ALIASES[variant]
        rows.append(row)
    return rows


def extract_mfma_snippets(isa_path: Path, out_path: Path, radius: int = 30) -> list[dict[str, object]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = isa_path.read_text(errors="replace").splitlines()
    snippets: list[dict[str, object]] = []
    out_lines = [f"# MFMA ISA snippets for `{isa_path.name}`\n"]

    mfma_count = 0
    for idx, line in enumerate(lines):
        if "v_mfma" not in line:
            continue
        mfma_count += 1
        start = max(0, idx - radius)
        end = min(len(lines), idx + radius + 1)
        body = "\n".join(lines[start:end])
        kind = "mfma32_pred" if "32x32x8_bf16" in line else "mfma16_update" if "16x16x16_bf16" in line else "mfma_other"
        local_vgpr = max_index(body, VGPR_RE)
        local_acc = max_index(body, ACC_RE)
        snippets.append(
            {
                "ordinal": mfma_count,
                "line": idx + 1,
                "kind": kind,
                "mnemonic": line.strip(),
                "snippet_start_line": start + 1,
                "snippet_end_line": end,
                "local_max_vgpr_index": local_vgpr,
                "local_max_acc_index": local_acc,
            }
        )
        out_lines.append(f"## MFMA {mfma_count}: {kind}, ISA line {idx + 1}\n")
        out_lines.append(f"- local max VGPR index: `{local_vgpr}`")
        out_lines.append(f"- local max ACC index: `{local_acc}`\n")
        out_lines.append("```asm")
        out_lines.append(body)
        out_lines.append("```\n")

    out_path.write_text("\n".join(out_lines) + "\n")
    return snippets


def write_ir_status(out_dir: Path) -> Path:
    path = out_dir / "ir_dump_status.md"
    path.write_text(
        "\n".join(
            [
                "# IR Dump Status",
                "",
                "This audit collected HSACO/ISA and rocprof counters for the four L6 variants.",
                "",
                "Per-pass IR dumps were not available through the current Python profiling harness.",
                "The public `MLIRGenerator.get_mlir()` hook can expose the initial AveLang module,",
                "but this script does not have a stable compiler API for stopping after AveLang-to-memref,",
                "GPU outlining, ROCDL, or LLVM lowering while preserving the same JIT path.",
                "",
                "For this pass, AMDGPU ISA plus rocprof metadata are the authoritative artifacts.",
                "If more granularity is needed, the next compiler-side step is to add pass-manager",
                "dump hooks for the JIT pipeline instead of creating more source variants.",
            ]
        )
        + "\n"
    )
    return path


def write_artifact_summary(
    out_dir: Path,
    smoke: list[dict[str, object]],
    rocprof: dict[str, dict[str, object]],
    isa: dict[str, dict[str, object]],
    snippets: dict[str, list[dict[str, object]]],
) -> Path:
    path = out_dir / "artifact_summary.md"
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
    lines = ["# L6 Lowering Audit Artifact Summary\n"]
    lines.append("## Variant Aliases\n")
    for variant in KEY_VARIANTS:
        lines.append(f"- `{ALIASES[variant]}` -> `{variant}`")
    lines.append("\n## Smoke\n")
    lines.append("| variant | alias | latency_ms | finite | checksum |")
    lines.append("|:---|:---|---:|:---:|---:|")
    for row in smoke:
        lines.append(
            f"| {row.get('variant')} | {row.get('audit_alias')} | {row.get('latency_ms')} | "
            f"{row.get('sink_finite')} | {row.get('sink_checksum_abs')} |"
        )
    lines.append("\n## Rocprof\n")
    lines.append("| variant | alias | " + " | ".join(cols) + " |")
    lines.append("|:---|:---|" + "|".join(["---:"] * len(cols)) + "|")
    for variant in KEY_VARIANTS:
        row = rocprof.get(variant, {})
        lines.append(
            f"| {variant} | {ALIASES[variant]} | " + " | ".join(str(row.get(c, "")) for c in cols) + " |"
        )
    lines.append("\n## Static ISA\n")
    lines.append("| variant | alias | " + " | ".join(ISA_METRICS) + " |")
    lines.append("|:---|:---|" + "|".join(["---:"] * len(ISA_METRICS)) + "|")
    for variant in KEY_VARIANTS:
        row = isa.get(variant, {})
        lines.append(
            f"| {variant} | {ALIASES[variant]} | "
            + " | ".join(str(row.get(c, "")) for c in ISA_METRICS)
            + " |"
        )
    lines.append("\n## MFMA Snippet Files\n")
    for variant in KEY_VARIANTS:
        lines.append(f"- `{variant}`: `{len(snippets.get(variant, []))}` MFMA snippets")
    path.write_text("\n".join(lines) + "\n")
    return path


def maybe_copy(path: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / path.name
    if path.resolve() != dst.resolve():
        shutil.copy2(path, dst)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--hsaco-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--rocprof-warmup", type=int, default=2)
    parser.add_argument("--rocprof-repeat", type=int, default=5)
    parser.add_argument("--skip-rocprof", action="store_true")
    parser.add_argument("--skip-hsaco", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.hsaco_dir = args.hsaco_dir or (args.out_dir / "hsaco")
    args.hsaco_dir.mkdir(parents=True, exist_ok=True)

    run_cmd([sys.executable, "-m", "py_compile", str(REPRO)])

    smoke = run_smoke(args)
    (args.out_dir / "smoke.json").write_text(json.dumps(smoke, indent=2, sort_keys=True))

    rocprof: dict[str, dict[str, object]] = {}
    if not args.skip_rocprof:
        for variant in KEY_VARIANTS:
            rocprof[variant] = run_rocprof(args, variant)
        (args.out_dir / "rocprof_summary.json").write_text(json.dumps(rocprof, indent=2, sort_keys=True))

    isa: dict[str, dict[str, object]] = {}
    snippets: dict[str, list[dict[str, object]]] = {}
    if not args.skip_hsaco:
        snippet_dir = args.out_dir / "mfma_snippets"
        hsaco_copy_dir = args.out_dir / "hsaco_copies"
        isa_copy_dir = args.out_dir / "isa"
        for variant in KEY_VARIANTS:
            hsaco = dump_hsaco(args, variant)
            isa_path = disassemble(hsaco)
            hsaco_copy = maybe_copy(hsaco, hsaco_copy_dir)
            isa_copy = maybe_copy(isa_path, isa_copy_dir)
            row = analyze_isa(isa_copy)
            row["hsaco"] = str(hsaco_copy)
            row["isa"] = str(isa_copy)
            isa[variant] = row
            snippets[variant] = extract_mfma_snippets(
                isa_copy,
                snippet_dir / f"{variant}_mfma_snippets.md",
            )
        (args.out_dir / "isa_summary.json").write_text(json.dumps(isa, indent=2, sort_keys=True))
        (args.out_dir / "mfma_region_summary.json").write_text(json.dumps(snippets, indent=2, sort_keys=True))

    write_ir_status(args.out_dir)
    summary = write_artifact_summary(args.out_dir, smoke, rocprof, isa, snippets)
    print(f"out_dir={args.out_dir}")
    print(f"summary={summary}")
    print(f"pmcs={' '.join(PMCS)}")
    print(f"kernel_regex={KERNEL}")


if __name__ == "__main__":
    main()
