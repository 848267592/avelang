#!/usr/bin/env python3
"""Profile and evidence builder for mfma_region_lifetime_v3.

This driver intentionally keeps the repro small and repeatable:

* runs all correctness/latency variants;
* profiles each variant with rocprofv3;
* dumps/disassembles HSACO for selected variants;
* counts important ISA mnemonics;
* records compiler grep/audit commands related to FullOp, memory effects,
  private allocations, bufferization and MFMA lowering.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[4]
REPRO = SCRIPT_DIR / "repro_mfma_region_lifetime_v3.py"
DEFAULT_OUT_DIR = PROJECT_ROOT / "test/examples/linear_attention/rocprof_outputs/mfma_region_lifetime_v3"
DEFAULT_HSACO_DIR = PROJECT_ROOT / "test/examples/linear_attention/rocprof_outputs/mfma_region_lifetime_v3_hsaco"

PMCS = [
    "SQ_INSTS_MFMA",
    "SQ_INSTS_VALU",
    "SQ_INSTS_SALU",
    "SQ_INSTS_VMEM",
    "SQ_INSTS_LDS",
    "OccupancyPercent",
]

KEY_METADATA = [
    "Grid_Size",
    "Workgroup_Size",
    "LDS_Block_Size",
    "Scratch_Size",
    "VGPR_Count",
    "Accum_VGPR_Count",
    "SGPR_Count",
]

IMPORTANT_HSACO_VARIANTS = [
    "update16_only",
    "pred16_only_sink",
    "pred32_only_sink",
    "pred32_then_update16",
    "pred32_unpacked_to_lds_then_update16",
    "pred32_two_regions_then_update16",
    "pred32_then_update16_with_dummy_barrier",
    "pred32_then_update16_with_source_lifetime_hint",
]

GREP_COMMANDS = [
    ("FullOp", ["grep", "-R", "def FullOp", "-n", "include", "lib", "python"]),
    ("MemoryEffects", ["grep", "-R", "MemoryEffects", "-n", "include", "lib"]),
    ("Pure_or_speculatable", ["grep", "-R", r"NoMemoryEffect\|Pure\|AlwaysSpeculatable", "-n", "include", "lib"]),
    ("MFMA", ["grep", "-R", r"mfma_32x32x8\|mfma_16x16x16", "-n", "include", "lib", "python"]),
    ("Bufferization", ["grep", "-R", r"bufferize\|Bufferize\|bufferization", "-n", "include", "lib"]),
    ("Private_alloc", ["grep", "-R", r"alloca\|alloc\|private", "-n", "include", "lib"]),
    ("Lifetime", ["grep", "-R", r"lifetime\|Lifetime\|dealloc\|alloca_scope", "-n", "include", "lib"]),
]


def run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def load_repro_module():
    spec = importlib.util.spec_from_file_location("repro_mfma_region_lifetime_v3", REPRO)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {REPRO}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_correctness(args: argparse.Namespace) -> list[dict[str, object]]:
    cmd = [
        sys.executable,
        str(REPRO),
        "--mode",
        "all",
        "--seed",
        str(args.seed),
        "--scale",
        str(args.scale),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--json",
    ]
    result = run_cmd(cmd, cwd=PROJECT_ROOT)
    text = result.stdout
    start = text.find("[")
    if start < 0:
        raise RuntimeError(f"repro did not emit JSON:\n{text}")
    rows = json.loads(text[start:])
    return rows


def run_rocprof(args: argparse.Namespace, mode: str) -> dict[str, object]:
    out_dir = Path(args.out_dir) / f"rocprof_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "/opt/rocm/bin/rocprofv3",
        "--kernel-trace",
        "--pmc",
        *PMCS,
        "--kernel-include-regex",
        "_mfma_region_lifetime_v3_kernel",
        "-d",
        str(out_dir),
        "-o",
        f"{mode}_counters",
        "-f",
        "csv",
        "--",
        sys.executable,
        str(REPRO),
        "--mode",
        mode,
        "--seed",
        str(args.seed),
        "--scale",
        str(args.scale),
        "--warmup",
        str(args.rocprof_warmup),
        "--repeat",
        str(args.rocprof_repeat),
    ]
    result = run_cmd(cmd, cwd=PROJECT_ROOT, check=False)
    summary = parse_rocprof_dir(out_dir, "_mfma_region_lifetime_v3_kernel")
    summary["rocprof_returncode"] = result.returncode
    summary["rocprof_stdout_tail"] = "\n".join(result.stdout.splitlines()[-20:])
    return summary


def _to_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_rocprof_dir(out_dir: Path, kernel_substr: str) -> dict[str, object]:
    summary: dict[str, object] = {}
    for path in sorted(out_dir.glob("*.csv")):
        name = path.name
        if "kernel_trace" in name:
            trace = parse_trace_csv(path, kernel_substr)
            summary.update(trace)
            summary["trace_csv"] = str(path)
        elif "counter" in name or "counters" in name:
            counters = parse_counter_csv(path, kernel_substr)
            if counters:
                summary.update(counters)
                summary["counter_csv"] = str(path)
    return summary


def parse_trace_csv(path: Path, kernel_substr: str) -> dict[str, object]:
    durations: list[float] = []
    metadata: dict[str, object] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel = row.get("Kernel_Name") or row.get("Name") or ""
            if kernel_substr and kernel_substr not in kernel:
                continue
            for key in KEY_METADATA:
                if row.get(key):
                    metadata[key] = row[key]
            duration = _to_float(row.get("Duration") or row.get("DurationNs"))
            if duration is not None:
                # rocprofv3 Duration is usually ns in current CSVs.
                durations.append(duration / 1000.0)
                continue
            start = _to_float(row.get("Start_Timestamp", ""))
            end = _to_float(row.get("End_Timestamp", ""))
            if start is not None and end is not None:
                durations.append((end - start) / 1000.0)
    if durations:
        metadata["trace_median_us"] = statistics.median(durations)
        metadata["trace_count"] = len(durations)
    return metadata


def parse_counter_csv(path: Path, kernel_substr: str) -> dict[str, object]:
    metrics: dict[str, list[float]] = {}
    metadata: dict[str, object] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel = row.get("Kernel_Name") or row.get("Name") or ""
            if kernel_substr and kernel_substr not in kernel:
                continue
            for key in KEY_METADATA:
                if row.get(key):
                    metadata[key] = row[key]
            metric_name = row.get("Counter_Name") or row.get("Metric") or row.get("Name")
            metric_value = row.get("Value") or row.get("Counter_Value")
            number = _to_float(metric_value)
            if metric_name and number is not None:
                metrics.setdefault(metric_name, []).append(number)
    for name, values in metrics.items():
        metadata[name] = statistics.median(values)
    return metadata


def dump_hsaco(args: argparse.Namespace, mode: str) -> Path:
    import torch
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    repro = load_repro_module()
    out_dir = Path(args.hsaco_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped: list[Path] = []

    def wrapped_compile(self, src, target, options=None):
        binary = original_compile(self, src, target, options)
        kernel_name = src.fn.fn.__name__
        if "_mfma_region_lifetime_v3_kernel" in kernel_name and not dumped:
            path = out_dir / f"{mode}.{kernel_name}.hsaco"
            path.write_bytes(binary)
            dumped.append(path)
            print(f"dumped_hsaco={path}")
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        repro.run_variant(mode, seed=args.seed, scale=args.scale, warmup=1, repeat=1, atol=args.atol, rtol=args.rtol)
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile

    if not dumped:
        raise RuntimeError(f"no hsaco dumped for mode={mode}")
    return dumped[0]


def find_objdump() -> str:
    candidates = [
        os.environ.get("LLVM_OBJDUMP"),
        "/opt/rocm/llvm/bin/llvm-objdump",
        "/opt/rocm/bin/llvm-objdump",
        shutil.which("llvm-objdump"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise RuntimeError("llvm-objdump not found; set LLVM_OBJDUMP")


def disassemble(hsaco: Path) -> Path:
    objdump = find_objdump()
    isa = hsaco.with_suffix(".isa")
    result = run_cmd([objdump, "-d", "--no-show-raw-insn", str(hsaco)], cwd=PROJECT_ROOT)
    isa.write_text(result.stdout)
    return isa


MNEMONIC_RE = re.compile(r"^\s*(?:[0-9a-fA-F]+:\s*(?:[0-9a-fA-F]{2}\s+)*)?([A-Za-z_][A-Za-z0-9_.$]*)\b")
VGPR_RE = re.compile(r"\bv(?:gpr)?(\d+)\b|\bv\[(\d+):(\d+)\]")
ACC_RE = re.compile(r"\b(?:acc|a)(\d+)\b|\b(?:acc|a)\[(\d+):(\d+)\]")


def iter_mnemonics(text: str) -> Iterable[str]:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("Disassembly", "file format")) or stripped.endswith(">:"):
            continue
        match = MNEMONIC_RE.match(line)
        if match:
            yield match.group(1)


def count_isa(isa: Path) -> dict[str, object]:
    text = isa.read_text(errors="replace")
    mnemonics = list(iter_mnemonics(text))
    counts = Counter(mnemonics)
    categories = {
        "v_mfma_f32_16x16x16_bf16": counts.get("v_mfma_f32_16x16x16_bf16", 0),
        "v_mfma_f32_32x32x8_bf16": counts.get("v_mfma_f32_32x32x8_bf16", 0),
        "v_mfma_total": sum(c for m, c in counts.items() if m.startswith("v_mfma")),
        "s_barrier": counts.get("s_barrier", 0),
        "s_waitcnt": counts.get("s_waitcnt", 0),
        "ds_read": sum(c for m, c in counts.items() if m.startswith("ds_read")),
        "ds_write": sum(c for m, c in counts.items() if m.startswith("ds_write")),
        "global_or_buffer_load": sum(c for m, c in counts.items() if m.startswith(("global_load", "flat_load", "buffer_load"))),
        "global_or_buffer_store": sum(c for m, c in counts.items() if m.startswith(("global_store", "flat_store", "buffer_store"))),
        "v_add": sum(c for m, c in counts.items() if m.startswith("v_add")),
        "v_add3": counts.get("v_add3_u32", 0),
        "v_lshl": sum(c for m, c in counts.items() if m.startswith("v_lshl")),
        "v_lshl_add": sum(c for m, c in counts.items() if m.startswith("v_lshl_add")),
        "v_or": sum(c for m, c in counts.items() if m.startswith("v_or")),
        "v_bfe": sum(c for m, c in counts.items() if m.startswith("v_bfe")),
    }
    max_vgpr = max_index(text, VGPR_RE)
    max_acc = max_index(text, ACC_RE)
    categories["max_vgpr_index_static_best_effort"] = max_vgpr
    categories["max_acc_index_static_best_effort"] = max_acc
    categories["total_instructions"] = len(mnemonics)
    categories["top_mnemonics"] = dict(counts.most_common(25))
    return categories


def max_index(text: str, regex: re.Pattern[str]) -> int | None:
    max_seen: int | None = None
    for match in regex.finditer(text):
        nums = [g for g in match.groups() if g is not None]
        if not nums:
            continue
        if len(nums) >= 3 and nums[-2].isdigit() and nums[-1].isdigit():
            value = int(nums[-1])
        else:
            value = max(int(n) for n in nums if n.isdigit())
        max_seen = value if max_seen is None else max(max_seen, value)
    return max_seen


def run_grep_audit(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir) / "compiler_grep"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, object] = {}
    for label, cmd in GREP_COMMANDS:
        result = run_cmd(cmd, cwd=PROJECT_ROOT, check=False)
        path = out_dir / f"{label}.txt"
        path.write_text("$ " + " ".join(cmd) + "\n\n" + result.stdout)
        rows[label] = {
            "cmd": " ".join(cmd),
            "returncode": result.returncode,
            "path": str(path),
            "first_lines": result.stdout.splitlines()[:20],
        }
    return rows


def expected_vs_observed(correctness: list[dict[str, object]], rocprof: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pred32 = rocprof.get("pred32_only_sink", {})
    update16 = rocprof.get("update16_only", {})
    for mode, data in rocprof.items():
        if not data:
            continue
        row: dict[str, object] = {"mode": mode}
        for metric in ["VGPR_Count", "Accum_VGPR_Count", "Scratch_Size", "LDS_Block_Size", "SQ_INSTS_MFMA", "SQ_INSTS_VALU", "SQ_INSTS_LDS"]:
            observed = _to_float(data.get(metric))
            p = _to_float(pred32.get(metric))
            u = _to_float(update16.get(metric))
            row[metric] = observed
            if p is not None and u is not None:
                row[f"{metric}_max_pred32_update16"] = max(p, u)
                row[f"{metric}_sum_pred32_update16"] = p + u
        rows.append(row)
    return rows


def write_report(
    args: argparse.Namespace,
    correctness: list[dict[str, object]],
    rocprof: dict[str, dict[str, object]],
    isa_rows: dict[str, dict[str, object]],
    grep_rows: dict[str, object],
) -> Path:
    out = SCRIPT_DIR / "mfma_region_lifetime_v3_report.md"
    evs = expected_vs_observed(correctness, rocprof)
    lines: list[str] = []
    lines.append("# MFMA Region Lifetime v3 Report\n")
    lines.append("## Summary\n")
    lines.append(
        "This report is generated by `profile_mfma_region_lifetime_v3.py`. "
        "It isolates independent pred/update MFMA regions in one minimal Avelang kernel. "
        "The key question is whether combined variants allocate/register-live closer to "
        "`max(pred_only, update_only)` or closer to `pred_only + update_only`.\n"
    )
    lines.append("## Correctness And Runtime\n")
    lines.append("| mode | ok | update_expected | latency_ms | max_abs | mean_abs | max_rel | sink_checksum_abs |")
    lines.append("|:---|:---:|:---:|---:|---:|---:|---:|---:|")
    for row in correctness:
        lines.append(
            "| {mode} | {ok} | {update_expected} | {latency_ms:.6f} | {max_abs} | {mean_abs} | {max_rel} | {sink_checksum_abs:.6g} |".format(
                **row
            )
        )
    lines.append("\n## Rocprof Resource Table\n")
    metric_cols = [
        "trace_median_us",
        "Workgroup_Size",
        "Grid_Size",
        "LDS_Block_Size",
        "Scratch_Size",
        "VGPR_Count",
        "Accum_VGPR_Count",
        "SGPR_Count",
        "SQ_INSTS_MFMA",
        "SQ_INSTS_VALU",
        "SQ_INSTS_SALU",
        "SQ_INSTS_VMEM",
        "SQ_INSTS_LDS",
        "OccupancyPercent",
    ]
    lines.append("| mode | " + " | ".join(metric_cols) + " |")
    lines.append("|:---|" + "|".join(["---:"] * len(metric_cols)) + "|")
    for mode, data in rocprof.items():
        vals = [fmt(data.get(col)) for col in metric_cols]
        lines.append(f"| {mode} | " + " | ".join(vals) + " |")
    lines.append("\n## Expected Vs Observed Resource Combination\n")
    lines.append("| mode | AccVGPR obs | AccVGPR max(pred32,update16) | AccVGPR sum | VGPR obs | VGPR max | VGPR sum | Scratch obs |")
    lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|")
    for row in evs:
        lines.append(
            "| {mode} | {acc} | {acc_max} | {acc_sum} | {vgpr} | {vgpr_max} | {vgpr_sum} | {scratch} |".format(
                mode=row["mode"],
                acc=fmt(row.get("Accum_VGPR_Count")),
                acc_max=fmt(row.get("Accum_VGPR_Count_max_pred32_update16")),
                acc_sum=fmt(row.get("Accum_VGPR_Count_sum_pred32_update16")),
                vgpr=fmt(row.get("VGPR_Count")),
                vgpr_max=fmt(row.get("VGPR_Count_max_pred32_update16")),
                vgpr_sum=fmt(row.get("VGPR_Count_sum_pred32_update16")),
                scratch=fmt(row.get("Scratch_Size")),
            )
        )
    lines.append("\n## ISA Static Counts\n")
    lines.append("| mode | mfma16 | mfma32 | mfma_total | s_barrier | s_waitcnt | ds_read | ds_write | global/buffer load | global/buffer store | max VGPR idx | max ACC idx |")
    lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mode, row in isa_rows.items():
        lines.append(
            "| {mode} | {mf16} | {mf32} | {mft} | {bar} | {wait} | {dsr} | {dsw} | {gload} | {gstore} | {vgpr} | {acc} |".format(
                mode=mode,
                mf16=fmt(row.get("v_mfma_f32_16x16x16_bf16")),
                mf32=fmt(row.get("v_mfma_f32_32x32x8_bf16")),
                mft=fmt(row.get("v_mfma_total")),
                bar=fmt(row.get("s_barrier")),
                wait=fmt(row.get("s_waitcnt")),
                dsr=fmt(row.get("ds_read")),
                dsw=fmt(row.get("ds_write")),
                gload=fmt(row.get("global_or_buffer_load")),
                gstore=fmt(row.get("global_or_buffer_store")),
                vgpr=fmt(row.get("max_vgpr_index_static_best_effort")),
                acc=fmt(row.get("max_acc_index_static_best_effort")),
            )
        )
    lines.append("\n## Compiler Grep Audit\n")
    for label, row in grep_rows.items():
        lines.append(f"### {label}\n")
        lines.append(f"Command: `{row['cmd']}`\n")
        lines.append(f"Output: `{row['path']}`\n")
        first = row.get("first_lines", [])
        if first:
            lines.append("First matches:\n")
            lines.append("```text")
            lines.extend(str(x) for x in first[:12])
            lines.append("```\n")
        else:
            lines.append("No matches in first output block.\n")
    lines.append("## Initial Interpretation Checklist\n")
    lines.append("- `update16_only` is the correctness baseline for update output.")
    lines.append("- `pred32_then_update16` should be compared against `max(pred32_only_sink, update16_only)` for VGPR/AccVGPR/Scratch.")
    lines.append("- If barriers or source lifetime hint do not reduce resources, evidence points away from simple source ordering and toward lowering/register-lifetime behavior.")
    lines.append("- Static max VGPR/ACC index extraction is best-effort only; rocprof `VGPR_Count`/`Accum_VGPR_Count` is authoritative.\n")
    lines.append("## Exact Commands\n")
    lines.append("```bash")
    lines.append(f"python3 {REPRO.relative_to(PROJECT_ROOT)} --mode all --warmup {args.warmup} --repeat {args.repeat} --json")
    lines.append(f"python3 {Path(__file__).relative_to(PROJECT_ROOT)} --run-all --warmup {args.warmup} --repeat {args.repeat}")
    lines.append("```")
    out.write_text("\n".join(lines) + "\n")
    return out


def fmt(value: object) -> str:
    if value is None:
        return ""
    number = _to_float(value)
    if number is None:
        return str(value)
    if abs(number) >= 1000:
        return f"{number:.0f}"
    if abs(number) >= 1:
        return f"{number:.4f}"
    return f"{number:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--hsaco-dir", type=Path, default=DEFAULT_HSACO_DIR)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--scale", type=float, default=0.73)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--rocprof-warmup", type=int, default=2)
    parser.add_argument("--rocprof-repeat", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--skip-rocprof", action="store_true")
    parser.add_argument("--skip-hsaco", action="store_true")
    parser.add_argument("--skip-grep", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.hsaco_dir.mkdir(parents=True, exist_ok=True)

    correctness = run_correctness(args)
    (args.out_dir / "correctness.json").write_text(json.dumps(correctness, indent=2, sort_keys=True))

    mode_names = [row["mode"] for row in correctness]
    rocprof: dict[str, dict[str, object]] = {}
    if not args.skip_rocprof:
        for mode in mode_names:
            rocprof[mode] = run_rocprof(args, mode)
        (args.out_dir / "rocprof_summary.json").write_text(json.dumps(rocprof, indent=2, sort_keys=True))

    isa_rows: dict[str, dict[str, object]] = {}
    if not args.skip_hsaco:
        for mode in IMPORTANT_HSACO_VARIANTS:
            hsaco = dump_hsaco(args, mode)
            isa = disassemble(hsaco)
            row = count_isa(isa)
            row["hsaco"] = str(hsaco)
            row["isa"] = str(isa)
            isa_rows[mode] = row
        (args.out_dir / "isa_summary.json").write_text(json.dumps(isa_rows, indent=2, sort_keys=True))

    grep_rows: dict[str, object] = {}
    if not args.skip_grep:
        grep_rows = run_grep_audit(args)
        (args.out_dir / "compiler_grep_summary.json").write_text(json.dumps(grep_rows, indent=2, sort_keys=True))

    report = write_report(args, correctness, rocprof, isa_rows, grep_rows)
    print(f"report={report}")


if __name__ == "__main__":
    main()
