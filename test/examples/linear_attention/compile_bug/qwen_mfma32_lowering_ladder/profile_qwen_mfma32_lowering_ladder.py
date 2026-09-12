#!/usr/bin/env python3
"""Profile Qwen-shaped MFMA32 lowering ladder."""

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
REPRO = SCRIPT_DIR / "repro_qwen_mfma32_lowering_ladder.py"
OUT_DIR = PROJECT_ROOT / "test/examples/linear_attention/rocprof_outputs/qwen_mfma32_lowering_ladder"
HSACO_DIR = PROJECT_ROOT / "test/examples/linear_attention/rocprof_outputs/qwen_mfma32_lowering_ladder_hsaco"

PMCS = ["SQ_INSTS_MFMA", "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_VMEM", "SQ_INSTS_LDS", "OccupancyPercent"]
META = ["Grid_Size", "Workgroup_Size", "LDS_Block_Size", "Scratch_Size", "VGPR_Count", "Accum_VGPR_Count", "SGPR_Count"]
LEVEL_ORDER = [
    "L0_pred32_mfma_only",
    "L1_pred32_unpack_to_pred_partial",
    "L2_pred32_unpack_reduce",
    "L3_pred32_ucorr_epilogue_no_store",
    "L4_pred32_vdecay_shared_stage",
    "L5_pred32_vdecay_plus_k_stage",
    "L5_alt_token_major_k_stage",
    "L6_pred32_vdecay_update16_one_ktile",
    "L7_pred32_vdecay_update16_full_k_no_feedback",
    "L8_pred32_update16_full_k_state_writeback",
    "L9_pred32_update16_with_h_vn_global_materialization",
    "L9_alt_grouped_v4_materialization",
]
TRANSITIONS = [
    ("L0 -> L1", "L0_pred32_mfma_only", "L1_pred32_unpack_to_pred_partial", "accumulator unpack cost"),
    ("L1 -> L2", "L1_pred32_unpack_to_pred_partial", "L2_pred32_unpack_reduce", "pred_partial LDS read/reduction cost"),
    ("L2 -> L3", "L2_pred32_unpack_reduce", "L3_pred32_ucorr_epilogue_no_store", "u load/sub cost"),
    ("L3 -> L4", "L3_pred32_ucorr_epilogue_no_store", "L4_pred32_vdecay_shared_stage", "v_decay shared staging cost"),
    ("L4 -> L5", "L4_pred32_vdecay_shared_stage", "L5_pred32_vdecay_plus_k_stage", "k staging/shared view cost"),
    ("L5 -> L6", "L5_pred32_vdecay_plus_k_stage", "L6_pred32_vdecay_update16_one_ktile", "first dependent update MFMA cost"),
    ("L6 -> L7", "L6_pred32_vdecay_update16_one_ktile", "L7_pred32_vdecay_update16_full_k_no_feedback", "full K update cost"),
    ("L7 -> L8", "L7_pred32_vdecay_update16_full_k_no_feedback", "L8_pred32_update16_full_k_state_writeback", "state writeback cost"),
    ("L8 -> L9", "L8_pred32_update16_full_k_state_writeback", "L9_pred32_update16_with_h_vn_global_materialization", "h/vn global materialization cost"),
]


def run_cmd(cmd: list[str], *, cwd: Path = PROJECT_ROOT, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def load_repro():
    spec = importlib.util.spec_from_file_location("repro_qwen_mfma32_lowering_ladder", REPRO)
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
            "--level",
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
    rows = json.loads(text[text.find("[") :])
    return rows


def run_rocprof(args: argparse.Namespace, level: str) -> dict[str, object]:
    out_dir = Path(args.out_dir) / f"rocprof_{level}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "/opt/rocm/bin/rocprofv3",
        "--kernel-trace",
        "--pmc",
        *PMCS,
        "--kernel-include-regex",
        "_qwen_mfma32_lowering_ladder_kernel",
        "-d",
        str(out_dir),
        "-o",
        f"{level}_counters",
        "-f",
        "csv",
        "--",
        sys.executable,
        str(REPRO),
        "--level",
        level,
        "--seed",
        str(args.seed),
        "--warmup",
        str(args.rocprof_warmup),
        "--repeat",
        str(args.rocprof_repeat),
    ]
    result = run_cmd(cmd, check=False)
    summary = parse_rocprof_dir(out_dir, "_qwen_mfma32_lowering_ladder_kernel")
    summary["returncode"] = result.returncode
    summary["stdout_tail"] = "\n".join(result.stdout.splitlines()[-12:])
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
        if "kernel_trace" in path.name:
            summary.update(parse_trace(path, kernel_substr))
            summary["trace_csv"] = str(path)
        elif "counter" in path.name:
            counters = parse_counters(path, kernel_substr)
            if counters:
                summary.update(counters)
                summary["counter_csv"] = str(path)
    return summary


def parse_trace(path: Path, kernel_substr: str) -> dict[str, object]:
    durations = []
    meta: dict[str, object] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel = row.get("Kernel_Name") or row.get("Name") or ""
            if kernel_substr not in kernel:
                continue
            for key in META:
                if row.get(key):
                    meta[key] = row[key]
            duration = _to_float(row.get("Duration") or row.get("DurationNs"))
            if duration is not None:
                durations.append(duration / 1000.0)
            else:
                start = _to_float(row.get("Start_Timestamp"))
                end = _to_float(row.get("End_Timestamp"))
                if start is not None and end is not None:
                    durations.append((end - start) / 1000.0)
    if durations:
        meta["trace_median_us"] = statistics.median(durations)
        meta["trace_count"] = len(durations)
    return meta


def parse_counters(path: Path, kernel_substr: str) -> dict[str, object]:
    meta: dict[str, object] = {}
    metrics: dict[str, list[float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel = row.get("Kernel_Name") or row.get("Name") or ""
            if kernel_substr not in kernel:
                continue
            for key in META:
                if row.get(key):
                    meta[key] = row[key]
            name = row.get("Counter_Name") or row.get("Metric") or row.get("Name")
            val = _to_float(row.get("Value") or row.get("Counter_Value"))
            if name and val is not None:
                metrics.setdefault(name, []).append(val)
    for name, vals in metrics.items():
        meta[name] = statistics.median(vals)
    return meta


def dump_hsaco(args: argparse.Namespace, level: str) -> Path:
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
        if "_qwen_mfma32_lowering_ladder_kernel" in kernel_name and not dumped:
            path = out_dir / f"{level}.{kernel_name}.hsaco"
            path.write_bytes(binary)
            dumped.append(path)
            print(f"dumped_hsaco={path}")
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        tensors = repro.make_inputs(args.seed)
        repro.launch_level(level, tensors)
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile
    if not dumped:
        raise RuntimeError(f"no hsaco dumped for {level}")
    return dumped[0]


def find_objdump() -> str:
    candidates = [os.environ.get("LLVM_OBJDUMP"), "/opt/rocm/llvm/bin/llvm-objdump", "/opt/rocm/bin/llvm-objdump", shutil.which("llvm-objdump")]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    raise RuntimeError("llvm-objdump not found")


def disassemble(hsaco: Path) -> Path:
    result = run_cmd([find_objdump(), "-d", "--no-show-raw-insn", str(hsaco)])
    isa = hsaco.with_suffix(".isa")
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


def max_index(text: str, regex: re.Pattern[str]) -> int | None:
    out: int | None = None
    for m in regex.finditer(text):
        nums = [g for g in m.groups() if g and g.isdigit()]
        if nums:
            value = max(int(x) for x in nums)
            out = value if out is None else max(out, value)
    return out


def analyze_isa(isa: Path) -> dict[str, object]:
    text = isa.read_text(errors="replace")
    counts = Counter(iter_mnemonics(text))
    return {
        "v_mfma_f32_32x32x8_bf16": counts.get("v_mfma_f32_32x32x8_bf16", 0),
        "v_mfma_f32_16x16x16_bf16": counts.get("v_mfma_f32_16x16x16_bf16", 0),
        "global_load": sum(c for m, c in counts.items() if m.startswith("global_load")),
        "global_store": sum(c for m, c in counts.items() if m.startswith("global_store")),
        "buffer_load": sum(c for m, c in counts.items() if m.startswith("buffer_load")),
        "buffer_store": sum(c for m, c in counts.items() if m.startswith("buffer_store")),
        "ds_read": sum(c for m, c in counts.items() if m.startswith("ds_read")),
        "ds_write": sum(c for m, c in counts.items() if m.startswith("ds_write")),
        "s_barrier": counts.get("s_barrier", 0),
        "s_waitcnt": counts.get("s_waitcnt", 0),
        "v_add": sum(c for m, c in counts.items() if m.startswith("v_add")),
        "v_add3": sum(c for m, c in counts.items() if m.startswith("v_add3")),
        "v_lshl": sum(c for m, c in counts.items() if m.startswith("v_lshl")),
        "v_lshl_add": sum(c for m, c in counts.items() if m.startswith("v_lshl_add")),
        "v_or": sum(c for m, c in counts.items() if m.startswith("v_or")),
        "v_bfe": sum(c for m, c in counts.items() if m.startswith("v_bfe")),
        "max_vgpr_index_static_best_effort": max_index(text, VGPR_RE),
        "max_acc_index_static_best_effort": max_index(text, ACC_RE),
        "total_instructions": sum(counts.values()),
        "isa": str(isa),
    }


def delta_rows(rocprof: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    metrics = ["trace_median_us", "VGPR_Count", "Accum_VGPR_Count", "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_VMEM", "SQ_INSTS_LDS", "LDS_Block_Size"]
    for label, a, b, purpose in TRANSITIONS:
        row: dict[str, object] = {"transition": label, "purpose": purpose}
        for metric in metrics:
            av = _to_float(rocprof.get(a, {}).get(metric))
            bv = _to_float(rocprof.get(b, {}).get(metric))
            row[metric] = None if av is None or bv is None else bv - av
        rows.append(row)
    return rows


def find_biggest_jump(deltas: list[dict[str, object]]) -> dict[str, object] | None:
    best = None
    best_score = -1.0
    for row in deltas:
        score = 0.0
        for metric, weight in [
            ("Accum_VGPR_Count", 3.0),
            ("VGPR_Count", 2.0),
            ("SQ_INSTS_VALU", 0.005),
            ("SQ_INSTS_SALU", 0.01),
            ("SQ_INSTS_VMEM", 0.02),
            ("SQ_INSTS_LDS", 0.01),
            ("trace_median_us", 4.0),
            ("LDS_Block_Size", 0.005),
        ]:
            val = _to_float(row.get(metric))
            if val and val > 0:
                score += val * weight
        if score > best_score:
            best = row
            best_score = score
    return best


def fmt(x: object) -> str:
    val = _to_float(x)
    if val is None:
        return "" if x is None else str(x)
    if abs(val) >= 1000:
        return f"{val:.0f}"
    if abs(val) >= 1:
        return f"{val:.4f}"
    return f"{val:.6g}"


def write_report(smoke: list[dict[str, object]], rocprof: dict[str, dict[str, object]], isa: dict[str, dict[str, object]]) -> Path:
    deltas = delta_rows(rocprof)
    biggest = find_biggest_jump(deltas)
    report = SCRIPT_DIR / "qwen_mfma32_lowering_ladder_report.md"
    lines = []
    lines.append("# Qwen MFMA32 Lowering Ladder Report\n")
    lines.append("## Purpose\n")
    lines.append("This ladder follows the negative `mfma_region_lifetime_v3` result: independent sequential MFMA regions do not additively accumulate AccVGPR. The goal here is Qwen-shaped delta debugging for v28/v29-like MFMA32 pred/update source patterns.\n")
    lines.append("## Summary Conclusion\n")
    if biggest:
        lines.append(f"Biggest weighted transition: `{biggest['transition']}` ({biggest['purpose']}).\n")
    lines.append("The data below should be read as lowering/resource attribution, not full Qwen correctness. Each level uses fixed Qwen-like BT64/BV32/Hv8/K128 tensors and deterministic inputs.\n")
    lines.append("## Ladder Variants\n")
    for name in LEVEL_ORDER:
        lines.append(f"- `{name}`")
    lines.append("\n## Smoke/Checksum\n")
    lines.append("| level | latency_ms | finite | checksum |")
    lines.append("|:---|---:|:---:|---:|")
    for row in smoke:
        lines.append(f"| {row['level']} | {row['latency_ms']:.6f} | {row['sink_finite']} | {row['sink_checksum_abs']:.6g} |")
    lines.append("\n## Rocprof Resource Table\n")
    cols = ["trace_median_us", "VGPR_Count", "Accum_VGPR_Count", "SGPR_Count", "Scratch_Size", "LDS_Block_Size", "SQ_INSTS_MFMA", "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_VMEM", "SQ_INSTS_LDS", "OccupancyPercent"]
    lines.append("| level | " + " | ".join(cols) + " |")
    lines.append("|:---|" + "|".join(["---:"] * len(cols)) + "|")
    for name in LEVEL_ORDER:
        row = rocprof.get(name, {})
        lines.append(f"| {name} | " + " | ".join(fmt(row.get(c)) for c in cols) + " |")
    lines.append("\n## Static ISA Table\n")
    isa_cols = ["v_mfma_f32_32x32x8_bf16", "v_mfma_f32_16x16x16_bf16", "global_load", "global_store", "buffer_load", "buffer_store", "ds_read", "ds_write", "s_barrier", "s_waitcnt", "v_add", "v_add3", "v_lshl", "v_lshl_add", "v_or", "v_bfe", "max_vgpr_index_static_best_effort", "max_acc_index_static_best_effort"]
    lines.append("| level | " + " | ".join(isa_cols) + " |")
    lines.append("|:---|" + "|".join(["---:"] * len(isa_cols)) + "|")
    for name in LEVEL_ORDER:
        row = isa.get(name, {})
        lines.append(f"| {name} | " + " | ".join(fmt(row.get(c)) for c in isa_cols) + " |")
    lines.append("\n## Delta Table\n")
    dcols = ["trace_median_us", "VGPR_Count", "Accum_VGPR_Count", "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_VMEM", "SQ_INSTS_LDS", "LDS_Block_Size"]
    lines.append("| transition | purpose | " + " | ".join(dcols) + " |")
    lines.append("|:---|:---|" + "|".join(["---:"] * len(dcols)) + "|")
    for row in deltas:
        lines.append(f"| {row['transition']} | {row['purpose']} | " + " | ".join(fmt(row.get(c)) for c in dcols) + " |")
    lines.append("\n## Disproportionate Jump Analysis\n")
    if biggest:
        lines.append(f"The first/strongest weighted jump is `{biggest['transition']}`: {biggest['purpose']}.\n")
        acc = _to_float(biggest.get("Accum_VGPR_Count")) or 0.0
        vgpr = _to_float(biggest.get("VGPR_Count")) or 0.0
        if acc > 64 or vgpr > 32:
            classification = "Compiler/lowering issue candidate"
        elif (_to_float(biggest.get("SQ_INSTS_VALU")) or 0) > 1000 or (_to_float(biggest.get("SQ_INSTS_LDS")) or 0) > 1000:
            classification = "Mixed: source schedule growth plus possible scalarized lowering"
        else:
            classification = "Source schedule issue: no single huge register jump"
        lines.append(f"Classification: **{classification}**.\n")
    lines.append("## Workaround Result\n")
    l5 = rocprof.get("L5_pred32_vdecay_plus_k_stage", {})
    l5_alt = rocprof.get("L5_alt_token_major_k_stage", {})
    if l5 and l5_alt:
        lines.append("Primary workaround candidate for the L4->L5 jump: `L5_alt_token_major_k_stage`, staging K as token-major `[BT,K]` instead of update-oriented transposed `[K,BT]`.")
        lines.append("")
        lines.append("| metric | L5 transposed | L5_alt token-major | delta |")
        lines.append("|:---|---:|---:|---:|")
        for c in cols:
            bv = _to_float(l5.get(c))
            av = _to_float(l5_alt.get(c))
            lines.append(f"| {c} | {fmt(bv)} | {fmt(av)} | {fmt(None if bv is None or av is None else av - bv)} |")
        lines.append("")
    base = rocprof.get("L9_pred32_update16_with_h_vn_global_materialization", {})
    alt = rocprof.get("L9_alt_grouped_v4_materialization", {})
    if base and alt:
        lines.append("Secondary workaround candidate: `L9_alt_grouped_v4_materialization`, using flat grouped-v4 VN stores instead of scalar 4D tensor indexing.")
        lines.append("")
        lines.append("| metric | L9 | L9_alt | delta |")
        lines.append("|:---|---:|---:|---:|")
        for c in cols:
            bv = _to_float(base.get(c))
            av = _to_float(alt.get(c))
            lines.append(f"| {c} | {fmt(bv)} | {fmt(av)} | {fmt(None if bv is None or av is None else av - bv)} |")
    lines.append("\n## Targeted Compiler/Lowering Diagnosis\n")
    lines.append("This ladder avoids broad compiler grep. Interpret the biggest jump as follows:")
    lines.append("- If the jump is L0->L1 or L1->L2, inspect MFMA32 accumulator unpack and pred_partial LDS indexing lowering.")
    lines.append("- If the jump is L3->L4 or L4->L5, inspect shared memref layouts, `al.view`, and DS read/write/index generation.")
    lines.append("- If the jump is L5->L6, inspect dependent pred->update liveness and mixed MFMA32/MFMA16 accumulator allocation.")
    lines.append("- If the jump is L8->L9, inspect global store lowering and whether contiguous VN stores can be generated as vector stores or `raw_buffer_store_x4` helpers.")
    lines.append("\n## Exact Commands\n")
    lines.append("```bash")
    lines.append(f"python3 {REPRO.relative_to(PROJECT_ROOT)} --level all --warmup 5 --repeat 20")
    lines.append(f"python3 {Path(__file__).relative_to(PROJECT_ROOT)} --warmup 5 --repeat 20 --rocprof-warmup 2 --rocprof-repeat 5")
    lines.append("```")
    report.write_text("\n".join(lines) + "\n")
    return report


def write_decision_report(rocprof: dict[str, dict[str, object]], isa: dict[str, dict[str, object]]) -> Path:
    deltas = delta_rows(rocprof)
    biggest = find_biggest_jump(deltas)
    out = PROJECT_ROOT / "test/examples/linear_attention/vllm_compare/qwen_gdn_next_decision_qwen_shaped_lowering.md"
    lines = ["# Qwen GDN Next Decision After Qwen-Shaped Lowering Ladder\n"]
    lines.append("## Summary\n")
    lines.append("Independent sequential MFMA lifetime was not the generic issue in `mfma_region_lifetime_v3`; this Qwen-shaped ladder attributes resource growth to concrete source features in a v29-like BT64/BV32 MFMA32 schedule.\n")
    if biggest:
        lines.append(f"Biggest weighted transition observed: `{biggest['transition']}` ({biggest['purpose']}).\n")
    lines.append("## Decision Questions\n")
    lines.append("1. Since independent sequential MFMA lifetime is not the issue, what Qwen-shaped pattern is the issue?\n")
    if biggest:
        lines.append(f"   - Current evidence points first to `{biggest['transition']}`: {biggest['purpose']}.\n")
    lines.append("2. Is there evidence for a small compiler/helper fix?\n")
    lines.append("   - If the largest jump is materialization/global-store related and grouped-v4 helps, a helper for vectorized contiguous stores is plausible. If the largest jump is shared staging/view related, inspect layout/index lowering before proposing a broad block-dot rewrite.\n")
    lines.append("3. Which source pattern should be fixed first?\n")
    if biggest:
        lines.append(f"   - Start with the source pattern added by `{biggest['transition']}`.\n")
    lines.append("4. Should we continue v29/BT64/MFMA32 after this?\n")
    lines.append("   - Continue only if the culprit has a local workaround/helper with measurable resource improvement. Otherwise current v29 source schedule is broadly too heavy and v24 remains the production path.\n")
    lines.append("5. Exact next action:\n")
    lines.append("   - Use `qwen_mfma32_lowering_ladder_report.md` delta table to pick one helper-sized change, then test it in the isolated ladder before returning to full Qwen.\n")
    out.write_text("\n".join(lines))
    return out


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
        for level in LEVEL_ORDER:
            rocprof[level] = run_rocprof(args, level)
        (args.out_dir / "rocprof_summary.json").write_text(json.dumps(rocprof, indent=2, sort_keys=True))

    isa: dict[str, dict[str, object]] = {}
    if not args.skip_hsaco:
        for level in LEVEL_ORDER:
            hsaco = dump_hsaco(args, level)
            isa_path = disassemble(hsaco)
            row = analyze_isa(isa_path)
            row["hsaco"] = str(hsaco)
            isa[level] = row
        (args.out_dir / "isa_summary.json").write_text(json.dumps(isa, indent=2, sort_keys=True))

    report = write_report(smoke, rocprof, isa)
    decision = write_decision_report(rocprof, isa)
    print(f"report={report}")
    print(f"decision={decision}")


if __name__ == "__main__":
    main()
