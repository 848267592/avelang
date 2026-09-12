#!/usr/bin/env python3
"""Summarize Stage 5E rocprof traces without changing the measured graph."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
CASE_RE = re.compile(
    r"(?:(full|tail)_)?t(\d+)_(none|warm|perturb|prime)_(v18|hierarchical_fp32_v1)$"
)
BASIC_COUNTERS = {
    "SQ_INSTS_MFMA",
    "SQ_INSTS_VALU",
    "SQ_INSTS_SALU",
    "SQ_INSTS_VMEM",
    "SQ_INSTS_LDS",
    "OccupancyPercent",
}
STAGE_ORDER = ("cumsum", "kkt", "solve", "w", "u", "asm", "chunk_o", "cast")


def stage_of(name: str) -> str | None:
    if "_qwen_gdn_chunk_cumsum_kernel" in name:
        return "cumsum"
    if "_qwen_gdn_kkt_bf16_kernel" in name:
        return "kkt"
    if "_qwen_gdn_solve_kernel_v18_parallel" in name or "_qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1" in name:
        return "solve"
    if "_qwen_gdn_w_bf16_kernel_bt64_mfma_v2_s0" in name:
        return "w"
    if "_qwen_gdn_u_bf16_kernel_bt64_mfma_v2_s0" in name:
        return "u"
    if name == "qwen_gdn_bt64_gfx942_asm_v0":
        return "asm"
    if "_qwen_gdn_chunk_o_bf16_kernel_bt64_mfma_v2_s0" in name:
        return "chunk_o"
    if "bfloat16_copy_kernel_cuda" in name:
        return "cast"
    return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nN/A\n")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def selected_stage_rows(trace: list[dict[str, str]], operation: str, control: str) -> list[dict[str, str]]:
    rows = [row for row in trace if stage_of(row["Kernel_Name"]) is not None]
    first_stage = "cumsum" if operation == "full" else "solve"
    first = next((index for index, row in enumerate(rows) if stage_of(row["Kernel_Name"]) == first_stage), len(rows))
    rows = rows[first:]
    if operation == "tail":
        rows = [row for row in rows if stage_of(row["Kernel_Name"]) not in ("cumsum", "kkt")]
    if operation == "tail" and control == "warm":
        counts: dict[str, int] = defaultdict(int)
        chosen: list[dict[str, str]] = []
        for row in rows:
            stage = stage_of(row["Kernel_Name"])
            if stage == "solve":
                chosen.append(row)
                continue
            index = counts[stage]
            counts[stage] += 1
            if index % 2 == 1:
                chosen.append(row)
        rows = chosen
    return rows


def analyze_case(directory: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    match = CASE_RE.fullmatch(directory.name)
    if not match:
        return [], [], []
    operation = match.group(1) or "tail"
    t = int(match.group(2))
    control = match.group(3)
    solve_impl = match.group(4)
    trace = selected_stage_rows(read_csv(directory / "stage5e_kernel_trace.csv"), operation, control)
    counters = read_csv(directory / "stage5e_counter_collection.csv")
    counter_by_dispatch: dict[str, dict[str, float]] = defaultdict(dict)
    for row in counters:
        counter_by_dispatch[row["Dispatch_Id"]][row["Counter_Name"]] = float(row["Counter_Value"])

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in trace:
        grouped[stage_of(row["Kernel_Name"])].append(row)

    summary: list[dict[str, object]] = []
    cache: list[dict[str, object]] = []
    for stage in STAGE_ORDER:
        stage_rows = grouped.get(stage, [])
        if not stage_rows:
            continue
        common = {
            "operation": operation,
            "T": t,
            "control": control,
            "solve_impl": solve_impl,
            "stage": stage,
            "dispatches": len(stage_rows),
            "trace_median_us": median([
                (int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1000.0
                for row in stage_rows
            ]),
            "workgroup_size": stage_rows[-1]["Workgroup_Size_X"],
            "grid_size": stage_rows[-1]["Grid_Size_X"],
            "lds_block_size": stage_rows[-1]["LDS_Block_Size"],
            "scratch_size": stage_rows[-1]["Scratch_Size"],
            "vgpr_count": stage_rows[-1]["VGPR_Count"],
            "accvgpr_count": stage_rows[-1]["Accum_VGPR_Count"],
            "sgpr_count": stage_rows[-1]["SGPR_Count"],
        }
        names = sorted({name for dispatch in stage_rows for name in counter_by_dispatch.get(dispatch["Dispatch_Id"], {})})
        for name in names:
            values = [counter_by_dispatch[row["Dispatch_Id"]][name] for row in stage_rows if name in counter_by_dispatch.get(row["Dispatch_Id"], {})]
            common[name] = median(values)
        summary.append(common)
        for name in names:
            if name not in BASIC_COUNTERS:
                cache.append({**{key: common[key] for key in ("operation", "T", "control", "solve_impl", "stage")}, "counter_name": name, "counter_median": common[name]})

    gaps: list[dict[str, object]] = []
    for previous, current in zip(trace, trace[1:]):
        prev_stage = stage_of(previous["Kernel_Name"])
        stage = stage_of(current["Kernel_Name"])
        if STAGE_ORDER.index(stage) == STAGE_ORDER.index(prev_stage) + 1:
            gaps.append({
                "operation": operation,
                "T": t,
                "control": control,
                "solve_impl": solve_impl,
                "from_stage": prev_stage,
                "to_stage": stage,
                "gap_us": (int(current["Start_Timestamp"]) - int(previous["End_Timestamp"])) / 1000.0,
            })
    gap_summary: list[dict[str, object]] = []
    by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in gaps:
        by_pair[(str(row["from_stage"]), str(row["to_stage"]))].append(float(row["gap_us"]))
    for (from_stage, to_stage), values in by_pair.items():
        gap_summary.append({
            "operation": operation,
            "T": t,
            "control": control,
            "solve_impl": solve_impl,
            "from_stage": from_stage,
            "to_stage": to_stage,
            "samples": len(values),
            "gap_median_us": median(values),
        })
    return summary, cache, gap_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rocprof-dir", type=Path, default=HERE / "rocprof")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    summaries: list[dict[str, object]] = []
    cache_rows: list[dict[str, object]] = []
    gaps: list[dict[str, object]] = []
    for directory in sorted(path for path in args.rocprof_dir.iterdir() if path.is_dir()):
        summary, cache, gap = analyze_case(directory)
        summaries.extend(summary)
        cache_rows.extend(cache)
        gaps.extend(gap)
    write_csv(args.out_dir / "downstream_counter_comparison.csv", [row for row in summaries if row["operation"] == "tail"])
    write_csv(args.out_dir / "whole_graph_trace_comparison.csv", [row for row in summaries if row["operation"] == "full"])
    write_csv(args.out_dir / "cache_counter_comparison.csv", cache_rows)
    write_csv(args.out_dir / "rocprof" / "dispatch_gap_comparison.csv", gaps)


if __name__ == "__main__":
    main()
