#!/usr/bin/env python3
"""Summarize only steady-state Stage5A rocprof dispatches.

Triton profiling processes also contain autotune trials.  For each target
kernel this script intentionally selects the last eight matching dispatches,
which are the eight direct, selected-config calls made by the harness.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROC = HERE / "rocprof"
TARGETS = [
    ("avelang_v18", 512, ROC / "avelang_v18_T512", "_qwen_gdn_solve_kernel_v18_parallel"),
    ("avelang_v18", 2048, ROC / "avelang_T2048", "_qwen_gdn_solve_kernel_v18_parallel"),
    ("avelang_v18", 8192, ROC / "avelang_v18_T8192", "_qwen_gdn_solve_kernel_v18_parallel"),
    ("vllm", 512, ROC / "vllm_selected_T512", "merge_16x16_to_64x64_inverse_kernel"),
    ("vllm", 2048, ROC / "vllm_selected_T2048", "merge_16x16_to_64x64_inverse_kernel"),
    ("vllm", 8192, ROC / "vllm_selected_T8192", "merge_16x16_to_64x64_inverse_kernel"),
]


def first(path: Path, pattern: str) -> Path:
    matches = list(path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"{pattern} under {path}")
    return matches[0]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    resources: list[dict[str, object]] = []
    counters: list[dict[str, object]] = []
    timeline: list[dict[str, object]] = []
    for implementation, tokens, directory, needle in TARGETS:
        trace_rows = [row for row in csv.DictReader(first(directory, "*kernel_trace.csv").open()) if needle in row["Kernel_Name"]]
        selected = trace_rows[-8:]
        ids = {row["Dispatch_Id"] for row in selected}
        durations = [(int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1000.0 for row in selected]
        exemplar = selected[-1]
        resources.append({
            "implementation": implementation,
            "T": tokens,
            "selected_dispatches": len(selected),
            "kernel": needle,
            "trace_median_us": statistics.median(durations),
            "trace_p10_us": sorted(durations)[(len(durations) - 1) // 10],
            "trace_p90_us": sorted(durations)[(len(durations) - 1) * 9 // 10],
            "workgroup": exemplar["Workgroup_Size_X"],
            "grid_x_global_work_items": exemplar["Grid_Size_X"],
            "grid_y": exemplar["Grid_Size_Y"],
            "lds_block_bytes": exemplar["LDS_Block_Size"],
            "scratch_bytes": exemplar["Scratch_Size"],
            "vgpr": exemplar["VGPR_Count"],
            "accvgpr": exemplar["Accum_VGPR_Count"],
            "sgpr": exemplar["SGPR_Count"],
        })
        for row in selected:
            timeline.append({"implementation": implementation, "T": tokens, "dispatch_id": row["Dispatch_Id"],
                             "duration_us": (int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1000.0,
                             "kernel": needle})
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in csv.DictReader(first(directory, "*counter_collection.csv").open()):
            if row["Dispatch_Id"] in ids:
                grouped[row["Counter_Name"]].append(float(row["Counter_Value"]))
        for name, values in sorted(grouped.items()):
            counters.append({"implementation": implementation, "T": tokens, "counter": name,
                             "selected_dispatches": len(selected), "median": statistics.median(values),
                             "minimum": min(values), "maximum": max(values)})
    write_csv(HERE / "resource_comparison.csv", resources)
    write_csv(HERE / "dynamic_counter_comparison.csv", counters)
    write_csv(HERE / "dispatch_timeline.csv", timeline)


if __name__ == "__main__":
    main()
