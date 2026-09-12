#!/usr/bin/env python3
"""Summarize Stage 3 rocprof traces and HIP-event CSV sessions without guessing."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
STAGE2 = HERE.parent / "codex_qwen_bt64_full_pipeline_stage2"
COUNTERS = ("SQ_INSTS_MFMA", "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_VMEM", "SQ_INSTS_LDS", "OccupancyPercent")
NATIVE = {
    "w": (HERE / "native_wu/rocprof/wu_w", "_qwen_gdn_w_bf16_kernel_bt64_from_v24_mfma_v1"),
    "u": (HERE / "native_wu/rocprof/wu_u", "_qwen_gdn_u_bf16_kernel_bt64_from_v24_mfma_v1"),
    "chunk_o": (HERE / "native_chunko/rocprof/chunk_o", "_qwen_gdn_chunk_o_bf16_kernel_bt64_from_v24_mfma_v1"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def as_number(value: str) -> int | float:
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def profile(directory: Path, kernel: str) -> dict[str, object]:
    trace_path = next(directory.glob("*kernel_trace*.csv"))
    counters_path = next(directory.glob("*counter_collection*.csv"))
    traces = [row for row in read_csv(trace_path) if row["Kernel_Name"] == kernel]
    if not traces:
        raise RuntimeError(f"no matching traces for {kernel} in {trace_path}")
    tail = traces[-5:]
    dispatch_ids = {row["Dispatch_Id"] for row in tail}
    durations = [(int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1000.0 for row in tail]
    counters: dict[str, list[float]] = defaultdict(list)
    for row in read_csv(counters_path):
        if row["Kernel_Name"] == kernel and row["Dispatch_Id"] in dispatch_ids:
            counters[row["Counter_Name"]].append(float(row["Counter_Value"]))
    meta = tail[-1]
    workgroup = int(meta.get("Workgroup_Size", meta.get("Workgroup_Size_X", "0")))
    grid = int(meta.get("Grid_Size", "0"))
    if not grid:
        grid = (
            int(meta.get("Grid_Size_X", "1"))
            * int(meta.get("Grid_Size_Y", "1"))
            * int(meta.get("Grid_Size_Z", "1"))
        )
    result: dict[str, object] = {
        "kernel": kernel,
        "trace_median_us_last_5": statistics.median(durations),
        "trace_us_last_5": durations,
        "workgroup": workgroup,
        "grid_work_items": grid,
        "lds_block_bytes": int(meta["LDS_Block_Size"]),
        "scratch_bytes": int(meta["Scratch_Size"]),
        "vgpr": int(meta["VGPR_Count"]),
        "accvgpr": int(meta["Accum_VGPR_Count"]),
        "sgpr": int(meta["SGPR_Count"]),
    }
    for counter in COUNTERS:
        values = counters.get(counter)
        if not values:
            raise RuntimeError(f"missing {counter} for {kernel}")
        result[counter] = statistics.median(values)
    return result


def session_summary() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for path in sorted(HERE.glob("full_pipeline_benchmark_*.csv")):
        for row in read_csv(path):
            if row["scope"] == "full":
                grouped[(int(row["T"]), row["implementation"])].append(float(row["median_ms"]))
    for (t, implementation), values in sorted(grouped.items()):
        rows.append({
            "T": t,
            "implementation": implementation,
            "session_count": len(values),
            "median_of_session_medians_ms": statistics.median(values),
            "min_session_median_ms": min(values),
            "max_session_median_ms": max(values),
            "session_medians_ms": values,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    resources = {name: profile(directory, kernel) for name, (directory, kernel) in NATIVE.items()}
    stage2 = json.loads((STAGE2 / "resource_profile.json").read_text())
    result = {"native": resources, "stage2_fallback": {"w_u": stage2["w_u"], "chunk_o": stage2["chunk_o"]}, "full_sessions": session_summary()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "resource_profile.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = ["stage", "kernel", "trace_median_us_last_5", "workgroup", "grid_work_items", "lds_block_bytes", "scratch_bytes", "vgpr", "accvgpr", "sgpr", *COUNTERS]
    with (args.out / "resource_profile.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for stage, record in resources.items():
            writer.writerow({"stage": stage, **record})
    with (args.out / "full_pipeline_benchmark.csv").open("w", newline="") as stream:
        fields = ["T", "implementation", "session_count", "median_of_session_medians_ms", "min_session_median_ms", "max_session_median_ms", "session_medians_ms"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["full_sessions"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
