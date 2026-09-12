#!/usr/bin/env python3
"""Run and summarize Stage 6A rocprof full/body replay captures."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
TRACE = HERE / "stage6a_trace_replay.py"
ROCPROF = Path("/opt/rocm/bin/rocprofv3")
IMPL_A = "avelang_stage4_bt64_hierarchical_v1"
IMPL_B = "vllm_authoritative_bt64"
IMPLEMENTATIONS = (IMPL_A, IMPL_B)
STAGES = ("cumsum", "kkt", "solve", "wu", "recurrence", "chunk_o", "cast")
COUNTERS = ("SQ_INSTS_MFMA", "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_VMEM", "SQ_INSTS_LDS", "OccupancyPercent")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def command(target: dict[str, str], out: Path, args: argparse.Namespace) -> list[str]:
    return [
        str(ROCPROF), "--kernel-trace", "--pmc", *COUNTERS, "-d", str(out), "-o", "stage6a", "-f", "csv", "--",
        sys.executable, str(TRACE), "--scope", target["scope"], "--implementation", target["implementation"],
        "--T", str(args.T), "--replay", str(args.replay),
        *(["--stage", target["stage"]] if target.get("stage") else []),
    ]


def periodic_tail(names: list[str], replay: int) -> list[str]:
    """Find the final repeated replay sequence without adding marker kernels."""
    for width in range(1, len(names) // replay + 1):
        candidate = names[-width:]
        if all(names[-(index + 1) * width:-index * width if index else None] == candidate for index in range(replay)):
            return candidate
    return names[-min(len(names), 32):]


def summarize(target: dict[str, str], directory: Path, replay: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trace_path = directory / "stage6a_kernel_trace.csv"
    counter_path = directory / "stage6a_counter_collection.csv"
    trace = read_csv(trace_path)
    counters = read_csv(counter_path)
    ordered = sorted(trace, key=lambda row: int(row["Start_Timestamp"]))
    names = [row["Kernel_Name"] for row in ordered]
    tail = periodic_tail(names, replay)
    graph_rows = [{"scope": target["scope"], "stage": target.get("stage", "total"), "implementation": target["implementation"],
                   "ordinal": ordinal, "kernel_name": name} for ordinal, name in enumerate(tail)]

    recent_ids_by_kernel: dict[str, set[int]] = defaultdict(set)
    for row in ordered[-max(len(tail) * replay, 1):]:
        recent_ids_by_kernel[row["Kernel_Name"]].add(int(row["Dispatch_Id"]))
    by_kernel_counter: dict[tuple[str, str], list[float]] = defaultdict(list)
    metadata: dict[str, dict[str, str]] = {}
    for row in counters:
        kernel = row["Kernel_Name"]
        if int(row["Dispatch_Id"]) not in recent_ids_by_kernel.get(kernel, set()):
            continue
        by_kernel_counter[(kernel, row["Counter_Name"])].append(float(row["Counter_Value"]))
        metadata[kernel] = row
    counter_rows = []
    for kernel, meta in metadata.items():
        durations = [(int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1000.0 for row in ordered if row["Kernel_Name"] == kernel][-replay:]
        item: dict[str, object] = {
            "scope": target["scope"], "stage": target.get("stage", "total"), "implementation": target["implementation"],
            "kernel_name": kernel, "trace_median_us": statistics.median(durations) if durations else None,
            "workgroup": meta.get("Workgroup_Size"), "grid_work_items": meta.get("Grid_Size"),
            "lds_block_bytes": meta.get("LDS_Block_Size"), "scratch_bytes": meta.get("Scratch_Size"),
            "vgpr": meta.get("VGPR_Count"), "accvgpr": meta.get("Accum_VGPR_Count"), "sgpr": meta.get("SGPR_Count"),
        }
        for counter in COUNTERS:
            values = by_kernel_counter.get((kernel, counter), [])
            item[counter] = statistics.median(values) if values else None
        counter_rows.append(item)
    return graph_rows, counter_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--replay", type=int, default=3)
    parser.add_argument("--scope", choices=("full", "bodies", "all"), default="all")
    parser.add_argument("--out-dir", type=Path, default=HERE / "rocprof")
    args = parser.parse_args()
    targets = [{"scope": "full", "implementation": implementation} for implementation in IMPLEMENTATIONS]
    if args.scope == "bodies":
        targets = []
    if args.scope in ("bodies", "all"):
        targets.extend({"scope": "body", "implementation": implementation, "stage": stage}
                       for implementation in IMPLEMENTATIONS for stage in STAGES
                       if not (implementation == IMPL_B and stage == "cast"))
    graph_rows: list[dict[str, object]] = []
    counter_rows: list[dict[str, object]] = []
    manifest = []
    for target in targets:
        label = "_".join(value for value in (target["scope"], target["implementation"], target.get("stage")) if value)
        destination = args.out_dir / label
        destination.mkdir(parents=True, exist_ok=True)
        cmd = command(target, destination, args)
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        graph, counters = summarize(target, destination, args.replay)
        graph_rows.extend(graph)
        counter_rows.extend(counters)
        manifest.append({**target, "directory": str(destination), "command": cmd})
    write_csv(args.out_dir / "actual_dispatch_graph.csv", graph_rows)
    write_csv(args.out_dir / "kernel_counters.csv", counter_rows)
    (args.out_dir / "profile_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
