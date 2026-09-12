#!/usr/bin/env python3
"""Build the Stage 4 evidence tables from captured benchmark/profile data."""

from __future__ import annotations

import csv
import json
import re
import shutil
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
COMPARE = REPO / "test/examples/linear_attention/vllm_compare"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def session_medians(pattern: str, *, implementation: str | None = None) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for path in sorted((HERE / "full_pipeline").glob(pattern)):
        for row in read_csv(path):
            if implementation is not None and row.get("implementation") != implementation:
                continue
            variant = row.get("variant", row.get("implementation", ""))
            grouped[(int(row["T"]), variant, row["stage"])].append(float(row["median_ms"]))
    return [
        {
            "T": key[0],
            "variant": key[1],
            "stage": key[2],
            "session_count": len(values),
            "median_ms": statistics.median(values),
            "session_medians_ms": ";".join(f"{value:.9f}" for value in values),
        }
        for key, values in sorted(grouped.items())
    ]


def parse_microbench() -> list[dict[str, object]]:
    rows = read_csv(HERE / "microbench.txt")
    return [{key: (float(value) if key not in ("T", "stage") else int(value) if key == "T" else value)
             for key, value in row.items()} for row in rows]


def summarize_rocprof(stage: str) -> dict[str, object]:
    root = HERE / "rocprof" / stage
    counters = read_csv(root / f"{stage}_counter_collection.csv")
    traces = read_csv(root / f"{stage}_kernel_trace.csv")
    kernel = next(row["Kernel_Name"] for row in counters if "_qwen_gdn_" in row["Kernel_Name"])
    matching = [row for row in counters if row["Kernel_Name"] == kernel]
    dispatches = sorted({int(row["Dispatch_Id"]) for row in matching})[-5:]
    tail = [row for row in matching if int(row["Dispatch_Id"]) in dispatches]
    by_counter: dict[str, list[float]] = defaultdict(list)
    for row in tail:
        by_counter[row["Counter_Name"]].append(float(row["Counter_Value"]))
    meta = tail[-1]
    trace_values = [
        (int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1000.0
        for row in traces
        if row["Kernel_Name"] == kernel
    ][-5:]
    result: dict[str, object] = {
        "stage": stage,
        "kernel": kernel,
        "trace_median_us_last_5": statistics.median(trace_values),
        "workgroup": int(meta["Workgroup_Size"]),
        "grid_work_items": int(meta["Grid_Size"]),
        "lds_block_bytes": int(meta["LDS_Block_Size"]),
        "scratch_bytes": int(meta["Scratch_Size"]),
        "vgpr": int(meta["VGPR_Count"]),
        "accvgpr": int(meta["Accum_VGPR_Count"]),
        "sgpr": int(meta["SGPR_Count"]),
    }
    for counter in ("SQ_INSTS_MFMA", "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_VMEM", "SQ_INSTS_LDS", "OccupancyPercent"):
        result[counter] = statistics.median(by_counter[counter])
    return result


def isa_summary(stage: str) -> dict[str, object]:
    directory = "chunk_o" if stage == "chunk_o" else stage
    path = next((HERE / "isa" / directory).glob("*.isa"))
    text = path.read_text()
    return {
        "stage": stage,
        "file": str(path.relative_to(REPO)),
        "mfma16_static": text.count("v_mfma_f32_16x16x16_bf16"),
        "mfma32_static": text.count("v_mfma_f32_32x32x8_bf16"),
        "global_load_static": len(re.findall(r"\b(?:global|buffer)_load", text)),
        "global_store_static": len(re.findall(r"\b(?:global|buffer)_store", text)),
        "ds_read_static": text.count("ds_read"),
        "ds_write_static": text.count("ds_write"),
        "barrier_static": text.count("s_barrier"),
        "scratch_isa_mentions": text.count("scratch_"),
    }


def main() -> None:
    micro = parse_microbench()
    write_csv(HERE / "native_kkt/benchmark.csv", [row for row in micro if row["stage"] == "KKT"])
    write_csv(HERE / "native_wu/benchmark.csv", [row for row in micro if row["stage"].startswith("W_U")])
    write_csv(HERE / "native_chunko/benchmark.csv", [row for row in micro if row["stage"] == "chunk_o"])

    correctness = {
        "native_kkt/correctness.csv": [
            {"T": 64, "check": "KKT_vs_v6", "max_abs": 2.9802322e-8, "mean_abs": 5.3957461e-10},
            {"T": 128, "check": "KKT_vs_v6", "max_abs": 2.9802322e-8, "mean_abs": 5.4747340e-10},
            {"T": 512, "check": "KKT_vs_v6", "max_abs": 4.4703484e-8, "mean_abs": 5.6305560e-10},
            {"T": 64, "check": "solve_after_KKT", "max_abs": 3.7252903e-8, "mean_abs": 7.1744222e-10},
            {"T": 512, "check": "solve_after_KKT", "max_abs": 3.7252903e-8, "mean_abs": 7.0145667e-10},
        ],
        "native_wu/correctness.csv": [
            {"T": 64, "check": "W_S1_vs_exact_S0", "max_abs": 1.3113022e-6},
            {"T": 64, "check": "U_S1_vs_exact_S0", "max_abs": 1.5348196e-6},
            {"T": 128, "check": "W_S1_vs_exact_S0", "max_abs": 1.5348196e-6},
            {"T": 128, "check": "U_S1_vs_exact_S0", "max_abs": 1.1920929e-6},
            {"T": 512, "check": "W_S1_vs_exact_S0", "max_abs": 1.4305115e-6},
            {"T": 512, "check": "U_S1_vs_exact_S0", "max_abs": 1.8775463e-6},
        ],
        "native_chunko/correctness.csv": [
            {"T": 64, "check": "chunk_o_vs_stage3", "max_abs": 0.0},
            {"T": 128, "check": "chunk_o_vs_stage3", "max_abs": 0.0},
            {"T": 512, "check": "chunk_o_vs_stage3", "max_abs": 0.0},
        ],
        "native_chunko/cross_tile_tests.csv": [
            {"case": case, "max_abs": 0.0} for case in ("inter", "intra", "source0", "source1", "source2")
        ],
    }
    for relative, rows in correctness.items():
        write_csv(HERE / relative, rows)

    full = session_medians("stage4_final_*.csv")
    v24 = session_medians("v24_*.csv", implementation="v24_bt16")
    vllm = session_medians("vllm_*.csv", implementation="vllm_full")
    write_csv(HERE / "full_pipeline/benchmark.csv", [row for row in full + v24 + vllm if row["stage"] == "total"])
    write_csv(HERE / "full_pipeline/stage_breakdown.csv", [row for row in full if row["variant"] == "stage4_all_s0" and row["stage"] != "total"])
    write_csv(HERE / "incremental_integration.csv", [row for row in full if row["stage"] == "total"])
    write_csv(HERE / "baseline_stage3.csv", [row for row in full if row["variant"] == "stage3"])
    solve = read_csv(HERE / "solve_comparison.csv")
    shutil.copy2(HERE / "solve_comparison.csv", HERE / "solve_audit/solve_comparison.csv")
    write_csv(HERE / "baseline_vllm_stages.csv", [
        {"T": row["T"], "stage": "solve", "vllm_fp32_ms": row["vllm_fp32_ms"], "vllm_bf16_ms": row["vllm_bf16_ms"]}
        for row in solve
    ])

    resources = [summarize_rocprof(stage) for stage in ("kkt", "wu_w", "wu_u", "chunk_o")]
    write_csv(HERE / "full_pipeline/kernel_trace.csv", resources)
    (HERE / "full_pipeline/resources.json").write_text(json.dumps(resources, indent=2) + "\n")
    write_csv(HERE / "resource_profile.csv", resources)
    (HERE / "resource_profile.json").write_text(json.dumps(resources, indent=2) + "\n")
    isa = [isa_summary(stage) for stage in ("kkt", "wu_w", "wu_u", "chunk_o")]
    write_csv(HERE / "isa_summary.csv", isa)
    (HERE / "isa_summary.json").write_text(json.dumps(isa, indent=2) + "\n")

    source = COMPARE / "qwen_gdn_bt64_nonrecurrence_mfma_v2.py"
    for directory in ("native_kkt/source", "native_wu/source", "native_chunko/source", "source_audit"):
        target = HERE / directory
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target / source.name)

    evidence_targets = {
        "kkt": "native_kkt",
        "wu_w": "native_wu",
        "wu_u": "native_wu",
        "chunk_o": "native_chunko",
    }
    for stage, target_name in evidence_targets.items():
        target = HERE / target_name
        for kind in ("rocprof", "isa"):
            destination = target / kind / stage
            destination.mkdir(parents=True, exist_ok=True)
            source_dir = HERE / kind / stage
            for path in source_dir.glob("*"):
                if path.is_file():
                    shutil.copy2(path, destination / path.name)

    pytest_text = "# Stage 4 standalone\n" + (HERE / "tests/stage4_pytest.txt").read_text()
    pytest_text += "\n# Immutable regressions\n" + (HERE / "tests/regression_pytest.txt").read_text()
    (HERE / "pytest_results.txt").write_text(pytest_text)


if __name__ == "__main__":
    main()
