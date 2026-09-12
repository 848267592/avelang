#!/usr/bin/env python3
"""Normalize rocprof v0/context rows into the requested comparison CSV."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FULLSEQ = ROOT.parent / "codex_triton_fullseq_asm_opt_audit"
RESOURCE_FIELDS = ("Grid_Size", "Workgroup_Size", "LDS_Block_Size", "Scratch_Size", "VGPR_Count", "Accum_VGPR_Count", "SGPR_Count")


def _load_profile(directory: Path, kernel: str) -> dict[str, object] | None:
    traces = []
    counters: list[dict[str, str]] = []
    for trace_file in directory.glob("run/*kernel_trace*.csv"):
        for row in csv.DictReader(trace_file.open()):
            if kernel in row.get("Kernel_Name", ""):
                traces.append((int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1_000)
    for counter_file in directory.glob("run/*counter_collection*.csv"):
        counters.extend(row for row in csv.DictReader(counter_file.open()) if kernel in row.get("Kernel_Name", ""))
    if not counters:
        return None
    first = counters[0]
    result: dict[str, object] = {
        "trace_median_us": statistics.median(traces) if traces else None,
        **{field: first.get(field, "") for field in RESOURCE_FIELDS},
    }
    result.update({row["Counter_Name"]: row["Counter_Value"] for row in counters})
    return result


def main() -> None:
    fullseq = json.loads((FULLSEQ / "fullseq_resource_counters.json").read_text())
    rows: list[dict[str, object]] = []
    for implementation, profile_dir in (
        ("golden_triton_original_hsaco", "golden_original"),
        ("golden_triton_rebuilt_hsaco", "golden_rebuilt"),
    ):
        data = _load_profile(ROOT / "rocprof" / f"t2048_{profile_dir}", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64")
        if data is None:
            golden = fullseq.get("t2048_original", {})
            data = {"trace_median_us": golden.get("trace_median_us"), **golden.get("resources", {}), **golden.get("counters", {})}
            source = "existing_fullseq_rocprof_fallback"
        else:
            source = "current_same_harness_rocprof"
        rows.append({"implementation": implementation, "source": source, **data})
    profiles = (
        ("avelang_asm_v0", "asm_v0", "qwen_gdn_bt64_gfx942_asm_v0"),
        ("old_full_v29_context", "v29", "_qwen_gdn_fused_chunk_gdr_full_bf16_kernel_v29_mfma32"),
        ("v31_context", "v31", "_qwen_gdn_fused_chunk_gdr_full_bf16_kernel_v31_bt64_bv32_hierarchical_mfma16_pred"),
    )
    for name, profile_dir, kernel in profiles:
        data = _load_profile(ROOT / "rocprof" / f"t2048_{profile_dir}", kernel)
        if data is None:
            data = {"trace_median_us": "not_profiled"}
        rows.append({"implementation": name, "source": "current_asm_v0_audit", **data})
    fields = ["implementation", "source", "trace_median_us", *RESOURCE_FIELDS, "SQ_INSTS_MFMA", "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_VMEM", "SQ_INSTS_LDS", "OccupancyPercent"]
    with (ROOT / "resource_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "resource_comparison.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
