#!/usr/bin/env python3
"""Generate deterministic local audit summaries from captured Triton/rocprof files."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KERNEL = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def matching_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return [row for row in csv.DictReader(stream) if KERNEL in row.get("Kernel_Name", "")]


def rocprof_summary(label: str, folder: str, prefix: str) -> dict[str, object]:
    trace = matching_rows(ROOT / folder / f"{prefix}_kernel_trace.csv")
    counters = matching_rows(ROOT / folder / f"{prefix}_counter_collection.csv")
    durations = [(int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1_000 for row in trace]
    if not trace or not counters:
        raise RuntimeError(f"no {KERNEL} rows in {folder}")
    first = trace[0]
    counter_values = {row["Counter_Name"]: float(row["Counter_Value"]) for row in counters}
    return {
        "label": label,
        "samples": len(durations),
        "trace_us": {"median": statistics.median(durations), "values": durations},
        "resources": {
            "grid_workitems": int(first["Grid_Size_X"]) * int(first["Grid_Size_Y"]) * int(first["Grid_Size_Z"]),
            "workgroup": int(first["Workgroup_Size_X"]) * int(first["Workgroup_Size_Y"]) * int(first["Workgroup_Size_Z"]),
            "lds_block_size": int(first["LDS_Block_Size"]),
            "scratch_size": int(first["Scratch_Size"]),
            "vgpr": int(first["VGPR_Count"]),
            "accvgpr": int(first["Accum_VGPR_Count"]),
            "sgpr": int(first["SGPR_Count"]),
        },
        "counters": counter_values,
    }


def main() -> None:
    files = sorted(path for path in (ROOT / "triton_cache_exact").rglob("*") if path.is_file())
    manifest = {
        "selected_cache_key": "JM5FXOJP4XDF5LGYXIQP2FVT7DVYQMES3VOVFQDZJT7WNOWXH2MQ",
        "selected_hsaco_sha256": sha256(ROOT / "extracted/original_triton_kernel.hsaco"),
        "files": [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files],
    }
    (ROOT / "cache_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "hashes.sha256").write_text("".join(f"{entry['sha256']}  {entry['path']}\n" for entry in manifest["files"]))
    (ROOT / "cache_inventory.txt").write_text("".join(f"{entry['path']}\t{entry['bytes']}\n" for entry in manifest["files"]))

    summaries = {
        "wrapper": rocprof_summary("vLLM Triton wrapper", "rocprof_wrapper", "wrapper"),
        "original": rocprof_summary("extracted original HSACO", "rocprof_original_hsaco", "original"),
        "rebuilt": rocprof_summary("reassembled compiler-stage AMDGCN", "rocprof_rebuilt_hsaco", "rebuilt"),
    }
    (ROOT / "rocprof_wrapper.json").write_text(json.dumps(summaries["wrapper"], indent=2) + "\n")
    (ROOT / "rocprof_standalone.json").write_text(json.dumps(summaries["original"], indent=2) + "\n")
    (ROOT / "resource_comparison.json").write_text(json.dumps(summaries, indent=2) + "\n")

    rows = [
        ("vLLM wrapper selected dispatch", summaries["wrapper"]["trace_us"]["median"]),
        ("extracted HSACO HIP module", summaries["original"]["trace_us"]["median"]),
        ("reassembled compiler-stage AMDGCN", summaries["rebuilt"]["trace_us"]["median"]),
    ]
    with (ROOT / "standalone_benchmark.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "median_device_kernel_trace_us"])
        writer.writerows(rows)
    wrapper_us = rows[0][1]
    original_delta = 100 * (rows[1][1] / wrapper_us - 1)
    rebuilt_delta = 100 * (rows[2][1] / rows[1][1] - 1)
    (ROOT / "performance_comparison.md").write_text(
        "# Device Kernel Trace Comparison\n\n"
        "These are rocprof kernel-trace medians for the same fixed dispatch, not Python wrapper wall time. "
        "The standalone C++ event loop has host submission jitter and is intentionally not used as the ABI/performance gate.\n\n"
        "| path | median trace us | delta |\n|:--|--:|--:|\n"
        f"| vLLM selected Triton dispatch | {wrapper_us:.3f} | baseline |\n"
        f"| extracted HSACO via HIP module | {rows[1][1]:.3f} | {original_delta:+.3f}% |\n"
        f"| rebuilt compiler-stage AMDGCN via HIP module | {rows[2][1]:.3f} | {rebuilt_delta:+.3f}% vs extracted |\n"
    )


if __name__ == "__main__":
    main()
