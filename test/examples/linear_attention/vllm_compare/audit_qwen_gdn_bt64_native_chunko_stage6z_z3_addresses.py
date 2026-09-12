#!/usr/bin/env python3
"""Host-only address audit for the experimental Z3 WG128 chunk-o.

This does not launch a GPU kernel.  It expands the source-level index formulas
for the fixed Z3 contract and records the first out-of-bounds global access and
all shared phase ranges.  It is intentionally diagnostic and is useful before
any long-text GPU repro after a prior device fault.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BT = 64
BK = 32
H_K = 4
H_V = 8
V_DIM = 128
K_DIM = 128
WORKGROUP = 128


def _record(records: list[dict[str, object]], **values: object) -> None:
    records.append(dict(values))


def audit(t: int, *, include_buggy_phase_b: bool = False) -> dict[str, object]:
    if t < BT or t % BT:
        raise ValueError("T must be divisible by 64")
    num_chunks = t // BT
    programs = num_chunks * H_V * 2
    records: list[dict[str, object]] = []
    phase_ranges: list[dict[str, object]] = []
    first_global_oob: dict[str, object] | None = None

    def global_check(phase: str, tensor: str, token: int, extent: int, **extra: object) -> None:
        nonlocal first_global_oob
        valid = 0 <= token < extent
        item = {
            "phase": phase,
            "tensor": tensor,
            "token": token,
            "extent": extent,
            "valid": valid,
            **extra,
        }
        _record(records, **item)
        if not valid and first_global_oob is None:
            first_global_oob = item

    # The launch formula is pid -> [chunk, value head, value block].
    launch = {
        "programs": programs,
        "pid_min": 0,
        "pid_max": programs - 1,
        "last_mapping": {
            "pid": programs - 1,
            "chunk_idx": num_chunks - 1,
            "value_head_idx": H_V - 1,
            "v_block_idx": 1,
        },
    }

    # Phase A: Q/H writes.  Every WG128 thread/repetition maps to 64 rows x 32 K.
    for rep in range(16):
        for tid in (0, WORKGROUP - 1):
            idx = tid + rep * WORKGROUP
            row, col = divmod(idx, BK)
            phase_ranges.append({"phase": "A", "tensor": "phase", "row": row, "col": col})
            global_check("A", "q", row, t, row=row, col=col, rep=rep, tid=tid)
            global_check("A", "h", row, num_chunks * BT, row=row, col=col, rep=rep, tid=tid)

    # Phase B: Q is 64 rows, but the consumer-visible K plane is 32 rows.
    for source_half in range(2):
        for rep in range(16):
            for tid in (0, WORKGROUP - 1):
                idx = tid + rep * WORKGROUP
                row, col = divmod(idx, BK)
                phase_ranges.append({"phase": "B", "tensor": "phase", "row": source_half * 128 + row, "col": col})
                global_check("B", "q", row, t, source_half=source_half, row=row, col=col, rep=rep, tid=tid)
        repetitions = 16 if include_buggy_phase_b else 8
        for rep in range(repetitions):
            for tid in (0, WORKGROUP - 1):
                idx = tid + rep * WORKGROUP
                row, col = divmod(idx, BK)
                source_token = source_half * 32 + row
                global_check(
                    "B", "k", (num_chunks - 1) * BT + source_token, t,
                    source_half=source_half,
                    row=row,
                    col=col,
                    source_token=source_token,
                    rep=rep,
                    tid=tid,
                    buggy_formula=include_buggy_phase_b,
                )

    # Phase C: V-new writes map 64 value rows x 64 tokens into phase rows 128..255.
    for rep in range(32):
        for tid in (0, WORKGROUP - 1):
            idx = tid + rep * WORKGROUP
            value_offset, token_offset = divmod(idx, BT)
            phase_row = 128 + value_offset * 2 + token_offset // 32
            phase_ranges.append({"phase": "C", "tensor": "phase", "row": phase_row, "col": token_offset % 32})
            global_check("C", "v_new", token_offset, t, value_offset=value_offset, token_offset=token_offset, rep=rep, tid=tid)

    phase_rows = [int(item["row"]) for item in phase_ranges]
    phase_summary = {
        "min_row": min(phase_rows),
        "max_row": max(phase_rows),
        "row_extent": 256,
        "all_rows_in_bounds": min(phase_rows) >= 0 and max(phase_rows) < 256,
        "writes_checked": len(phase_ranges),
    }
    valid_records = sum(1 for item in records if item["valid"])
    return {
        "contract": {"T": t, "BT": BT, "BK": BK, "WG": WORKGROUP, "H_V": H_V, "K": K_DIM, "V": V_DIM},
        "launch": launch,
        "phase_summary": phase_summary,
        "global_checks": {"count": len(records), "valid": valid_records, "invalid": len(records) - valid_records},
        "first_global_oob": first_global_oob,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, action="append", required=True)
    parser.add_argument("--include-buggy-phase-b", action="store_true")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()
    payload = {
        "diagnostic_only": True,
        "formula": "Z3 Phase B K source_half*32 + row; fixed path has rep=8, historical path has rep=16",
        "cases": [audit(t, include_buggy_phase_b=args.include_buggy_phase_b) for t in args.T],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["T", "phase", "tensor", "token", "extent", "valid", "source_half", "row", "source_token", "buggy_formula"])
            writer.writeheader()
            for case in payload["cases"]:
                for row in case["records"]:
                    writer.writerow({"T": case["contract"]["T"], **{key: row.get(key) for key in writer.fieldnames if key != "T"}})
    print(json.dumps({"json_out": str(args.json_out), "cases": payload["cases"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
