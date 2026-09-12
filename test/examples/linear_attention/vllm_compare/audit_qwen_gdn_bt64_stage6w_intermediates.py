#!/usr/bin/env python3
"""Static Stage 6W global-intermediate accounting; no kernels are changed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BT = 64
H_V = 8
K = 128
V = 128


def bytes_for(shape: tuple[int, ...], dtype: str) -> int:
    element_size = {"bf16": 2, "fp32": 4}[dtype]
    count = 1
    for dim in shape:
        count *= dim
    return count * element_size


def rows_for_t(t: int) -> list[dict[str, object]]:
    chunks = t // BT
    specs = [
        ("g_cumsum", (1, t, H_V), "fp32", "cumsum", "KKT; W/U; recurrence; chunk-o", 4, False, "multiple consumers; no direct one-edge elimination"),
        ("a", (1, t, H_V, BT), "fp32", "KKT", "solve", 1, True, "largest source-native upstream one-use edge"),
        ("a_solved_bf16", (1, t, H_V, BT), "bf16", "solve", "fused W/U", 1, True, "same CTA geometry; smaller than a"),
        ("w_bf16", (1, t, H_V, K), "bf16", "fused W/U", "immutable recurrence HSACO", 1, False, "external recurrence ABI boundary"),
        ("u_bf16", (1, t, H_V, V), "bf16", "fused W/U", "immutable recurrence HSACO", 1, False, "external recurrence ABI boundary"),
        ("h_bf16", (1, chunks, H_V, V, K), "bf16", "immutable recurrence HSACO", "chunk-o", 1, False, "largest edge, but recurrence HSACO must change/fuse"),
        ("v_new_bf16", (1, t, H_V, V), "bf16", "immutable recurrence HSACO", "chunk-o", 1, False, "Stage 6W removed only its FP32 expansion; ABI producer remains external"),
        ("final_state", (1, H_V, V, K), "fp32", "immutable recurrence HSACO", "public optional output", 0, False, "public result, not an eliminable intermediate"),
        ("output_bf16", (1, t, H_V, V), "bf16", "chunk-o", "public output", 0, False, "Stage 6W direct final store; no final cast dispatch"),
    ]
    rows = []
    for name, shape, dtype, producer, consumer, uses, feasible, note in specs:
        size = bytes_for(shape, dtype)
        rows.append({
            "T": t,
            "chunks": chunks,
            "tensor": name,
            "dtype": dtype,
            "shape": "x".join(map(str, shape)),
            "bytes": size,
            "MiB": size / (1024 * 1024),
            "producer": producer,
            "consumer": consumer,
            "consumer_uses": uses,
            "single_use": uses == 1,
            "producer_already_writes_consumer_dtype_layout": True,
            "candidate_source_native_elimination": feasible,
            "one_write_plus_one_read_bytes": 2 * size if uses == 1 else size * (1 + uses),
            "note": note,
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if any(t < BT or t % BT for t in args.T):
        raise ValueError("Stage 6W accounting requires T divisible by 64.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for t in args.T for row in rows_for_t(t)]
    dispatch_graph = {
        "stage6w_dispatch_count": 6,
        "dispatches": [
            "cumsum -> g_cumsum FP32",
            "KKT -> a FP32",
            "solve -> a_solved BF16",
            "fused W/U -> w_bf16 + u_bf16",
            "immutable recurrence HSACO -> h_bf16 + v_new_bf16 + final_state",
            "chunk-o -> public output BF16",
        ],
        "removed_by_stage6w": ["v_new BF16-to-FP32 cast dispatch", "FP32 output-to-BF16 final cast dispatch"],
        "selected_next_single_experiment": {
            "edge": "KKT FP32 a -> BF16 solve",
            "reason": "At T=2048 it is a 4 MiB one-use global intermediate (8 MiB write+read traffic), and unlike W/U or recurrence outputs both endpoints are Avelang source kernels rather than immutable external-HSACO ABI boundaries.",
            "not_selected": "h/v_new and W/U have larger traffic but cross the immutable recurrence HSACO ABI; g_cumsum has four consumers; a_solved is feasible but only half the a traffic.",
        },
    }
    write_csv(args.out_dir / "stage6w_intermediate_accounting.csv", rows)
    (args.out_dir / "stage6w_dispatch_graph.json").write_text(json.dumps(dispatch_graph, indent=2) + "\n")
    print(json.dumps(dispatch_graph, indent=2))


if __name__ == "__main__":
    main()
