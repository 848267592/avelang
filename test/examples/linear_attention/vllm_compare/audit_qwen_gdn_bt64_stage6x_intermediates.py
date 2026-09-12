#!/usr/bin/env python3
"""Static Stage 6Y accounting for the frozen five-dispatch Stage 6X graph.

This tool performs no GPU work and changes no kernel. It records only the
post-X2 graph so an old Stage 6W six-dispatch accounting cannot accidentally
be used to choose the next experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BT = 64
H_V = 8
K_DIM = 128
V_DIM = 128


def tensor_bytes(shape: tuple[int, ...], dtype: str) -> int:
    bytes_per_element = {"bf16": 2, "fp32": 4}[dtype]
    elements = 1
    for extent in shape:
        elements *= extent
    return elements * bytes_per_element


def rows_for_t(t: int) -> list[dict[str, object]]:
    chunks = t // BT
    specs = (
        ("g_cumsum", (1, t, H_V), "fp32", "cumsum", "fused KKT+solve; fused W/U; recurrence; chunk-o", 4, False, "four consumers; not a one-edge handoff"),
        ("a_solved_bf16", (1, t, H_V, BT), "bf16", "fused KKT+solve", "fused W/U", 1, True, "largest remaining source-native one-use handoff"),
        ("w_bf16", (1, t, H_V, K_DIM), "bf16", "fused W/U", "immutable recurrence HSACO", 1, False, "crosses frozen recurrence ABI"),
        ("u_bf16", (1, t, H_V, V_DIM), "bf16", "fused W/U", "immutable recurrence HSACO", 1, False, "crosses frozen recurrence ABI"),
        ("h_bf16", (1, chunks, H_V, V_DIM, K_DIM), "bf16", "immutable recurrence HSACO", "chunk-o", 1, False, "largest edge; crosses frozen recurrence ABI"),
        ("v_new_bf16", (1, t, H_V, V_DIM), "bf16", "immutable recurrence HSACO", "chunk-o", 1, False, "crosses frozen recurrence ABI"),
        ("final_state", (1, H_V, V_DIM, K_DIM), "fp32", "immutable recurrence HSACO", "public optional output", 0, False, "public result, not eliminable"),
        ("output_bf16", (1, t, H_V, V_DIM), "bf16", "chunk-o", "public output", 0, False, "public result, already directly stored"),
    )
    rows: list[dict[str, object]] = []
    for name, shape, dtype, producer, consumer, consumer_uses, source_native, note in specs:
        size = tensor_bytes(shape, dtype)
        rows.append({
            "T": t,
            "chunks": chunks,
            "tensor": name,
            "dtype": dtype,
            "shape": "x".join(str(extent) for extent in shape),
            "bytes": size,
            "MiB": size / (1024 * 1024),
            "producer": producer,
            "consumer": consumer,
            "consumer_uses": consumer_uses,
            "single_use": consumer_uses == 1,
            "producer_already_writes_consumer_dtype_layout": True,
            "candidate_source_native_elimination": source_native,
            "one_write_plus_one_read_bytes": 2 * size if consumer_uses == 1 else size * (1 + consumer_uses),
            "note": note,
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if any(t < BT or t % BT for t in args.T):
        raise ValueError("Stage 6X accounting requires T >= 64 and divisible by 64.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for t in args.T for row in rows_for_t(t)]
    graph = {
        "stage": "Stage 6Y updated full-gap audit input",
        "frozen_baseline": "Stage 6X X2",
        "dispatch_count": 5,
        "dispatches": [
            "cumsum -> g_cumsum FP32",
            "fused KKT+solve -> a_solved BF16",
            "fused W/U -> w_bf16 + u_bf16",
            "immutable recurrence HSACO -> h_bf16 + v_new_bf16 + final_state",
            "chunk-o -> public output BF16",
        ],
        "removed_by_stage6x": [
            "KKT -> global FP32 a",
            "global FP32 a -> solve",
            "standalone solve dispatch",
        ],
        "selection_rule": "Do not select the next kernel change from traffic alone. First obtain current-X2 versus vLLM standalone body slopes and dispatch identity.",
        "static_candidates": [
            {
                "edge": "a_solved_bf16 -> fused W/U",
                "status": "largest remaining source-native single-use edge",
                "not_yet_selected": True,
                "reason": "fusing it would combine X2's 40 KiB LDS/AccVGPR=164 kernel with W/U; body/resource evidence is required first."
            },
            {
                "edge": "recurrence -> chunk-o (h_bf16 + v_new_bf16)",
                "status": "largest overall traffic",
                "not_yet_selected": True,
                "reason": "crosses immutable recurrence HSACO ABI; only a new recurrence design or explicit bridge change can remove it."
            }
        ]
    }
    write_csv(args.out_dir / "stage6x_intermediate_accounting.csv", rows)
    (args.out_dir / "stage6x_dispatch_graph.json").write_text(json.dumps(graph, indent=2) + "\n")
    print(json.dumps(graph, indent=2))


if __name__ == "__main__":
    main()
