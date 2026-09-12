#!/usr/bin/env python3
"""Low-perturbation, audit-only Stage 5F measurement driver.

This file deliberately imports the frozen Stage 5E direct-common-out graph.
It neither imports production paths nor changes the graph: the only A/B delta
is the already-audited solve symbol/workgroup.  HIP events surround one whole
tail or one whole graph, never an internal stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
STAGE5E = HERE.parent / "codex_qwen_bt64_direct_common_out_stage5e"
sys.path.insert(0, str(STAGE5E))

from direct_common_out_harness import (  # noqa: E402
    SOLVE_A,
    SOLVE_B,
    Stage5ERunner,
    benchmark_operation,
    patch_rocm_autotune,
    setup_runner,
    write_json,
)


def _normalise(dispatches: list[dict[str, object]]) -> list[dict[str, object]]:
    """Remove the sole declared A/B difference for a structural comparison."""
    result = []
    for item in dispatches:
        copied = dict(item)
        if copied.get("stage") == "solve":
            copied["symbol"] = "<solve-symbol>"
            copied["workgroup"] = "<solve-workgroup>"
        result.append(copied)
    return result


def write_frozen_contract(out_dir: Path, t: int, seed: int) -> None:
    runner = setup_runner(t, seed)
    graph = runner.graph_contract()
    graph_a = {
        "name": "GRAPH-A",
        "solve_impl": SOLVE_A,
        "common_data_ptr": int(runner.solved_common.data_ptr()),
        "dispatches": graph[SOLVE_A]["dispatches"],
        "allocation_between_solve_and_wu": False,
        "copy_between_solve_and_wu": False,
        "fill_between_solve_and_wu": False,
    }
    graph_b = {
        "name": "GRAPH-B",
        "solve_impl": SOLVE_B,
        "common_data_ptr": int(runner.solved_common.data_ptr()),
        "dispatches": graph[SOLVE_B]["dispatches"],
        "allocation_between_solve_and_wu": False,
        "copy_between_solve_and_wu": False,
        "fill_between_solve_and_wu": False,
    }
    same_downstream = _normalise(graph_a["dispatches"]) == _normalise(graph_b["dispatches"])
    solve_a = graph_a["dispatches"][2]
    solve_b = graph_b["dispatches"][2]
    if not same_downstream:
        raise RuntimeError("frozen Stage 5E graphs differ beyond solve symbol/workgroup")
    write_json(out_dir / "graph_a.json", graph_a)
    write_json(out_dir / "graph_b.json", graph_b)
    (out_dir / "graph_diff.md").write_text(
        "# Frozen Graph Difference\n\n"
        "The structural comparison after replacing the solve symbol/workgroup is `true`.\n\n"
        f"- GRAPH-A solve: `{solve_a['symbol']}`, WG `{solve_a['workgroup']}`\n"
        f"- GRAPH-B solve: `{solve_b['symbol']}`, WG `{solve_b['workgroup']}`\n"
        f"- exact common output pointer for this audit process: `0x{int(runner.solved_common.data_ptr()):x}`\n"
        "- no copy, fill, or allocation occurs between solve and W/U.\n"
    )
    (out_dir / "frozen_graph_contract.md").write_text(
        "# Stage 5F Frozen Direct-Common-Out Graph\n\n"
        "This audit reuses the Stage 5E graph in one process and one stream. Both paths use the "
        "same preallocated solve-output tensor. There are no internal stage events, allocations, "
        "copies, fills, dummy kernels, or production-path changes.\n\n"
        "```text\n"
        "cumsum -> KKT -> solve_direct_common_out -> W -> U -> asm-v0 -> chunk-o -> cast\n"
        "```\n\n"
        "The sole A/B difference is the solve implementation and its fixed workgroup. See "
        "`graph_a.json`, `graph_b.json`, and `graph_diff.md`.\n"
    )


def calibrate(args: argparse.Namespace) -> None:
    # `benchmark_operation` is the Stage 5E ABBA/order-balanced HIP-event timer.
    # It creates all inputs and the fixed common solve-output allocation outside events.
    all_summaries = []
    all_raw = []
    pointers = []
    for operation in args.operation:
        summary, raw, pointer_rows = benchmark_operation(
            operation=operation,
            t_values=tuple(args.T),
            args=args,
            output_summary=f"{args.label}_{operation}_summary.csv",
            output_raw=f"{args.label}_{operation}_raw.csv",
        )
        all_summaries.extend(summary)
        all_raw.extend(raw)
        pointers.extend(pointer_rows)
    metadata = {
        "label": args.label,
        "T": args.T,
        "operations": args.operation,
        "sessions": args.sessions,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "timer": "HIP event around one whole tail/full graph only",
        "internal_stage_events": False,
        "same_process_per_ab_pair": True,
        "same_common_pointer_per_session": True,
        "summary_rows": len(all_summaries),
        "raw_rows": len(all_raw),
        "pointer_rows": len(pointers),
    }
    write_json(args.out_dir / f"{args.label}_measurement_metadata.json", metadata)
    print(json.dumps(metadata, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contract", "calibrate"), required=True)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--label", default="hip_event")
    parser.add_argument("--T", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--operation", choices=("tail", "full"), nargs="+", default=["tail", "full"])
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=200)
    # Required by the shared Stage 5E benchmark API but unused for this no-control audit.
    parser.add_argument("--flush-mib", type=int, default=512)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    patch_rocm_autotune()
    if args.mode == "contract":
        write_frozen_contract(args.out_dir, args.T[0], args.seed)
    else:
        calibrate(args)


if __name__ == "__main__":
    main()
