#!/usr/bin/env python3
"""Mechanically verify the captured Triton K64 packet/rotating-LDS map.

This is an audit-only companion for Direct-K64 S0-C.  It evaluates the exact
layout bases from the captured TTGIR and the concrete K0 LDS store/load
address expressions from its LLVM IR.  It does not compile or launch a kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BT = 64
K64 = 64
WORKGROUP = 128
BF16_BYTES = 2

REPO_ROOT = Path(__file__).resolve().parents[5]
TRITON_DIR = (
    REPO_ROOT
    / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
    / "codex_qwen_bt64_recurrence_reconciliation_stage6r/current_kernels/vllm"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_pipeline_stage_s0c/audit"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_lane(token: int, feature: int) -> int:
    """TTGIR #linear1 inverse: lane basis plus warp basis."""

    return ((feature >> 5) << 6) | (((feature & 31) >> 2) << 3) | (token >> 3)


def native_register(token: int, feature: int) -> int:
    """TTGIR #linear1 inverse: register bases [col bits, row bits]."""

    return (feature & 3) | ((token & 7) << 2)


def native_lds_byte_address(lane: int, token_low: int, feature_low: int) -> int:
    """LLVM %513..%520 plus the eight packet-store offsets.

    This is the K0 `#shared1` formula at `kernel.llir:608..673`.  The packet
    store is `<4 x bf16>`; its fourth argument is the feature-low lane.
    """

    base = (
        ((0 if (lane & 1) == 0 else 1088) | ((lane & 6) << 2))
        ^ (lane & 120)
    ) | ((lane & 6) << 10)
    return (base ^ (136 * token_low)) + BF16_BYTES * feature_low


def consumer_base(lane: int) -> int:
    """LLVM %2749..%2755 for the K0 steady-state local load."""

    return (
        (2056 if (lane & 16) else 0)
        ^ ((4112 if (lane & 64) else 0) | ((lane & 32) >> 2))
        ^ ((lane & 15) << 3)
    ) | ((lane & 15) << 7)


def s0_packet(token: int, feature: int) -> dict[str, int]:
    """Current S0 `emitPacketLoads` ownership for a semantic K[token, feature]."""

    return {
        "lane": 8 * (token & 15) + (feature // 8),
        "packet": token // 16,
        "bf16_index": feature & 7,
    }


def build_mapping() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_address: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    # TTGIR #blocked -> amdg.in_thread_transpose -> #linear1:
    # pre-transpose packet = feature & 3 and element = token & 7.  The
    # transpose produces register = packet + 4 * element.  LLVM then writes
    # one four-BF16 packet per token-low value.
    for token in range(BT):
        for feature in range(K64):
            lane = native_lane(token, feature)
            reg = native_register(token, feature)
            load_packet = feature & 3
            packet_element = token & 7
            assert reg == load_packet + 4 * packet_element

            address = native_lds_byte_address(lane, packet_element, feature & 3)
            if address in by_address:
                raise AssertionError(f"duplicate native LDS byte {address}")

            row: dict[str, Any] = {
                "source": {"token": token, "k": feature},
                "native": {
                    "lane": lane,
                    "wave": lane // 64,
                    "lane_in_wave": lane & 63,
                    "load_packet": load_packet,
                    "load_packet_bf16_index": packet_element,
                    "post_in_thread_transpose_register": reg,
                    "repack_packet": packet_element,
                    "repack_bf16_lane": feature & 3,
                    "lds_byte_address": address,
                },
                "s0_current_source_packet": s0_packet(token, feature),
            }
            by_address[address] = row
            rows.append(row)

    expected_addresses = set(range(0, BT * K64 * BF16_BYTES, BF16_BYTES))
    if set(by_address) != expected_addresses:
        missing = sorted(expected_addresses - set(by_address))
        extra = sorted(set(by_address) - expected_addresses)
        raise AssertionError(f"native LDS coverage mismatch missing={missing[:8]} extra={extra[:8]}")

    reads: list[dict[str, int]] = []
    for lane in range(WORKGROUP):
        base = consumer_base(lane)
        for mfma_k8 in range(8):
            packet_address = base ^ (16 * mfma_k8)
            for word in range(4):
                address = packet_address + BF16_BYTES * word
                row = by_address.get(address)
                if row is None:
                    raise AssertionError(
                        f"consumer load points outside native K64 LDS: lane={lane} "
                        f"mfma_k8={mfma_k8} word={word} address={address}"
                    )
                consumer = {
                    "lane": lane,
                    "wave": lane // 64,
                    "lane_in_wave": lane & 63,
                    "mfma_k8_index": mfma_k8,
                    "mfma_bf16_word": word,
                    "lds_byte_address": address,
                }
                row["consumer"] = consumer
                reads.append(consumer)

    if len(reads) != BT * K64:
        raise AssertionError(f"expected 4096 K consumer words, saw {len(reads)}")
    if any("consumer" not in row for row in rows):
        raise AssertionError("some source elements have no MFMA B consumer")

    # A native BF16x8 load packet is a fixed K feature across eight consecutive
    # tokens.  The current S0 packet is a fixed token across eight K features.
    # This proves the required retile spans eight S0 lanes, not just registers
    # in one lane.
    native_packet_sources: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        native = row["native"]
        native_packet_sources[(native["lane"], native["load_packet"])].append(row)
    cross_lane_widths: Counter[int] = Counter()
    for source_rows in native_packet_sources.values():
        lanes = {item["s0_current_source_packet"]["lane"] for item in source_rows}
        if len(source_rows) != 8:
            raise AssertionError("native BF16x8 packet did not contain exactly eight values")
        cross_lane_widths[len(lanes)] += 1

    summary = {
        "tile": {"tokens": BT, "k": K64, "bf16_elements": BT * K64},
        "native_mapping": {
            "ttgir_linear1": {
                "register_bases": [[0, 1], [0, 2], [1, 0], [2, 0], [4, 0]],
                "lane_bases": [[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]],
                "warp_bases": [[0, 32]],
            },
            "lds_coverage_elements": len(by_address),
            "lds_min_byte": min(by_address),
            "lds_max_byte": max(by_address),
            "consumer_words": len(reads),
            "consumer_unique_source_elements": len(
                {(row["source"]["token"], row["source"]["k"]) for row in rows}
            ),
            "inverse_map_exact": True,
        },
        "s0_source_packet_incompatibility": {
            "s0_packet_shape": "one token x eight consecutive K values",
            "native_packet_shape": "one K value x eight consecutive tokens",
            "s0_lanes_per_native_bf16x8_packet_histogram": dict(sorted(cross_lane_widths.items())),
            "requires_cross_lane_retile": all(width > 1 for width in cross_lane_widths),
            "legal_under_s0c_constraints": False,
            "reason": (
                "The current S0 early BF16x8 packets and the recovered native packets "
                "have orthogonal axes. Every native BF16x8 packet draws from eight "
                "distinct current-S0 lanes. A lane-local transpose/repack cannot create "
                "it; doing so requires a cross-lane exchange or a changed LDS consumer "
                "layout, both explicitly forbidden by S0-C."
            ),
        },
    }
    return rows, summary


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    first = rows[:8]
    sample = "\n".join(
        "| {token} | {k} | {lane} | {packet}:{element} | {register} | {address} | "
        "{consumer_lane}:{mfma}:{word} | {s0_lane}:{s0_packet}:{s0_element} |".format(
            token=row["source"]["token"],
            k=row["source"]["k"],
            lane=row["native"]["lane"],
            packet=row["native"]["load_packet"],
            element=row["native"]["load_packet_bf16_index"],
            register=row["native"]["post_in_thread_transpose_register"],
            address=row["native"]["lds_byte_address"],
            consumer_lane=row["consumer"]["lane"],
            mfma=row["consumer"]["mfma_k8_index"],
            word=row["consumer"]["mfma_bf16_word"],
            s0_lane=row["s0_current_source_packet"]["lane"],
            s0_packet=row["s0_current_source_packet"]["packet"],
            s0_element=row["s0_current_source_packet"]["bf16_index"],
        )
        for row in first
    )
    path.write_text(
        "# Triton K64 Packet Commit Mapping (S0-C)\n\n"
        "此文件由 `audit_qwen_k64_triton_packet_commit_mapping_s0c.py` 从捕获的\n"
        "Stage-6R TTGIR 与 LLVM 地址表达式机械生成。\n\n"
        "## Exactness\n\n"
        f"- K tile: `{summary['tile']['tokens']} x {summary['tile']['k']}` BF16, "
        f"`{summary['tile']['bf16_elements']}` elements.\n"
        f"- Native rotating-LDS store covers `{summary['native_mapping']['lds_coverage_elements']}` "
        f"distinct BF16 byte locations from `{summary['native_mapping']['lds_min_byte']}` to "
        f"`{summary['native_mapping']['lds_max_byte']}`.\n"
        f"- The captured K0 consumer reads `{summary['native_mapping']['consumer_words']}` BF16 words; "
        "each resolves to exactly one source element, and the inverse map is exact.\n\n"
        "The producer map comes from `#linear1` in TTGIR and `kernel.llir:608..725`; "
        "the consumer map comes from `kernel.llir:3237..3266`.\n\n"
        "| source token | source K | native lane | native load packet:index | post-transpose reg | LDS byte | MFMA consumer lane:k8:word | current-S0 lane:packet:index |\n"
        "|---:|---:|---:|:---|---:|---:|:---|:---|\n"
        f"{sample}\n\n"
        "## S0-C feasibility result\n\n"
        "The native packet has one fixed K value across eight consecutive tokens. "
        "The current S0 early packet has one fixed token across eight consecutive K values. "
        "For every recovered native BF16x8 packet, its eight values belong to eight distinct "
        "current-S0 lanes. Therefore a lane-local `in_thread_transpose` or register repack cannot "
        "produce the native packet. It would require cross-lane exchange, or changing the LDS/consumer "
        "layout. Both are forbidden by the S0-C contract.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    ttgir = TRITON_DIR / "kernel.ttgir"
    llir = TRITON_DIR / "kernel.llir"
    if not ttgir.exists() or not llir.exists():
        raise FileNotFoundError(f"missing captured Triton artifacts under {TRITON_DIR}")

    rows, summary = build_mapping()
    payload = {
        "schema": "qwen-k64-triton-packet-commit-mapping-v1",
        "artifacts": {
            "ttgir": str(ttgir.relative_to(REPO_ROOT)),
            "ttgir_sha256": sha256(ttgir),
            "llvm": str(llir.relative_to(REPO_ROOT)),
            "llvm_sha256": sha256(llir),
        },
        "summary": summary,
        "elements": rows,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "qwen_k64_triton_packet_commit_mapping.json"
    md_path = args.out_dir / "qwen_k64_triton_packet_commit_mapping.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, summary, rows)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
