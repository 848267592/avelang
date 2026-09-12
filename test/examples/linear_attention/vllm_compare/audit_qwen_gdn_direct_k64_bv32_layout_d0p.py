#!/usr/bin/env python3
"""Extract only source-verifiable Triton direct-K64 layout evidence for D0-P.

The script intentionally distinguishes layout facts emitted by TTGIR from
information that cannot be reconstructed after Triton has lowered a memdesc
to scalar AMDGPU address arithmetic.  It never infers a lane-to-LDS or
lane-to-fragment formula from variable names.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
DEFAULT_KERNEL_DIR = (
    REPO
    / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/"
    "codex_qwen_bt64_recurrence_reconciliation_stage6r/current_kernels/vllm"
)
DEFAULT_OUT_DIR = REPO / "test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_layout_d0p/audit"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def _blocked_k_rows() -> list[dict[str, int | str]]:
    """Enumerate the exact TTGIR #blocked ownership for tensor<64x64>.

    The source encoding is:
      sizePerThread=[8,4], threadsPerWarp=[8,8], warpsPerCTA=[1,2],
      order=[0,1].

    This maps a lane's 8x4 register tile directly.  The physical LDS offset
    and MFMA fragment word are deliberately not filled here: TTGIR records
    those as #shared1 / #ttg.dot_op encodings but the final per-lane address
    calculation is no longer represented as an element map in the dump.
    """

    rows: list[dict[str, int | str]] = []
    for warp in range(2):
        for lane in range(64):
            lane_row_group = lane % 8
            lane_col_group = lane // 8
            for row_item in range(8):
                for col_item in range(4):
                    token = lane_row_group * 8 + row_item
                    feature = warp * 32 + lane_col_group * 4 + col_item
                    rows.append(
                        {
                            "operand": "K64xT64_global_load",
                            "warp": warp,
                            "lane": lane,
                            "lane_row_group": lane_row_group,
                            "lane_col_group": lane_col_group,
                            "row_item": row_item,
                            "col_item": col_item,
                            "logical_token": token,
                            "logical_feature_in_k64": feature,
                            "bf16_byte_offset_in_token_major_k64": 2 * (token * 64 + feature),
                            "lds_byte_offset": "not materialized by TTGIR element map",
                            "mfma_fragment_word": "not materialized by TTGIR element map",
                            "evidence": "#blocked sizePerThread=[8,4], threadsPerWarp=[8,8], warpsPerCTA=[1,2], order=[0,1]",
                        }
                    )
    return rows


def _extract_dot_sites(ttgir: str) -> list[dict[str, str | int]]:
    sites: list[dict[str, str | int]] = []
    for line_no, line in enumerate(ttgir.splitlines(), start=1):
        if "tt.dot" not in line:
            continue
        match = re.search(r"#ttg\.dot_op<\{opIdx = (\d), parent = #mma, kWidth = (\d+)\}>", line)
        sites.append(
            {
                "line": line_no,
                "ttgir": line.strip(),
                "op_idx": int(match.group(1)) if match else -1,
                "k_width": int(match.group(2)) if match else -1,
            }
        )
    return sites


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-dir", type=Path, default=DEFAULT_KERNEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    kernel_dir = args.kernel_dir.resolve()
    ttir_path = kernel_dir / "kernel.ttir"
    ttgir_path = kernel_dir / "kernel.ttgir"
    isa_path = kernel_dir / "kernel.amdgcn"
    for path in (ttir_path, ttgir_path, isa_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    ttir = ttir_path.read_text()
    ttgir = ttgir_path.read_text()
    isa = isa_path.read_text()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    layouts = [line.strip() for line in ttgir.splitlines()[:20] if line.startswith("#")]
    local_ops = [
        {"line": number, "text": line.strip()}
        for number, line in enumerate(ttgir.splitlines(), start=1)
        if "ttg.local_store" in line
        or "ttg.local_load" in line
        or "amdg.in_thread_transpose" in line
        or "ttg.local_alloc" in line
    ]
    isa_counts = {
        "buffer_load_dwordx4": _count(isa, r"\bbuffer_load_dwordx4\b"),
        "ds_write_b16": _count(isa, r"\bds_write_b16\b"),
        "ds_write_b32": _count(isa, r"\bds_write_b32\b"),
        "ds_write_b64": _count(isa, r"\bds_write_b64\b"),
        "ds_write2st64_b64": _count(isa, r"\bds_write2st64_b64\b"),
        "ds_read_b64": _count(isa, r"\bds_read_b64\b"),
        "ds_read2_b64": _count(isa, r"\bds_read2_b64\b"),
        "ds_read2st64_b32": _count(isa, r"\bds_read2st64_b32\b"),
        "v_perm_b32": _count(isa, r"\bv_perm_b32\b"),
        "ds_bpermute_b32": _count(isa, r"\bds_bpermute_b32\b"),
        "v_mov_b32_dpp": _count(isa, r"\bv_mov_b32_dpp\b"),
        "v_mfma_f32_32x32x8_bf16": _count(isa, r"\bv_mfma_f32_32x32x8_bf16\b"),
        "s_barrier": _count(isa, r"\bs_barrier\b"),
    }
    ttir_dots = [line.strip() for line in ttir.splitlines() if "tt.dot" in line]
    rows = _blocked_k_rows()
    dot_sites = _extract_dot_sites(ttgir)

    with (out_dir / "triton_k64_lane_global_mapping.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema": "qwen_direct_k64_bv32_d0p_triton_mapping_v1",
        "artifact": {
            "kernel_dir": str(kernel_dir),
            "ttir_sha256": _sha256(ttir_path),
            "ttgir_sha256": _sha256(ttgir_path),
            "amdgcn_sha256": _sha256(isa_path),
        },
        "ttir_update_dots": ttir_dots,
        "ttgir_layouts": layouts,
        "ttgir_local_ops": local_ops,
        "ttgir_dot_sites": dot_sites,
        "derived_global_mapping": {
            "source_layout": "#blocked",
            "formula": {
                "token": "(lane % 8) * 8 + row_item, row_item in [0,7]",
                "feature_in_k64": "warp * 32 + (lane // 8) * 4 + col_item, col_item in [0,3]",
            },
            "row_count": len(rows),
        },
        "ldsmem_and_fragment_status": {
            "shared_k_encoding": "#ttg.amd_rotating_shared<{vec=4, perPhase=1, maxPhase=16, order=[1,0]}>",
            "dot_operand_encoding": "#ttg.dot_op<{opIdx=0/1, parent=#mma, kWidth=4}>",
            "exact_lane_to_lds_byte_offset": "unavailable in the retained TTGIR element map; do not infer from names",
            "exact_lane_to_mfma_fragment_word": "unavailable in the retained TTGIR element map; do not infer from names",
            "required_next_evidence": "source compile gate plus a dedicated operand-layout IR that preserves this mapping",
        },
        "static_isa_counts": isa_counts,
        "summary": {
            "has_in_thread_transpose": "amdg.in_thread_transpose" in ttgir,
            "has_swizzled_shared": "#ttg.swizzled_shared" in ttgir,
            "has_rotating_shared": "#ttg.amd_rotating_shared" in ttgir,
            "has_dot_operand_layout": "#ttg.dot_op" in ttgir,
            "has_mfma32": isa_counts["v_mfma_f32_32x32x8_bf16"] > 0,
            "has_permute": isa_counts["v_perm_b32"] > 0,
            "has_ds_bpermute": isa_counts["ds_bpermute_b32"] > 0,
            "has_dpp": isa_counts["v_mov_b32_dpp"] > 0,
        },
    }
    (out_dir / "triton_k64_mapping.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out_dir": str(out_dir), "isa": isa_counts, "lane_rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
