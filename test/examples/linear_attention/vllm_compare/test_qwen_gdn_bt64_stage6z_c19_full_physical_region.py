"""Static C19 ownership and artifact contract checks.

Runtime C19 correctness and the PMC capture were executed in the gfx942
Docker.  These checks intentionally reject a mixed C18/P2 ownership source
before a future runtime invocation can silently fall back.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = HERE / "qwen_gdn_bt64_native_chunko_stage6z_c19_full_physical_region.py"
LOWERING = REPO / "lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc"
EVIDENCE = REPO / (
    "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/"
    "stage6z_c19_machine_evidence.json"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_c19_has_one_source_arena_and_no_legacy_source_owner() -> None:
    source = _text(SOURCE)
    assert "AVELANG_STAGE6Z_FULL_PHYSICAL_REGION" in source
    assert "C19_SHARED_ROWS = 3 * BT * 2" in source
    assert source.count("al.make_shared((C19_SHARED_ROWS, BK), al.bf16)") == 1
    assert "q_cache" not in source
    assert "phase_vec" not in source
    assert "c18" not in source.lower()
    assert source.count("block_dot_bf16_f32_logical") >= 3
    assert "score_acc0" in source and "score_acc1" in source


def test_c19_compiler_hard_fails_mixed_legacy_ownership() -> None:
    lowering = _text(LOWERING)
    assert "emitC19FullPhysicalProducer" in lowering
    assert "avelang.stage6z.c19.legacy_owners_forbidden" in lowering
    assert "C19 encountered a legacy physical owner attribute" in lowering
    assert "C19 requires compiler-owned Q/H/K/V producers and consumers" in lowering
    assert "c19.legacy_owner" in lowering


def test_c19_captured_object_has_real_24k_allocation_and_no_spill() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    obj = evidence["code_object"]
    assert obj["group_segment_fixed_size"] == 24576
    assert obj["private_segment_fixed_size"] == 0
    assert obj["vgpr_spill_count"] == 0
    assert obj["sgpr_spill_count"] == 0
    assert evidence["static_isa"]["ds_bpermute"] == 0
    assert evidence["machine_distinct"]["vs_P2"] is True
    assert evidence["machine_distinct"]["vs_C18"] is True
