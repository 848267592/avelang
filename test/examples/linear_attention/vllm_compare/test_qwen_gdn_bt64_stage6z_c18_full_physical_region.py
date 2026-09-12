"""Static contract checks for the experimental C18 full-region lowering gate."""

from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = HERE / "qwen_gdn_bt64_native_chunko_stage6z_c18_full_physical_region.py"
LOWERING = REPO / "lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc"
INTRINSICS = REPO / "lib/IR/Intrinsics/amdgpu_module.cc"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_c18_is_experimental_and_uses_the_existing_logical_block_dot() -> None:
    source = _text(SOURCE)
    assert "AVELANG_STAGE6Z_FULL_PHYSICAL_REGION" in source
    assert "block_dot_bf16_f32_logical" in source
    assert "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c18_full_physical_region" in source
    assert "phase_vec" not in source
    assert "al.amdgpu.mfma_32x32x8_bf16_f32" not in source


def test_c18_routes_v_by_source_identity_into_the_internal_plan() -> None:
    lowering = _text(LOWERING)
    assert "useC18FullPhysicalRegion" in lowering
    assert "FullPhysicalRegionPlan" in lowering
    assert "c18.v_producer_owner" in lowering
    assert "c18.full_physical_region" in lowering
    assert "resolved_args[2] == resolved_args[3]" in _text(INTRINSICS)


def test_c18_keeps_production_and_allocator_outside_the_gate() -> None:
    source = _text(SOURCE)
    lowering = _text(LOWERING)
    assert "experimental-only" in source
    assert "allocator" not in lowering.lower()
    assert "qwen_c18" not in lowering
