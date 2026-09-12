"""Static contract tests for the C17 full physical-plan gate."""

from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LOWERING = REPO / "lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc"
SOURCE = HERE / "qwen_gdn_bt64_native_chunko_stage6z_c17_full_physical_plan.py"
BDV2 = HERE / "qwen_gdn_bt64_native_chunko_stage6z_block_dot_v2_full_scope.py"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_c17_is_an_experimental_gate_over_the_existing_full_scope_contract() -> None:
    source = _text(SOURCE)
    bdv2 = _text(BDV2)
    assert '"AVELANG_STAGE6Z_FULL_PHYSICAL_PLAN"' in source
    assert '"c17"' in source
    assert "qwen_gdn_bt64_native_chunko_stage6z_block_dot_v2_full_scope" in source
    assert "block_dot_bf16_f32_logical" in bdv2
    assert "Q_CACHE_ROWS = 4 * BT" in bdv2
    assert "qwen_chunk_o_c17" not in source


def test_c17_creates_one_plan_and_uses_static_affine_ownership() -> None:
    lowering = _text(LOWERING)
    assert "useC17FullPhysicalPlan" in lowering
    assert "ChunkOPhysicalPlan::makeC12T2048WG256()" in lowering
    assert "makeC17StaticAffineLayoutPlan" in lowering
    assert "c17Plan_" in lowering
    assert "source_Q_H_K_then_score_V" in lowering
    assert '"c17.full_physical_plan"' in lowering
    assert "arith::DivUIOp::create" in lowering
    assert "arith::RemUIOp::create" in lowering


def test_c17_plan_attrs_reach_the_first_class_operand_boundary() -> None:
    lowering = _text(LOWERING)
    assert "annotateC17PhysicalPlan(planned" in lowering
    assert "c17.operand_lifetime" in lowering
    assert "c17.phase_boundary" in lowering
    assert "c17.q_dual_consumer" in lowering


def test_c17_does_not_modify_production_or_create_a_qwen_specific_op() -> None:
    source = _text(SOURCE)
    lowering = _text(LOWERING)
    assert "production" in source
    assert "AMDGPUBlockDotBF16F32Op" in lowering
    assert "qwen_c17" not in lowering
    assert "C17Q" not in lowering
