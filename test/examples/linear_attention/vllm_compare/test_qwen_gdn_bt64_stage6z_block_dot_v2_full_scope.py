"""Static compiler-contract tests for the reusable full-scope block-dot API.

The GPU correctness and benchmark drivers are intentionally separate because
they compile a fresh JIT module and require a gfx942 device.  These tests run
without a device and protect the source/IR contract and the legacy API.
"""

from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = HERE / "qwen_gdn_bt64_native_chunko_stage6z_block_dot_v2_full_scope.py"
INTRINSICS = REPO / "lib/IR/Intrinsics/amdgpu_module.cc"
OPS = REPO / "lib/Dialect/AveLang/IR/AveLangOps.td"
LOWERING = REPO / "lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_full_scope_source_uses_one_generic_logical_block_dot_contract() -> None:
    source = _text(SOURCE)
    assert "block_dot_bf16_f32_logical_transposed" in source
    assert "block_dot_bf16_f32_logical(" in source
    assert "Q_CACHE_ROWS = 4 * BT" in source
    assert "Q cache" in source
    assert "if value_half == 0" in source
    assert "value_half = wave_id & 1" in source
    assert "qwen_chunk_o_dot" not in source
    assert "chunk_o_operand_op" not in source
    assert "z8_special_dot" not in source


def test_logical_api_keeps_one_operation_identity_and_generic_fallback() -> None:
    intrinsics = _text(INTRINSICS)
    ops = _text(OPS)
    lowering = _text(LOWERING)

    assert '"block_dot_bf16_f32"' in intrinsics
    assert '"block_dot_bf16_f32_logical"' in intrinsics
    assert '"block_dot_bf16_f32_logical_transposed"' in intrinsics
    assert '"full_scope"' in intrinsics
    assert '"logical_block"' in intrinsics
    assert "AMDGPUBlockDotBF16F32Op" in intrinsics

    assert "full-scope logical-block mode" in ops
    assert "target-independent global logical B source" in ops
    assert "backward-compatible construction" in ops

    assert "isFullScopeOperandMode" in lowering
    assert "emitFullScopeProducer" in lowering
    assert "emitGenericOperandBPair" in lowering
    assert "global_bf16x8" in lowering
    assert "generic_scalar" in lowering
    assert "valueHalf" in lowering


def test_k_and_h_share_the_same_full_scope_lowering_path() -> None:
    lowering = _text(LOWERING)
    source = _text(SOURCE)

    # K and H are selected by generic source-role metadata, not by separate
    # Qwen-specific lowering functions or source-level staging loops.
    assert 'sourceRole == "H"' in lowering
    assert 'sourceRole == "K"' in lowering
    assert "if (specialized)" in lowering
    assert "if (isH)" in lowering
    assert "sourceType.getRank() != 4" in lowering
    assert "sourceType.getRank() != 5" in lowering
    assert source.count("block_dot_bf16_f32_logical_transposed") == 1
    assert source.count("block_dot_bf16_f32_logical(") == 1


def test_full_scope_does_not_turn_resident_q_into_a_global_producer() -> None:
    source = _text(SOURCE)
    lowering = _text(LOWERING)

    # The only Q pointer access in the BDV2 source is the frozen cache fill;
    # the logical block-dot calls receive q_cache as their resident A block.
    assert source.count("q[0, chunk_start") == 1
    assert "q_cache,\n            phase,\n            h" in source
    assert "q_cache,\n                phase,\n                k" in source
    assert "lhs_residency" in _text(INTRINSICS)


def test_p1_is_one_common_planner_and_typed_fragment_arm() -> None:
    source = _text(SOURCE)
    lowering = _text(LOWERING)
    assert "set_block_dot_planner" in source
    assert "bdv2_p1_affine" in source
    assert "useLogicalBlockLayoutPlan" in lowering
    assert "LogicalBlockLayoutPlan" in lowering
    assert "typed_fragment_direct" in lowering
    # The P1 path must remain shared by both logical source roles.
    assert "emitGenericOperandBPair" in lowering
    assert 'sourceRole == "H"' in lowering
    assert 'sourceRole == "K"' in lowering


def test_p2_preserves_a_first_class_mfma_operand_until_late_gpu_lowering() -> None:
    source = _text(SOURCE)
    lowering = _text(LOWERING)
    gpu_pipeline = _text(REPO / "lib/Target/GPU/lower_to_llvm.cc")

    # P2 is a compiler-only selector on the same source.  The internal op is
    # intentionally absent from the public Python API and is materialized only
    # after GPU outlining, where its packed LDS consumer can be audited.
    assert "set_block_dot_operand_preservation" in source
    assert '"p2_first_class"' in source
    assert "AMDGPUBlockDotMfmaOperandOp" in lowering
    assert "first_class_lds_b64_i64" in lowering
    assert "createLowerQwenBlockDotMfmaOperandPass" in gpu_pipeline
    assert "pre_block_dot_operand_materialization" in gpu_pipeline
    assert "post_block_dot_operand_materialization" in gpu_pipeline


def test_p3_keeps_first_class_operand_and_selects_packed_consumer_groups() -> None:
    source = _text(SOURCE)
    lowering = _text(LOWERING)
    dump = _text(
        HERE / "dump_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope_machine.py"
    )
    bench = _text(
        HERE / "bench_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py"
    )

    assert '"p3_packed_reuse"' in source
    assert '"p3_packed_reuse"' in dump
    assert '"bdv2_p3_specialized"' in bench
    assert "usePackedOperandReusePlan" in lowering
    assert "first_class_lds_b128_group" in lowering
    assert "consumer_group" in lowering
    assert "ExtractStridedSliceOp" in lowering
    # P3 stays compiler-internal; it must not create a Qwen-specific op.
    assert "qwen_p3" not in lowering
    assert "chunk_o_p3" not in lowering


def test_p4_is_generic_accumulator_forwarding_on_the_same_source() -> None:
    source = _text(SOURCE)
    lowering = _text(LOWERING)
    bench = _text(HERE / "bench_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py")

    assert '"p4_accumulator_reuse"' in source
    assert "useAccumulatorReusePlan" in lowering
    assert "AccumulatorForwardingMap" in lowering
    assert "hasEnclosingAccumulatorReset" in lowering
    assert '"p4_ssa_forwarding"' in lowering
    assert '"bdv2_p4_specialized"' in bench
    # P4 is an internal planner choice, not a new Qwen/chunk-o operation.
    assert "qwen_p4" not in lowering
    assert "chunk_o_p4" not in lowering
