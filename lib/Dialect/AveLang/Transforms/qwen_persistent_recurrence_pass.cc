#include "qwen_persistent_recurrence_pass.h"

#include "Dialect/AveLang/IR/AveLangOps.h"
#include "qwen_recurrence_schedule_plan.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/raw_ostream.h>

#include <array>
#include <functional>
#include <optional>
#include <string>

namespace causalflow::avelang::dialect {
namespace {

mlir::DictionaryAttr makePlan(mlir::MLIRContext *context,
                              QwenRecurrenceScheduleKind kind) {
    mlir::Builder builder(context);
    QwenRecurrenceSchedulePlan plan;
    if (kind == QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV1) {
        plan.kind = kind;
        plan.distributedTile = QwenDistributedTileEncoding::Bv32TwoWave;
        plan.shared = QwenSharedEncoding::JointV1SingleWkBank;
        plan.dotOperand = QwenDotOperandEncoding::TypedBf16x8;
        plan.nextChunkStage = QwenNextChunkStagePolicy::OneChunkAheadTailCommit;
    } else if (kind == QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV2) {
        plan.kind = kind;
        plan.distributedTile = QwenDistributedTileEncoding::Bv32TwoWave;
        plan.shared = QwenSharedEncoding::JointV2RotatingTypedBank;
        plan.dotOperand = QwenDotOperandEncoding::RotatingSharedTypedDot;
        plan.nextChunkStage =
            QwenNextChunkStagePolicy::OneChunkAheadInterleavedTailCommit;
    } else if (kind == QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV3) {
        plan.kind = kind;
        plan.distributedTile = QwenDistributedTileEncoding::Bv32TwoWave;
        plan.shared = QwenSharedEncoding::JointV3FullTypedRotatingBank;
        plan.dotOperand = QwenDotOperandEncoding::FullTypedRotatingDot;
        plan.nextChunkStage =
            QwenNextChunkStagePolicy::OneChunkAheadInterleavedTailCommit;
    } else if (kind == QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV4) {
        plan.kind = kind;
        plan.distributedTile = QwenDistributedTileEncoding::Bv32TwoWave;
        plan.shared = QwenSharedEncoding::JointV4LdsMediatedRetileBank;
        plan.dotOperand = QwenDotOperandEncoding::LdsMediatedRetileDot;
        plan.nextChunkStage =
            QwenNextChunkStagePolicy::OneChunkAheadInterleavedTailCommit;
    } else if (kind == QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV5) {
        plan.kind = kind;
        plan.distributedTile = QwenDistributedTileEncoding::Bv32TwoWave;
        plan.shared = QwenSharedEncoding::JointV5SuperblockBank;
        plan.dotOperand = QwenDotOperandEncoding::SuperblockLdsMediatedRetileDot;
        plan.nextChunkStage =
            QwenNextChunkStagePolicy::OneChunkAheadInterleavedTailCommit;
    } else if (kind == QwenRecurrenceScheduleKind::Gfx942Bt64Bv32SoftwarePipeline) {
        // Reuse the R4 producer/consumer encoding. The new fact here is loop
        // structure (a distance-one packet ring), not a sixth joint layout.
        plan.kind = kind;
        plan.distributedTile = QwenDistributedTileEncoding::Bv32TwoWave;
        plan.shared = QwenSharedEncoding::JointV4LdsMediatedRetileBank;
        plan.dotOperand = QwenDotOperandEncoding::LdsMediatedRetileDot;
        plan.nextChunkStage = QwenNextChunkStagePolicy::CompilerOwnedDeferred;
    }
    const auto schedule = kind == QwenRecurrenceScheduleKind::LegacyB0
                              ? "legacy_b0"
                              : kind == QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV1
                                    ? "gfx942_bt64_bv32_joint_v1"
                                    : kind == QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV2
                                          ? "gfx942_bt64_bv32_joint_v2"
                                          : kind == QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV3
                                                ? "gfx942_bt64_bv32_joint_v3"
                                                : kind == QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV4
                                                      ? "gfx942_bt64_bv32_joint_v4"
                                                      : kind == QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV5
                                                            ? "gfx942_bt64_bv32_joint_v5"
                                                            : "gfx942_bt64_bv32_software_pipeline";
    const auto distributed = plan.distributedTile == QwenDistributedTileEncoding::Bv32TwoWave
                                 ? "bv32_two_wave"
                                 : "deferred";
    const auto shared = plan.shared == QwenSharedEncoding::LegacyB0
                            ? "legacy_b0"
                            : plan.shared == QwenSharedEncoding::JointV1SingleWkBank
                                  ? "joint_v1_single_wk_bank"
                                  : plan.shared == QwenSharedEncoding::JointV2RotatingTypedBank
                                        ? "joint_v2_rotating_typed_bank"
                                        : plan.shared == QwenSharedEncoding::JointV3FullTypedRotatingBank
                                              ? "joint_v3_full_typed_rotating_bank"
                                        : plan.shared == QwenSharedEncoding::JointV4LdsMediatedRetileBank
                                              ? "joint_v4_lds_mediated_retile_bank"
                                        : plan.shared == QwenSharedEncoding::JointV5SuperblockBank
                                              ? "joint_v5_superblock_bank"
                                        : "deferred";
    const auto dotOperand = plan.dotOperand == QwenDotOperandEncoding::TypedBf16x8
                                ? "typed_bf16x8"
                                : plan.dotOperand == QwenDotOperandEncoding::RotatingSharedTypedDot
                                      ? "rotating_shared_typed_dot"
                                      : plan.dotOperand == QwenDotOperandEncoding::FullTypedRotatingDot
                                            ? "full_typed_rotating_dot"
                                      : plan.dotOperand == QwenDotOperandEncoding::LdsMediatedRetileDot
                                            ? "lds_mediated_retile_dot"
                                      : plan.dotOperand == QwenDotOperandEncoding::SuperblockLdsMediatedRetileDot
                                            ? "superblock_lds_mediated_retile_dot"
                                      : "deferred";
    const auto nextChunk = plan.nextChunkStage == QwenNextChunkStagePolicy::OneChunkAheadTailCommit
                               ? "one_chunk_ahead_tail_commit"
                               : plan.nextChunkStage == QwenNextChunkStagePolicy::OneChunkAheadInterleavedTailCommit
                                     ? "one_chunk_ahead_interleaved_tail_commit"
                                     : plan.nextChunkStage == QwenNextChunkStagePolicy::CompilerOwnedDeferred
                                           ? "compiler_owned_deferred"
                                           : "none";
    return builder.getDictionaryAttr({
        builder.getNamedAttr("target", builder.getStringAttr(plan.target)),
        builder.getNamedAttr("bt", builder.getI64IntegerAttr(plan.bt)),
        builder.getNamedAttr("bv", builder.getI64IntegerAttr(plan.bv)),
        builder.getNamedAttr("workgroup_size", builder.getI64IntegerAttr(plan.workgroupSize)),
        builder.getNamedAttr("waves", builder.getI64IntegerAttr(plan.waves)),
        builder.getNamedAttr("state_type", builder.getStringAttr(plan.stateType)),
        builder.getNamedAttr("boundary_type", builder.getStringAttr(plan.boundaryType)),
        builder.getNamedAttr("dot_geometry",
                             builder.getStringAttr("mfma_32x32x8_bf16_f32")),
        builder.getNamedAttr("schedule", builder.getStringAttr(schedule)),
        builder.getNamedAttr("pipeline_stage_token", builder.getBoolAttr(plan.supportsPipelineStageToken)),
        builder.getNamedAttr("distributed_layout", builder.getStringAttr(distributed)),
        builder.getNamedAttr("shared_encoding", builder.getStringAttr(shared)),
        builder.getNamedAttr("dot_operand_encoding", builder.getStringAttr(dotOperand)),
        builder.getNamedAttr("next_chunk_stage", builder.getStringAttr(nextChunk)),
        builder.getNamedAttr("pred_phase", builder.getStringAttr("mfma32_bf16")),
        builder.getNamedAttr("vnew_boundary", builder.getStringAttr("bf16_round_trip")),
        builder.getNamedAttr("update_phase", builder.getStringAttr("direct_k64_mfma32")),
        builder.getNamedAttr("feedback", builder.getStringAttr("fp32_loop_carried")),
        builder.getNamedAttr("stage_operands", builder.getStringAttr("w0,w1,k0,k1")),
        builder.getNamedAttr(
            "bank_lifetime",
            builder.getStringAttr(kind == QwenRecurrenceScheduleKind::Gfx942Bt64Bv32SoftwarePipeline
                                      ? "loop_carried_packet_ring_single_lds_bank"
                                      : kind == QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV2 ||
                                          kind == QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV3 ||
                                          kind == QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV4 ||
                                          kind == QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV5
                                      ? "rotating_wk_bank_interleaved_tail_commit"
                                      : "single_wk_bank_tail_commit")),
    });
}

constexpr llvm::StringLiteral kMicrotileModePrefix =
    "gfx942_bt64_bv32_microtile_experimental_";
constexpr llvm::StringLiteral kCoreLastUseModePrefix =
    "gfx942_bt64_bv32_core_lastuse_experimental_";
constexpr llvm::StringLiteral kBvConsumeTailIssueModePrefix =
    "gfx942_bt64_bv";

// The physical MFMA is deliberately fixed at V32.  BV16 is consequently a
// padded/masked V32 consumer and BV64 is two serial V32 consumers in the
// *same* recurrence iteration.  This parser makes that distinction explicit
// in the first-class plan instead of silently treating a launch-grid change as
// a tile-size experiment.
std::optional<int64_t> parseBvConsumeTailIssuePlan(llvm::StringRef mode) {
    if (!mode.starts_with(kBvConsumeTailIssueModePrefix)) {
        return std::nullopt;
    }
    mode = mode.drop_front(kBvConsumeTailIssueModePrefix.size());
    int64_t bv = 0;
    auto suffix = mode.take_until([](char c) { return c == '_'; });
    if (suffix.getAsInteger(10, bv) ||
        (bv != 16 && bv != 32 && bv != 64) ||
        mode.drop_front(suffix.size()) != "_joint_v4_tail_issue") {
        return std::nullopt;
    }
    return bv;
}

mlir::DictionaryAttr makeBvConsumeTailIssuePlan(mlir::MLIRContext *context,
                                                 int64_t bv,
                                                 llvm::StringRef mode) {
    mlir::Builder builder(context);
    mlir::NamedAttrList attrs(makePlan(
        context, QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV4).getValue());
    attrs.set("schedule", builder.getStringAttr(mode));
    attrs.set("bv", builder.getI64IntegerAttr(bv));
    attrs.set("bv_consume", builder.getI64IntegerAttr(bv));
    attrs.set("physical_mfma_bv", builder.getI64IntegerAttr(32));
    attrs.set("physical_v32_subtiles_per_recurrence_iteration",
              builder.getI64IntegerAttr((bv + 31) / 32));
    if (bv == 64) {
        attrs.set("workgroup_size", builder.getI64IntegerAttr(256));
        attrs.set("waves", builder.getI64IntegerAttr(4));
        attrs.set("distributed_layout",
                  builder.getStringAttr("two_bv32_pairs_four_wave"));
    }
    attrs.set("bv_consume_mapping",
              builder.getStringAttr(bv < 32
                                        ? "v32_padded_masked_consumer"
                                        : bv == 32
                                              ? "one_v32_consumer"
                                              : "two_serial_v32_consumers"));
    attrs.set("tail_issue", builder.getStringAttr(
                               "next_wk_global_to_lds_after_full_current_core"));
    return attrs.getDictionary(context);
}

struct ParsedMicrotilePlan {
    QwenMicrotileSchedulePlan schedule;
    std::string name;
};

struct ParsedCoreLastUsePlan {
    QwenCoreLastUseSchedulePlan schedule;
    std::string name;
};

std::optional<int64_t> parsePacketGroup(llvm::StringRef value,
                                        llvm::StringRef prefix) {
    if (!value.consume_front(prefix)) {
        return std::nullopt;
    }
    int64_t packets = 0;
    if (value.getAsInteger(10, packets) ||
        (packets != 1 && packets != 2 && packets != 4)) {
        return std::nullopt;
    }
    return packets;
}

std::optional<ParsedMicrotilePlan> parseMicrotilePlan(llvm::StringRef mode) {
    if (!mode.starts_with(kMicrotileModePrefix)) {
        return std::nullopt;
    }
    llvm::SmallVector<llvm::StringRef> fields;
    mode.drop_front(kMicrotileModePrefix.size()).split(fields, '_', -1,
                                                        /*KeepEmpty=*/false);
    if (fields.size() != 4) {
        return std::nullopt;
    }
    auto wPackets = parsePacketGroup(fields[0], "w");
    auto kPackets = parsePacketGroup(fields[1], "k");
    if (!wPackets || !kPackets ||
        (fields[2] != "tail" && fields[2] != "lastuse") ||
        (fields[3] != "d0" && fields[3] != "d1")) {
        return std::nullopt;
    }
    ParsedMicrotilePlan result;
    result.schedule.wPacketsPerGroup = *wPackets;
    result.schedule.kPacketsPerGroup = *kPackets;
    result.schedule.issuePlacement = fields[2] == "tail"
                                         ? QwenMicrotileIssuePlacement::Tail
                                         : QwenMicrotileIssuePlacement::LastUseStaggered;
    result.schedule.vgprResidentMicroGroups = fields[3] == "d1" ? 1 : 0;
    result.name = mode.str();
    return result;
}

std::optional<ParsedCoreLastUsePlan> parseCoreLastUsePlan(llvm::StringRef mode) {
    if (!mode.starts_with(kCoreLastUseModePrefix)) {
        return std::nullopt;
    }
    llvm::SmallVector<llvm::StringRef> fields;
    mode.drop_front(kCoreLastUseModePrefix.size()).split(fields, '_', -1,
                                                         /*KeepEmpty=*/false);
    if (fields.size() != 3) {
        return std::nullopt;
    }
    auto wPackets = parsePacketGroup(fields[0], "w");
    auto kPackets = parsePacketGroup(fields[1], "k");
    if (!wPackets || !kPackets ||
        (fields[2] != "immediate" && fields[2] != "delay1")) {
        return std::nullopt;
    }
    ParsedCoreLastUsePlan result;
    result.schedule.wPacketsPerGroup = *wPackets;
    result.schedule.kPacketsPerGroup = *kPackets;
    result.schedule.issuePlacement = fields[2] == "immediate"
        ? QwenCoreLastUseIssuePlacement::Immediate
        : QwenCoreLastUseIssuePlacement::DelayedOneConsumerGroup;
    result.name = mode.str();
    return result;
}

mlir::DictionaryAttr makeMicrotilePlan(mlir::MLIRContext *context,
                                       const ParsedMicrotilePlan &microtile) {
    mlir::Builder builder(context);
    mlir::NamedAttrList attrs(makePlan(
        context, QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV4).getValue());
    const auto placement = microtile.schedule.issuePlacement ==
                                   QwenMicrotileIssuePlacement::Tail
                               ? "tail"
                               : "last_use_staggered";
    attrs.set("schedule", builder.getStringAttr(microtile.name));
    attrs.set("microtile_w_packets_per_group",
              builder.getI64IntegerAttr(microtile.schedule.wPacketsPerGroup));
    attrs.set("microtile_k_packets_per_group",
              builder.getI64IntegerAttr(microtile.schedule.kPacketsPerGroup));
    attrs.set("microtile_issue_placement", builder.getStringAttr(placement));
    attrs.set("microtile_vgpr_resident_groups",
              builder.getI64IntegerAttr(microtile.schedule.vgprResidentMicroGroups));
    attrs.set("microtile_current_consumer_order",
              builder.getStringAttr(microtile.schedule.currentConsumerOrder));
    attrs.set("microtile_lds_region_last_use",
              builder.getStringAttr(microtile.schedule.ldsRegionLastUse));
    attrs.set("microtile_same_region_commit",
              builder.getStringAttr(microtile.schedule.sameRegionCommit));
    attrs.set("microtile_cross_wave_lifetime",
              builder.getStringAttr("verified_no_per_group_barrier"));
    return attrs.getDictionary(context);
}

mlir::DictionaryAttr makeCoreLastUsePlan(
    mlir::MLIRContext *context, const ParsedCoreLastUsePlan &coreLastUse) {
    mlir::Builder builder(context);
    mlir::NamedAttrList attrs(makePlan(
        context, QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV4).getValue());
    const auto placement = coreLastUse.schedule.issuePlacement ==
                                   QwenCoreLastUseIssuePlacement::Immediate
                               ? "immediate_after_group_last_use"
                               : "delayed_one_consumer_group";
    attrs.set("schedule", builder.getStringAttr(coreLastUse.name));
    attrs.set("core_lastuse_w_packets_per_group",
              builder.getI64IntegerAttr(coreLastUse.schedule.wPacketsPerGroup));
    attrs.set("core_lastuse_k_packets_per_group",
              builder.getI64IntegerAttr(coreLastUse.schedule.kPacketsPerGroup));
    attrs.set("core_lastuse_issue_placement", builder.getStringAttr(placement));
    attrs.set("core_lastuse_current_consumer_order",
              builder.getStringAttr(coreLastUse.schedule.currentConsumerOrder));
    attrs.set("core_lastuse_w_last_use",
              builder.getStringAttr(coreLastUse.schedule.wLastUse));
    attrs.set("core_lastuse_k_last_use",
              builder.getStringAttr(coreLastUse.schedule.kLastUse));
    attrs.set("core_lastuse_same_region_commit",
              builder.getStringAttr(coreLastUse.schedule.sameRegionCommit));
    attrs.set("core_lastuse_cross_wave_lifetime",
              builder.getStringAttr("commit_only_after_existing_tail_barrier"));
    return attrs.getDictionary(context);
}

std::optional<int64_t> qwenOperandHeadCount(mlir::Value source) {
    auto type = mlir::dyn_cast<mlir::MemRefType>(source.getType());
    if (!type || type.getRank() != 4 || type.getShape()[0] != 1 ||
        type.getShape()[3] != 128 || !type.getElementType().isBF16()) {
        return std::nullopt;
    }
    return type.getShape()[2];
}

bool tagJointStages(AMDGPUQwenPersistentRecurrenceOp recurrence, bool jointV2,
                    bool jointV3, bool jointV4, bool jointV5) {
    llvm::SmallVector<AMDGPUQwenK64PipelineStageLoadOp> loads;
    llvm::SmallVector<AMDGPUQwenK64PipelineStageCommitOp> commits;
    recurrence.walk([&](AMDGPUQwenK64PipelineStageLoadOp op) {
        loads.push_back(op);
    });
    recurrence.walk([&](AMDGPUQwenK64PipelineStageCommitOp op) {
        commits.push_back(op);
    });
    // The source schedule has four prologue operands (W0/W1/K0/K1) and the
    // same four opaque next-chunk stages. This is schedule structure, not a
    // lane or element mapping; packet ownership remains in the late lowering.
    if (loads.size() != 8 || commits.size() != 8) {
        recurrence.emitError("joint schedule requires four prologue and four next "
                             "W/K stage load/commit pairs");
        return false;
    }
    int wCount = 0;
    int kCount = 0;
    for (auto [index, stage] : llvm::enumerate(loads)) {
        auto heads = qwenOperandHeadCount(stage.getSourceK());
        if (!heads || (*heads != 4 && *heads != 8)) {
            stage.emitError("joint schedule source must be BF16 K/W [1,T,4|8,128]");
            return false;
        }
        const char *operand = *heads == 8 ? "w" : "k";
        *heads == 8 ? ++wCount : ++kCount;
        const bool interleaved = jointV2 || jointV3 || jointV4 || jointV5;
        const char *phase = index < 4
                                ? "current_prologue"
                                : interleaved && *heads == 8
                                      ? "next_w_before_pred"
                                      : interleaved ? "next_k_after_pred"
                                                : "next_deferred";
        const std::string prefix = jointV5 ? "avelang.qwen.joint_v5."
                                   : jointV4 ? "avelang.qwen.joint_v4."
                                   : jointV3 ? "avelang.qwen.joint_v3."
                                   : jointV2 ? "avelang.qwen.joint_v2."
                                             : "avelang.qwen.joint_v1.";
        stage->setAttr(prefix + "operand",
                       mlir::StringAttr::get(recurrence.getContext(), operand));
        stage->setAttr(prefix + "stage",
                       mlir::StringAttr::get(recurrence.getContext(), phase));
        stage->setAttr(
            prefix + "placement",
            mlir::StringAttr::get(recurrence.getContext(),
                                  interleaved ? "distributed_register_packet" : "distributed"));
        if (jointV5) {
            stage->setAttr("avelang.qwen.joint_v5.shared_encoding",
                           mlir::StringAttr::get(recurrence.getContext(), "lds_mediated_retile"));
            stage->setAttr("avelang.qwen.joint_v5.dot_operand",
                           mlir::StringAttr::get(recurrence.getContext(),
                                                 *heads == 8 ? "typed_pred_operand"
                                                             : "token_major_lds_gather_fragment"));
            stage->setAttr("avelang.qwen.joint_v5.producer_layout",
                           mlir::StringAttr::get(recurrence.getContext(), "token_major_bf16x8"));
        } else if (jointV4) {
            stage->setAttr("avelang.qwen.joint_v4.shared_encoding",
                           mlir::StringAttr::get(recurrence.getContext(), "lds_mediated_retile"));
            stage->setAttr("avelang.qwen.joint_v4.dot_operand",
                           mlir::StringAttr::get(recurrence.getContext(),
                                                 *heads == 8 ? "typed_pred_operand"
                                                             : "token_major_lds_gather_fragment"));
            stage->setAttr("avelang.qwen.joint_v4.producer_layout",
                           mlir::StringAttr::get(recurrence.getContext(), "token_major_bf16x8"));
        } else if (jointV3) {
            stage->setAttr("avelang.qwen.joint_v3.shared_encoding",
                           mlir::StringAttr::get(recurrence.getContext(), "full_typed_rotating"));
            stage->setAttr("avelang.qwen.joint_v3.dot_operand",
                           mlir::StringAttr::get(recurrence.getContext(),
                                                 *heads == 8 ? "typed_pred_operand"
                                                             : "typed_mfma32_fragment"));
            stage->setAttr("avelang.qwen.joint_v3.producer_layout",
                           mlir::StringAttr::get(recurrence.getContext(),
                                                 *heads == 8 ? "distributed_bf16x8"
                                                             : "subgroup_8x8_transposed_bf16x8"));
        } else if (jointV2) {
            stage->setAttr("avelang.qwen.joint_v2.shared_encoding",
                           mlir::StringAttr::get(recurrence.getContext(), "rotating_typed"));
            stage->setAttr("avelang.qwen.joint_v2.dot_operand",
                           mlir::StringAttr::get(recurrence.getContext(), "typed_bf16x8"));
        }
    }
    if (wCount != 4 || kCount != 4) {
        recurrence.emitError("joint schedule requires two current and two next "
                             "stages for each of W and K");
        return false;
    }
    for (auto [index, commit] : llvm::enumerate(commits)) {
        const char *phase = index < 4 ? "current_prologue" : "next_tail_commit";
        const std::string prefix = jointV5 ? "avelang.qwen.joint_v5."
                                   : jointV4 ? "avelang.qwen.joint_v4."
                                   : jointV3 ? "avelang.qwen.joint_v3."
                                   : jointV2 ? "avelang.qwen.joint_v2."
                                             : "avelang.qwen.joint_v1.";
        commit->setAttr(prefix + "stage",
                        mlir::StringAttr::get(recurrence.getContext(), phase));
        if (jointV5) {
            commit->setAttr("avelang.qwen.joint_v5.shared_encoding",
                            mlir::StringAttr::get(recurrence.getContext(), "lds_mediated_retile"));
        } else if (jointV4) {
            commit->setAttr("avelang.qwen.joint_v4.shared_encoding",
                            mlir::StringAttr::get(recurrence.getContext(), "lds_mediated_retile"));
        } else if (jointV3) {
            commit->setAttr("avelang.qwen.joint_v3.shared_encoding",
                            mlir::StringAttr::get(recurrence.getContext(), "full_typed_rotating"));
        } else if (jointV2) {
            commit->setAttr("avelang.qwen.joint_v2.shared_encoding",
                            mlir::StringAttr::get(recurrence.getContext(), "rotating_typed"));
        }
    }
    return true;
}

// A stage token is a pure producer recipe until the late K64 pass materializes
// it.  In R4 the two next-K recipes remain after corrected/V-new formation.
// R5 schedules those recipes into the pred window, so their global packet
// loads can be issued while the current pred MFMA is executing.  The same
// packets are still committed only after the current update releases the LDS
// bank.  This intentionally does not change arithmetic, tile ownership, or
// any shared allocation.
bool scheduleJointV5Superblock(AMDGPUQwenPersistentRecurrenceOp recurrence) {
    llvm::SmallVector<AMDGPUQwenK64PipelineStageLoadOp> loads;
    recurrence.walk([&](AMDGPUQwenK64PipelineStageLoadOp op) {
        loads.push_back(op);
    });
    if (loads.size() != 8) {
        recurrence.emitError("joint_v5 superblock requires eight opaque W/K stages");
        return false;
    }

    auto nextW0 = loads[4];
    auto nextW1 = loads[5];
    auto nextK0 = loads[6];
    auto nextK1 = loads[7];
    if (nextW0->getBlock() != nextW1->getBlock() ||
        nextW0->getBlock() != nextK0->getBlock() ||
        nextW0->getBlock() != nextK1->getBlock()) {
        recurrence.emitError("joint_v5 next W/K stage recipes must share one loop block");
        return false;
    }

    auto moveIssueBundleAfter = [&](mlir::Operation *stage,
                                    mlir::Operation *anchor) -> bool {
        llvm::SmallVector<mlir::Operation *> directOperands;
        for (mlir::Value operand : stage->getOperands()) {
            auto *definition = operand.getDefiningOp();
            if (!definition || definition->getBlock() != stage->getBlock()) {
                continue;
            }
            // Only pull the local scalar/index recipes that lie between the
            // desired issue point and the opaque stage. Captured tensors and
            // constants already dominate both locations and remain in place.
            if (anchor->isBeforeInBlock(definition) &&
                definition->isBeforeInBlock(stage)) {
                directOperands.push_back(definition);
            }
        }

        llvm::SmallVector<mlir::Operation *> ordered;
        for (auto *candidate = anchor->getNextNode(); candidate && candidate != stage;
             candidate = candidate->getNextNode()) {
            if (llvm::is_contained(directOperands, candidate)) {
                ordered.push_back(candidate);
            }
        }
        if (ordered.size() != directOperands.size()) {
            recurrence.emitError("joint_v5 could not form a dominance-safe next-K issue bundle");
            return false;
        }
        auto *cursor = anchor;
        for (auto *operand : ordered) {
            operand->moveAfter(cursor);
            cursor = operand;
        }
        stage->moveAfter(cursor);
        return true;
    };

    // Preserve W0/W1 order, then issue K0/K1 before pred. The tail commits
    // remain in place, after update, and are not moved by this schedule pass.
    if (!moveIssueBundleAfter(nextK0, nextW1) ||
        !moveIssueBundleAfter(nextK1, nextK0)) {
        return false;
    }
    for (auto stage : {nextW0, nextW1}) {
        stage->setAttr("avelang.qwen.joint_v5.superblock_issue",
                       mlir::StringAttr::get(recurrence.getContext(), "next_w_before_pred"));
    }
    for (auto stage : {nextK0, nextK1}) {
        stage->setAttr("avelang.qwen.joint_v5.superblock_issue",
                       mlir::StringAttr::get(recurrence.getContext(), "next_k_before_pred"));
    }
    recurrence->setAttr("avelang.qwen.joint_v5.superblock",
                        mlir::StringAttr::get(recurrence.getContext(),
                                              "next_wk_issue_pred_vnew_update_tail_commit"));
    recurrence->setAttr("avelang.qwen.joint_v5.phase_lifetime",
                        mlir::StringAttr::get(recurrence.getContext(),
                                              "pred->bf16_vnew->vdecay->update->fp32_feedback"));
    return true;
}

// This is a causality control, not another joint layout or a software
// pipeline.  Start from the exact R4 plan and move only the four next-packet
// global-to-LDS paths to the lexical end of the same recurrence loop.  The
// packet's scalar address recipe stays where R4 formed it, while the opaque
// stage is late-lowered into the actual global BF16x8 loads at its new point.
//
// A packet path has four operations in the R4 frontend IR:
//   stage-load -> private token store -> private token load -> LDS commit.
// Moving fewer than all four would either violate SSA dominance or cause the
// commit to consume the prior iteration's token.  Keeping the group intact
// makes this a strict position-only control for the direct-LDS candidate.
bool scheduleJointV4TailIssueControl(
    AMDGPUQwenPersistentRecurrenceOp recurrence) {
    llvm::SmallVector<AMDGPUQwenK64PipelineStageLoadOp> loads;
    llvm::SmallVector<AMDGPUQwenK64PipelineStageCommitOp> commits;
    recurrence.walk([&](AMDGPUQwenK64PipelineStageLoadOp op) {
        loads.push_back(op);
    });
    recurrence.walk([&](AMDGPUQwenK64PipelineStageCommitOp op) {
        commits.push_back(op);
    });
    if (loads.size() != 8 || commits.size() != 8) {
        recurrence.emitError(
            "R4 tail-issue control requires four prologue and four next W/K stages");
        return false;
    }

    struct PacketPath {
        AMDGPUQwenK64PipelineStageLoadOp stage;
        mlir::memref::StoreOp tokenStore;
        mlir::memref::LoadOp tokenLoad;
        AMDGPUQwenK64PipelineStageCommitOp commit;
    };
    llvm::SmallVector<PacketPath> paths;
    paths.reserve(4);
    auto *block = loads[4]->getBlock();
    for (unsigned index = 0; index != 4; ++index) {
        auto stage = loads[index + 4];
        auto commit = commits[index + 4];
        if (stage->getBlock() != block || commit->getBlock() != block) {
            recurrence.emitError(
                "R4 tail-issue control requires next W/K stages in one loop block");
            return false;
        }

        mlir::memref::StoreOp tokenStore;
        for (mlir::OpOperand &use : stage.getStageToken().getUses()) {
            auto candidate = mlir::dyn_cast<mlir::memref::StoreOp>(use.getOwner());
            if (!candidate || candidate.getValue() != stage.getStageToken() ||
                candidate->getBlock() != block || tokenStore) {
                recurrence.emitError(
                    "R4 tail-issue control requires one local token store per stage");
                return false;
            }
            tokenStore = candidate;
        }
        auto tokenLoad = mlir::dyn_cast_or_null<mlir::memref::LoadOp>(
            commit.getStageToken().getDefiningOp());
        if (!tokenStore || !tokenLoad || tokenLoad->getBlock() != block) {
            recurrence.emitError(
                "R4 tail-issue control requires one local token load per commit");
            return false;
        }
        paths.push_back({stage, tokenStore, tokenLoad, commit});
    }

    // R4 has a barrier immediately after the four old tail commits.  The
    // control replaces it with the loop-entry barrier already used by R4 and
    // a new barrier immediately before the relocated producer group, exactly
    // as the direct-LDS candidate does.  Do not leave both barriers behind:
    // that would make synchronization, rather than issue position, a second
    // experimental variable.
    auto oldPostCommitBarrier =
        mlir::dyn_cast_or_null<mlir::gpu::BarrierOp>(commits.back()->getNextNode());
    if (!oldPostCommitBarrier) {
        recurrence.emitError(
            "R4 tail-issue control requires the R4 post-commit barrier");
        return false;
    }
    oldPostCommitBarrier.erase();

    auto *tail = block->getTerminator();
    if (!tail) {
        recurrence.emitError("R4 tail-issue control requires a loop terminator");
        return false;
    }
    mlir::OpBuilder builder(tail);
    auto reuseBarrier = mlir::gpu::BarrierOp::create(builder, tail->getLoc());
    mlir::Operation *anchor = reuseBarrier.getOperation();
    for (auto path : paths) {
        path.stage->moveAfter(anchor);
        path.tokenStore->moveAfter(path.stage);
        path.tokenLoad->moveAfter(path.tokenStore);
        path.commit->moveAfter(path.tokenLoad);
        for (mlir::Operation *op : {path.stage.getOperation(),
                                    path.tokenStore.getOperation(),
                                    path.tokenLoad.getOperation(),
                                    path.commit.getOperation()}) {
            op->setAttr("avelang.qwen.joint_v4.tail_issue_control",
                        mlir::UnitAttr::get(recurrence.getContext()));
        }
        anchor = path.commit.getOperation();
    }
    recurrence->setAttr("avelang.qwen.joint_v4.tail_issue_control",
                        mlir::StringAttr::get(
                            recurrence.getContext(),
                            "r4_next_wk_global_to_lds_after_full_current_core"));
    return true;
}

// The microtile schedule operates only on the four next-tile global-to-LDS
// paths.  It keeps the surrounding single recurrence loop and the one opaque
// pred/update core intact.  A stage group has a direct SSA token to its
// commit; it intentionally does not create a packet ring or a local payload
// memref.  The late pass expands the stage into exactly the selected BF16x8
// packet range.
bool scheduleJointV4Microtiles(AMDGPUQwenPersistentRecurrenceOp recurrence,
                               const ParsedMicrotilePlan &plan) {
    if (plan.schedule.issuePlacement == QwenMicrotileIssuePlacement::Tail &&
        plan.schedule.vgprResidentMicroGroups != 0) {
        recurrence.emitError(
            "microtile tail placement with d1 is contradictory: a resident group needs a pre-core issue");
        return false;
    }
    if (!scheduleJointV4TailIssueControl(recurrence)) {
        return false;
    }

    llvm::SmallVector<AMDGPUQwenK64PipelineStageLoadOp> loads;
    llvm::SmallVector<AMDGPUQwenK64PipelineStageCommitOp> commits;
    recurrence.walk([&](AMDGPUQwenK64PipelineStageLoadOp op) {
        loads.push_back(op);
    });
    recurrence.walk([&](AMDGPUQwenK64PipelineStageCommitOp op) {
        commits.push_back(op);
    });
    if (loads.size() != 8 || commits.size() != 8) {
        recurrence.emitError("microtile schedule requires the exact R4 four next W/K paths");
        return false;
    }

    AMDGPUBlockDotBF16F32Op core;
    recurrence.walk([&](AMDGPUBlockDotBF16F32Op op) {
        if (!core) {
            core = op;
        }
    });
    if (!core) {
        recurrence.emitError("microtile schedule requires one current pred/update core");
        return false;
    }

    struct PacketPath {
        AMDGPUQwenK64PipelineStageLoadOp stage;
        mlir::memref::StoreOp tokenStore;
        mlir::memref::LoadOp tokenLoad;
        AMDGPUQwenK64PipelineStageCommitOp commit;
        llvm::StringRef region;
        bool wOperand;
    };
    constexpr std::array<llvm::StringLiteral, 4> kRegions = {
        "w0", "w1", "k0", "k1"};
    llvm::SmallVector<PacketPath> paths;
    paths.reserve(4);
    auto *loopBlock = loads[4]->getBlock();
    for (unsigned index = 0; index != 4; ++index) {
        auto stage = loads[index + 4];
        auto commit = commits[index + 4];
        if (stage->getBlock() != loopBlock || commit->getBlock() != loopBlock) {
            recurrence.emitError("microtile schedule requires all next paths in one loop block");
            return false;
        }
        mlir::memref::StoreOp tokenStore;
        for (mlir::OpOperand &use : stage.getStageToken().getUses()) {
            auto candidate = mlir::dyn_cast<mlir::memref::StoreOp>(use.getOwner());
            if (!candidate || candidate.getValue() != stage.getStageToken() ||
                candidate->getBlock() != loopBlock || tokenStore) {
                recurrence.emitError("microtile schedule requires one R4 token store per next stage");
                return false;
            }
            tokenStore = candidate;
        }
        auto tokenLoad = mlir::dyn_cast_or_null<mlir::memref::LoadOp>(
            commit.getStageToken().getDefiningOp());
        if (!tokenStore || !tokenLoad || tokenLoad->getBlock() != loopBlock) {
            recurrence.emitError("microtile schedule requires one R4 token load per next commit");
            return false;
        }
        const bool wOperand = index < 2;
        paths.push_back({stage, tokenStore, tokenLoad, commit, kRegions[index],
                         wOperand});
    }

    // Cross-wave lifetime proof.  The current W/K bank is read inside the
    // opaque block-dot core.  Every next-tile write is constructed below the
    // core and therefore follows the already-validated tail barrier.  An
    // optional d1 group is only a global load above the core, never an LDS
    // write, so it cannot overwrite a still-read region.  Any plan that needs
    // a group commit before the core would need an extra workgroup barrier and
    // is deliberately not expressible here.
    // The source keeps the K-half update traversal as a nested scf.for.  Its
    // enclosing operation, rather than the leaf block-dot itself, is the
    // physical last-use boundary in the recurrence loop block.
    mlir::Operation *coreLifetimeBoundary = core.getOperation();
    while (coreLifetimeBoundary->getBlock() != loopBlock) {
        coreLifetimeBoundary = coreLifetimeBoundary->getParentOp();
        if (!coreLifetimeBoundary) {
            recurrence.emitError("microtile schedule cannot find the enclosing core lifetime boundary");
            return false;
        }
    }
    auto *tail = loopBlock->getTerminator();
    if (!tail) {
        recurrence.emitError("microtile schedule requires a loop terminator");
        return false;
    }
    auto tailBarrier = mlir::dyn_cast_or_null<mlir::gpu::BarrierOp>(
        paths.front().stage->getPrevNode());
    if (!tailBarrier || !coreLifetimeBoundary->isBeforeInBlock(tailBarrier)) {
        recurrence.emitError("microtile schedule requires the R4 core-exit tail barrier");
        return false;
    }

    // `d1` keeps exactly one BF16x8 packet group resident in VGPRs: W0's
    // first group.  It is issued before the opaque core, but its same-region
    // commit stays below the core-exit barrier.  All other groups are tail
    // issue/commit pairs.  This is the only early issue that the current
    // first-class op can prove without opening the opaque core or adding a
    // per-group barrier.
    const bool oneEarlyGroup =
        plan.schedule.issuePlacement == QwenMicrotileIssuePlacement::LastUseStaggered &&
        plan.schedule.vgprResidentMicroGroups == 1;
    if (oneEarlyGroup) {
        for (mlir::Value operand : paths.front().stage->getOperands()) {
            if (auto *definition = operand.getDefiningOp(); definition &&
                definition->getBlock() == loopBlock &&
                coreLifetimeBoundary->isBeforeInBlock(definition)) {
                recurrence.emitError(
                    "microtile d1 W0 issue has a non-dominating scalar recipe; would require a private ring");
                return false;
            }
        }
    }

    mlir::OpBuilder tailBuilder(tail);
    bool issuedEarly = false;
    for (PacketPath &path : paths) {
        const int64_t packetsPerGroup = path.wOperand
                                            ? plan.schedule.wPacketsPerGroup
                                            : plan.schedule.kPacketsPerGroup;
        for (int64_t packetStart = 0; packetStart < 4;
             packetStart += packetsPerGroup) {
            const bool early = oneEarlyGroup && !issuedEarly;
            mlir::OpBuilder earlyBuilder(coreLifetimeBoundary);
            auto &builder = early ? earlyBuilder : tailBuilder;
            auto stage = AMDGPUQwenK64PipelineStageLoadOp::create(
                builder, path.stage.getLoc(), path.stage.getStageToken().getType(),
                path.stage.getSourceK(), path.stage.getThreadId(),
                path.stage.getChunkStart(), path.stage.getKeyHead(), path.stage.getKHalf());
            stage->setAttrs(path.stage->getAttrs());
            stage->setAttr("avelang.qwen.microtile.plan",
                           builder.getStringAttr(plan.name));
            stage->setAttr("avelang.qwen.microtile.region",
                           builder.getStringAttr(path.region));
            stage->setAttr("avelang.qwen.microtile.packet_start",
                           builder.getI64IntegerAttr(packetStart));
            stage->setAttr("avelang.qwen.microtile.packet_count",
                           builder.getI64IntegerAttr(packetsPerGroup));
            stage->setAttr("avelang.qwen.microtile.issue",
                           builder.getStringAttr(early ? "last_use_staggered_d1"
                                                       : "tail_after_core_exit"));
            stage->setAttr("avelang.qwen.microtile.vgpr_distance",
                           builder.getI64IntegerAttr(early ? 1 : 0));

            auto commit = AMDGPUQwenK64PipelineStageCommitOp::create(
                tailBuilder, path.commit.getLoc(), stage.getStageToken(),
                path.commit.getSharedKBank());
            commit->setAttrs(path.commit->getAttrs());
            commit->setAttr("avelang.qwen.microtile.plan",
                            tailBuilder.getStringAttr(plan.name));
            commit->setAttr("avelang.qwen.microtile.region",
                            tailBuilder.getStringAttr(path.region));
            commit->setAttr("avelang.qwen.microtile.packet_start",
                            tailBuilder.getI64IntegerAttr(packetStart));
            commit->setAttr("avelang.qwen.microtile.packet_count",
                            tailBuilder.getI64IntegerAttr(packetsPerGroup));
            commit->setAttr("avelang.qwen.microtile.last_use",
                            tailBuilder.getStringAttr("single_core_exit"));
            commit->setAttr("avelang.qwen.microtile.commit_boundary",
                            tailBuilder.getStringAttr("after_existing_core_exit_barrier"));
            issuedEarly = issuedEarly || early;
        }
    }
    for (PacketPath &path : paths) {
        path.commit.erase();
        path.tokenLoad.erase();
        path.tokenStore.erase();
        path.stage.erase();
    }
    recurrence->setAttr("avelang.qwen.microtile.plan",
                        mlir::StringAttr::get(recurrence.getContext(), plan.name));
    recurrence->setAttr("avelang.qwen.microtile.single_core",
                        mlir::UnitAttr::get(recurrence.getContext()));
    recurrence->setAttr("avelang.qwen.microtile.cross_wave_lifetime",
                        mlir::StringAttr::get(
                            recurrence.getContext(),
                            "all_next_lds_commits_after_single_core_exit_barrier"));
    return true;
}

// Core-last-use scheduling deliberately uses static, *inner* recurrence
// micro-groups as the planning boundary.  It expands the two fixed K halves
// and the four fixed W consumer groups in the one existing recurrence body;
// it does not construct prologue/steady/epilogue copies or another recurrence
// loop.  This gives a core issue a normal vector SSA definition and a tail
// commit in the same block, avoiding both a private packet ring and lexical
// SSA escapes through scf.for.
struct StaticUnrollIteration {
    int64_t inductionValue;
    llvm::SmallVector<mlir::Operation *> operations;
};

std::optional<int64_t> constantIndexValue(mlir::Value value) {
    if (auto constant = value.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
        return constant.value();
    }
    return std::nullopt;
}

bool hasStaticTripCount(mlir::scf::ForOp loop, int64_t expectedTripCount) {
    auto lower = constantIndexValue(loop.getLowerBound());
    auto upper = constantIndexValue(loop.getUpperBound());
    auto step = constantIndexValue(loop.getStep());
    return lower && upper && step && *step > 0 &&
           (*upper - *lower) / *step == expectedTripCount &&
           (*upper - *lower) % *step == 0;
}

std::optional<llvm::SmallVector<StaticUnrollIteration>>
unrollNoResultStaticFor(mlir::scf::ForOp loop) {
    if (loop.getNumResults() != 0 || loop.getNumRegionIterArgs() != 0) {
        return std::nullopt;
    }
    auto lower = constantIndexValue(loop.getLowerBound());
    auto upper = constantIndexValue(loop.getUpperBound());
    auto step = constantIndexValue(loop.getStep());
    if (!lower || !upper || !step || *step <= 0 ||
        (*upper - *lower) % *step != 0) {
        return std::nullopt;
    }
    llvm::SmallVector<mlir::Operation *> bodyOps;
    for (mlir::Operation &op : loop.getBody()->without_terminator()) {
        bodyOps.push_back(&op);
    }
    if (bodyOps.empty()) {
        return std::nullopt;
    }

    mlir::OpBuilder builder(loop);
    llvm::SmallVector<StaticUnrollIteration> result;
    for (int64_t iv = *lower; iv < *upper; iv += *step) {
        mlir::IRMapping mapping;
        auto constant = mlir::arith::ConstantIndexOp::create(
            builder, loop.getLoc(), iv);
        mapping.map(loop.getInductionVar(), constant.getResult());
        StaticUnrollIteration iteration{iv, {}};
        for (mlir::Operation *bodyOp : bodyOps) {
            iteration.operations.push_back(builder.clone(*bodyOp, mapping));
        }
        result.push_back(std::move(iteration));
    }
    loop.erase();
    return result;
}

struct CoreLastUsePacketPath {
    AMDGPUQwenK64PipelineStageLoadOp stage;
    mlir::memref::StoreOp tokenStore;
    mlir::memref::LoadOp tokenLoad;
    AMDGPUQwenK64PipelineStageCommitOp commit;
    llvm::StringRef region;
    bool isW;
    int64_t half;
};

struct CoreLastUsePacket {
    CoreLastUsePacketPath *path;
    int64_t packet;
    AMDGPUQwenK64CoreIssueOp issue;
};

AMDGPUQwenK64CoreIssueOp createCoreLastUseIssue(
    mlir::OpBuilder &builder, CoreLastUsePacketPath &path, int64_t packet,
    int64_t groupStart, int64_t groupSize, int64_t issueAfterGroup,
    llvm::StringRef boundary, llvm::StringRef planName, bool deferredBlockDot) {
    auto packetType = mlir::VectorType::get({8}, builder.getBF16Type());
    auto issue = AMDGPUQwenK64CoreIssueOp::create(
        builder, path.stage.getLoc(), packetType, path.stage.getSourceK(),
        path.stage.getThreadId(), path.stage.getChunkStart(),
        path.stage.getKeyHead(), path.stage.getKHalf());
    issue->setAttrs(path.stage->getAttrs());
    issue->setAttr("avelang.qwen.core_staggered.plan",
                   builder.getStringAttr(planName));
    issue->setAttr("avelang.qwen.core_staggered.region",
                   builder.getStringAttr(path.region));
    issue->setAttr("avelang.qwen.core_staggered.packet_start",
                   builder.getI64IntegerAttr(packet));
    issue->setAttr("avelang.qwen.core_staggered.group_start",
                   builder.getI64IntegerAttr(groupStart));
    issue->setAttr("avelang.qwen.core_staggered.group_size",
                   builder.getI64IntegerAttr(groupSize));
    issue->setAttr("avelang.qwen.core_staggered.issue_after_consumer_group",
                   builder.getI64IntegerAttr(issueAfterGroup));
    issue->setAttr("avelang.qwen.core_staggered.last_use_boundary",
                   builder.getStringAttr(boundary));
    issue->setAttr("avelang.qwen.core_staggered.register_distance",
                   builder.getI64IntegerAttr(0));
    if (deferredBlockDot) {
        issue->setAttr("avelang.qwen.core_staggered.deferred_block_dot",
                       mlir::UnitAttr::get(builder.getContext()));
    }
    return issue;
}

void createCoreLastUseCommit(mlir::OpBuilder &builder, CoreLastUsePacket packet,
                             llvm::StringRef planName) {
    auto commit = AMDGPUQwenK64CoreCommitOp::create(
        builder, packet.path->commit.getLoc(), packet.issue.getPacket(),
        packet.path->stage.getThreadId(), packet.path->stage.getKHalf(),
        packet.path->commit.getSharedKBank());
    commit->setAttrs(packet.path->commit->getAttrs());
    commit->setAttr("avelang.qwen.core_staggered.plan",
                    builder.getStringAttr(planName));
    commit->setAttr("avelang.qwen.core_staggered.region",
                    builder.getStringAttr(packet.path->region));
    commit->setAttr("avelang.qwen.core_staggered.packet_start",
                    builder.getI64IntegerAttr(packet.packet));
    commit->setAttr("avelang.qwen.core_staggered.commit_boundary",
                    builder.getStringAttr("after_existing_core_tail_barrier"));
}

bool scheduleJointV4CoreLastUse(AMDGPUQwenPersistentRecurrenceOp recurrence,
                                 const ParsedCoreLastUsePlan &plan) {
    if (!scheduleJointV4TailIssueControl(recurrence)) {
        return false;
    }

    llvm::SmallVector<AMDGPUQwenK64PipelineStageLoadOp> loads;
    llvm::SmallVector<AMDGPUQwenK64PipelineStageCommitOp> commits;
    recurrence.walk([&](AMDGPUQwenK64PipelineStageLoadOp op) {
        loads.push_back(op);
    });
    recurrence.walk([&](AMDGPUQwenK64PipelineStageCommitOp op) {
        commits.push_back(op);
    });
    if (loads.size() != 8 || commits.size() != 8) {
        recurrence.emitError(
            "core-last-use schedule requires the exact four R4 next W/K paths");
        return false;
    }

    auto *loopBlock = loads[4]->getBlock();
    llvm::SmallVector<CoreLastUsePacketPath> paths;
    constexpr std::array<llvm::StringLiteral, 4> kRegions = {
        "w0", "w1", "k0", "k1"};
    for (unsigned index = 0; index != 4; ++index) {
        auto stage = loads[index + 4];
        auto commit = commits[index + 4];
        if (stage->getBlock() != loopBlock || commit->getBlock() != loopBlock) {
            recurrence.emitError(
                "core-last-use schedule requires all next paths in the recurrence chunk block");
            return false;
        }
        mlir::memref::StoreOp tokenStore;
        for (mlir::OpOperand &use : stage.getStageToken().getUses()) {
            auto store = mlir::dyn_cast<mlir::memref::StoreOp>(use.getOwner());
            if (!store || store.getValue() != stage.getStageToken() ||
                store->getBlock() != loopBlock || tokenStore) {
                recurrence.emitError(
                    "core-last-use schedule requires one R4 token store per next stage");
                return false;
            }
            tokenStore = store;
        }
        auto tokenLoad = mlir::dyn_cast_or_null<mlir::memref::LoadOp>(
            commit.getStageToken().getDefiningOp());
        if (!tokenStore || !tokenLoad || tokenLoad->getBlock() != loopBlock) {
            recurrence.emitError(
                "core-last-use schedule requires one R4 token load per next commit");
            return false;
        }
        paths.push_back({stage, tokenStore, tokenLoad, commit, kRegions[index],
                         index < 2, static_cast<int64_t>(index % 2)});
    }

    AMDGPUBlockDotBF16F32Op firstBlockDot;
    recurrence.walk([&](AMDGPUBlockDotBF16F32Op op) {
        if (!firstBlockDot) {
            firstBlockDot = op;
        }
    });
    if (!firstBlockDot) {
        recurrence.emitError("core-last-use schedule requires one update block-dot");
        return false;
    }
    auto updateLoop = firstBlockDot->getParentOfType<mlir::scf::ForOp>();
    if (!updateLoop || updateLoop->getBlock() != loopBlock ||
        !hasStaticTripCount(updateLoop, 2)) {
        recurrence.emitError(
            "core-last-use schedule requires the fixed two-half update loop");
        return false;
    }

    mlir::scf::ForOp predLoop;
    for (mlir::Operation &candidate : *loopBlock) {
        auto loop = mlir::dyn_cast<mlir::scf::ForOp>(&candidate);
        if (!loop || !hasStaticTripCount(loop, 2) ||
            !loop->isBeforeInBlock(updateLoop)) {
            continue;
        }
        bool hasPredGroups = false;
        loop.walk([&](mlir::scf::ForOp nested) {
            if (hasStaticTripCount(nested, 4)) {
                hasPredGroups = true;
            }
        });
        if (hasPredGroups) {
            predLoop = loop;
            break;
        }
    }
    if (!predLoop) {
        recurrence.emitError(
            "core-last-use schedule cannot find the two-half pred consumer loop");
        return false;
    }

    const int64_t delayedGroups = plan.schedule.issuePlacement ==
                                          QwenCoreLastUseIssuePlacement::Immediate
                                      ? 0
                                      : 1;
    llvm::SmallVector<CoreLastUsePacket> packets;
    auto emitGroupAtBoundary = [&](mlir::OpBuilder &builder,
                                   CoreLastUsePacketPath &path, int64_t boundary,
                                   int64_t packetsPerGroup,
                                   llvm::StringRef lastUse,
                                   bool deferredBlockDot) {
        for (int64_t groupStart = 0; groupStart < 4;
             groupStart += packetsPerGroup) {
            const int64_t groupEnd = groupStart + packetsPerGroup - 1;
            const int64_t issueAfter = std::min<int64_t>(
                3, groupEnd + delayedGroups);
            if (issueAfter != boundary) {
                continue;
            }
            for (int64_t packet = groupStart; packet <= groupEnd; ++packet) {
                auto issue = createCoreLastUseIssue(
                    builder, path, packet, groupStart, packetsPerGroup,
                    issueAfter, lastUse, plan.name, deferredBlockDot);
                packets.push_back({&path, packet, issue});
            }
        }
    };

    auto predIterations = unrollNoResultStaticFor(predLoop);
    if (!predIterations || predIterations->size() != 2) {
        recurrence.emitError(
            "core-last-use schedule cannot statically expose the two pred halves");
        return false;
    }
    for (auto &predIteration : *predIterations) {
        CoreLastUsePacketPath &path = paths[predIteration.inductionValue];
        mlir::scf::ForOp predGroupLoop;
        for (mlir::Operation *op : predIteration.operations) {
            if (auto candidate = mlir::dyn_cast<mlir::scf::ForOp>(op);
                candidate && hasStaticTripCount(candidate, 4)) {
                predGroupLoop = candidate;
                break;
            }
        }
        if (!predGroupLoop) {
            recurrence.emitError(
                "core-last-use schedule cannot expose four pred MFMA groups");
            return false;
        }
        auto groupIterations = unrollNoResultStaticFor(predGroupLoop);
        if (!groupIterations || groupIterations->size() != 4) {
            recurrence.emitError(
                "core-last-use schedule failed to unroll pred consumer groups");
            return false;
        }
        for (auto &groupIteration : *groupIterations) {
            mlir::OpBuilder builder(groupIteration.operations.back());
            builder.setInsertionPointAfter(groupIteration.operations.back());
            emitGroupAtBoundary(builder, path, groupIteration.inductionValue,
                                plan.schedule.wPacketsPerGroup,
                                "pred_mfma_group_last_use",
                                /*deferredBlockDot=*/false);
        }
    }

    auto updateIterations = unrollNoResultStaticFor(updateLoop);
    if (!updateIterations || updateIterations->size() != 2) {
        recurrence.emitError(
            "core-last-use schedule cannot statically expose the two update halves");
        return false;
    }
    for (auto &updateIteration : *updateIterations) {
        CoreLastUsePacketPath &path = paths[2 + updateIteration.inductionValue];
        AMDGPUBlockDotBF16F32Op blockDot;
        for (mlir::Operation *op : updateIteration.operations) {
            op->walk([&](AMDGPUBlockDotBF16F32Op candidate) {
                if (!blockDot) {
                    blockDot = candidate;
                }
            });
        }
        if (!blockDot || blockDot->getBlock() != loopBlock) {
            recurrence.emitError(
                "core-last-use schedule requires an unrolled direct update block-dot");
            return false;
        }
        mlir::OpBuilder builder(blockDot);
        builder.setInsertionPointAfter(blockDot);
        for (int64_t boundary = 0; boundary != 4; ++boundary) {
            emitGroupAtBoundary(builder, path, boundary,
                                plan.schedule.kPacketsPerGroup,
                                "update_mfma_group_last_use",
                                /*deferredBlockDot=*/true);
        }
    }

    if (packets.size() != 16) {
        recurrence.emitError("core-last-use schedule did not create 16 next W/K packets");
        return false;
    }
    auto tailBarrier = mlir::dyn_cast_or_null<mlir::gpu::BarrierOp>(
        paths.front().stage->getPrevNode());
    if (!tailBarrier) {
        recurrence.emitError(
            "core-last-use schedule requires the existing safe tail barrier");
        return false;
    }
    mlir::OpBuilder tailBuilder(tailBarrier);
    tailBuilder.setInsertionPointAfter(tailBarrier);
    for (auto packet : packets) {
        createCoreLastUseCommit(tailBuilder, packet, plan.name);
    }

    for (auto &path : paths) {
        path.commit.erase();
        path.tokenLoad.erase();
        path.tokenStore.erase();
        path.stage.erase();
    }
    recurrence->setAttr("avelang.qwen.core_staggered.plan",
                        mlir::StringAttr::get(recurrence.getContext(), plan.name));
    recurrence->setAttr("avelang.qwen.core_staggered.single_core",
                        mlir::UnitAttr::get(recurrence.getContext()));
    recurrence->setAttr("avelang.qwen.core_staggered.cross_wave_lifetime",
                        mlir::StringAttr::get(
                            recurrence.getContext(),
                            "no_lds_write_before_existing_tail_barrier"));
    return true;
}

class FormQwenPersistentRecurrencePass
    : public mlir::PassWrapper<FormQwenPersistentRecurrencePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FormQwenPersistentRecurrencePass)

    llvm::StringRef getArgument() const final {
        return "form-qwen-persistent-recurrence";
    }

    llvm::StringRef getDescription() const final {
        return "Form a persistent Qwen recurrence region from R0 delimiters";
    }

    void runOnOperation() override {
        auto func = getOperation();
        llvm::SmallVector<AMDGPUQwenPersistentRecurrenceOp> begins;
        func.walk([&](AMDGPUQwenPersistentRecurrenceOp op) {
            begins.push_back(op);
        });
        if (begins.empty()) {
            return;
        }

        for (auto begin : begins) {
            if (!begin->hasAttr("avelang.qwen.persistent_recurrence.frontend_marker")) {
                begin.emitError("persistent recurrence region was not formed by the R0 frontend");
                signalPassFailure();
                return;
            }
            if (begin.getBody().empty() || !llvm::hasSingleElement(begin.getBody()) ||
                !llvm::hasSingleElement(begin.getBody().front()) ||
                !mlir::isa<AMDGPUQwenPersistentRecurrenceYieldOp>(
                    begin.getBody().front().back())) {
                begin.emitError("persistent recurrence frontend region must contain only its yield");
                signalPassFailure();
                return;
            }

            auto *parent = begin->getBlock();
            if (!parent) {
                begin.emitError("persistent recurrence must be in a block");
                signalPassFailure();
                return;
            }
            auto iter = std::next(mlir::Block::iterator(begin));
            AMDGPUQwenPersistentRecurrenceEndOp end;
            for (; iter != parent->end(); ++iter) {
                if (auto candidate = mlir::dyn_cast<AMDGPUQwenPersistentRecurrenceEndOp>(&*iter)) {
                    end = candidate;
                    break;
                }
            }
            if (!end) {
                begin.emitError("missing qwen_persistent_recurrence_end delimiter");
                signalPassFailure();
                return;
            }

            llvm::SmallVector<mlir::Operation *> bodyOps;
            for (auto bodyIter = std::next(mlir::Block::iterator(begin));
                 &*bodyIter != end.getOperation(); ++bodyIter) {
                bodyOps.push_back(&*bodyIter);
            }
            if (bodyOps.empty()) {
                begin.emitError("persistent recurrence body is empty");
                signalPassFailure();
                return;
            }
            auto *body = &begin.getBody().front();
            body->back().erase();
            for (auto *bodyOp : bodyOps) {
                bodyOp->moveBefore(body, body->end());
            }
            mlir::OpBuilder bodyBuilder(body, body->end());
            AMDGPUQwenPersistentRecurrenceYieldOp::create(bodyBuilder,
                                                           begin.getLoc());
            end.erase();
            begin->removeAttr("avelang.qwen.persistent_recurrence.frontend_marker");
            begin->setAttr("avelang.amdgpu.recurrence_plan",
                           makePlan(&getContext(), QwenRecurrenceScheduleKind::LegacyB0));
            begin->setAttr("avelang.qwen.persistent_recurrence.formed",
                           mlir::UnitAttr::get(&getContext()));
        }
    }
};

class PlanQwenPersistentRecurrencePass
    : public mlir::PassWrapper<PlanQwenPersistentRecurrencePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanQwenPersistentRecurrencePass)

    llvm::StringRef getArgument() const final {
        return "plan-qwen-persistent-recurrence";
    }

    llvm::StringRef getDescription() const final {
        return "Attach one validated target plan to a complete persistent Qwen recurrence";
    }

    void runOnOperation() override {
        const auto mode = llvm::sys::Process::GetEnv(
                              "AVELANG_PERSISTENT_RECURRENCE_LOWERING")
                              .value_or("legacy_b0");
        if (mode == "legacy_b0") {
            return;
        }
        const bool jointV1 = mode == "gfx942_bt64_bv32_joint_v1";
        const bool jointV2 = mode == "gfx942_bt64_bv32_joint_v2";
        const bool jointV3 = mode == "gfx942_bt64_bv32_joint_v3";
        const bool jointV4 = mode == "gfx942_bt64_bv32_joint_v4";
        const bool tailIssueControl =
            mode == "gfx942_bt64_bv32_joint_v4_tail_issue";
        const auto bvConsumeTailIssue = parseBvConsumeTailIssuePlan(mode);
        const bool jointV5 = mode == "gfx942_bt64_bv32_joint_v5";
        const bool softwarePipeline = mode == "gfx942_bt64_bv32_software_pipeline";
        const auto microtile = parseMicrotilePlan(mode);
        const auto coreLastUse = parseCoreLastUsePlan(mode);
        if (!jointV1 && !jointV2 && !jointV3 && !jointV4 && !tailIssueControl && !bvConsumeTailIssue && !jointV5 &&
            !softwarePipeline && !microtile && !coreLastUse) {
            getOperation().emitError()
                << "unsupported persistent recurrence plan mode " << mode;
            signalPassFailure();
            return;
        }

        bool foundPersistent = false;
        bool planned = false;
        getOperation().walk([&](AMDGPUQwenPersistentRecurrenceOp op) {
            foundPersistent = true;
            if (!op->hasAttr("avelang.qwen.persistent_recurrence.formed")) {
                op.emitError("joint planner requires a formed persistent recurrence");
                signalPassFailure();
                return;
            }
            if (op.getBody().empty() || !llvm::hasSingleElement(op.getBody())) {
                op.emitError("joint planner requires one recurrence body block");
                signalPassFailure();
                return;
            }
            // The pipeline reuses R4's typed producer and LDS-mediated K
            // retile attributes. Its scheduler owns loop structure separately.
            if (!tagJointStages(op, jointV2, jointV3,
                                jointV4 || tailIssueControl || bvConsumeTailIssue || softwarePipeline || microtile || coreLastUse,
                                jointV5)) {
                signalPassFailure();
                return;
            }
            if (jointV5 && !scheduleJointV5Superblock(op)) {
                signalPassFailure();
                return;
            }
            if ((tailIssueControl || bvConsumeTailIssue) &&
                !scheduleJointV4TailIssueControl(op)) {
                signalPassFailure();
                return;
            }
            if (microtile && !scheduleJointV4Microtiles(op, *microtile)) {
                signalPassFailure();
                return;
            }
            if (coreLastUse && !scheduleJointV4CoreLastUse(op, *coreLastUse)) {
                signalPassFailure();
                return;
            }
            // The entire body is the scheduling unit.  The plan records the
            // pred/BF16-boundary/update/feedback contract without lowering an
            // individual phase first.
            op->setAttr("avelang.amdgpu.recurrence_plan",
                        makePlan(&getContext(),
                                         softwarePipeline ? QwenRecurrenceScheduleKind::Gfx942Bt64Bv32SoftwarePipeline
                                         : jointV5 ? QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV5
                                         : (microtile || coreLastUse) ? QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV4
                                         : (jointV4 || tailIssueControl)
                                               ? QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV4
                                         : jointV3 ? QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV3
                                         : jointV2 ? QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV2
                                                   : QwenRecurrenceScheduleKind::Gfx942Bt64Bv32JointV1));
            if (microtile) {
                op->setAttr("avelang.amdgpu.recurrence_plan",
                            makeMicrotilePlan(&getContext(), *microtile));
            }
            if (coreLastUse) {
                op->setAttr("avelang.amdgpu.recurrence_plan",
                            makeCoreLastUsePlan(&getContext(), *coreLastUse));
            }
            if (bvConsumeTailIssue) {
                op->setAttr("avelang.amdgpu.recurrence_plan",
                            makeBvConsumeTailIssuePlan(&getContext(),
                                                        *bvConsumeTailIssue,
                                                        mode));
                op->setAttr("avelang.qwen.bv_consume",
                            mlir::IntegerAttr::get(
                                mlir::IntegerType::get(&getContext(), 64),
                                *bvConsumeTailIssue));
            }
            op->setAttr(softwarePipeline
                                ? "avelang.qwen.persistent_recurrence.software_pipeline_planned"
                                : coreLastUse ? "avelang.qwen.persistent_recurrence.core_lastuse_planned"
                                : microtile ? "avelang.qwen.persistent_recurrence.microtile_planned"
                                : jointV5 ? "avelang.qwen.persistent_recurrence.joint_v5_planned"
                                : (jointV4 || tailIssueControl || bvConsumeTailIssue)
                                      ? "avelang.qwen.persistent_recurrence.joint_v4_planned"
                                : jointV3 ? "avelang.qwen.persistent_recurrence.joint_v3_planned"
                                : jointV2 ? "avelang.qwen.persistent_recurrence.joint_v2_planned"
                                          : "avelang.qwen.persistent_recurrence.joint_v1_planned",
                        mlir::UnitAttr::get(&getContext()));
            planned = true;
        });
        if (foundPersistent && !planned) {
            getOperation().emitError("joint planner found no persistent recurrence");
            signalPassFailure();
        }
    }
};

class LegacyB0PersistentRecurrenceLowering
    : public mlir::OpRewritePattern<AMDGPUQwenPersistentRecurrenceOp> {
  public:
    using mlir::OpRewritePattern<AMDGPUQwenPersistentRecurrenceOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(AMDGPUQwenPersistentRecurrenceOp op,
                    mlir::PatternRewriter &rewriter) const override {
        if (!op->hasAttr("avelang.qwen.persistent_recurrence.formed")) {
            return rewriter.notifyMatchFailure(op, "missing formation marker");
        }
        if (op.getBody().empty() || !llvm::hasSingleElement(op.getBody())) {
            return rewriter.notifyMatchFailure(op, "expected one formed body block");
        }
        auto &body = op.getBody().front();
        if (!body.getArguments().empty()) {
            return rewriter.notifyMatchFailure(op, "legacy_b0 body cannot have block arguments");
        }
        if (!mlir::isa<AMDGPUQwenPersistentRecurrenceYieldOp>(body.back())) {
            return rewriter.notifyMatchFailure(op, "missing region yield");
        }
        body.back().erase();
        // This is intentionally a structural lowering: the region is the
        // validated B0 body and is moved back verbatim before block-dot late
        // lowering. No schedule, memory layout, or arithmetic is changed.
        rewriter.inlineBlockBefore(&body, op, mlir::ValueRange{});
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

class LowerQwenPersistentRecurrencePass
    : public mlir::PassWrapper<LowerQwenPersistentRecurrencePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerQwenPersistentRecurrencePass)

    llvm::StringRef getArgument() const final {
        return "lower-qwen-persistent-recurrence";
    }

    llvm::StringRef getDescription() const final {
        return "Lower a persistent Qwen recurrence semantic region";
    }

    void runOnOperation() override {
        bool hasPersistent = false;
        bool hasEndDelimiter = false;
        getOperation().walk([&](AMDGPUQwenPersistentRecurrenceOp) {
            hasPersistent = true;
        });
        getOperation().walk([&](AMDGPUQwenPersistentRecurrenceEndOp) {
            hasEndDelimiter = true;
        });
        if (hasEndDelimiter) {
            getOperation().emitError("unformed qwen persistent recurrence end delimiter");
            signalPassFailure();
            return;
        }
        if (!hasPersistent) {
            return;
        }

        const auto mode = llvm::sys::Process::GetEnv(
                              "AVELANG_PERSISTENT_RECURRENCE_LOWERING")
                              .value_or("legacy_b0");
        if (mode != "legacy_b0" && mode != "gfx942_bt64_bv32_joint_v1" &&
            mode != "gfx942_bt64_bv32_joint_v2" &&
            mode != "gfx942_bt64_bv32_joint_v3" &&
            mode != "gfx942_bt64_bv32_joint_v4" &&
            mode != "gfx942_bt64_bv32_joint_v4_tail_issue" &&
            mode != "gfx942_bt64_bv32_joint_v5" &&
            mode != "gfx942_bt64_bv32_software_pipeline" &&
            !parseMicrotilePlan(mode) && !parseCoreLastUsePlan(mode) &&
            !parseBvConsumeTailIssuePlan(mode)) {
            getOperation().emitError()
                << "AVELANG_PERSISTENT_RECURRENCE_LOWERING must be legacy_b0, got "
                << mode;
            signalPassFailure();
            return;
        }
        if (mode == "gfx942_bt64_bv32_joint_v1" || mode == "gfx942_bt64_bv32_joint_v2" ||
            mode == "gfx942_bt64_bv32_joint_v3" || mode == "gfx942_bt64_bv32_joint_v4" ||
            mode == "gfx942_bt64_bv32_joint_v5" ||
            mode == "gfx942_bt64_bv32_software_pipeline" || parseMicrotilePlan(mode) ||
            parseCoreLastUsePlan(mode)) {
            bool missingPlan = false;
            getOperation().walk([&](AMDGPUQwenPersistentRecurrenceOp op) {
                const auto marker = mode == "gfx942_bt64_bv32_software_pipeline"
                                        ? "avelang.qwen.persistent_recurrence.software_pipeline_planned"
                                        : parseMicrotilePlan(mode)
                                        ? "avelang.qwen.persistent_recurrence.microtile_planned"
                                        : parseCoreLastUsePlan(mode)
                                        ? "avelang.qwen.persistent_recurrence.core_lastuse_planned"
                                        : mode == "gfx942_bt64_bv32_joint_v5"
                                        ? "avelang.qwen.persistent_recurrence.joint_v5_planned"
                                        : mode == "gfx942_bt64_bv32_joint_v4"
                                        ? "avelang.qwen.persistent_recurrence.joint_v4_planned"
                                        : mode == "gfx942_bt64_bv32_joint_v3"
                                        ? "avelang.qwen.persistent_recurrence.joint_v3_planned"
                                        : mode == "gfx942_bt64_bv32_joint_v2"
                                              ? "avelang.qwen.persistent_recurrence.joint_v2_planned"
                                              : "avelang.qwen.persistent_recurrence.joint_v1_planned";
                if (!op->hasAttr(marker)) {
                    op.emitError("joint lowering requires the matching planner marker");
                    missingPlan = true;
                }
            });
            if (missingPlan) {
                signalPassFailure();
                return;
            }
        }
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<LegacyB0PersistentRecurrenceLowering>(&getContext());
        if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                     std::move(patterns)))) {
            signalPassFailure();
            return;
        }
        bool remaining = false;
        getOperation().walk([&](AMDGPUQwenPersistentRecurrenceOp op) {
            op.emitError("persistent recurrence survived legacy_b0 lowering");
            remaining = true;
        });
        if (remaining) {
            signalPassFailure();
            return;
        }
        getOperation()->setAttr("avelang.qwen.persistent_recurrence.lowering",
                                mlir::StringAttr::get(&getContext(), mode));
        llvm::errs() << "[qwen-persistent-recurrence] mode=" << mode
                     << (mode == "legacy_b0"
                             ? " inlined the legacy B0 semantic region before block-dot lowering\n"
                             : " lowered a complete joint recurrence region before block-dot lowering\n");
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createFormQwenPersistentRecurrencePass() {
    return std::make_unique<FormQwenPersistentRecurrencePass>();
}

std::unique_ptr<mlir::Pass> createPlanQwenPersistentRecurrencePass() {
    return std::make_unique<PlanQwenPersistentRecurrencePass>();
}

std::unique_ptr<mlir::Pass> createLowerQwenPersistentRecurrencePass() {
    return std::make_unique<LowerQwenPersistentRecurrencePass>();
}

} // namespace causalflow::avelang::dialect
