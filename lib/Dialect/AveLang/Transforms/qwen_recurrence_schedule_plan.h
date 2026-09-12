#pragma once

#include <cstdint>
#include <string_view>

namespace causalflow::avelang::dialect {

// R0 keeps this target-facing description deliberately small.  It is not a
// Qwen schedule implementation and does not encode lane/LDS addresses.  R1
// will consume one plan for the complete recurrence rather than independent
// pred and update plans.
enum class QwenDistributedTileEncoding {
    Deferred,
    Bv32TwoWave,
};

enum class QwenSharedEncoding {
    Deferred,
    LegacyB0,
    JointV1SingleWkBank,
    JointV2RotatingTypedBank,
    JointV3FullTypedRotatingBank,
    JointV4LdsMediatedRetileBank,
    JointV5SuperblockBank,
};

enum class QwenDotOperandEncoding {
    Deferred,
    TypedBf16x8,
    RotatingSharedTypedDot,
    FullTypedRotatingDot,
    LdsMediatedRetileDot,
    SuperblockLdsMediatedRetileDot,
};

enum class QwenRecurrenceScheduleKind {
    LegacyB0,
    Gfx942Bt64Bv32JointV1,
    Gfx942Bt64Bv32JointV2,
    Gfx942Bt64Bv32JointV3,
    Gfx942Bt64Bv32JointV4,
    Gfx942Bt64Bv32JointV5,
    Gfx942Bt64Bv32SoftwarePipeline,
};

enum class QwenNextChunkStagePolicy {
    None,
    CompilerOwnedDeferred,
    OneChunkAheadTailCommit,
    OneChunkAheadInterleavedTailCommit,
};

// An experimental schedule is still one complete recurrence plan.  These
// fields deliberately describe ownership and lifetime at the granularity at
// which the late packet lowering can act: one BF16x8 packet per lane.  They do
// not introduce another W/K LDS allocation or another recurrence core.
enum class QwenMicrotileIssuePlacement {
    Tail,
    LastUseStaggered,
};

// Core-last-use plans are separate from the retired full-core d1 experiment:
// the issue site is one of the real MFMA consumer groups, while the LDS
// commit remains at the existing tail boundary.
enum class QwenCoreLastUseIssuePlacement {
    Immediate,
    DelayedOneConsumerGroup,
};

struct QwenCoreLastUseSchedulePlan {
    int64_t wPacketsPerGroup = 1;
    int64_t kPacketsPerGroup = 1;
    QwenCoreLastUseIssuePlacement issuePlacement =
        QwenCoreLastUseIssuePlacement::Immediate;
    std::string_view currentConsumerOrder =
        "w_pred_group[0..3]->bf16_vnew_vdecay->k_update_group[0..3]->fp32_feedback";
    std::string_view wLastUse = "pred_mfma_group[p]";
    std::string_view kLastUse = "update_mfma_group[p]";
    std::string_view sameRegionCommit = "after_existing_core_tail_barrier";
};

struct QwenMicrotileSchedulePlan {
    int64_t wPacketsPerGroup = 4;
    int64_t kPacketsPerGroup = 4;
    QwenMicrotileIssuePlacement issuePlacement =
        QwenMicrotileIssuePlacement::Tail;
    int64_t vgprResidentMicroGroups = 0;
    // The existing full-recurrence op exposes one opaque pred/update core.
    // This order is semantic and must remain coupled to the physical LDS
    // last-use boundary below.
    std::string_view currentConsumerOrder =
        "w0,w1->pred->bf16_vnew_vdecay->k0,k1->update->fp32_feedback";
    std::string_view ldsRegionLastUse =
        "w0,w1,k0,k1:single_core_exit";
    std::string_view sameRegionCommit = "after_core_exit_existing_barrier";
};

struct QwenRecurrenceSchedulePlan {
    std::string_view target = "gfx942";
    int64_t bt = 64;
    int64_t bv = 32;
    int64_t workgroupSize = 128;
    int64_t waves = 2;
    std::string_view stateType = "f32";
    std::string_view boundaryType = "bf16";
    QwenDistributedTileEncoding distributedTile =
        QwenDistributedTileEncoding::Deferred;
    QwenSharedEncoding shared = QwenSharedEncoding::Deferred;
    QwenDotOperandEncoding dotOperand =
        QwenDotOperandEncoding::Deferred;
    bool supportsPipelineStageToken = true;
    QwenRecurrenceScheduleKind kind = QwenRecurrenceScheduleKind::LegacyB0;
    QwenNextChunkStagePolicy nextChunkStage = QwenNextChunkStagePolicy::None;
};

} // namespace causalflow::avelang::dialect
