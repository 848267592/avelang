#include "lower_qwen_k64_pipeline_stage_pass.h"

#include "Dialect/AveLang/IR/AveLangOps.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Process.h>

#include <array>
#include <optional>
#include <string>

namespace causalflow::avelang::dialect {
namespace {

constexpr int64_t kWorkgroupSize = 128;
constexpr int64_t kPacketsPerLane = 4;
constexpr int64_t kPacketElements = 8;

bool usesBv64FourWaveProducer() {
    return llvm::sys::Process::GetEnv("AVELANG_PERSISTENT_RECURRENCE_LOWERING") ==
           std::optional<std::string>("gfx942_bt64_bv64_joint_v4_tail_issue");
}

bool useDistributedPlacement() {
    return llvm::sys::Process::GetEnv("AVELANG_QWEN_K64_PIPELINE_LOWERING") !=
           std::optional<std::string>("immediate");
}

bool useJointV1Placement(AMDGPUQwenK64PipelineStageLoadOp stage) {
    auto placement = stage->getAttrOfType<mlir::StringAttr>(
        "avelang.qwen.joint_v1.placement");
    return placement && placement.getValue() == "distributed";
}

bool useJointV2Placement(AMDGPUQwenK64PipelineStageLoadOp stage) {
    auto placement = stage->getAttrOfType<mlir::StringAttr>(
        "avelang.qwen.joint_v2.placement");
    return placement && placement.getValue() == "distributed_register_packet";
}

bool useJointV3Placement(AMDGPUQwenK64PipelineStageLoadOp stage) {
    auto placement = stage->getAttrOfType<mlir::StringAttr>(
        "avelang.qwen.joint_v3.placement");
    return placement && placement.getValue() == "distributed_register_packet";
}

bool useJointV4Placement(AMDGPUQwenK64PipelineStageLoadOp stage) {
    auto placement = stage->getAttrOfType<mlir::StringAttr>(
        "avelang.qwen.joint_v4.placement");
    return placement && placement.getValue() == "distributed_register_packet";
}

bool useJointV5Placement(AMDGPUQwenK64PipelineStageLoadOp stage) {
    auto placement = stage->getAttrOfType<mlir::StringAttr>(
        "avelang.qwen.joint_v5.placement");
    return placement && placement.getValue() == "distributed_register_packet";
}

bool isJointWOperand(AMDGPUQwenK64PipelineStageLoadOp stage) {
    auto operand = stage->getAttrOfType<mlir::StringAttr>(
        "avelang.qwen.joint_v1.operand");
    if (operand && operand.getValue() == "w") {
        return true;
    }
    operand = stage->getAttrOfType<mlir::StringAttr>(
        "avelang.qwen.joint_v2.operand");
    if (operand && operand.getValue() == "w") {
        return true;
    }
    operand = stage->getAttrOfType<mlir::StringAttr>(
        "avelang.qwen.joint_v3.operand");
    if (operand && operand.getValue() == "w") {
        return true;
    }
    operand = stage->getAttrOfType<mlir::StringAttr>(
        "avelang.qwen.joint_v4.operand");
    if (operand && operand.getValue() == "w") {
        return true;
    }
    operand = stage->getAttrOfType<mlir::StringAttr>(
        "avelang.qwen.joint_v5.operand");
    return operand && operand.getValue() == "w";
}

bool isJointV2(AMDGPUQwenK64PipelineStageLoadOp stage) {
    return stage->hasAttr("avelang.qwen.joint_v2.shared_encoding");
}

bool isJointV3(AMDGPUQwenK64PipelineStageLoadOp stage) {
    return stage->hasAttr("avelang.qwen.joint_v3.shared_encoding");
}

bool isJointV4(AMDGPUQwenK64PipelineStageLoadOp stage) {
    return stage->hasAttr("avelang.qwen.joint_v4.shared_encoding");
}

bool isJointV5(AMDGPUQwenK64PipelineStageLoadOp stage) {
    return stage->hasAttr("avelang.qwen.joint_v5.shared_encoding");
}

void copyJointStageAttrs(mlir::Operation *from, mlir::Operation *to) {
    for (llvm::StringRef name : {"avelang.qwen.software_pipeline.role",
                                 "avelang.qwen.software_pipeline.part",
                                 "avelang.qwen.software_pipeline.stage",
                                 "avelang.qwen.software_pipeline.distance",
                                 "avelang.qwen.software_pipeline.slot"}) {
        if (auto attr = from->getAttr(name)) {
            to->setAttr(name, attr);
        }
    }
    for (llvm::StringRef name : {"avelang.qwen.joint_v1.operand",
                                 "avelang.qwen.joint_v1.stage",
                                 "avelang.qwen.joint_v2.operand",
                                 "avelang.qwen.joint_v2.stage",
                                 "avelang.qwen.joint_v2.shared_encoding",
                                 "avelang.qwen.joint_v2.dot_operand",
                                 "avelang.qwen.joint_v3.operand",
                                 "avelang.qwen.joint_v3.stage",
                                 "avelang.qwen.joint_v3.placement",
                                 "avelang.qwen.joint_v3.shared_encoding",
                                 "avelang.qwen.joint_v3.dot_operand",
                                 "avelang.qwen.joint_v3.producer_layout"}) {
        if (auto attr = from->getAttr(name)) {
            to->setAttr(name, attr);
        }
    }
    for (llvm::StringRef name : {"avelang.qwen.joint_v4.operand",
                                 "avelang.qwen.joint_v4.stage",
                                 "avelang.qwen.joint_v4.placement",
                                 "avelang.qwen.joint_v4.shared_encoding",
                                 "avelang.qwen.joint_v4.dot_operand",
                                 "avelang.qwen.joint_v4.producer_layout"}) {
        if (auto attr = from->getAttr(name)) {
            to->setAttr(name, attr);
        }
    }
    for (llvm::StringRef name : {"avelang.qwen.microtile.plan",
                                 "avelang.qwen.microtile.region",
                                 "avelang.qwen.microtile.packet_start",
                                 "avelang.qwen.microtile.packet_count",
                                 "avelang.qwen.microtile.issue",
                                 "avelang.qwen.microtile.vgpr_distance",
                                 "avelang.qwen.microtile.last_use",
                                 "avelang.qwen.microtile.commit_boundary"}) {
        if (auto attr = from->getAttr(name)) {
            to->setAttr(name, attr);
        }
    }
    for (llvm::StringRef name : {"avelang.qwen.joint_v5.operand",
                                 "avelang.qwen.joint_v5.stage",
                                 "avelang.qwen.joint_v5.placement",
                                 "avelang.qwen.joint_v5.shared_encoding",
                                 "avelang.qwen.joint_v5.dot_operand",
                                 "avelang.qwen.joint_v5.producer_layout",
                                 "avelang.qwen.joint_v5.superblock_issue"}) {
        if (auto attr = from->getAttr(name)) {
            to->setAttr(name, attr);
        }
    }
}

struct StageProducer {
    AMDGPUQwenK64PipelineStageLoadOp stage;
    llvm::SmallVector<mlir::Operation *> wrappers;
};

constexpr llvm::StringLiteral kSoftwarePipelineRoleAttr =
    "avelang.qwen.software_pipeline.role";
constexpr llvm::StringLiteral kSoftwarePipelinePartAttr =
    "avelang.qwen.software_pipeline.part";
constexpr llvm::StringLiteral kSoftwarePipelineSlotAttr =
    "avelang.qwen.software_pipeline.slot";
constexpr llvm::StringLiteral kLogicalLdsConsumerAttr =
    "avelang.qwen.software_pipeline.logical_lds_consumer";
constexpr llvm::StringLiteral kDirectLdsCommitAttr =
    "avelang.qwen.software_pipeline.direct_lds_commit";

bool isSoftwarePipelineStage(AMDGPUQwenK64PipelineStageLoadOp stage) {
    return stage->hasAttr(kSoftwarePipelineRoleAttr);
}

// The generic scheduler carries opaque stage tokens through an scf.for.  The
// late target lowering must turn that symbolic edge into an actual per-lane
// packet ring.  Static shapes may already have been unrolled at this point;
// retain the existing direct SSA lowering for those cases.
bool isModuloLoopStage(AMDGPUQwenK64PipelineStageLoadOp stage) {
    auto loop = stage->getParentOfType<mlir::scf::ForOp>();
    return loop && !loop.getInitArgs().empty();
}

mlir::Value indexConstant(mlir::PatternRewriter &rewriter, mlir::Location loc,
                           int64_t value);
llvm::SmallVector<mlir::Value>
emitPacketLoads(mlir::PatternRewriter &rewriter, mlir::Location loc,
                AMDGPUQwenK64PipelineStageLoadOp stage, bool distributed);

bool hasSameIndices(mlir::OperandRange lhs, mlir::OperandRange rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (auto [left, right] : llvm::zip(lhs, rhs)) {
        if (left != right) {
            return false;
        }
    }
    return true;
}

StageProducer findStageProducer(mlir::Value value) {
    StageProducer result;
    // A modulo-pipelined packet is intentionally represented as an scf.for
    // iter_arg/result pair. Follow that recurrence edge to the single static
    // stage producer. This is the missing connection in the old late pass,
    // which only understood frontend scalar spill/reload wrappers.
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
        auto *owner = argument.getOwner();
        if (auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(owner->getParentOp());
            loop && owner == loop.getBody() && argument.getArgNumber() > 0) {
            value = loop.getYieldedValues()[argument.getArgNumber() - 1];
        }
    }
    while (auto *producer = value.getDefiningOp()) {
        if (auto stage = mlir::dyn_cast<AMDGPUQwenK64PipelineStageLoadOp>(producer)) {
            result.stage = stage;
            return result;
        }
        if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(producer)) {
            if (cast.getInputs().size() == 1) {
                result.wrappers.push_back(cast);
                value = cast.getInputs().front();
                continue;
            }
        }
        if (auto cast = mlir::dyn_cast<mlir::arith::IndexCastOp>(producer)) {
            result.wrappers.push_back(cast);
            value = cast.getIn();
            continue;
        }
        if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(producer)) {
            auto result = mlir::cast<mlir::OpResult>(value);
            if (result.getResultNumber() < loop.getNumResults()) {
                value = loop.getYieldedValues()[result.getResultNumber()];
                continue;
            }
        }
        if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(producer)) {
            // The JIT represents a scalar local as a private memref store/load
            // pair. This is the only wrapper accepted for the opaque stage
            // token, and it is erased once the late pass reconnects the
            // producer with its unique commit.
            mlir::memref::StoreOp matchingStore;
            for (auto it = load->getIterator(); it != load->getBlock()->begin();) {
                --it;
                if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(&*it);
                    store && store.getMemRef() == load.getMemRef() &&
                    hasSameIndices(store.getIndices(), load.getIndices())) {
                    matchingStore = store;
                    break;
                }
            }
            if (matchingStore) {
                result.wrappers.push_back(load);
                result.wrappers.push_back(matchingStore);
                value = matchingStore.getValue();
                continue;
            }
        }
        break;
    }
    return result;
}

mlir::Value stagePacketStorage(mlir::PatternRewriter &rewriter,
                               AMDGPUQwenK64PipelineStageLoadOp stage) {
    auto packetType = mlir::VectorType::get({kPacketElements},
                                             rewriter.getBF16Type());
    auto privateSpace = mlir::gpu::AddressSpaceAttr::get(
        rewriter.getContext(), mlir::gpu::AddressSpace::Private);
    auto storageType = mlir::MemRefType::get(
        {kPacketsPerLane}, packetType, mlir::MemRefLayoutAttrInterface(),
        privateSpace);
    rewriter.setInsertionPoint(stage);
    return mlir::memref::AllocaOp::create(rewriter, stage.getLoc(), storageType,
                                           mlir::ValueRange{}, mlir::IntegerAttr{});
}

AMDGPUQwenK64PipelineStageLoadOp
findSoftwarePipelineInitialIssue(mlir::Operation *scope,
                                AMDGPUQwenK64PipelineStageLoadOp steadyStage) {
    auto slot = steadyStage->getAttrOfType<mlir::IntegerAttr>(
        kSoftwarePipelineSlotAttr);
    if (!slot) {
        return {};
    }
    AMDGPUQwenK64PipelineStageLoadOp initialIssue;
    scope->walk([&](AMDGPUQwenK64PipelineStageLoadOp candidate) {
        if (initialIssue || !isSoftwarePipelineStage(candidate) ||
            isModuloLoopStage(candidate)) {
            return;
        }
        auto part = candidate->getAttrOfType<mlir::StringAttr>(
            kSoftwarePipelinePartAttr);
        auto candidateSlot = candidate->getAttrOfType<mlir::IntegerAttr>(
            kSoftwarePipelineSlotAttr);
        // This is the cloned first-body issue that initializes the modulo
        // loop.  It is intentionally distinct from the original packet-0
        // `prologue` stage, which is lowered by the existing R4 path.
        if (part && candidateSlot &&
            part.getValue() == "steady_state_issue" &&
            candidateSlot.getInt() == slot.getInt()) {
            initialIssue = candidate;
        }
    });
    return initialIssue;
}

void emitSoftwarePipelineStage(mlir::PatternRewriter &rewriter,
                               AMDGPUQwenK64PipelineStageLoadOp stage,
                               mlir::Value storage) {
    rewriter.setInsertionPoint(stage);
    auto packets = emitPacketLoads(rewriter, stage.getLoc(), stage,
                                   /*distributed=*/true);
    for (auto [index, packet] : llvm::enumerate(packets)) {
        auto store = mlir::memref::StoreOp::create(
            rewriter, stage.getLoc(), packet, storage,
            mlir::ValueRange{indexConstant(rewriter, stage.getLoc(), index)});
        store->setAttr("avelang.qwen.software_pipeline.packet_ring",
                       rewriter.getStringAttr("per_lane_distance_1"));
    }
    auto tokenType = mlir::cast<mlir::IntegerType>(stage.getStageToken().getType());
    auto placeholder = mlir::arith::ConstantIntOp::create(
        rewriter, stage.getLoc(), 0, tokenType.getWidth());
    stage.getStageToken().replaceAllUsesWith(placeholder);
}

llvm::SmallVector<mlir::Value>
loadSoftwarePipelinePackets(mlir::PatternRewriter &rewriter, mlir::Location loc,
                            mlir::Value storage) {
    llvm::SmallVector<mlir::Value> packets;
    packets.reserve(kPacketsPerLane);
    for (int64_t index = 0; index < kPacketsPerLane; ++index) {
        packets.push_back(mlir::memref::LoadOp::create(
            rewriter, loc, storage,
            mlir::ValueRange{indexConstant(rewriter, loc, index)}));
    }
    return packets;
}

mlir::Value asIndex(mlir::PatternRewriter &rewriter, mlir::Location loc,
                    mlir::Value value) {
    if (value.getType().isIndex()) {
        return value;
    }
    return mlir::arith::IndexCastOp::create(rewriter, loc,
                                             rewriter.getIndexType(), value);
}

mlir::Value indexConstant(mlir::PatternRewriter &rewriter, mlir::Location loc,
                           int64_t value) {
    return mlir::arith::ConstantIndexOp::create(rewriter, loc, value);
}

mlir::Value indexAdd(mlir::PatternRewriter &rewriter, mlir::Location loc,
                     mlir::Value lhs, mlir::Value rhs) {
    return mlir::arith::AddIOp::create(rewriter, loc, lhs, rhs);
}

mlir::Value indexMul(mlir::PatternRewriter &rewriter, mlir::Location loc,
                     mlir::Value lhs, mlir::Value rhs) {
    return mlir::arith::MulIOp::create(rewriter, loc, lhs, rhs);
}

mlir::Value i32Constant(mlir::PatternRewriter &rewriter, mlir::Location loc,
                        int32_t value) {
    return mlir::arith::ConstantIntOp::create(rewriter, loc, value, 32);
}

mlir::Value asI32(mlir::PatternRewriter &rewriter, mlir::Location loc,
                  mlir::Value value) {
    if (value.getType() == rewriter.getI32Type()) {
        return value;
    }
    return mlir::arith::IndexCastOp::create(rewriter, loc, rewriter.getI32Type(),
                                             value);
}

struct PacketRange {
    int64_t start = 0;
    int64_t count = kPacketsPerLane;
};

PacketRange packetRange(AMDGPUQwenK64PipelineStageLoadOp stage) {
    auto start = stage->getAttrOfType<mlir::IntegerAttr>(
        "avelang.qwen.microtile.packet_start");
    auto count = stage->getAttrOfType<mlir::IntegerAttr>(
        "avelang.qwen.microtile.packet_count");
    if (!start && !count) {
        return {};
    }
    const int64_t first = start ? start.getInt() : 0;
    const int64_t size = count ? count.getInt() : kPacketsPerLane;
    // The recurrence planner validates these values before this late pass.
    // Keep an empty/default range from silently producing a partial tile if a
    // malformed external IR bypasses that planner.
    if (first < 0 || size <= 0 || first + size > kPacketsPerLane) {
        return {};
    }
    return {first, size};
}

/// Materialize four BF16x8 packets per lane.  Across 128 lanes this is the
/// exact 64 x 64 K half: packet = token * 8 + (K / 8).  The values deliberately
/// remain small vector SSA values rather than a source-level local K array.
llvm::SmallVector<mlir::Value>
emitPacketLoads(mlir::PatternRewriter &rewriter, mlir::Location loc,
                AMDGPUQwenK64PipelineStageLoadOp stage, bool distributed) {
    auto c0 = indexConstant(rewriter, loc, 0);
    auto c8 = indexConstant(rewriter, loc, kPacketElements);
    auto c64 = indexConstant(rewriter, loc, 64);
    auto tid = asIndex(rewriter, loc, stage.getThreadId());
    if (usesBv64FourWaveProducer()) {
        // The extra V32 pair reuses the validated 128-lane R4 packet owner.
        // Its waves may observe the packet, but must not become distinct
        // global producers.
        tid = mlir::arith::RemUIOp::create(
            rewriter, loc, tid, indexConstant(rewriter, loc, kWorkgroupSize));
    }
    auto chunkStart = asIndex(rewriter, loc, stage.getChunkStart());
    auto keyHead = asIndex(rewriter, loc, stage.getKeyHead());
    auto kHalf = asIndex(rewriter, loc, stage.getKHalf());
    auto bf16x8 = mlir::VectorType::get({kPacketElements}, rewriter.getBF16Type());

    llvm::SmallVector<mlir::Value> packets;
    const auto range = packetRange(stage);
    packets.reserve(range.count);
    for (int64_t packet = range.start; packet < range.start + range.count; ++packet) {
        auto linear = indexAdd(rewriter, loc, tid,
                               indexConstant(rewriter, loc,
                                             packet * kWorkgroupSize));
        auto tokenOffset = mlir::arith::DivUIOp::create(rewriter, loc, linear, c8);
        auto kPacket = mlir::arith::RemUIOp::create(rewriter, loc, linear, c8);
        auto token = indexAdd(rewriter, loc, chunkStart, tokenOffset);
        auto kColumn = indexAdd(
            rewriter, loc, indexMul(rewriter, loc, kHalf, c64),
            indexMul(rewriter, loc, kPacket, c8));
        auto load = mlir::vector::LoadOp::create(
            rewriter, loc, bf16x8, stage.getSourceK(),
            mlir::ValueRange{c0, token, keyHead, kColumn});
        load->setAttr("avelang.qwen.k64.pipeline.global_packet",
                      rewriter.getStringAttr(distributed ? "distributed" : "immediate"));
        copyJointStageAttrs(stage, load);
        packets.push_back(load.getResult());
    }
    return packets;
}

struct RotatingKPacket {
    mlir::Value value;
    mlir::Value row;
    mlir::Value tokenBase;
};

// R3 keeps the existing C0 consumer-major [half, K, token] shared contract,
// but changes the producer ownership before the shared write. Every lane
// loads eight contiguous K BF16 values for one token. The eight lanes in a
// subgroup then form one K-row x token-eight packet through a fixed IDX
// shuffle. This is the Triton-audited 8x8 producer formula, expressed from
// lane/group arithmetic rather than a table of element addresses.
llvm::SmallVector<RotatingKPacket>
emitJointV3RotatingKPackets(mlir::PatternRewriter &rewriter, mlir::Location loc,
                            AMDGPUQwenK64PipelineStageLoadOp stage) {
    auto c0 = indexConstant(rewriter, loc, 0);
    auto c2 = indexConstant(rewriter, loc, 2);
    auto c8 = indexConstant(rewriter, loc, 8);
    auto c64 = indexConstant(rewriter, loc, 64);
    auto originalTid = asIndex(rewriter, loc, stage.getThreadId());
    auto tid = originalTid;
    const bool ownerWaveOnly = usesBv64FourWaveProducer();
    if (ownerWaveOnly) {
        tid = mlir::arith::RemUIOp::create(
            rewriter, loc, tid, indexConstant(rewriter, loc, kWorkgroupSize));
    }
    auto chunkStart = asIndex(rewriter, loc, stage.getChunkStart());
    auto keyHead = asIndex(rewriter, loc, stage.getKeyHead());
    auto kHalf = asIndex(rewriter, loc, stage.getKHalf());
    auto lane64 = mlir::arith::RemUIOp::create(rewriter, loc, tid, c64);
    auto lane8 = mlir::arith::RemUIOp::create(rewriter, loc, lane64, c8);
    auto subgroup = mlir::arith::DivUIOp::create(rewriter, loc, lane64, c8);
    auto globalSubgroup = mlir::arith::DivUIOp::create(rewriter, loc, tid, c8);
    auto pair = mlir::arith::DivUIOp::create(rewriter, loc, lane8, c2);
    auto pairI32 = asI32(rewriter, loc, pair);
    auto laneParity = mlir::arith::RemUIOp::create(rewriter, loc, lane8, c2);
    auto isLow = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::eq, asI32(rewriter, loc, laneParity),
        i32Constant(rewriter, loc, 0));
    auto bf16x8 = mlir::VectorType::get({8}, rewriter.getBF16Type());
    auto i32x4 = mlir::VectorType::get({4}, rewriter.getI32Type());
    auto i32x1 = mlir::VectorType::get({1}, rewriter.getI32Type());
    auto bf16x2 = mlir::VectorType::get({2}, rewriter.getBF16Type());

    llvm::SmallVector<RotatingKPacket> packets;
    packets.reserve(kPacketsPerLane);
    for (int64_t pass = 0; pass < kPacketsPerLane; ++pass) {
        auto group = indexAdd(rewriter, loc, globalSubgroup,
                              indexConstant(rewriter, loc, pass * 16));
        auto featureBlock = mlir::arith::RemUIOp::create(rewriter, loc, group, c8);
        auto tokenBlock = mlir::arith::DivUIOp::create(rewriter, loc, group, c8);
        auto sourceToken = indexAdd(
            rewriter, loc,
            indexAdd(rewriter, loc, chunkStart, indexMul(rewriter, loc, tokenBlock, c8)),
            lane8);
        auto sourceColumn = indexAdd(
            rewriter, loc, indexMul(rewriter, loc, kHalf, c64),
            indexMul(rewriter, loc, featureBlock, c8));
        auto source = mlir::vector::LoadOp::create(
            rewriter, loc, bf16x8, stage.getSourceK(),
            mlir::ValueRange{c0, sourceToken, keyHead, sourceColumn});
        source->setAttr("avelang.qwen.joint_v3.typed_global_packet",
                        rewriter.getStringAttr("bf16x8"));
        source->setAttr("avelang.qwen.joint_v3.packet_mapping",
                        rewriter.getStringAttr("subgroup_8x8_token_to_krow"));
        copyJointStageAttrs(stage, source);

        auto words = mlir::vector::BitCastOp::create(rewriter, loc, i32x4,
                                                      source.getResult());
        std::array<mlir::Value, 4> wordValues;
        for (int64_t word = 0; word < 4; ++word) {
            wordValues[word] = mlir::vector::ExtractOp::create(rewriter, loc,
                                                                 words.getResult(), word);
        }
        llvm::SmallVector<mlir::Value> rowElements;
        rowElements.reserve(8);
        for (int64_t tokenInGroup = 0; tokenInGroup < 8; ++tokenInGroup) {
            auto sourceLane = indexAdd(
                rewriter, loc, indexMul(rewriter, loc, subgroup, c8),
                indexConstant(rewriter, loc, tokenInGroup));
            // A wave shuffle transports one scalar from a source lane. The
            // source lane owns a token, whereas the destination lane owns a
            // K row. Materialize four fixed word planes so the destination
            // selects its K-row pair *after* transport; selecting before the
            // shuffle would accidentally use the source token's lane id.
            mlir::Value selectedElement;
            for (int64_t word = 0; word < 4; ++word) {
                auto shuffled = mlir::gpu::ShuffleOp::create(
                    rewriter, loc, wordValues[word], asI32(rewriter, loc, sourceLane),
                    i32Constant(rewriter, loc, 64), mlir::gpu::ShuffleMode::IDX);
                shuffled->setAttr("avelang.qwen.joint_v3.cross_lane_transpose",
                                  rewriter.getStringAttr("idx_subgroup_8x8"));
                auto wordVector = mlir::vector::FromElementsOp::create(
                    rewriter, loc, i32x1,
                    mlir::ValueRange{shuffled.getShuffleResult()});
                auto pairValue = mlir::vector::BitCastOp::create(
                    rewriter, loc, bf16x2, wordVector.getResult());
                auto low = mlir::vector::ExtractOp::create(rewriter, loc,
                                                            pairValue.getResult(), 0);
                auto high = mlir::vector::ExtractOp::create(rewriter, loc,
                                                             pairValue.getResult(), 1);
                auto candidate = mlir::arith::SelectOp::create(rewriter, loc, isLow,
                                                                low, high);
                if (word == 0) {
                    selectedElement = candidate;
                } else {
                    auto selectWord = mlir::arith::CmpIOp::create(
                        rewriter, loc, mlir::arith::CmpIPredicate::eq, pairI32,
                        i32Constant(rewriter, loc, word));
                    selectedElement = mlir::arith::SelectOp::create(
                        rewriter, loc, selectWord, candidate, selectedElement);
                }
            }
            rowElements.push_back(selectedElement);
        }
        auto rowPacket = mlir::vector::FromElementsOp::create(rewriter, loc, bf16x8,
                                                               rowElements);
        rowPacket->setAttr("avelang.qwen.joint_v3.typed_register_tile",
                           rewriter.getStringAttr("bf16x8_krow"));
        packets.push_back({rowPacket.getResult(),
                           indexAdd(rewriter, loc,
                                    indexMul(rewriter, loc, featureBlock, c8), lane8),
                           indexMul(rewriter, loc, tokenBlock, c8)});
    }
    return packets;
}

void emitJointV3RotatingKCommit(mlir::PatternRewriter &rewriter,
                                mlir::Location loc,
                                AMDGPUQwenK64PipelineStageLoadOp stage,
                                AMDGPUQwenK64PipelineStageCommitOp commit,
                                llvm::ArrayRef<RotatingKPacket> packets) {
    auto kHalf = asIndex(rewriter, loc, stage.getKHalf());
    for (const auto &packet : packets) {
        auto store = mlir::vector::StoreOp::create(
            rewriter, loc, packet.value, commit.getSharedKBank(),
            mlir::ValueRange{kHalf, packet.row, packet.tokenBase});
        store->setAttr("avelang.qwen.joint_v3.typed_lds_packet",
                       rewriter.getStringAttr("bf16x8_krow"));
        store->setAttr("avelang.qwen.joint_v3.shared_layout",
                       rewriter.getStringAttr("rotating_k_major_dot_operand"));
        copyJointStageAttrs(stage, store);
    }
}

void emitPacketCommit(mlir::PatternRewriter &rewriter, mlir::Location loc,
                      AMDGPUQwenK64PipelineStageLoadOp stage,
                      AMDGPUQwenK64PipelineStageCommitOp commit,
                      llvm::ArrayRef<mlir::Value> packets, bool distributed) {
    auto c8 = indexConstant(rewriter, loc, kPacketElements);
    auto originalTid = asIndex(rewriter, loc, stage.getThreadId());
    auto tid = originalTid;
    const bool ownerWaveOnly = usesBv64FourWaveProducer();
    if (ownerWaveOnly) {
        tid = mlir::arith::RemUIOp::create(
            rewriter, loc, tid, indexConstant(rewriter, loc, kWorkgroupSize));
    }
    auto kHalf = asIndex(rewriter, loc, stage.getKHalf());
    const bool wOperand = isJointWOperand(stage);
    const bool jointV2 = isJointV2(stage);
    const bool jointV3 = isJointV3(stage);
    const bool jointV4 = isJointV4(stage);
    const bool jointV5 = isJointV5(stage);
    const auto range = packetRange(stage);
    for (auto [relativePacket, packetValue] : llvm::enumerate(packets)) {
        const int64_t packet = range.start + static_cast<int64_t>(relativePacket);
        auto linear = indexAdd(rewriter, loc, tid,
                               indexConstant(rewriter, loc,
                                             packet * kWorkgroupSize));
        auto tokenOffset = mlir::arith::DivUIOp::create(rewriter, loc, linear, c8);
        auto kPacket = mlir::arith::RemUIOp::create(rewriter, loc, linear, c8);
        // R2 preserves the same logical W/K elements but writes a W packet as
        // one typed LDS vector. K still uses the K-major consumer bank; its
        // ownership changes through the joint schedule rather than a hidden
        // source transpose. This lets the later dot lowering see a typed W
        // packet and a typed preloaded-K operand in the same plan.
        if ((jointV2 || jointV3 || jointV4 || jointV5) && (wOperand || jointV4 || jointV5)) {
            auto kBase = indexMul(rewriter, loc, kPacket, c8);
            auto emitStore = [&]() {
                auto store = mlir::vector::StoreOp::create(
                    rewriter, loc, packetValue, commit.getSharedKBank(),
                    mlir::ValueRange{kHalf, tokenOffset, kBase});
                store->setAttr(jointV5 ? "avelang.qwen.joint_v5.typed_lds_packet"
                                       : jointV4 ? "avelang.qwen.joint_v4.typed_lds_packet"
                                       : jointV3 ? "avelang.qwen.joint_v3.typed_lds_packet"
                                                 : "avelang.qwen.joint_v2.typed_lds_packet",
                               rewriter.getStringAttr("bf16x8"));
                if ((jointV4 || jointV5) && !wOperand) {
                    store->setAttr(jointV5 ? "avelang.qwen.joint_v5.shared_layout"
                                           : "avelang.qwen.joint_v4.shared_layout",
                                   rewriter.getStringAttr("token_major_physical"));
                }
                store->setAttr("avelang.qwen.k64.pipeline.lds_commit",
                               rewriter.getStringAttr(distributed ? "distributed" : "immediate"));
                copyJointStageAttrs(stage, store);
            };
            if (ownerWaveOnly) {
                auto owner = mlir::arith::CmpIOp::create(
                    rewriter, loc, mlir::arith::CmpIPredicate::ult, originalTid,
                    indexConstant(rewriter, loc, kWorkgroupSize));
                auto guarded = mlir::scf::IfOp::create(
                    rewriter, loc, owner, /*withElseRegion=*/false);
                rewriter.setInsertionPointToStart(&guarded.getThenRegion().front());
                emitStore();
                rewriter.setInsertionPointAfter(guarded);
            } else {
                emitStore();
            }
            continue;
        }
        for (int64_t element = 0; element < kPacketElements; ++element) {
            auto value = mlir::vector::ExtractOp::create(rewriter, loc,
                                                          packetValue, element);
            auto row = indexAdd(rewriter, loc,
                                indexMul(rewriter, loc, kPacket, c8),
                                indexConstant(rewriter, loc, element));
            // K's direct-K64 consumer reads [half, K, token]; P0 pred reads
            // its W bank as [half, token, K].  The joint planner owns this
            // distinction, so a W stage must not inherit the historical K
            // transpose merely because both banks have shape [2, 64, 64].
            llvm::SmallVector<mlir::Value, 3> indices;
            indices.push_back(kHalf);
            if (wOperand) {
                indices.push_back(tokenOffset);
                indices.push_back(row);
            } else {
                indices.push_back(row);
                indices.push_back(tokenOffset);
            }
            auto store = mlir::memref::StoreOp::create(
                rewriter, loc, value, commit.getSharedKBank(), indices);
            store->setAttr("avelang.qwen.k64.pipeline.lds_commit",
                           rewriter.getStringAttr(distributed ? "distributed" : "immediate"));
            copyJointStageAttrs(stage, store);
        }
    }
}

// The core-last-use scheduler has already made the packet itself explicit
// vector SSA.  Keep this lowering intentionally small: it shares the exact
// R4 per-lane packet/address formula, but does not re-introduce an opaque
// token, local array, or private packet ring merely because the producer and
// consumer are separated in the recurrence body.
mlir::Value emitCorePacketLoad(mlir::PatternRewriter &rewriter,
                               AMDGPUQwenK64CoreIssueOp issue) {
    auto loc = issue.getLoc();
    auto packetAttr = issue->getAttrOfType<mlir::IntegerAttr>(
        "avelang.qwen.core_staggered.packet_start");
    const int64_t packet = packetAttr.getInt();
    auto c0 = indexConstant(rewriter, loc, 0);
    auto c8 = indexConstant(rewriter, loc, kPacketElements);
    auto c64 = indexConstant(rewriter, loc, 64);
    auto tid = asIndex(rewriter, loc, issue.getThreadId());
    auto chunkStart = asIndex(rewriter, loc, issue.getChunkStart());
    auto keyHead = asIndex(rewriter, loc, issue.getKeyHead());
    auto kHalf = asIndex(rewriter, loc, issue.getKHalf());
    auto linear = indexAdd(rewriter, loc, tid,
                           indexConstant(rewriter, loc,
                                         packet * kWorkgroupSize));
    auto tokenOffset = mlir::arith::DivUIOp::create(rewriter, loc, linear, c8);
    auto kPacket = mlir::arith::RemUIOp::create(rewriter, loc, linear, c8);
    auto token = indexAdd(rewriter, loc, chunkStart, tokenOffset);
    auto kColumn = indexAdd(
        rewriter, loc, indexMul(rewriter, loc, kHalf, c64),
        indexMul(rewriter, loc, kPacket, c8));
    auto load = mlir::vector::LoadOp::create(
        rewriter, loc, issue.getPacket().getType(), issue.getSourceK(),
        mlir::ValueRange{c0, token, keyHead, kColumn});
    for (auto attr : issue->getAttrs()) {
        load->setAttr(attr.getName(), attr.getValue());
    }
    load->setAttr("avelang.qwen.core_staggered.global_packet",
                  rewriter.getStringAttr("bf16x8_register_live"));
    return load.getResult();
}

void emitCorePacketCommit(mlir::PatternRewriter &rewriter,
                          AMDGPUQwenK64CoreCommitOp commit,
                          mlir::Value packet) {
    auto loc = commit.getLoc();
    auto packetAttr = commit->getAttrOfType<mlir::IntegerAttr>(
        "avelang.qwen.core_staggered.packet_start");
    const int64_t packetIndex = packetAttr.getInt();
    auto c8 = indexConstant(rewriter, loc, kPacketElements);
    auto tid = asIndex(rewriter, loc, commit.getThreadId());
    auto kHalf = asIndex(rewriter, loc, commit.getKHalf());
    auto linear = indexAdd(rewriter, loc, tid,
                           indexConstant(rewriter, loc,
                                         packetIndex * kWorkgroupSize));
    auto tokenOffset = mlir::arith::DivUIOp::create(rewriter, loc, linear, c8);
    auto kPacket = mlir::arith::RemUIOp::create(rewriter, loc, linear, c8);
    auto kBase = indexMul(rewriter, loc, kPacket, c8);
    auto store = mlir::vector::StoreOp::create(
        rewriter, loc, packet, commit.getSharedKBank(),
        mlir::ValueRange{kHalf, tokenOffset, kBase});
    for (auto attr : commit->getAttrs()) {
        store->setAttr(attr.getName(), attr.getValue());
    }
    store->setAttr("avelang.qwen.core_staggered.lds_commit",
                   rewriter.getStringAttr("existing_r4_tail_bank"));
}

// C24's region representation owns a real vector payload rather than an
// opaque token.  Keeping the predicate at both ends is important: inactive
// waves neither publish duplicate LDS words nor need a private packet ring.
// The vector load is issued at the Issue op; its first true dependency is the
// vector store below, so AMDGPU is free to place the waitcnt at Commit.
mlir::Value emitRegionPendingPacketIssue(
    mlir::PatternRewriter &rewriter,
    AMDGPURegionPendingPacketIssueOp issue) {
    auto loc = issue.getLoc();
    auto packetType = mlir::cast<mlir::VectorType>(issue.getPacket().getType());
    auto guarded = mlir::scf::IfOp::create(
        rewriter, loc, mlir::TypeRange{packetType}, issue.getPredicate(),
        /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&guarded.getThenRegion().front());
    auto load = mlir::vector::LoadOp::create(rewriter, loc, packetType,
                                              issue.getSource(), issue.getIndices());
    for (auto attr : issue->getAttrs()) {
        load->setAttr(attr.getName(), attr.getValue());
    }
    load->setAttr("avelang.region_pending_packet.global_issue",
                  rewriter.getStringAttr("bf16x8_ssa"));
    mlir::scf::YieldOp::create(rewriter, loc, load.getResult());
    rewriter.setInsertionPointToStart(&guarded.getElseRegion().front());
    auto zero = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getBF16Type(),
        rewriter.getFloatAttr(rewriter.getBF16Type(), 0.0));
    auto empty = mlir::vector::SplatOp::create(rewriter, loc, packetType, zero);
    mlir::scf::YieldOp::create(rewriter, loc, empty.getResult());
    rewriter.setInsertionPointAfter(guarded);
    return guarded.getResult(0);
}

void emitRegionPendingPacketCommit(
    mlir::PatternRewriter &rewriter,
    AMDGPURegionPendingPacketCommitOp commit, mlir::Value packet) {
    auto loc = commit.getLoc();
    auto guarded = mlir::scf::IfOp::create(rewriter, loc, commit.getPredicate(),
                                            /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&guarded.getThenRegion().front());
    auto store = mlir::vector::StoreOp::create(
        rewriter, loc, packet, commit.getDestination(), commit.getIndices());
    for (auto attr : commit->getAttrs()) {
        store->setAttr(attr.getName(), attr.getValue());
    }
    store->setAttr("avelang.region_pending_packet.lds_commit",
                   rewriter.getStringAttr("delayed_true_dependency"));
    rewriter.setInsertionPointAfter(guarded);
    mlir::gpu::BarrierOp::create(rewriter, loc);
}

class LowerQwenK64PipelineStagePass
    : public mlir::PassWrapper<LowerQwenK64PipelineStagePass,
                               mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerQwenK64PipelineStagePass)

    llvm::StringRef getArgument() const final {
        return "lower-qwen-k64-pipeline-stage";
    }

    llvm::StringRef getDescription() const final {
        return "Lower opaque Qwen K64 packet stages after GPU outlining";
    }

    void runOnOperation() override {
        llvm::SmallVector<AMDGPURegionPendingPacketCommitOp> regionCommits;
        getOperation().walk([&](AMDGPURegionPendingPacketCommitOp op) {
            regionCommits.push_back(op);
        });
        llvm::SmallVector<AMDGPUQwenK64CoreCommitOp> coreCommits;
        getOperation().walk([&](AMDGPUQwenK64CoreCommitOp op) {
            coreCommits.push_back(op);
        });
        llvm::SmallVector<AMDGPUQwenK64PipelineStageCommitOp> commits;
        getOperation().walk([&](AMDGPUQwenK64PipelineStageCommitOp op) {
            commits.push_back(op);
        });
        if (commits.empty() && coreCommits.empty() && regionCommits.empty()) {
            return;
        }

        mlir::PatternRewriter rewriter(&getContext());

        // Region pending packets are lowered as one owned pair.  This is the
        // C24 boundary absent from C23: no greedy per-block-dot pattern owns
        // the vector or keeps a pointer-keyed side table after replacement.
        for (auto commit : regionCommits) {
            auto issue = mlir::dyn_cast_or_null<AMDGPURegionPendingPacketIssueOp>(
                commit.getPacket().getDefiningOp());
            if (!issue || issue->getBlock() != commit->getBlock() ||
                !issue.getPacket().hasOneUse()) {
                commit.emitError(
                    "region pending commit requires a unique direct same-block "
                    "pending-packet issue");
                signalPassFailure();
                return;
            }
            rewriter.setInsertionPoint(issue);
            auto packet = emitRegionPendingPacketIssue(rewriter, issue);
            issue.getPacket().replaceAllUsesWith(packet);
            rewriter.setInsertionPoint(commit);
            emitRegionPendingPacketCommit(rewriter, commit, packet);
            rewriter.eraseOp(commit);
            rewriter.eraseOp(issue);
        }

        // Core-last-use packets are intentionally constrained to direct SSA
        // producer/consumer edges in the unrolled single recurrence body.
        // Rejecting a wrapped edge here protects the no-private-ring rule.
        for (auto commit : coreCommits) {
            auto issue = mlir::dyn_cast_or_null<AMDGPUQwenK64CoreIssueOp>(
                commit.getPacket().getDefiningOp());
            if (!issue || issue->getBlock() != commit->getBlock()) {
                commit.emitError(
                    "core-last-use commit requires a direct same-block register SSA issue");
                signalPassFailure();
                return;
            }
            rewriter.setInsertionPoint(issue);
            auto packet = emitCorePacketLoad(rewriter, issue);
            rewriter.setInsertionPoint(commit);
            emitCorePacketCommit(rewriter, commit, packet);
            rewriter.eraseOp(commit);
            rewriter.eraseOp(issue);
        }

        struct ModuloPacketGroup {
            AMDGPUQwenK64PipelineStageLoadOp steadyIssue;
            llvm::SmallVector<AMDGPUQwenK64PipelineStageCommitOp> consumers;
        };
        llvm::DenseMap<mlir::Operation *, unsigned> moduloGroupIndex;
        llvm::SmallVector<ModuloPacketGroup> moduloGroups;
        llvm::SmallVector<AMDGPUQwenK64PipelineStageCommitOp> ordinaryCommits;

        // A rotating-LDS schedule has two representations of an edge:
        //  * a light i64 iter_arg, retained by the generic modulo scheduler;
        //  * this direct commit, which owns the BF16x8 packet and writes it
        //    to the physical bank that will be consumed next iteration.
        // The former is intentionally erased here without materializing a
        // private ring; the latter is lowered in place before its stage is
        // erased.  Consequently no memref<4xvector<8xbf16>, private> can
        // enter the LLVM conversion for a software-pipelined packet.
        for (auto commit : commits) {
            if (commit->hasAttr(kLogicalLdsConsumerAttr)) {
                rewriter.eraseOp(commit);
                continue;
            }

            auto producer = findStageProducer(commit.getStageToken());
            auto stage = producer.stage;
            if (!stage) {
                commit.emitError("K64 pipeline commit lost its unique stage token "
                                 "after frontend conversion");
                signalPassFailure();
                return;
            }

            if (commit->hasAttr(kDirectLdsCommitAttr)) {
                const bool distributed = useJointV1Placement(stage) ||
                                         useJointV2Placement(stage) ||
                                         useJointV3Placement(stage) ||
                                         useJointV4Placement(stage) ||
                                         useJointV5Placement(stage) ||
                                         useDistributedPlacement();
                llvm::SmallVector<mlir::Value> packets;
                if (isJointV3(stage) && !isJointWOperand(stage)) {
                    rewriter.setInsertionPoint(stage);
                    auto rotatingPackets = emitJointV3RotatingKPackets(
                        rewriter, stage.getLoc(), stage);
                    rewriter.setInsertionPoint(commit);
                    emitJointV3RotatingKCommit(rewriter, commit.getLoc(), stage,
                                               commit, rotatingPackets);
                } else {
                    rewriter.setInsertionPoint(stage);
                    packets = emitPacketLoads(rewriter, stage.getLoc(), stage,
                                              distributed);
                    rewriter.setInsertionPoint(commit);
                    emitPacketCommit(rewriter, commit.getLoc(), stage, commit,
                                     packets, distributed);
                }
                rewriter.setInsertionPoint(stage);
                auto tokenType =
                    mlir::cast<mlir::IntegerType>(stage.getStageToken().getType());
                auto placeholder = mlir::arith::ConstantIntOp::create(
                    rewriter, stage.getLoc(), 0, tokenType.getWidth());
                stage.getStageToken().replaceAllUsesWith(placeholder);
                rewriter.eraseOp(commit);
                rewriter.eraseOp(stage);
                continue;
            }

            if (isSoftwarePipelineStage(stage) && isModuloLoopStage(stage)) {
                stage.emitError(
                    "rotating LDS pipeline reached the legacy private packet-ring path");
                signalPassFailure();
                return;
            }

            ordinaryCommits.push_back(commit);
        }

        for (auto commit : ordinaryCommits) {
            auto producer = findStageProducer(commit.getStageToken());
            auto stage = producer.stage;
            if (!stage) {
                commit.emitError("K64 pipeline commit lost its unique stage token "
                                 "after frontend conversion");
                signalPassFailure();
                continue;
            }

            if (!stage.getStageToken().hasOneUse()) {
                commit.emitError("K64 pipeline commit lost its unique stage token "
                                 "after frontend conversion");
                signalPassFailure();
                return;
            }

            const bool distributed = useJointV1Placement(stage) ||
                                     useJointV2Placement(stage) ||
                                     useJointV3Placement(stage) ||
                                     useJointV4Placement(stage) ||
                                     useJointV5Placement(stage) ||
                                     useDistributedPlacement();
            llvm::SmallVector<mlir::Value> packets;
            if (isJointV3(stage) && !isJointWOperand(stage)) {
                rewriter.setInsertionPoint(stage);
                auto rotatingPackets = emitJointV3RotatingKPackets(
                    rewriter, stage.getLoc(), stage);
                rewriter.setInsertionPoint(commit);
                emitJointV3RotatingKCommit(rewriter, commit.getLoc(), stage,
                                           commit, rotatingPackets);
            } else if (distributed) {
                rewriter.setInsertionPoint(stage);
                packets = emitPacketLoads(rewriter, stage.getLoc(), stage, true);
                rewriter.setInsertionPoint(commit);
                emitPacketCommit(rewriter, commit.getLoc(), stage, commit,
                                 packets, true);
            } else {
                rewriter.setInsertionPoint(commit);
                packets = emitPacketLoads(rewriter, commit.getLoc(), stage, false);
                emitPacketCommit(rewriter, commit.getLoc(), stage, commit,
                                 packets, false);
            }
            rewriter.eraseOp(commit);
            for (auto *wrapper : producer.wrappers) {
                if (wrapper->use_empty()) {
                    rewriter.eraseOp(wrapper);
                }
            }
            rewriter.eraseOp(stage);
        }

        // Materialize each symbolic scf.iter_arg edge as a generic per-lane
        // BF16x8 packet ring.  This is deliberately expressed in MLIR
        // memref/vector operations rather than as a Qwen instruction recipe:
        // global packets are issued and written at stage 0, while both the
        // next steady iteration and the peeled epilogue read the same ring at
        // stage 1 before committing to the R4 LDS bank.
        for (auto &group : moduloGroups) {
            auto initialIssue = findSoftwarePipelineInitialIssue(
                getOperation(), group.steadyIssue);
            if (!initialIssue) {
                group.steadyIssue.emitError(
                    "modulo pipeline stage has no initial issue for its iter_arg");
                signalPassFailure();
                return;
            }
            auto storage = stagePacketStorage(rewriter, initialIssue);
            emitSoftwarePipelineStage(rewriter, initialIssue, storage);
            emitSoftwarePipelineStage(rewriter, group.steadyIssue, storage);
            for (auto commit : group.consumers) {
                rewriter.setInsertionPoint(commit);
                auto packets = loadSoftwarePipelinePackets(
                    rewriter, commit.getLoc(), storage);
                // Commit indexing depends only on the lane/k-half ownership,
                // not on the next chunk address.  Use the initial issue's
                // loop-invariant ownership operands so the epilogue does not
                // reference values defined inside the steady scf.for.
                emitPacketCommit(rewriter, commit.getLoc(), initialIssue,
                                 commit, packets, /*distributed=*/true);
                rewriter.eraseOp(commit);
            }
            rewriter.eraseOp(initialIssue);
            rewriter.eraseOp(group.steadyIssue);
        }

        // Static Qwen shapes are deliberately unrolled by GPU outlining
        // before this late pass. Their remaining direct stages took the
        // generic path above; modulo-loop stages took the packet-ring path.

        bool remaining = false;
        getOperation().walk([&](mlir::Operation *op) {
            if (mlir::isa<AMDGPUQwenK64PipelineStageLoadOp,
                           AMDGPUQwenK64PipelineStageCommitOp,
                           AMDGPUQwenK64CoreIssueOp,
                           AMDGPUQwenK64CoreCommitOp,
                           AMDGPURegionPendingPacketIssueOp,
                           AMDGPURegionPendingPacketCommitOp>(op)) {
                op->emitError("Qwen K64 pipeline stage survived late lowering");
                remaining = true;
            }
        });
        if (remaining) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerQwenK64PipelineStagePass() {
    return std::make_unique<LowerQwenK64PipelineStagePass>();
}

} // namespace causalflow::avelang::dialect
