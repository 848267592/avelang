#include "lower_qwen_block_dot_pass.h"

#include "Dialect/AveLang/IR/AveLangOps.h"
#include "Dialect/AveLang/IR/static_physical_layout.h"
#include "IR/Intrinsics/intrinsic_support.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/raw_ostream.h>

#include <array>
#include <optional>
#include <string>
#include <utility>

namespace causalflow::avelang::dialect {
namespace {

constexpr int64_t kThreads = 128;

enum class LoweringKind { Generic, Specialized };

// This is deliberately narrower than the block-dot scheduling mode. Both
// choices keep the same BV32 shared tiles, barriers, MFMA consumers, and CTA
// ownership. They differ only in the global-to-LDS producer for K/V operands.
enum class OperandStagingKind {
    Scalar,
    TypedVector,
    PersistentTypedBlock,
    PersistentTypedLdsLayout,
};

// C14 is deliberately opt-in at the internal operand boundary.  A source
// block_dot operation is eligible only when the planner has attached all five
// typed C13 attributes.  This keeps the old Qwen paths byte-for-byte inert and
// prevents a missing recipe from silently falling back to the generic layout
// planner.
struct C14StaticPhysicalPlan {
    c13::DistributedEncoding distributed;
    c13::SharedEncoding shared;
    c13::MfmaEncoding mfma;
    c13::DotOperandEncoding dot;
    c13::StaticLayoutTransform transform;
};

std::optional<C14StaticPhysicalPlan>
getC14StaticPhysicalPlan(AMDGPUBlockDotMfmaOperandOp op) {
    if (!op->hasAttr("c14.static_physical"))
        return std::nullopt;
    auto distributed = op->getAttrOfType<DistributedEncodingAttr>(
        "c13.distributed");
    auto shared = op->getAttrOfType<SharedEncodingAttr>("c13.shared");
    auto mfma = op->getAttrOfType<MfmaEncodingAttr>("c13.mfma");
    auto dot = op->getAttrOfType<DotOperandEncodingAttr>("c13.dot");
    auto transform = op->getAttrOfType<StaticTransformAttr>("c13.transform");
    if (!distributed || !shared || !mfma || !dot || !transform)
        return std::nullopt;
    auto distributedValue = c13::DistributedEncoding::fromAttr(distributed);
    auto sharedValue = c13::SharedEncoding::fromAttr(shared);
    auto mfmaValue = c13::MfmaEncoding::fromAttr(mfma);
    auto dotValue = c13::DotOperandEncoding::fromAttr(dot);
    auto transformValue = c13::StaticLayoutTransform::fromAttr(transform);
    if (!distributedValue || !sharedValue || !mfmaValue || !dotValue ||
        !transformValue)
        return std::nullopt;
    if (dotValue->parent != *mfmaValue)
        return std::nullopt;
    return C14StaticPhysicalPlan{*distributedValue, *sharedValue,
                                 *mfmaValue, *dotValue, *transformValue};
}

bool isCooperativeBv32(AMDGPUBlockDotBF16F32Op op) {
    auto type = mlir::dyn_cast<mlir::MemRefType>(op.getAStage().getType());
    return type && type.getRank() == 3 &&
           (type.getShape() == llvm::ArrayRef<int64_t>({1, 32, 32}) ||
            type.getShape() == llvm::ArrayRef<int64_t>({1, 32, 64}));
}

bool isPrecomputedVDecay(AMDGPUBlockDotBF16F32Op op) {
    return op->hasAttr("avelang.block_dot.precomputed_vdecay");
}

// F0 uses the existing persistent C0 operand schedule, but the producer has
// already materialized the BF16 V-decay block in shared memory. This is a
// dataflow boundary, not another LDS layout choice.
bool isStagedVDecay(AMDGPUBlockDotBF16F32Op op) {
    return op->hasAttr("avelang.block_dot.staged_vdecay");
}

// B2 owns the one-chunk lookahead in source. The K operand has already been
// loaded from global memory and written into the single reusable LDS bank, so
// late block-dot lowering must consume it directly rather than issue another
// source-K producer sequence.
bool isPreloadedK(AMDGPUBlockDotBF16F32Op op) {
    return op->hasAttr("avelang.block_dot.preloaded_k");
}

// The StateKV variant has the same staged operands and K-half ownership as
// R4.  Only the interpretation of the MFMA output changes: output rows are
// K and output columns are V, so the physical FP32 feedback fragment can stay
// KxV instead of being transposed back into the legacy VxK fragment.
bool isStateKV(AMDGPUBlockDotBF16F32Op op) {
    return op->hasAttr("avelang.block_dot.state_kv");
}

bool isGenericOperandMode(AMDGPUBlockDotBF16F32Op op) {
    return op->hasAttr("avelang.block_dot.operand_mode");
}

bool isFullScopeOperandMode(AMDGPUBlockDotBF16F32Op op) {
    auto mode = op->getAttrOfType<mlir::StringAttr>(
        "avelang.block_dot.operand_mode");
    return mode && mode.getValue() == "full_scope" &&
           op->hasAttr("avelang.block_dot.scope");
}

llvm::StringRef genericOperandRole(AMDGPUBlockDotBF16F32Op op) {
    if (auto role = op->getAttrOfType<mlir::StringAttr>(
            "avelang.block_dot.operand_role"))
        return role.getValue();
    return "B";
}

llvm::StringRef genericOperandSourceRole(AMDGPUBlockDotBF16F32Op op) {
    if (llvm::sys::Process::GetEnv("AVELANG_STAGE6Z_FULL_PHYSICAL_REGION") ==
            std::optional<std::string>("c19") ||
        llvm::sys::Process::GetEnv("AVELANG_STAGE6Z_FULL_PHYSICAL_REGION") ==
            std::optional<std::string>("c21")) {
        if (op.getSourceK() == op.getSourceVNew()) {
            if (auto type = mlir::dyn_cast<mlir::MemRefType>(
                    op.getSourceK().getType())) {
                if (type.getRank() == 4 && type.getShape()[2] == 4)
                    return "Q";
                if (type.getRank() == 4 && type.getShape()[2] == 8)
                    return "V";
            }
        }
    }
    // C18 uses the existing logical block-dot ABI for the V producer and
    // consumer by passing the same BF16 V-new SSA value in both logical
    // source slots.  Treat that identity as a compiler-internal role
    // selection before consulting the legacy H/K attribute.
    if (llvm::sys::Process::GetEnv("AVELANG_STAGE6Z_FULL_PHYSICAL_REGION") ==
            std::optional<std::string>("c18") &&
        op.getSourceK() == op.getSourceVNew())
        return "V";
    // C22 is schedule-preserving: Phase-C passes V-new in both source slots
    // solely to identify its physical B-operand recipe.  It must not inherit
    // the C18/C19 region owner or scheduling behaviour.
    if (llvm::sys::Process::GetEnv(
            "AVELANG_STAGE6Z_SCHEDULE_PRESERVING_PHYSICAL") ==
            std::optional<std::string>("c22") &&
        op.getSourceK() == op.getSourceVNew())
        return "V";
    if (auto role = op->getAttrOfType<mlir::StringAttr>(
            "avelang.block_dot.source_role"))
        return role.getValue();
    return "K";
}

bool useC22SchedulePreservingPhysical() {
    return llvm::sys::Process::GetEnv(
               "AVELANG_STAGE6Z_SCHEDULE_PRESERVING_PHYSICAL") ==
           std::optional<std::string>("c22");
}

// The full-scope contract has a static logical shape and a static ownership
// rule, but the old lowering rebuilt the same affine thread map once for the
// global producer and again for the LDS/MFMA consumer.  Keep the map as an
// internal planner object so K and H share one implementation and one set of
// SSA index values.  This is deliberately not a Qwen-specific address table:
// the role and transpose only select the logical block interpretation.
mlir::Value toIndex(mlir::PatternRewriter &rewriter, mlir::Location loc,
                    mlir::Value value);
mlir::Value indexConstant(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          int64_t value);
mlir::Value indexAdd(mlir::PatternRewriter &rewriter, mlir::Location loc,
                     mlir::Value lhs, mlir::Value rhs);
mlir::Value indexMul(mlir::PatternRewriter &rewriter, mlir::Location loc,
                     mlir::Value lhs, mlir::Value rhs);

int64_t staticShiftForPowerOfTwo(int64_t value) {
    if (value <= 0 || (value & (value - 1)) != 0)
        return -1;
    int64_t shift = 0;
    while ((int64_t{1} << shift) != value)
        ++shift;
    return shift;
}

mlir::Value c14I64(mlir::PatternRewriter &rewriter, mlir::Location loc,
                   mlir::Value value) {
    if (value.getType().isSignlessInteger(64))
        return value;
    return mlir::arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI64Type(), value);
}

mlir::Value c14I64Constant(mlir::PatternRewriter &rewriter,
                           mlir::Location loc, int64_t value) {
    return mlir::arith::ConstantIntOp::create(rewriter, loc, value, 64);
}

// Lower the finite C13 shared algebra with shifts/masks/xor.  The supported
// C12 recipes all have power-of-two vec/perPhase/maxPhase/group parameters;
// rejecting anything else is intentional because C14 must not reintroduce a
// generic DivUI/RemUI physical planner.
mlir::FailureOr<mlir::Value> emitC14SharedElementOffset(
    mlir::PatternRewriter &rewriter, mlir::Location loc,
    const C14StaticPhysicalPlan &plan, mlir::Value row, mlir::Value col) {
    const auto &shared = plan.shared;
    const auto &shape = plan.distributed.logicalShape;
    const int64_t innerExtent = shape[shared.order[1]];
    const int64_t groups = innerExtent / shared.vec;
    const int64_t vecShift = staticShiftForPowerOfTwo(shared.vec);
    const int64_t phaseShift = staticShiftForPowerOfTwo(shared.perPhase);
    const int64_t maxPhaseMask = shared.maxPhase - 1;
    const int64_t groupsMask = groups - 1;
    if (vecShift < 0 || phaseShift < 0 ||
        staticShiftForPowerOfTwo(shared.maxPhase) < 0 ||
        staticShiftForPowerOfTwo(groups) < 0 || maxPhaseMask < 0 ||
        groupsMask < 0)
        return mlir::failure();

    auto logical = std::array<mlir::Value, 2>{row, col};
    auto outer = logical[shared.order[0]];
    auto inner = logical[shared.order[1]];
    auto outerI64 = c14I64(rewriter, loc, outer);
    auto innerI64 = c14I64(rewriter, loc, inner);

    mlir::Value phase = c14I64Constant(rewriter, loc, 0);
    if (shared.maxPhase != 1) {
        phase = mlir::arith::ShRUIOp::create(
            rewriter, loc, outerI64,
            c14I64Constant(rewriter, loc, phaseShift)).getResult();
        phase = mlir::arith::AndIOp::create(
            rewriter, loc, phase,
            c14I64Constant(rewriter, loc, maxPhaseMask)).getResult();
    }
    mlir::Value swizzle = phase;
    if (shared.kind == "amd_rotating_shared") {
        const int64_t combinedShift =
            staticShiftForPowerOfTwo(shared.maxPhase * shared.perPhase);
        if (combinedShift < 0)
            return mlir::failure();
        mlir::Value blockNo = mlir::arith::ShRUIOp::create(
            rewriter, loc, outerI64,
            c14I64Constant(rewriter, loc, combinedShift)).getResult();
        blockNo = mlir::arith::AndIOp::create(
            rewriter, loc, blockNo,
            c14I64Constant(rewriter, loc, maxPhaseMask)).getResult();
        swizzle = mlir::arith::XOrIOp::create(rewriter, loc, phase, blockNo)
                      .getResult();
    }
    swizzle = mlir::arith::AndIOp::create(
        rewriter, loc, swizzle,
        c14I64Constant(rewriter, loc, groupsMask)).getResult();
    mlir::Value innerGroup = mlir::arith::ShRUIOp::create(
        rewriter, loc, innerI64,
        c14I64Constant(rewriter, loc, vecShift)).getResult();
    mlir::Value physicalGroup = mlir::arith::XOrIOp::create(
        rewriter, loc,
        innerGroup, swizzle).getResult();
    mlir::Value physicalInner = mlir::arith::ShLIOp::create(
        rewriter, loc, physicalGroup,
        c14I64Constant(rewriter, loc, vecShift)).getResult();
    mlir::Value intra = mlir::arith::AndIOp::create(
        rewriter, loc, innerI64,
        c14I64Constant(rewriter, loc, shared.vec - 1)).getResult();
    physicalInner = mlir::arith::OrIOp::create(rewriter, loc, physicalInner,
                                               intra).getResult();
    mlir::Value elementOffset = mlir::arith::AddIOp::create(
        rewriter, loc,
        mlir::arith::MulIOp::create(
            rewriter, loc, outerI64,
            c14I64Constant(rewriter, loc, innerExtent)).getResult(),
        physicalInner).getResult();
    return elementOffset;
}

// The selected Triton Q/H/K shared encodings use the first order entry as
// the physical inner dimension.  C13's generic utility predates this native
// consumer contract and interprets order[1] as inner; keep that utility
// unchanged for C14/V and expose the selected Q/H/K interpretation here.
// The resulting algebra is still finite and target-plan driven: it does not
// introduce a runtime layout planner or a role-specific address table.
mlir::FailureOr<mlir::Value> emitC16NativeSharedElementOffset(
    mlir::PatternRewriter &rewriter, mlir::Location loc,
    const C14StaticPhysicalPlan &plan, mlir::Value row, mlir::Value col) {
    const auto &shared = plan.shared;
    const auto &shape = plan.distributed.logicalShape;
    const int64_t innerExtent = shape[shared.order[0]];
    const int64_t groups = innerExtent / shared.vec;
    const int64_t vecShift = staticShiftForPowerOfTwo(shared.vec);
    const int64_t phaseShift = staticShiftForPowerOfTwo(shared.perPhase);
    const int64_t maxPhaseMask = shared.maxPhase - 1;
    const int64_t groupsMask = groups - 1;
    if (vecShift < 0 || phaseShift < 0 ||
        staticShiftForPowerOfTwo(shared.maxPhase) < 0 ||
        staticShiftForPowerOfTwo(groups) < 0 || maxPhaseMask < 0 ||
        groupsMask < 0)
        return mlir::failure();

    auto logical = std::array<mlir::Value, 2>{row, col};
    auto outer = logical[shared.order[1]];
    auto inner = logical[shared.order[0]];
    auto outerI64 = c14I64(rewriter, loc, outer);
    auto innerI64 = c14I64(rewriter, loc, inner);

    mlir::Value phase = c14I64Constant(rewriter, loc, 0);
    if (shared.maxPhase != 1) {
        phase = mlir::arith::ShRUIOp::create(
            rewriter, loc, outerI64,
            c14I64Constant(rewriter, loc, phaseShift));
        phase = mlir::arith::AndIOp::create(
            rewriter, loc, phase,
            c14I64Constant(rewriter, loc, maxPhaseMask));
    }
    mlir::Value swizzle = phase;
    if (shared.kind == "amd_rotating_shared") {
        const int64_t combinedShift =
            staticShiftForPowerOfTwo(shared.maxPhase * shared.perPhase);
        if (combinedShift < 0)
            return mlir::failure();
        mlir::Value blockNo = mlir::arith::ShRUIOp::create(
            rewriter, loc, outerI64,
            c14I64Constant(rewriter, loc, combinedShift));
        blockNo = mlir::arith::AndIOp::create(
            rewriter, loc, blockNo,
            c14I64Constant(rewriter, loc, maxPhaseMask));
        swizzle = mlir::arith::XOrIOp::create(rewriter, loc, phase, blockNo);
    }
    swizzle = mlir::arith::AndIOp::create(
        rewriter, loc, swizzle,
        c14I64Constant(rewriter, loc, groupsMask));
    mlir::Value innerGroup = mlir::arith::ShRUIOp::create(
        rewriter, loc, innerI64,
        c14I64Constant(rewriter, loc, vecShift));
    mlir::Value physicalGroup = mlir::arith::XOrIOp::create(
        rewriter, loc, innerGroup, swizzle);
    mlir::Value physicalInner = mlir::arith::ShLIOp::create(
        rewriter, loc, physicalGroup,
        c14I64Constant(rewriter, loc, vecShift));
    mlir::Value intra = mlir::arith::AndIOp::create(
        rewriter, loc, innerI64,
        c14I64Constant(rewriter, loc, shared.vec - 1));
    physicalInner = mlir::arith::OrIOp::create(
        rewriter, loc, physicalInner, intra);
    return mlir::arith::AddIOp::create(
        rewriter, loc,
        mlir::arith::MulIOp::create(
            rewriter, loc, outerI64,
            c14I64Constant(rewriter, loc, innerExtent)),
        physicalInner)
        .getResult();
}

// The selected native #shared2 producer is a contiguous eight-BF16 packet.
// LLVM shows its byte base as (logical_col * 32 + logical_row_base) * 2;
// the #shared2 phase is consumed by the dot load, not by a scalar producer
// scatter.  Keep this finite packet-order map separate from the generic
// logical-element swizzle so C16 can verify the producer/consumer pair.
mlir::FailureOr<mlir::Value> emitC16NativeKPacketElementOffset(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value row,
    mlir::Value col) {
    auto rowI64 = c14I64(rewriter, loc, row);
    auto colI64 = c14I64(rewriter, loc, col);
    return mlir::arith::AddIOp::create(
        rewriter, loc,
        mlir::arith::MulIOp::create(
            rewriter, loc, colI64, c14I64Constant(rewriter, loc, 32)),
        rowI64)
        .getResult();
}

bool useC15RealTile() {
    return llvm::sys::Process::GetEnv("AVELANG_C15_REAL_TILE") ==
           std::optional<std::string>("1");
}

bool isC15RealTile(AMDGPUBlockDotBF16F32Op op) {
    auto tile = op->getAttrOfType<mlir::StringAttr>("c15.real_tile");
    return useC15RealTile() && tile && tile.getValue() == "V";
}

std::optional<std::string> c16RealTileRole() {
    auto role = llvm::sys::Process::GetEnv("AVELANG_C16_REAL_TILE_ROLE");
    if (!role || (*role != "Q" && *role != "H" && *role != "K"))
        return std::nullopt;
    return role;
}

bool useC16QDualConsumer() {
    return llvm::sys::Process::GetEnv("AVELANG_C16_Q_DUAL") ==
           std::optional<std::string>("1");
}

// The selected native T8192 Q/H contract is a separate two-wave physical
// contract.  Keep it opt-in so the verified four-wave C16 path remains
// byte-for-byte unchanged for existing callers.
bool useC16WG128QH() {
    return llvm::sys::Process::GetEnv("AVELANG_C16_WG128_QH") ==
           std::optional<std::string>("1");
}

// The WG128 native parent represents a complete 64x64 dot with two
// independent 16-lane accumulator halves per wave.  Keep this contract
// behind a second opt-in gate until the source oracle proves its row/column
// ownership; the existing WG128 producer/readback closure remains unchanged.
bool useC16WG128QHFull64() {
    return useC16WG128QH() &&
           llvm::sys::Process::GetEnv("AVELANG_C16_WG128_QH_FULL64") ==
               std::optional<std::string>("1");
}

// Experimental Q@H-only arm: keep Q's already-closed packed path, but let
// the B/H operand use the native packed shared address contract as well.
// This is deliberately separate from the existing Q/H gate so the baseline
// oracle remains byte-for-byte reproducible.
bool useC16WG128QHPackedH() {
    return useC16WG128QHFull64() &&
           llvm::sys::Process::GetEnv("AVELANG_C16_WG128_QH_PACKED_H") ==
               std::optional<std::string>("1");
}

// The native T8192 Q@K consumer uses the same two-wave MFMA parent as Q@H,
// but its B operand is the [32,64] K tile.  Keep this independent from the
// already-closed Q/H gate: K needs four producer packets per thread to cover
// all 32x64 BF16 elements and a distinct B-row recipe at the consumer.
bool useC16WG128QKFull64() {
    return llvm::sys::Process::GetEnv("AVELANG_C16_WG128_QK_FULL64") ==
           std::optional<std::string>("1");
}

// The C16 dual-consumer repro marks the second generic block-dot call with
// the existing transposed operand spelling.  Under the explicit Q-dual gate
// that spelling is treated only as a consumer identity: both calls still
// carry the same Q role and same physical stage.  This avoids depending on
// rewrite traversal order when deciding which operation owns the producer.
bool isC16QDualConsumerOnly(AMDGPUBlockDotBF16F32Op op) {
    if (!useC16QDualConsumer())
        return false;
    auto transpose = op->getAttrOfType<mlir::StringAttr>(
        "avelang.block_dot.transpose");
    return transpose && transpose.getValue() == "rhs_transposed";
}

bool isC16RealTile(AMDGPUBlockDotBF16F32Op op) {
    auto role = op->getAttrOfType<mlir::StringAttr>("c16.real_tile_role");
    auto enabled = c16RealTileRole();
    return enabled && role && role.getValue() == *enabled;
}

std::optional<C14StaticPhysicalPlan>
getC16StaticPhysicalPlan(llvm::StringRef role) {
    auto physical = c13::ChunkOPhysicalPlan::makeC12T2048WG256();
    std::string error;
    if (!physical.verify(&error))
        return std::nullopt;
    const c13::PhysicalBlockPlan *block = nullptr;
    if (role == "Q")
        block = &physical.q;
    else if (role == "H")
        block = &physical.h;
    else if (role == "K")
        block = &physical.k;
    if (!block)
        return std::nullopt;
    C14StaticPhysicalPlan result{block->distributed, block->shared,
                                 physical.mfma, block->dot, block->transform};
    if (useC16WG128QH() && (role == "Q" || role == "H")) {
        // [64,32] = [2,8] registers * [16,4] lanes * [2,1] waves.
        // This is the smallest C13 representation that covers the complete
        // Q/H tile with two wave64 waves and matches the selected native MFMA
        // parent shape [1,2].
        result.distributed.sizePerThread = {2, 8};
        result.distributed.wavesPerCTA = {2, 1};
        result.mfma.warpsPerCTA = {1, 2};
        result.dot.parent = result.mfma;
    }
    if (useC16WG128QKFull64() && role == "K") {
        // Native TTGIR #blocked1: sizePerThread=[8,1],
        // threadsPerWarp=[4,16], warpsPerCTA=[1,2].
        // The native TTGIR advertises [8,1] for the dot operand because its
        // kWidth=4 dot encoding supplies the second physical packet.  The
        // source-facing C16 plan must describe the complete [32,64] logical
        // K tile before that dot expansion, so its explicit coverage is
        // [8,2] * [4,16] * [1,2] = [32,64].
        result.distributed.sizePerThread = {8, 2};
        result.distributed.wavesPerCTA = {1, 2};
        result.mfma.warpsPerCTA = {1, 2};
        result.dot.parent = result.mfma;
    }
    if (!result.distributed.verify(&error) ||
        !result.shared.verifyMapping(result.distributed.logicalShape, 2,
                                     &error) ||
        !result.mfma.verify(&error) || !result.dot.verify(&error) ||
        !result.transform.verify(&error))
        return std::nullopt;
    return result;
}

// C22 consumes the already validated C13-C16 layouts from the ordinary Z5B
// phase graph.  This selector is deliberately physical-only: it carries no
// phase, owner, or lifetime decision.
std::optional<C14StaticPhysicalPlan>
getC22StaticPhysicalPlan(llvm::StringRef role) {
    auto physical = c13::ChunkOPhysicalPlan::makeC12T2048WG256();
    std::string error;
    if (!physical.verify(&error))
        return std::nullopt;
    const c13::PhysicalBlockPlan *block = nullptr;
    if (role == "Q")
        block = &physical.q;
    else if (role == "H")
        block = &physical.h;
    else if (role == "K")
        block = &physical.k;
    else if (role == "V")
        block = &physical.v;
    if (!block)
        return std::nullopt;
    C14StaticPhysicalPlan result{block->distributed, block->shared,
                                 physical.mfma, block->dot, block->transform};
    if (!result.distributed.verify(&error) ||
        !result.shared.verifyMapping(result.distributed.logicalShape, 2,
                                     &error) ||
        !result.mfma.verify(&error) || !result.dot.verify(&error) ||
        !result.transform.verify(&error))
        return std::nullopt;
    return result;
}

// C15 uses the already recovered V recipe.  The distributed V encoding is
// fixed at [64,64], sizePerThread=[4,8], threadsPerWave=[8,8],
// wavesPerCTA=[2,1], order=[1,0].  Express that recipe with shifts/masks so
// this path never re-enters the generic DivUI/RemUI layout planner.
struct C15VSourceCoord {
    mlir::Value rowBase;
    mlir::Value col;
};

C15VSourceCoord emitC15VSourceCoord(mlir::PatternRewriter &rewriter,
                                    mlir::Location loc, mlir::Value tid,
                                    int64_t registerOne) {
    auto tid64 = c14I64(rewriter, loc, tid);
    auto wave = mlir::arith::ShRUIOp::create(
        rewriter, loc, tid64, c14I64Constant(rewriter, loc, 6));
    auto lane = mlir::arith::AndIOp::create(
        rewriter, loc, tid64, c14I64Constant(rewriter, loc, 63));
    auto laneRow = mlir::arith::ShRUIOp::create(
        rewriter, loc, lane, c14I64Constant(rewriter, loc, 3));
    auto laneCol = mlir::arith::AndIOp::create(
        rewriter, loc, lane, c14I64Constant(rewriter, loc, 7));
    auto rowBase = mlir::arith::AddIOp::create(
        rewriter, loc,
        mlir::arith::ShLIOp::create(
            rewriter, loc, wave, c14I64Constant(rewriter, loc, 5)),
        mlir::arith::ShLIOp::create(
            rewriter, loc, laneRow, c14I64Constant(rewriter, loc, 2)));
    auto col = mlir::arith::AddIOp::create(
        rewriter, loc,
        mlir::arith::ShLIOp::create(
            rewriter, loc, laneCol, c14I64Constant(rewriter, loc, 3)),
        c14I64Constant(rewriter, loc, registerOne));
    return {rowBase, col};
}

mlir::Value c15Index(mlir::PatternRewriter &rewriter, mlir::Location loc,
                     mlir::Value value) {
    if (value.getType().isIndex())
        return value;
    return mlir::arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), value);
}

struct C16SourceCoord {
    mlir::Value row;
    mlir::Value col;
};

// The selected T2048 TTGIR blocked encodings are the standard Triton
// distributed algebra:
//   #blocked2: row = wave*16 + lane/4, col = (lane%4)*8 + reg;
//   #blocked1: row = (lane%4)*8 + reg, col = wave*16 + lane/4.
// C16 uses the same equations for both producer and consumer, rather than a
// role-specific address table.  The mapping JSON records the literal TTGIR
// encoding and this helper is the executable form of that finite algebra.
C16SourceCoord emitC16SourceCoord(mlir::PatternRewriter &rewriter,
                                  mlir::Location loc, mlir::Value tid,
                                  llvm::StringRef role, int64_t packet,
                                  int64_t element, bool wg128QH = false) {
    auto tid64 = c14I64(rewriter, loc, tid);
    auto wave = mlir::arith::ShRUIOp::create(
        rewriter, loc, tid64, c14I64Constant(rewriter, loc, 6));
    auto lane = mlir::arith::AndIOp::create(
        rewriter, loc, tid64, c14I64Constant(rewriter, loc, 63));
    auto laneLow = mlir::arith::AndIOp::create(
        rewriter, loc, lane, c14I64Constant(rewriter, loc, 3));
    auto laneHigh = mlir::arith::ShRUIOp::create(
        rewriter, loc, lane, c14I64Constant(rewriter, loc, 2));
    C16SourceCoord result;
    if (role == "K") {
        if (wg128QH) {
            // WG128 has two waves.  Each wave owns 32 K columns; packet
            // parity selects the two columns carried by a lane while the
            // packet pair selects the two four-element row groups.  This
            // covers [32,64] with 4 BF16x4 packets per thread.
            auto packetGroup = mlir::arith::ShRUIOp::create(
                rewriter, loc, c14I64Constant(rewriter, loc, packet),
                c14I64Constant(rewriter, loc, 1));
            auto row = mlir::arith::AddIOp::create(
                rewriter, loc,
                mlir::arith::AddIOp::create(
                    rewriter, loc,
                    mlir::arith::ShLIOp::create(
                        rewriter, loc, laneLow,
                        c14I64Constant(rewriter, loc, 3)),
                    mlir::arith::ShLIOp::create(
                        rewriter, loc, packetGroup,
                        c14I64Constant(rewriter, loc, 2))),
                c14I64Constant(rewriter, loc, element));
            auto col = mlir::arith::AddIOp::create(
                rewriter, loc,
                mlir::arith::AddIOp::create(
                    rewriter, loc,
                    mlir::arith::ShLIOp::create(
                        rewriter, loc, wave,
                        c14I64Constant(rewriter, loc, 5)),
                    mlir::arith::ShLIOp::create(
                        rewriter, loc, laneHigh,
                        c14I64Constant(rewriter, loc, 1))),
                mlir::arith::AndIOp::create(
                    rewriter, loc,
                    c14I64Constant(rewriter, loc, packet),
                    c14I64Constant(rewriter, loc, 1)));
            result = {row, col};
        } else {
            auto packetRow = mlir::arith::ShLIOp::create(
                rewriter, loc, laneLow, c14I64Constant(rewriter, loc, 3));
            auto row = mlir::arith::AddIOp::create(
                rewriter, loc, packetRow,
                c14I64Constant(rewriter, loc, packet * 4 + element));
            auto col = mlir::arith::AddIOp::create(
                rewriter, loc,
                mlir::arith::ShLIOp::create(
                    rewriter, loc, wave, c14I64Constant(rewriter, loc, 4)),
                laneHigh);
            result = {row, col};
        }
    } else if (wg128QH) {
        // Two-wave Q/H ownership: each lane owns two rows and four BF16x4
        // packets.  packet pairs select the two rows; packet parity selects
        // the contiguous 4-element column packet.  Across wave/lane/packet
        // this is a bijection over [64,32].
        auto packetRow = mlir::arith::ShRUIOp::create(
            rewriter, loc, c14I64Constant(rewriter, loc, packet),
            c14I64Constant(rewriter, loc, 1));
        auto row = mlir::arith::AddIOp::create(
            rewriter, loc,
            mlir::arith::AddIOp::create(
                rewriter, loc,
                mlir::arith::ShLIOp::create(
                    rewriter, loc, wave, c14I64Constant(rewriter, loc, 5)),
                mlir::arith::ShLIOp::create(
                    rewriter, loc, laneHigh,
                    c14I64Constant(rewriter, loc, 1))),
            packetRow);
        auto packetCol = mlir::arith::AndIOp::create(
            rewriter, loc, c14I64Constant(rewriter, loc, packet),
            c14I64Constant(rewriter, loc, 1));
        auto col = mlir::arith::AddIOp::create(
            rewriter, loc,
            mlir::arith::ShLIOp::create(
                rewriter, loc, laneLow, c14I64Constant(rewriter, loc, 3)),
            mlir::arith::AddIOp::create(
                rewriter, loc,
                mlir::arith::ShLIOp::create(
                    rewriter, loc, packetCol,
                    c14I64Constant(rewriter, loc, 2)),
                c14I64Constant(rewriter, loc, element)));
        result = {row, col};
    } else {
        auto row = mlir::arith::AddIOp::create(
            rewriter, loc,
            mlir::arith::ShLIOp::create(
                rewriter, loc, wave, c14I64Constant(rewriter, loc, 4)),
            laneHigh);
        auto col = mlir::arith::AddIOp::create(
            rewriter, loc,
            mlir::arith::ShLIOp::create(
                rewriter, loc, laneLow, c14I64Constant(rewriter, loc, 3)),
            c14I64Constant(rewriter, loc, packet * 4 + element));
        result = {row, col};
    }
    return result;
}

// Materialize the real V producer from an ordinary global BF16 [64,64]
// source.  Four consecutive source rows are packed into one LDS store.  This
// is the physical C13/C14 path: source ownership is distributed, the C14
// fixed transpose is applied, and the rotating shared recipe computes the
// physical destination.  The debug readback is intentional for C15 only and
// lets the numerical harness check the transport independently of MFMA.
mlir::LogicalResult emitC15RealVTile(
    AMDGPUBlockDotBF16F32Op op, mlir::PatternRewriter &rewriter,
    const C14StaticPhysicalPlan &plan) {
    auto loc = op.getLoc();
    auto source = mlir::dyn_cast<mlir::MemRefType>(op.getSourceK().getType());
    auto debug = mlir::dyn_cast<mlir::MemRefType>(op.getSourceVNew().getType());
    auto stage = mlir::dyn_cast<mlir::MemRefType>(op.getBStage().getType());
    if (!source || !debug || !stage || source.getRank() != 2 ||
        source.getShape() != llvm::ArrayRef<int64_t>({64, 64}) ||
        debug.getRank() != 2 ||
        debug.getShape() != llvm::ArrayRef<int64_t>({64, 64}) ||
        stage.getRank() != 2 ||
        stage.getShape() != llvm::ArrayRef<int64_t>({64, 64}) ||
        !source.getElementType().isBF16() ||
        !debug.getElementType().isBF16() ||
        !stage.getElementType().isBF16())
        return rewriter.notifyMatchFailure(
            op, "C15 V real-tile requires global/debug/stage BF16 [64,64]");
    if (plan.transform.permutation != std::array<int64_t, 2>{1, 0})
        return rewriter.notifyMatchFailure(
            op, "C15 V real-tile requires the recovered fixed transpose");

    auto tid = toIndex(rewriter, loc, op.getThreadId());
    auto bf16x4 = mlir::VectorType::get({4}, rewriter.getBF16Type());
    // One thread owns eight packets. Each packet spans four source rows and
    // therefore becomes one packed vector in the transposed physical LDS
    // inner dimension.
    for (int64_t packet = 0; packet < 8; ++packet) {
        auto coord = emitC15VSourceCoord(rewriter, loc, tid, packet);
        auto sourceRow = c15Index(rewriter, loc, coord.rowBase);
        auto sourceCol = c15Index(rewriter, loc, coord.col);
        llvm::SmallVector<mlir::Value> values;
        values.reserve(4);
        for (int64_t element = 0; element < 4; ++element) {
            auto row = indexAdd(rewriter, loc, sourceRow,
                                indexConstant(rewriter, loc, element));
            values.push_back(mlir::memref::LoadOp::create(
                rewriter, loc, op.getSourceK(),
                mlir::ValueRange{row, sourceCol}));
        }
        auto packed = mlir::vector::FromElementsOp::create(
            rewriter, loc, bf16x4, values);

        // The C14 consumer applies the recovered [1, 0] transform to its
        // MFMA logical (output-row, K-column) coordinates. Store the source
        // [K-row, output-column] coordinates in the shared recipe so that
        // the consumer's inverse view lands on this exact element.
        auto sharedOffset = emitC14SharedElementOffset(
            rewriter, loc, plan, coord.col, coord.rowBase);
        if (mlir::failed(sharedOffset))
            return rewriter.notifyMatchFailure(
                op, "C15 V real-tile cannot compute rotating LDS offset");
        auto physicalRow = mlir::arith::ShRUIOp::create(
            rewriter, loc, *sharedOffset,
            c14I64Constant(rewriter, loc, 6));
        auto physicalCol = mlir::arith::AndIOp::create(
            rewriter, loc, *sharedOffset,
            c14I64Constant(rewriter, loc, 63));
        auto store = mlir::vector::StoreOp::create(
            rewriter, loc, packed, op.getBStage(),
            mlir::ValueRange{c15Index(rewriter, loc, physicalRow),
                             c15Index(rewriter, loc, physicalCol)});
        store->setAttr("c15.real_tile_producer", rewriter.getUnitAttr());
        store->setAttr("c15.packed_lds_store", rewriter.getUnitAttr());
        store->setAttr("c15.mapping", rewriter.getStringAttr(
                                           "distributed->fixed_transpose->rotating_shared"));
    }

    mlir::gpu::BarrierOp::create(rewriter, loc);

    // Read the packed LDS vector through the same physical address and
    // scatter only into the diagnostic global tensor.  This is not the MFMA
    // operand path; it is a lane-level observation point for the C15 oracle.
    for (int64_t packet = 0; packet < 8; ++packet) {
        auto coord = emitC15VSourceCoord(rewriter, loc, tid, packet);
        auto sharedOffset = emitC14SharedElementOffset(
            rewriter, loc, plan, coord.col, coord.rowBase);
        if (mlir::failed(sharedOffset))
            return rewriter.notifyMatchFailure(
                op, "C15 V real-tile cannot compute readback offset");
        auto physicalRow = mlir::arith::ShRUIOp::create(
            rewriter, loc, *sharedOffset,
            c14I64Constant(rewriter, loc, 6));
        auto physicalCol = mlir::arith::AndIOp::create(
            rewriter, loc, *sharedOffset,
            c14I64Constant(rewriter, loc, 63));
        auto packed = mlir::vector::LoadOp::create(
            rewriter, loc, bf16x4, op.getBStage(),
            mlir::ValueRange{c15Index(rewriter, loc, physicalRow),
                             c15Index(rewriter, loc, physicalCol)});
        packed->setAttr("c15.real_tile_consumer_readback",
                        rewriter.getUnitAttr());
        for (int64_t element = 0; element < 4; ++element) {
            auto value = mlir::vector::ExtractOp::create(
                rewriter, loc, packed, element);
            auto row = indexAdd(
                rewriter, loc, c15Index(rewriter, loc, coord.rowBase),
                indexConstant(rewriter, loc, element));
            mlir::memref::StoreOp::create(
                rewriter, loc, value, op.getSourceVNew(),
                mlir::ValueRange{row, c15Index(rewriter, loc, coord.col)});
        }
    }
    return mlir::success();
}

// C16 real Q/H/K producer and transport closure.  This is intentionally a
// single role-parameterized helper: Q/H use #blocked2, K uses #blocked1, and
// all three consume the same C13 shared-offset algebra.  The diagnostic
// readback is part of the closure so a failed MFMA result can be separated
// from a producer/shared mapping failure.
mlir::LogicalResult emitC16RealQHKTile(
    AMDGPUBlockDotBF16F32Op op, mlir::PatternRewriter &rewriter,
    const C14StaticPhysicalPlan &plan, llvm::StringRef role) {
    auto loc = op.getLoc();
    auto source = mlir::dyn_cast<mlir::MemRefType>(op.getSourceK().getType());
    auto debug = mlir::dyn_cast<mlir::MemRefType>(op.getSourceVNew().getType());
    auto stageValue = role == "Q" ? op.getAStage() : op.getBStage();
    auto stage = mlir::dyn_cast<mlir::MemRefType>(stageValue.getType());
    const std::array<int64_t, 2> shapeStorage =
        role == "K" ? std::array<int64_t, 2>{32, 64}
                     : std::array<int64_t, 2>{64, 32};
    const llvm::ArrayRef<int64_t> shape(shapeStorage);
    if (!source || !debug || !stage || source.getRank() != 2 ||
        source.getShape() != shape || debug.getRank() != 2 ||
        debug.getShape() != shape || stage.getRank() != 2 ||
        stage.getShape() != llvm::ArrayRef<int64_t>({64, 64}) ||
        !source.getElementType().isBF16() ||
        !debug.getElementType().isBF16() || !stage.getElementType().isBF16())
        return rewriter.notifyMatchFailure(
            op, "C16 Q/H/K requires BF16 source/debug [64,32] or [32,64] "
                "and a [64,64] workgroup stage");

    auto tid = toIndex(rewriter, loc, op.getThreadId());
    auto bf16x4 = mlir::VectorType::get({4}, rewriter.getBF16Type());
    const bool wg128QH = useC16WG128QH() && role != "K";
    const bool wg128QK = useC16WG128QKFull64() && role == "K";
    const bool wg128 = wg128QH || wg128QK;
    const int64_t packetCount = wg128 ? 4 : 2;
    // Each selected native producer owns two four-BF16 packets.  Q/H are
    // contiguous in their logical innermost dimension and therefore use one
    // real vector load per packet.  K is transposed in global memory; its
    // four source elements are gathered according to the same distributed
    // ownership, then committed to LDS as one packed vector.  The physical
    // shared mapping is unchanged; only the transport representation is
    // widened so C16 does not claim a scalar-scatter producer.
    for (int64_t packet = 0; packet < packetCount; ++packet) {
        auto baseCoord = emitC16SourceCoord(rewriter, loc, tid, role, packet, 0,
                                            wg128);
        mlir::Value packed;
        if (role != "K") {
            packed = mlir::vector::LoadOp::create(
                rewriter, loc, bf16x4, op.getSourceK(),
                mlir::ValueRange{c15Index(rewriter, loc, baseCoord.row),
                                 c15Index(rewriter, loc, baseCoord.col)});
        } else {
            llvm::SmallVector<mlir::Value> values;
            values.reserve(4);
            for (int64_t element = 0; element < 4; ++element) {
                auto coord = emitC16SourceCoord(rewriter, loc, tid, role,
                                                packet, element, wg128);
                values.push_back(mlir::memref::LoadOp::create(
                    rewriter, loc, op.getSourceK(),
                    mlir::ValueRange{c15Index(rewriter, loc, coord.row),
                                     c15Index(rewriter, loc, coord.col)}));
            }
            packed = mlir::vector::FromElementsOp::create(
                rewriter, loc, bf16x4, values);
        }

        // H's fixed transpose is a source-to-shared transform. Q and K use
        // identity at this producer boundary; all three roles therefore
        // share the same plan-driven physical-offset algebra. The selected
        // shared encoding carries H's fixed transpose in its order/consumer
        // contract, so no second source-coordinate swap is applied here.
        auto sharedRow = baseCoord.row;
        auto sharedCol = baseCoord.col;
        const auto &producerPlan = plan;
        auto sharedOffset =
            role == "K"
                ? emitC16NativeKPacketElementOffset(
                      rewriter, loc, sharedRow, sharedCol)
                : emitC16NativeSharedElementOffset(
                      rewriter, loc, producerPlan, sharedRow, sharedCol);
        if (mlir::failed(sharedOffset))
            return rewriter.notifyMatchFailure(
                op, "C16 Q/H/K shared offset construction failed");
        auto physicalRow = mlir::arith::ShRUIOp::create(
            rewriter, loc, *sharedOffset,
            c14I64Constant(rewriter, loc, 6));
        auto physicalCol = mlir::arith::AndIOp::create(
            rewriter, loc, *sharedOffset,
            c14I64Constant(rewriter, loc, 63));
        auto store = mlir::vector::StoreOp::create(
            rewriter, loc, packed, stageValue,
            mlir::ValueRange{c15Index(rewriter, loc, physicalRow),
                             c15Index(rewriter, loc, physicalCol)});
        store->setAttr("c16.real_tile_producer", rewriter.getUnitAttr());
        store->setAttr("c16.packed_lds_store", rewriter.getUnitAttr());
        store->setAttr("c16.packet_width", rewriter.getI64IntegerAttr(4));
    }

    mlir::gpu::BarrierOp::create(rewriter, loc);
    for (int64_t packet = 0; packet < packetCount; ++packet) {
        for (int64_t element = 0; element < 4; ++element) {
                auto coord = emitC16SourceCoord(rewriter, loc, tid, role,
                                                packet, element, wg128);
                auto sharedRow = coord.row;
                auto sharedCol = coord.col;
                const auto &producerPlan = plan;
                auto sharedOffset =
                    role == "K"
                        ? emitC16NativeKPacketElementOffset(
                              rewriter, loc, sharedRow, sharedCol)
                        : emitC16NativeSharedElementOffset(
                              rewriter, loc, producerPlan, sharedRow,
                              sharedCol);
                if (mlir::failed(sharedOffset))
                    return rewriter.notifyMatchFailure(
                        op, "C16 Q/H readback offset construction failed");
                auto physicalRow = mlir::arith::ShRUIOp::create(
                    rewriter, loc, *sharedOffset,
                    c14I64Constant(rewriter, loc, 6));
                auto physicalCol = mlir::arith::AndIOp::create(
                    rewriter, loc, *sharedOffset,
                    c14I64Constant(rewriter, loc, 63));
                auto value = mlir::memref::LoadOp::create(
                    rewriter, loc, stageValue,
                    mlir::ValueRange{c15Index(rewriter, loc, physicalRow),
                                     c15Index(rewriter, loc, physicalCol)});
                mlir::memref::StoreOp::create(
                    rewriter, loc, value, op.getSourceVNew(),
                    mlir::ValueRange{c15Index(rewriter, loc, coord.row),
                                     c15Index(rewriter, loc, coord.col)});
        }
    }
    return mlir::success();
}

struct LogicalBlockLayoutPlan {
    bool isH = false;
    mlir::Value tid;
    mlir::Value kStage;
    mlir::Value wave;
    mlir::Value lane;
    mlir::Value laneCol;
    mlir::Value laneGroup;
    mlir::Value rowHalf;
    mlir::Value valueHalf;
    mlir::Value producerLinear;
    mlir::Value packetRow;
    mlir::Value packet;
    mlir::Value packetCol;
    mlir::Value feature;
};

bool useLogicalBlockLayoutPlan() {
    auto mode = llvm::sys::Process::GetEnv("AVELANG_BLOCK_DOT_LAYOUT_PLANNER");
    return mode && *mode == "bdv2_p1_affine";
}

// C17 is the first full-region consumer of ChunkOPhysicalPlan.  The plan is
// created once by the pass and shared by every H/K logical block-dot in the
// function.  It is intentionally an experimental gate: legacy and BDV2
// source paths do not observe these attributes or the static affine recipe.
bool useC17FullPhysicalPlan() {
    auto mode = llvm::sys::Process::GetEnv(
        "AVELANG_STAGE6Z_FULL_PHYSICAL_PLAN");
    return mode && *mode == "c17";
}

bool useC18FullPhysicalRegion() {
    auto mode = llvm::sys::Process::GetEnv(
        "AVELANG_STAGE6Z_FULL_PHYSICAL_REGION");
    return mode && *mode == "c18";
}

bool useC19FullPhysicalRegion() {
    auto mode = llvm::sys::Process::GetEnv(
        "AVELANG_STAGE6Z_FULL_PHYSICAL_REGION");
    return mode && *mode == "c19";
}

// C21 keeps C19's validated full-region ownership and score/V contract, but
// changes the Q/H/K source schedule to the freshly captured WG256 native
// shape: four waves and two rotating K32 Q slots.
bool useC21SelectedNativePipeline() {
    auto mode = llvm::sys::Process::GetEnv(
        "AVELANG_STAGE6Z_FULL_PHYSICAL_REGION");
    return mode && *mode == "c21";
}

// C23 is deliberately narrower than C21's schedule experiment. It does not
// alter the source superloop, ownership, shared layout, or MFMA sequence. It
// only permits one BF16x8 K packet to remain an outstanding VMEM result until
// the existing K producer reaches its LDS publication point.
bool useC23LatePipelineMaterialization() {
    auto mode = llvm::sys::Process::GetEnv(
        "AVELANG_STAGE6Z_LATE_PIPELINE_MATERIALIZATION");
    return mode && *mode == "c23";
}

// C24 is intentionally a compiler-infrastructure mode, not another
// source-level chunk-o candidate.  Its plan is formed over an entire C21
// region before greedy block-dot rewriting starts, so a pending vector SSA
// edge never has to be owned by a RewritePattern instance.
bool useC24RegionPendingPacketInfrastructure() {
    auto mode = llvm::sys::Process::GetEnv(
        "AVELANG_STAGE6Z_PENDING_PACKET_INFRA");
    return mode && *mode == "c24";
}

// C25 does not introduce another packet representation.  It uses the C24
// issue/commit SSA pair, but establishes the current H packet before issuing
// the next K packet.  The resulting region has an explicit READY H and a
// PENDING K without changing C21 ownership, mapping, or source math.
bool useC25CurrentReadyNextPending() {
    auto mode = llvm::sys::Process::GetEnv(
        "AVELANG_STAGE6Z_CURRENT_READY_NEXT_PENDING");
    return mode && *mode == "c25";
}

bool usesC19CompatibleFullPhysicalRegion() {
    return useC19FullPhysicalRegion() || useC21SelectedNativePipeline();
}

// Compiler-internal owner of the complete physical region.  The nested
// ChunkOPhysicalPlan remains the target-independent encoding source, while
// these ownership/lifetime fields are consumed by the C18 producer and
// consumer builders rather than emitted as passive operation metadata.
struct FullPhysicalRegionPlan {
    c13::ChunkOPhysicalPlan physical;
    int64_t qCacheRows = 256;
    // C19 uses one 384-row arena (24 KiB): Q occupies rows 0..255,
    // H/K reuse rows 256..319, and score/V reuse rows 0..127 after Q's
    // last use.  C18 retains its historical 512-row offsets.
    int64_t scoreBase = 0;
    int64_t scoreVBase = 128;
    int64_t phaseBase = 320;
    int64_t c19PhaseBase = 256;
    int64_t c19SharedRows = 384;
    bool qDualConsumer = true;
    bool vConsumerOwned = true;
    bool reuseSharedAfterSource = true;
};

// C21 is intentionally an internal scheduling plan, rather than a new Qwen
// operation.  The existing logical block-dot continues to carry role and
// layout identity; this plan tells its late lowering which physical slots may
// rotate and where a producer is permitted to issue relative to a consumer.
struct ChunkOPipelinePlan {
    FullPhysicalRegionPlan fullRegion;
    int64_t workgroup = 256;
    int64_t waves = 4;
    int64_t stages = 2;
    int64_t qSlotRows = 64;
    int64_t qSlotCount = 2;
    bool qhAndQkShareK32Superloop = true;
    bool nextProducerMayIssueBeforeCurrentLastMfma = true;

    bool verify(std::string *error) const {
        if (workgroup != 256 || waves != 4 || stages != 2 || qSlotRows != 64 ||
            qSlotCount != 2) {
            if (error)
                *error = "C21 selected-native pipeline requires WG256, 4 waves, "
                         "two 64-row Q slots, and two stages";
            return false;
        }
        return fullRegion.physical.verify(error);
    }
};

bool useFirstClassMfmaOperandPlan() {
    auto mode = llvm::sys::Process::GetEnv(
        "AVELANG_BLOCK_DOT_OPERAND_PRESERVATION");
    return mode && (*mode == "p2_first_class" ||
                    *mode == "p3_packed_reuse" ||
                    *mode == "p4_accumulator_reuse");
}

bool usePackedOperandReusePlan() {
    auto mode = llvm::sys::Process::GetEnv(
        "AVELANG_BLOCK_DOT_OPERAND_PRESERVATION");
    return mode && (*mode == "p3_packed_reuse" ||
                    *mode == "p4_accumulator_reuse");
}

// P4 keeps the existing packed operand plan and changes exactly one
// representation boundary: a full-scope block-dot may consume the previous
// vector accumulator SSA value instead of reloading the same 16-f32 local
// accumulator memref.  This is intentionally an internal planner selector;
// it does not change the source block-dot ABI or the producer/consumer map.
bool useAccumulatorReusePlan() {
    auto mode = llvm::sys::Process::GetEnv(
        "AVELANG_BLOCK_DOT_OPERAND_PRESERVATION");
    return mode && *mode == "p4_accumulator_reuse";
}

LogicalBlockLayoutPlan makeLogicalBlockLayoutPlan(
    AMDGPUBlockDotBF16F32Op op, mlir::PatternRewriter &rewriter,
    mlir::Location loc, bool isH) {
    LogicalBlockLayoutPlan plan;
    plan.isH = isH;
    auto c2 = indexConstant(rewriter, loc, 2);
    auto c4 = indexConstant(rewriter, loc, 4);
    auto c8 = indexConstant(rewriter, loc, 8);
    auto c32 = indexConstant(rewriter, loc, 32);
    auto c64 = indexConstant(rewriter, loc, 64);

    plan.tid = toIndex(rewriter, loc, op.getThreadId());
    plan.kStage = toIndex(rewriter, loc, op.getKHalf());
    plan.wave = mlir::arith::DivUIOp::create(rewriter, loc, plan.tid, c64);
    plan.lane = mlir::arith::RemUIOp::create(rewriter, loc, plan.tid, c64);
    plan.laneCol =
        mlir::arith::RemUIOp::create(rewriter, loc, plan.lane, c32);
    plan.laneGroup =
        mlir::arith::DivUIOp::create(rewriter, loc, plan.lane, c32);
    plan.rowHalf =
        mlir::arith::DivUIOp::create(rewriter, loc, plan.wave, c2);
    plan.valueHalf =
        mlir::arith::RemUIOp::create(rewriter, loc, plan.wave, c2);

    if (isH) {
        plan.packetRow =
            mlir::arith::DivUIOp::create(rewriter, loc, plan.tid, c4);
        plan.packet =
            mlir::arith::RemUIOp::create(rewriter, loc, plan.tid, c4);
    } else {
        // K producer ownership is wave-pair based.  The same producerLinear
        // is also the packet identity used by the shared consumer.
        auto wavePair = mlir::arith::DivUIOp::create(
            rewriter, loc, plan.wave, c2);
        plan.producerLinear = indexAdd(
            rewriter, loc, indexMul(rewriter, loc, wavePair, c64),
            plan.lane);
        plan.packetRow = mlir::arith::DivUIOp::create(
            rewriter, loc, plan.producerLinear, c4);
        plan.packet = mlir::arith::RemUIOp::create(
            rewriter, loc, plan.producerLinear, c4);
    }
    plan.packetCol =
        indexMul(rewriter, loc, plan.packet, c8);
    plan.feature = indexAdd(
        rewriter, loc, indexMul(rewriter, loc, plan.kStage, c32),
        plan.packetCol);
    return plan;
}

// The C17 target contract fixes a wave64/WG256 mapping.  Express that
// mapping with shifts and masks instead of rebuilding the same div/rem
// decomposition independently in each logical block-dot.  The physical
// semantics still come from ChunkOPhysicalPlan; these operations only
// materialize its static affine ownership map in SSA.
LogicalBlockLayoutPlan makeC17StaticAffineLayoutPlan(
    AMDGPUBlockDotBF16F32Op op, mlir::PatternRewriter &rewriter,
    mlir::Location loc, bool isH) {
    LogicalBlockLayoutPlan plan;
    plan.isH = isH;
    auto c1 = indexConstant(rewriter, loc, 1);
    auto c2 = indexConstant(rewriter, loc, 2);
    auto c3 = indexConstant(rewriter, loc, 3);
    auto c5 = indexConstant(rewriter, loc, 5);
    auto c6 = indexConstant(rewriter, loc, 6);
    auto c7 = indexConstant(rewriter, loc, 7);
    auto c31 = indexConstant(rewriter, loc, 31);
    auto c63 = indexConstant(rewriter, loc, 63);

    plan.tid = toIndex(rewriter, loc, op.getThreadId());
    plan.kStage = toIndex(rewriter, loc, op.getKHalf());
    plan.wave = mlir::arith::ShRUIOp::create(rewriter, loc, plan.tid, c6);
    plan.lane = mlir::arith::AndIOp::create(rewriter, loc, plan.tid, c63);
    plan.laneCol = mlir::arith::AndIOp::create(rewriter, loc, plan.tid, c31);
    plan.laneGroup = mlir::arith::AndIOp::create(
        rewriter, loc,
        mlir::arith::ShRUIOp::create(rewriter, loc, plan.tid, c5), c1);
    plan.rowHalf = mlir::arith::ShRUIOp::create(rewriter, loc, plan.tid, c7);
    plan.valueHalf = mlir::arith::AndIOp::create(
        rewriter, loc,
        mlir::arith::ShRUIOp::create(rewriter, loc, plan.tid, c6), c1);

    if (isH) {
        plan.packetRow = mlir::arith::ShRUIOp::create(
            rewriter, loc, plan.tid, c2);
        plan.packet = mlir::arith::AndIOp::create(rewriter, loc, plan.tid, c3);
    } else {
        // K producer ownership folds the wave-pair into a 128-packet linear
        // space.  For tid in [0,255], this is (tid>>7)*64 + (tid&63).
        auto wavePair = mlir::arith::ShRUIOp::create(
            rewriter, loc, plan.tid, c7);
        plan.producerLinear = mlir::arith::AddIOp::create(
            rewriter, loc,
            mlir::arith::ShLIOp::create(rewriter, loc, wavePair,
                                        indexConstant(rewriter, loc, 6)),
            plan.lane);
        plan.packetRow = mlir::arith::ShRUIOp::create(
            rewriter, loc, plan.producerLinear, c2);
        plan.packet = mlir::arith::AndIOp::create(
            rewriter, loc, plan.producerLinear, c3);
    }
    plan.packetCol = mlir::arith::ShLIOp::create(
        rewriter, loc, plan.packet, indexConstant(rewriter, loc, 3));
    plan.feature = mlir::arith::AddIOp::create(
        rewriter, loc,
        mlir::arith::ShLIOp::create(rewriter, loc, plan.kStage,
                                    indexConstant(rewriter, loc, 5)),
        plan.packetCol);
    return plan;
}

const c13::PhysicalBlockPlan *getC17PhysicalBlockPlan(
    const c13::ChunkOPhysicalPlan &plan, llvm::StringRef role) {
    if (role == "H")
        return &plan.h;
    if (role == "K")
        return &plan.k;
    if (role == "Q")
        return &plan.q;
    if (role == "V")
        return &plan.v;
    return nullptr;
}

void annotateC17PhysicalPlan(
    mlir::Operation *operation, const c13::ChunkOPhysicalPlan &plan,
    llvm::StringRef role, llvm::StringRef consumer) {
    auto *block = getC17PhysicalBlockPlan(plan, role);
    if (!block)
        return;
    auto *context = operation->getContext();
    operation->setAttr("c17.full_physical_plan",
                       mlir::StringAttr::get(context, "gfx942_bt64_bv64_wg256"));
    operation->setAttr("c17.plan_role", mlir::StringAttr::get(context, role));
    operation->setAttr("c17.consumer", mlir::StringAttr::get(context, consumer));
    operation->setAttr("c17.distributed", block->distributed.toAttr(context));
    operation->setAttr("c17.shared", block->shared.toAttr(context));
    operation->setAttr("c17.dot", block->dot.toAttr(context));
    operation->setAttr("c17.transform", block->transform.toAttr(context));
    operation->setAttr("c17.mfma", plan.mfma.toAttr(context));
    operation->setAttr("c17.source_lifetime",
                       mlir::StringAttr::get(context, "source_prologue_to_release"));
    operation->setAttr("c17.q_dual_consumer",
                       mlir::StringAttr::get(context, "Q_to_Q@H_and_Q@K"));
}

void annotateC18PhysicalRegion(
    mlir::Operation *operation, const FullPhysicalRegionPlan &plan,
    llvm::StringRef role, llvm::StringRef consumer) {
    auto *block = getC17PhysicalBlockPlan(plan.physical, role);
    if (!block)
        return;
    auto *context = operation->getContext();
    operation->setAttr("c18.full_physical_region",
                       mlir::StringAttr::get(context,
                                             "gfx942_bt64_bv64_wg256"));
    operation->setAttr("c18.plan_role", mlir::StringAttr::get(context, role));
    operation->setAttr("c18.consumer",
                       mlir::StringAttr::get(context, consumer));
    operation->setAttr("c18.producer_owner",
                       mlir::StringAttr::get(context, "FullPhysicalRegionPlan"));
    operation->setAttr("c18.shared_region",
                       mlir::StringAttr::get(context, "q_cache_plus_phase"));
    operation->setAttr("c18.shared_lifetime",
                       mlir::StringAttr::get(
                           context, plan.reuseSharedAfterSource
                               ? "source_then_score_v_reuse"
                               : "source_to_release"));
    operation->setAttr("c18.q_dual_consumer",
                       mlir::BoolAttr::get(context, plan.qDualConsumer));
    operation->setAttr("c18.v_source_consumer_owned",
                       mlir::BoolAttr::get(context, plan.vConsumerOwned));
    operation->setAttr("c18.q_cache_rows",
                       mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                              plan.qCacheRows));
    operation->setAttr("c18.score_v_base",
                       mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                              plan.scoreVBase));
}

void annotateC19PhysicalRegion(
    mlir::Operation *operation, const FullPhysicalRegionPlan &plan,
    llvm::StringRef role, llvm::StringRef consumer) {
    auto *block = getC17PhysicalBlockPlan(plan.physical, role);
    if (!block)
        return;
    auto *context = operation->getContext();
    operation->setAttr(
        "c19.full_physical_region",
        mlir::StringAttr::get(context, "gfx942_bt64_bv64_wg256"));
    operation->setAttr("c19.plan_role", mlir::StringAttr::get(context, role));
    operation->setAttr("c19.consumer",
                       mlir::StringAttr::get(context, consumer));
    operation->setAttr(
        "c19.owner", mlir::StringAttr::get(context, "FullPhysicalRegionPlan"));
    operation->setAttr("c19.shared_arena",
                       mlir::StringAttr::get(context, "single_source_score_arena"));
    operation->setAttr(
        "c19.q_dual_consumer",
        mlir::BoolAttr::get(context, plan.qDualConsumer));
    operation->setAttr(
        "c19.v_source_consumer_owned",
        mlir::BoolAttr::get(context, plan.vConsumerOwned));
    operation->setAttr(
        "c19.shared_lifetime",
        mlir::StringAttr::get(context, "source_Q_H_K_then_score_V"));
    operation->setAttr(
        "c19.legacy_owner", mlir::BoolAttr::get(context, false));
}

void annotateC21SelectedNativePipeline(
    mlir::Operation *operation, const ChunkOPipelinePlan &plan,
    llvm::StringRef role, llvm::StringRef consumer) {
    auto *context = operation->getContext();
    operation->setAttr(
        "c21.selected_native_pipeline",
        mlir::StringAttr::get(context, "gfx942_bt64_bv64_wg256_stage2"));
    operation->setAttr("c21.plan_owner",
                       mlir::StringAttr::get(context, "ChunkOPipelinePlan"));
    operation->setAttr("c21.plan_role", mlir::StringAttr::get(context, role));
    operation->setAttr("c21.consumer",
                       mlir::StringAttr::get(context, consumer));
    operation->setAttr("c21.workgroup",
                       mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                              plan.workgroup));
    operation->setAttr("c21.waves",
                       mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                              plan.waves));
    operation->setAttr("c21.stages",
                       mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                              plan.stages));
    operation->setAttr(
        "c21.q_slot",
        mlir::StringAttr::get(context,
                              role == "Q" || role == "H" || role == "K"
                                  ? "k_stage_mod_2"
                                  : "score_v_after_qhk_release"));
    operation->setAttr(
        "c21.qh_qk_superloop",
        mlir::BoolAttr::get(context, plan.qhAndQkShareK32Superloop));
    operation->setAttr(
        "c21.next_issue_policy",
        mlir::StringAttr::get(
            context, plan.nextProducerMayIssueBeforeCurrentLastMfma
                         ? "eligible_before_current_last_mfma"
                         : "after_current_last_mfma"));
}

// The planner places these placeholders immediately after the opaque
// block-dot so their vector result reaches the existing recurrence-tail
// commit.  Only this late lowering knows the four actual update MFMA groups;
// it relocates each placeholder to the selected group last-use.
bool isCoreLastUseDeferredIssue(AMDGPUQwenK64CoreIssueOp op) {
    return op->hasAttr("avelang.qwen.core_staggered.deferred_block_dot");
}

// R4 is one complete recurrence-plan mode.  Its K producer writes the
// preloaded bank as token-major BF16x8 packets, so only the update dot's
// preloaded-K consumer changes physical LDS interpretation.
bool usesJointV4LdsRetile() {
    const auto mode = llvm::sys::Process::GetEnv("AVELANG_PERSISTENT_RECURRENCE_LOWERING");
    return mode == std::optional<std::string>("gfx942_bt64_bv32_joint_v4") ||
           // Tail-issue is a strict R4 ordering control: it retains the
           // identical R4 LDS-mediated K consumer.
           mode == std::optional<std::string>("gfx942_bt64_bv32_joint_v4_tail_issue") ||
           // BV-consume candidates retain the R4 token-major K producer and
           // the same physical V32 MFMA consumer.  Only the number of V32
           // consumers represented by one recurrence CTA changes.
           (mode && (llvm::StringRef(*mode) ==
                         "gfx942_bt64_bv16_joint_v4_tail_issue" ||
                     llvm::StringRef(*mode) ==
                         "gfx942_bt64_bv64_joint_v4_tail_issue")) ||
           mode == std::optional<std::string>("gfx942_bt64_bv32_joint_v5") ||
           // The generic modulo scheduler reuses the validated R4 producer
           // and its LDS-mediated K retile; only its loop schedule differs.
           mode == std::optional<std::string>("gfx942_bt64_bv32_software_pipeline") ||
           // Experimental microtiles change only packet issue/commit timing
           // inside the existing R4 full recurrence plan.
           (mode && llvm::StringRef(*mode).starts_with(
                        "gfx942_bt64_bv32_microtile_experimental_")) ||
           (mode && llvm::StringRef(*mode).starts_with(
                        "gfx942_bt64_bv32_core_lastuse_experimental_"));
}

// BV64 fuses two native V32 consumers into one 256-thread CTA.  Waves
// (0,1) own the first V32 and (2,3) the second; each pair retains R4's
// K-half ownership.  This is deliberately mode-gated so the baseline two
// wave mapping remains byte-for-byte unchanged.
bool usesBv64FourWaveOwnership() {
    return llvm::sys::Process::GetEnv("AVELANG_PERSISTENT_RECURRENCE_LOWERING") ==
           std::optional<std::string>("gfx942_bt64_bv64_joint_v4_tail_issue");
}

mlir::Value toIndex(mlir::PatternRewriter &rewriter, mlir::Location loc,
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

mlir::Value makeZeroVector(mlir::PatternRewriter &rewriter, mlir::Location loc,
                           mlir::VectorType type) {
    auto zero = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(0.0F));
    return mlir::vector::SplatOp::create(rewriter, loc, type, zero);
}

mlir::Value materializeAccumulatorVector(mlir::PatternRewriter &rewriter,
                                         mlir::Location loc,
                                         mlir::Value accumulator) {
    if (mlir::isa<mlir::VectorType>(accumulator.getType())) {
        return accumulator;
    }
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(accumulator.getType());
    if (!memrefType || memrefType.getRank() != 1 ||
        memrefType.getShape() != llvm::ArrayRef<int64_t>({16}) ||
        !memrefType.getElementType().isF32()) {
        return {};
    }
    auto zero = indexConstant(rewriter, loc, 0);
    auto vectorType = mlir::VectorType::get({16}, rewriter.getF32Type());
    return mlir::vector::LoadOp::create(rewriter, loc, vectorType, accumulator,
                                        mlir::ValueRange{zero});
}

// A vector zeroing store inside an enclosing loop is the reset boundary for
// the source-level accumulator (the score accumulator has exactly this
// shape).  Scalar stores after a block dot are ordinary commit stores and do
// not invalidate forwarding.  The conservative check therefore forwards only
// chains whose enclosing loops contain no vector reset for this memref.
bool hasEnclosingAccumulatorReset(mlir::Operation *op,
                                  mlir::Value accumulator) {
    for (mlir::Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
        auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent);
        if (!loop)
            continue;
        bool reset = false;
        loop.walk([&](mlir::vector::StoreOp store) {
            if (store->getNumOperands() > 1 &&
                store->getOperand(1) == accumulator)
                reset = true;
        });
        if (reset)
            return true;
    }
    return false;
}

using AccumulatorForwardingMap = llvm::DenseMap<mlir::Value, mlir::Value>;

mlir::Value extractBF16x4(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::Value vector8, int64_t offset) {
    auto bf16 = rewriter.getBF16Type();
    auto resultType = mlir::VectorType::get({4}, bf16);
    llvm::SmallVector<mlir::Value> values;
    values.reserve(4);
    for (int64_t index = 0; index < 4; ++index) {
        values.push_back(mlir::vector::ExtractOp::create(rewriter, loc, vector8,
                                                         offset + index));
    }
    return mlir::vector::FromElementsOp::create(rewriter, loc, resultType,
                                                values);
}

mlir::Value makeWorkgroupScratch(mlir::PatternRewriter &rewriter,
                                 mlir::Location loc, mlir::Value prototype,
                                 llvm::ArrayRef<int64_t> shape) {
    auto prototypeType =
        mlir::dyn_cast<mlir::MemRefType>(prototype.getType());
    if (!prototypeType) {
        return {};
    }
    auto type = mlir::MemRefType::get(
        shape, rewriter.getBF16Type(), mlir::MemRefLayoutAttrInterface(),
        prototypeType.getMemorySpace());
    return mlir::memref::AllocaOp::create(
        rewriter, loc, type, mlir::ValueRange{},
        rewriter.getI64IntegerAttr(16));
}

mlir::Value emitMfmaPair(mlir::PatternRewriter &rewriter, mlir::Location loc,
                         mlir::Value aStage, mlir::Value bStage,
                         mlir::Value aWave, mlir::Value laneCol,
                         mlir::Value laneGroup, mlir::Value accumulator) {
    auto bf16x8 = mlir::VectorType::get({8}, rewriter.getBF16Type());
    auto c8 = indexConstant(rewriter, loc, 8);
    auto c2 = indexConstant(rewriter, loc, 2);
    auto firstWord = laneGroup;
    auto secondWord = indexAdd(rewriter, loc, laneGroup, c2);
    auto mfmaName = ir::intrinsics::MakeIntrinsicFuncName(
        "amdgpu", "rocdl_mfma_f32_32x32x8bf16_1k");

    for (mlir::Value word : {firstWord, secondWord}) {
        auto elementOffset = indexMul(rewriter, loc, word, c8);
        auto aRaw = mlir::vector::LoadOp::create(
            rewriter, loc, bf16x8, aStage,
            mlir::ValueRange{aWave, laneCol, elementOffset});
        auto bRaw = mlir::vector::LoadOp::create(
            rewriter, loc, bf16x8, bStage,
            mlir::ValueRange{laneCol, elementOffset});
        for (int64_t fragmentOffset : {int64_t{0}, int64_t{4}}) {
            auto aFrag = extractBF16x4(rewriter, loc, aRaw, fragmentOffset);
            auto bFrag = extractBF16x4(rewriter, loc, bRaw, fragmentOffset);
            auto call = mlir::func::CallOp::create(
                rewriter, loc, mfmaName, mlir::TypeRange{accumulator.getType()},
                mlir::ValueRange{bFrag, aFrag, accumulator});
            call->setAttr("avelang.block_dot.mfma32", rewriter.getUnitAttr());
            accumulator = call.getResult(0);
        }
    }
    return accumulator;
}

mlir::Value emitMfmaForOwnership(mlir::PatternRewriter &rewriter,
                                 mlir::Location loc, AMDGPUBlockDotBF16F32Op op,
                                 mlir::Value wave, mlir::Value laneCol,
                                 mlir::Value laneGroup, mlir::Value accumulator,
                                 bool cooperativeBv32) {
    if (!cooperativeBv32) {
        return emitMfmaPair(rewriter, loc, op.getAStage(), op.getBStage(), wave,
                            laneCol, laneGroup, accumulator);
    }

    // Each K64 block is assigned to one wave. The other wave still stages
    // shared operands and reaches the surrounding barriers, but does not
    // compute a duplicate V32xK64 delta that would be discarded by source.
    auto activeWave = toIndex(rewriter, loc, op.getKHalf());
    auto waveOwner = wave;
    if (usesBv64FourWaveOwnership()) {
        waveOwner = mlir::arith::RemUIOp::create(
            rewriter, loc, wave, indexConstant(rewriter, loc, 2));
    }
    auto active = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::eq, waveOwner, activeWave);
    auto guarded = mlir::scf::IfOp::create(
        rewriter, loc, mlir::TypeRange{accumulator.getType()}, active,
        /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&guarded.getThenRegion().front());
    auto zero = indexConstant(rewriter, loc, 0);
    auto updated = emitMfmaPair(rewriter, loc, op.getAStage(), op.getBStage(),
                                zero, laneCol, laneGroup, accumulator);
    mlir::scf::YieldOp::create(rewriter, loc, updated);
    rewriter.setInsertionPointToStart(&guarded.getElseRegion().front());
    mlir::scf::YieldOp::create(rewriter, loc, accumulator);
    rewriter.setInsertionPointAfter(guarded);
    return guarded.getResult(0);
}

mlir::Value emitPersistentMfmaPair(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value aBlock,
    mlir::Value bBlock, mlir::Value laneCol, mlir::Value laneGroup,
    mlir::Value tokenHalf, int64_t colHalf, mlir::Value accumulator,
    bool preloadedK, mlir::Value kHalf) {
    auto bf16x8 = mlir::VectorType::get({8}, rewriter.getBF16Type());
    auto c0 = indexConstant(rewriter, loc, 0);
    auto c1 = indexConstant(rewriter, loc, 1);
    auto c2 = indexConstant(rewriter, loc, 2);
    auto c8 = indexConstant(rewriter, loc, 8);
    auto c32 = indexConstant(rewriter, loc, 32);
    auto tokenBase = indexMul(rewriter, loc, tokenHalf, c32);
    auto kBase = indexConstant(rewriter, loc, colHalf * 32);
    auto firstWord = laneGroup;
    auto secondWord = indexAdd(rewriter, loc, laneGroup, c2);
    auto mfmaName = ir::intrinsics::MakeIntrinsicFuncName(
        "amdgpu", "rocdl_mfma_f32_32x32x8bf16_1k");

    for (mlir::Value word : {firstWord, secondWord}) {
        auto elementOffset = indexMul(rewriter, loc, word, c8);
        auto tokenOffset = indexAdd(rewriter, loc, tokenBase, elementOffset);
        auto aRaw = mlir::vector::LoadOp::create(
            rewriter, loc, bf16x8, aBlock,
            mlir::ValueRange{c0, laneCol, tokenOffset});
        auto bRow = indexAdd(rewriter, loc, kBase, laneCol);
        auto bRaw = preloadedK
                        ? mlir::vector::LoadOp::create(
                              rewriter, loc, bf16x8, bBlock,
                              mlir::ValueRange{kHalf, bRow, tokenOffset})
                        : mlir::vector::LoadOp::create(
                              rewriter, loc, bf16x8, bBlock,
                              mlir::ValueRange{bRow, tokenOffset});
        for (int64_t fragmentOffset : {int64_t{0}, int64_t{4}}) {
            auto aFrag = extractBF16x4(rewriter, loc, aRaw, fragmentOffset);
            auto bFrag = extractBF16x4(rewriter, loc, bRaw, fragmentOffset);
            auto call = mlir::func::CallOp::create(
                rewriter, loc, mfmaName, mlir::TypeRange{accumulator.getType()},
                mlir::ValueRange{bFrag, aFrag, accumulator});
            call->setAttr("avelang.block_dot.mfma32", rewriter.getUnitAttr());
            accumulator = call.getResult(0);
        }
    }
    return accumulator;
}

mlir::Value emitPersistentMfmaForOwnership(
    mlir::PatternRewriter &rewriter, mlir::Location loc,
    AMDGPUBlockDotBF16F32Op op, mlir::Value aBlock, mlir::Value bBlock,
    mlir::Value wave, mlir::Value laneCol, mlir::Value laneGroup,
    mlir::Value tokenHalf, int64_t colHalf, mlir::Value accumulator,
    bool preloadedK) {
    auto activeWave = toIndex(rewriter, loc, op.getKHalf());
    auto waveOwner = wave;
    if (usesBv64FourWaveOwnership()) {
        waveOwner = mlir::arith::RemUIOp::create(
            rewriter, loc, wave, indexConstant(rewriter, loc, 2));
    }
    auto active = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::eq, waveOwner, activeWave);
    auto guarded = mlir::scf::IfOp::create(
        rewriter, loc, mlir::TypeRange{accumulator.getType()}, active,
        /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&guarded.getThenRegion().front());
    auto kHalf = toIndex(rewriter, loc, op.getKHalf());
    auto updated = emitPersistentMfmaPair(rewriter, loc, aBlock, bBlock,
                                          laneCol, laneGroup, tokenHalf,
                                          colHalf, accumulator, preloadedK,
                                          kHalf);
    mlir::scf::YieldOp::create(rewriter, loc, updated);
    rewriter.setInsertionPointToStart(&guarded.getElseRegion().front());
    mlir::scf::YieldOp::create(rewriter, loc, accumulator);
    rewriter.setInsertionPointAfter(guarded);
    return guarded.getResult(0);
}

// A token-major R4 bank stores one producer-owned BF16x8 packet at
// [half, token, K]. MFMA still needs one K row across eight tokens. AveLang's
// current public vector load has no strided-BF16x8 fragment form, so this is
// intentionally the explicit LDS-mediated gather fallback. It lets the full
// recurrence measure whether an address-layout retile alone can replace R3's
// cross-lane bpermute path without pretending the gather is a typed wide read.
mlir::Value loadPreloadedTokenMajorKVector(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value kBlock,
    mlir::Value kHalf, mlir::Value tokenOffset, mlir::Value row) {
    auto bf16x8 = mlir::VectorType::get({8}, rewriter.getBF16Type());
    llvm::SmallVector<mlir::Value> values;
    values.reserve(8);
    for (int64_t element = 0; element < 8; ++element) {
        auto token = indexAdd(rewriter, loc, tokenOffset,
                              indexConstant(rewriter, loc, element));
        values.push_back(mlir::memref::LoadOp::create(
            rewriter, loc, kBlock, mlir::ValueRange{kHalf, token, row}));
    }
    auto result = mlir::vector::FromElementsOp::create(rewriter, loc, bf16x8,
                                                        values);
    result->setAttr("avelang.qwen.joint_v4.lds_gather_fragment",
                    rewriter.getUnitAttr());
    return result;
}

mlir::Value emitPersistentRetiledMfmaPair(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value aBlock,
    mlir::Value kBlock, mlir::Value laneCol, mlir::Value laneGroup,
    mlir::Value tokenHalf, int64_t colHalf, mlir::Value accumulator,
    mlir::Value kHalf, bool stateKV) {
    auto bf16x8 = mlir::VectorType::get({8}, rewriter.getBF16Type());
    auto c0 = indexConstant(rewriter, loc, 0);
    auto c2 = indexConstant(rewriter, loc, 2);
    auto c8 = indexConstant(rewriter, loc, 8);
    auto c32 = indexConstant(rewriter, loc, 32);
    auto tokenBase = indexMul(rewriter, loc, tokenHalf, c32);
    auto kBase = indexConstant(rewriter, loc, colHalf * 32);
    auto firstWord = laneGroup;
    auto secondWord = indexAdd(rewriter, loc, laneGroup, c2);
    auto mfmaName = ir::intrinsics::MakeIntrinsicFuncName(
        "amdgpu", "rocdl_mfma_f32_32x32x8bf16_1k");

    for (mlir::Value word : {firstWord, secondWord}) {
        auto elementOffset = indexMul(rewriter, loc, word, c8);
        auto tokenOffset = indexAdd(rewriter, loc, tokenBase, elementOffset);
        auto aRaw = mlir::vector::LoadOp::create(
            rewriter, loc, bf16x8, aBlock,
            mlir::ValueRange{c0, laneCol, tokenOffset});
        auto bRow = indexAdd(rewriter, loc, kBase, laneCol);
        auto bRaw = loadPreloadedTokenMajorKVector(rewriter, loc, kBlock,
                                                    kHalf, tokenOffset, bRow);
        for (int64_t fragmentOffset : {int64_t{0}, int64_t{4}}) {
            auto aFrag = extractBF16x4(rewriter, loc, aRaw, fragmentOffset);
            auto bFrag = extractBF16x4(rewriter, loc, bRaw, fragmentOffset);
            auto call = mlir::func::CallOp::create(
                rewriter, loc, mfmaName, mlir::TypeRange{accumulator.getType()},
                stateKV ? mlir::ValueRange{aFrag, bFrag, accumulator}
                        : mlir::ValueRange{bFrag, aFrag, accumulator});
            call->setAttr("avelang.block_dot.mfma32", rewriter.getUnitAttr());
            if (stateKV)
                call->setAttr("avelang.block_dot.state_kv",
                              rewriter.getUnitAttr());
            accumulator = call.getResult(0);
        }
    }
    return accumulator;
}

mlir::Value emitPersistentRetiledMfmaForOwnership(
    mlir::PatternRewriter &rewriter, mlir::Location loc,
    AMDGPUBlockDotBF16F32Op op, mlir::Value aBlock, mlir::Value kBlock,
    mlir::Value wave, mlir::Value laneCol, mlir::Value laneGroup,
    mlir::Value tokenHalf, int64_t colHalf, mlir::Value accumulator,
    bool stateKV) {
    auto activeWave = toIndex(rewriter, loc, op.getKHalf());
    auto waveOwner = wave;
    if (usesBv64FourWaveOwnership()) {
        waveOwner = mlir::arith::RemUIOp::create(
            rewriter, loc, wave, indexConstant(rewriter, loc, 2));
    }
    auto active = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::eq, waveOwner, activeWave);
    auto guarded = mlir::scf::IfOp::create(
        rewriter, loc, mlir::TypeRange{accumulator.getType()}, active,
        /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&guarded.getThenRegion().front());
    auto updated = emitPersistentRetiledMfmaPair(
        rewriter, loc, aBlock, kBlock, laneCol, laneGroup, tokenHalf, colHalf,
        accumulator, activeWave, stateKV);
    mlir::scf::YieldOp::create(rewriter, loc, updated);
    rewriter.setInsertionPointToStart(&guarded.getElseRegion().front());
    mlir::scf::YieldOp::create(rewriter, loc, accumulator);
    rewriter.setInsertionPointAfter(guarded);
    return guarded.getResult(0);
}

// C0.5 uses a producer-friendly token-major LDS orientation: a global
// BF16x8 slice contiguous in V/K can be stored to LDS as one vector.  MFMA
// consumes the transpose, so this control gathers a fixed V/K row across
// eight tokens.  The operation schedule and accumulation order remain C0.
mlir::Value loadPackedTokenVector(mlir::PatternRewriter &rewriter,
                                  mlir::Location loc, mlir::Value packedBlock,
                                  mlir::Value tokenOffset, mlir::Value row) {
    auto bf16x8 = mlir::VectorType::get({8}, rewriter.getBF16Type());
    llvm::SmallVector<mlir::Value> values;
    values.reserve(8);
    for (int64_t element = 0; element < 8; ++element) {
        auto token = indexAdd(rewriter, loc, tokenOffset,
                              indexConstant(rewriter, loc, element));
        values.push_back(mlir::memref::LoadOp::create(
            rewriter, loc, packedBlock, mlir::ValueRange{token, row}));
    }
    return mlir::vector::FromElementsOp::create(rewriter, loc, bf16x8,
                                                values);
}

mlir::Value emitPersistentPackedMfmaPair(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value aBlock,
    mlir::Value bBlock, mlir::Value laneCol, mlir::Value laneGroup,
    mlir::Value tokenHalf, int64_t colHalf, mlir::Value accumulator) {
    auto c2 = indexConstant(rewriter, loc, 2);
    auto c8 = indexConstant(rewriter, loc, 8);
    auto c32 = indexConstant(rewriter, loc, 32);
    auto tokenBase = indexMul(rewriter, loc, tokenHalf, c32);
    auto kBase = indexConstant(rewriter, loc, colHalf * 32);
    auto firstWord = laneGroup;
    auto secondWord = indexAdd(rewriter, loc, laneGroup, c2);
    auto bRow = indexAdd(rewriter, loc, kBase, laneCol);
    auto mfmaName = ir::intrinsics::MakeIntrinsicFuncName(
        "amdgpu", "rocdl_mfma_f32_32x32x8bf16_1k");

    for (mlir::Value word : {firstWord, secondWord}) {
        auto elementOffset = indexMul(rewriter, loc, word, c8);
        auto tokenOffset = indexAdd(rewriter, loc, tokenBase, elementOffset);
        auto aRaw = loadPackedTokenVector(rewriter, loc, aBlock, tokenOffset,
                                          laneCol);
        auto bRaw = loadPackedTokenVector(rewriter, loc, bBlock, tokenOffset,
                                          bRow);
        for (int64_t fragmentOffset : {int64_t{0}, int64_t{4}}) {
            auto aFrag = extractBF16x4(rewriter, loc, aRaw, fragmentOffset);
            auto bFrag = extractBF16x4(rewriter, loc, bRaw, fragmentOffset);
            auto call = mlir::func::CallOp::create(
                rewriter, loc, mfmaName, mlir::TypeRange{accumulator.getType()},
                mlir::ValueRange{bFrag, aFrag, accumulator});
            call->setAttr("avelang.block_dot.mfma32", rewriter.getUnitAttr());
            accumulator = call.getResult(0);
        }
    }
    return accumulator;
}

mlir::Value emitPersistentPackedMfmaForOwnership(
    mlir::PatternRewriter &rewriter, mlir::Location loc,
    AMDGPUBlockDotBF16F32Op op, mlir::Value aBlock, mlir::Value bBlock,
    mlir::Value wave, mlir::Value laneCol, mlir::Value laneGroup,
    mlir::Value tokenHalf, int64_t colHalf, mlir::Value accumulator) {
    auto activeWave = toIndex(rewriter, loc, op.getKHalf());
    auto active = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::eq, wave, activeWave);
    auto guarded = mlir::scf::IfOp::create(
        rewriter, loc, mlir::TypeRange{accumulator.getType()}, active,
        /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&guarded.getThenRegion().front());
    auto updated = emitPersistentPackedMfmaPair(
        rewriter, loc, aBlock, bBlock, laneCol, laneGroup, tokenHalf, colHalf,
        accumulator);
    mlir::scf::YieldOp::create(rewriter, loc, updated);
    rewriter.setInsertionPointToStart(&guarded.getElseRegion().front());
    mlir::scf::YieldOp::create(rewriter, loc, accumulator);
    rewriter.setInsertionPointAfter(guarded);
    return guarded.getResult(0);
}

// Stage one contiguous BF16x8 V slice per thread, then scatter it into the
// unchanged [wave, V, token] LDS tile. The scalar reference producer loads
// the exact same 1024 values; this variant only preserves their typed
// contiguous global-load form until the operand-local store.
void emitTypedVectorStageA(mlir::PatternRewriter &rewriter, mlir::Location loc,
                           AMDGPUBlockDotBF16F32Op op, mlir::Value tokenHalf,
                           bool precomputedVDecay) {
    auto c0 = indexConstant(rewriter, loc, 0);
    auto c4 = indexConstant(rewriter, loc, 4);
    auto c8 = indexConstant(rewriter, loc, 8);
    auto c32 = indexConstant(rewriter, loc, 32);
    auto tid = toIndex(rewriter, loc, op.getThreadId());
    auto chunkStart = toIndex(rewriter, loc, op.getChunkStart());
    auto valueHead = toIndex(rewriter, loc, op.getValueHead());
    auto valueBase = toIndex(rewriter, loc, op.getValueBase());
    auto tokenBase = indexAdd(rewriter, loc, chunkStart,
                              indexMul(rewriter, loc, tokenHalf, c32));
    auto aToken = mlir::arith::DivUIOp::create(rewriter, loc, tid, c4);
    auto rowGroup = mlir::arith::RemUIOp::create(rewriter, loc, tid, c4);
    auto rowBase = indexMul(rewriter, loc, rowGroup, c8);
    auto token = indexAdd(rewriter, loc, tokenBase, aToken);
    auto globalV = indexAdd(rewriter, loc, valueBase, rowBase);
    auto bf16x8 = mlir::VectorType::get({8}, rewriter.getBF16Type());
    mlir::Value staged = mlir::vector::LoadOp::create(
        rewriter, loc, bf16x8, op.getSourceVNew(),
        mlir::ValueRange{c0, token, valueHead, globalV});
    if (!precomputedVDecay) {
        auto gValue = mlir::memref::LoadOp::create(
            rewriter, loc, op.getSourceG(),
            mlir::ValueRange{c0, token, valueHead});
        auto diff =
            mlir::arith::SubFOp::create(rewriter, loc, op.getGLast(), gValue);
        auto decay = mlir::math::ExpOp::create(rewriter, loc, diff);
        auto f32x8 = mlir::VectorType::get({8}, rewriter.getF32Type());
        auto vF32 = mlir::arith::ExtFOp::create(
            rewriter, loc, f32x8, staged, mlir::arith::FastMathFlagsAttr{});
        auto scale = mlir::vector::SplatOp::create(rewriter, loc, f32x8, decay);
        auto scaled = mlir::arith::MulFOp::create(rewriter, loc, vF32, scale);
        staged = mlir::arith::TruncFOp::create(rewriter, loc, bf16x8, scaled);
    }
    for (int64_t element = 0; element < 8; ++element) {
        auto value =
            mlir::vector::ExtractOp::create(rewriter, loc, staged, element);
        auto row = indexAdd(rewriter, loc, rowBase,
                            indexConstant(rewriter, loc, element));
        mlir::memref::StoreOp::create(rewriter, loc, value, op.getAStage(),
                                      mlir::ValueRange{c0, row, aToken});
    }
}

// The K tile has the same 32x32 logical shape as the V tile. A thread owns a
// contiguous Kx8 segment for one token and scatters it into the existing
// [K, token] LDS tile consumed by emitMfmaPair unchanged.
void emitTypedVectorStageB(mlir::PatternRewriter &rewriter, mlir::Location loc,
                           AMDGPUBlockDotBF16F32Op op, mlir::Value tokenHalf,
                           int64_t colHalf) {
    auto c0 = indexConstant(rewriter, loc, 0);
    auto c4 = indexConstant(rewriter, loc, 4);
    auto c8 = indexConstant(rewriter, loc, 8);
    auto c32 = indexConstant(rewriter, loc, 32);
    auto c64 = indexConstant(rewriter, loc, 64);
    auto tid = toIndex(rewriter, loc, op.getThreadId());
    auto chunkStart = toIndex(rewriter, loc, op.getChunkStart());
    auto keyHead = toIndex(rewriter, loc, op.getKeyHead());
    auto kHalf = toIndex(rewriter, loc, op.getKHalf());
    auto tokenBase = indexAdd(rewriter, loc, chunkStart,
                              indexMul(rewriter, loc, tokenHalf, c32));
    auto kBase = indexAdd(rewriter, loc, indexMul(rewriter, loc, kHalf, c64),
                          indexConstant(rewriter, loc, colHalf * 32));
    auto bToken = mlir::arith::DivUIOp::create(rewriter, loc, tid, c4);
    auto rowGroup = mlir::arith::RemUIOp::create(rewriter, loc, tid, c4);
    auto rowBase = indexMul(rewriter, loc, rowGroup, c8);
    auto token = indexAdd(rewriter, loc, tokenBase, bToken);
    auto kColumn = indexAdd(rewriter, loc, kBase, rowBase);
    auto bf16x8 = mlir::VectorType::get({8}, rewriter.getBF16Type());
    auto staged = mlir::vector::LoadOp::create(
        rewriter, loc, bf16x8, op.getSourceK(),
        mlir::ValueRange{c0, token, keyHead, kColumn});
    for (int64_t element = 0; element < 8; ++element) {
        auto value =
            mlir::vector::ExtractOp::create(rewriter, loc, staged, element);
        auto row = indexAdd(rewriter, loc, rowBase,
                            indexConstant(rewriter, loc, element));
        mlir::memref::StoreOp::create(rewriter, loc, value, op.getBStage(),
                                      mlir::ValueRange{row, bToken});
    }
}

// The source block-dot op is textually inside the uniform k_half loop. GPU
// outlining turns these workgroup allocas into one CTA attribution, so the
// V block written for k_half=0 remains available to k_half=1. The K block is
// overwritten once for each half. MFMA still consumes the original token-half
// then K32 order below.
void emitPersistentTypedBlockStageA(
    mlir::PatternRewriter &rewriter, mlir::Location loc,
    AMDGPUBlockDotBF16F32Op op, mlir::Value aBlock,
    bool precomputedVDecay, bool packedTokenMajor) {
    auto c0 = indexConstant(rewriter, loc, 0);
    auto c4 = indexConstant(rewriter, loc, 4);
    auto c8 = indexConstant(rewriter, loc, 8);
    auto tid = toIndex(rewriter, loc, op.getThreadId());
    auto chunkStart = toIndex(rewriter, loc, op.getChunkStart());
    auto valueHead = toIndex(rewriter, loc, op.getValueHead());
    auto valueBase = toIndex(rewriter, loc, op.getValueBase());
    auto aToken = mlir::arith::DivUIOp::create(rewriter, loc, tid, c4);
    auto rowGroup = mlir::arith::RemUIOp::create(rewriter, loc, tid, c4);
    auto rowBase = indexMul(rewriter, loc, rowGroup, c8);
    auto bf16x8 = mlir::VectorType::get({8}, rewriter.getBF16Type());
    auto f32x8 = mlir::VectorType::get({8}, rewriter.getF32Type());

    for (int64_t tokenHalf = 0; tokenHalf < 2; ++tokenHalf) {
        auto tokenOffset = indexConstant(rewriter, loc, tokenHalf * 32);
        auto token = indexAdd(rewriter, loc,
                              indexAdd(rewriter, loc, chunkStart, tokenOffset),
                              aToken);
        auto globalV = indexAdd(rewriter, loc, valueBase, rowBase);
        mlir::Value staged = mlir::vector::LoadOp::create(
            rewriter, loc, bf16x8, op.getSourceVNew(),
            mlir::ValueRange{c0, token, valueHead, globalV});
        if (!precomputedVDecay) {
            auto gValue = mlir::memref::LoadOp::create(
                rewriter, loc, op.getSourceG(),
                mlir::ValueRange{c0, token, valueHead});
            auto diff = mlir::arith::SubFOp::create(rewriter, loc,
                                                    op.getGLast(), gValue);
            auto decay = mlir::math::ExpOp::create(rewriter, loc, diff);
            auto vF32 = mlir::arith::ExtFOp::create(
                rewriter, loc, f32x8, staged, mlir::arith::FastMathFlagsAttr{});
            auto scale =
                mlir::vector::SplatOp::create(rewriter, loc, f32x8, decay);
            auto scaled = mlir::arith::MulFOp::create(rewriter, loc, vF32, scale);
            staged = mlir::arith::TruncFOp::create(rewriter, loc, bf16x8, scaled);
        }
        auto aColumn = indexAdd(rewriter, loc, tokenOffset, aToken);
        if (packedTokenMajor) {
            mlir::vector::StoreOp::create(
                rewriter, loc, staged, aBlock,
                mlir::ValueRange{aColumn, rowBase});
        } else {
            // Global V is contiguous in V while MFMA consumes contiguous
            // tokens. The C0 row-major layout therefore needs this scalar
            // LDS transpose at the producer boundary.
            for (int64_t element = 0; element < 8; ++element) {
                auto value = mlir::vector::ExtractOp::create(
                    rewriter, loc, staged, element);
                auto row = indexAdd(rewriter, loc, rowBase,
                                    indexConstant(rewriter, loc, element));
                mlir::memref::StoreOp::create(
                    rewriter, loc, value, aBlock,
                    mlir::ValueRange{c0, row, aColumn});
            }
        }
    }
}

void emitPersistentTypedBlockStageB(mlir::PatternRewriter &rewriter,
                                    mlir::Location loc,
                                    AMDGPUBlockDotBF16F32Op op,
                                    mlir::Value bBlock,
                                    bool packedTokenMajor) {
    auto c0 = indexConstant(rewriter, loc, 0);
    auto c4 = indexConstant(rewriter, loc, 4);
    auto c8 = indexConstant(rewriter, loc, 8);
    auto c64 = indexConstant(rewriter, loc, 64);
    auto tid = toIndex(rewriter, loc, op.getThreadId());
    auto chunkStart = toIndex(rewriter, loc, op.getChunkStart());
    auto keyHead = toIndex(rewriter, loc, op.getKeyHead());
    auto kHalf = toIndex(rewriter, loc, op.getKHalf());
    auto bToken = mlir::arith::DivUIOp::create(rewriter, loc, tid, c4);
    auto rowGroup = mlir::arith::RemUIOp::create(rewriter, loc, tid, c4);
    auto rowBase = indexMul(rewriter, loc, rowGroup, c8);
    auto kBlockBase = indexMul(rewriter, loc, kHalf, c64);
    auto bf16x8 = mlir::VectorType::get({8}, rewriter.getBF16Type());

    for (int64_t tokenHalf = 0; tokenHalf < 2; ++tokenHalf) {
        auto tokenOffset = indexConstant(rewriter, loc, tokenHalf * 32);
        auto token = indexAdd(rewriter, loc,
                              indexAdd(rewriter, loc, chunkStart, tokenOffset),
                              bToken);
        auto bColumn = indexAdd(rewriter, loc, tokenOffset, bToken);
        for (int64_t colHalf = 0; colHalf < 2; ++colHalf) {
            auto colOffset = indexConstant(rewriter, loc, colHalf * 32);
            auto kColumn = indexAdd(
                rewriter, loc, indexAdd(rewriter, loc, kBlockBase, colOffset),
                rowBase);
            auto staged = mlir::vector::LoadOp::create(
                rewriter, loc, bf16x8, op.getSourceK(),
                mlir::ValueRange{c0, token, keyHead, kColumn});
            auto row = indexAdd(rewriter, loc,
                                indexAdd(rewriter, loc, colOffset, rowBase),
                                indexConstant(rewriter, loc, 0));
            if (packedTokenMajor) {
                mlir::vector::StoreOp::create(
                    rewriter, loc, staged, bBlock,
                    mlir::ValueRange{bColumn, row});
            } else {
                for (int64_t element = 0; element < 8; ++element) {
                    auto value = mlir::vector::ExtractOp::create(
                        rewriter, loc, staged, element);
                    auto elementRow = indexAdd(
                        rewriter, loc, row,
                        indexConstant(rewriter, loc, element));
                    mlir::memref::StoreOp::create(
                        rewriter, loc, value, bBlock,
                        mlir::ValueRange{elementRow, bColumn});
                }
            }
        }
    }
}

void emitStageA(mlir::PatternRewriter &rewriter, mlir::Location loc,
                AMDGPUBlockDotBF16F32Op op, mlir::Value tokenHalf,
                bool cooperativeBv32, bool precomputedVDecay,
                OperandStagingKind operandStaging) {
    if (operandStaging == OperandStagingKind::TypedVector) {
        emitTypedVectorStageA(rewriter, loc, op, tokenHalf, precomputedVDecay);
        return;
    }
    auto c0 = indexConstant(rewriter, loc, 0);
    auto c1 = indexConstant(rewriter, loc, 1);
    auto c16 = indexConstant(rewriter, loc, 16);
    auto c8 = indexConstant(rewriter, loc, 8);
    auto c32 = indexConstant(rewriter, loc, 32);
    auto c128 = indexConstant(rewriter, loc, kThreads);
    auto c1024 = indexConstant(rewriter, loc, 32 * 32);
    auto tid = toIndex(rewriter, loc, op.getThreadId());
    auto chunkStart = toIndex(rewriter, loc, op.getChunkStart());
    auto valueHead = toIndex(rewriter, loc, op.getValueHead());
    auto valueBase = toIndex(rewriter, loc, op.getValueBase());
    auto tokenBase = indexAdd(rewriter, loc, chunkStart,
                              indexMul(rewriter, loc, tokenHalf, c32));
    if (cooperativeBv32) {
        // All 128 threads stage one V32 tile once. The active-wave guard in
        // the MFMA phase maps K[0:64] to wave 0 and K[64:128] to wave 1.
        auto loop = mlir::scf::ForOp::create(rewriter, loc, c0, c8, c1);
        rewriter.setInsertionPointToStart(loop.getBody());
        auto linear =
            indexAdd(rewriter, loc, tid,
                     indexMul(rewriter, loc, loop.getInductionVar(), c128));
        auto aRow = mlir::arith::DivUIOp::create(rewriter, loc, linear, c32);
        auto aToken = mlir::arith::RemUIOp::create(rewriter, loc, linear, c32);
        auto token = indexAdd(rewriter, loc, tokenBase, aToken);
        auto globalV = indexAdd(rewriter, loc, valueBase, aRow);
        auto vValue = mlir::memref::LoadOp::create(
            rewriter, loc, op.getSourceVNew(),
            mlir::ValueRange{c0, token, valueHead, globalV});
        mlir::Value staged = vValue;
        if (!precomputedVDecay) {
            auto gValue = mlir::memref::LoadOp::create(
                rewriter, loc, op.getSourceG(),
                mlir::ValueRange{c0, token, valueHead});
            auto diff = mlir::arith::SubFOp::create(rewriter, loc,
                                                    op.getGLast(), gValue);
            auto decay = mlir::math::ExpOp::create(rewriter, loc, diff);
            auto vF32 = mlir::arith::ExtFOp::create(
                rewriter, loc, rewriter.getF32Type(), vValue,
                mlir::arith::FastMathFlagsAttr{});
            auto scaled =
                mlir::arith::MulFOp::create(rewriter, loc, vF32, decay);
            staged = mlir::arith::TruncFOp::create(
                rewriter, loc, rewriter.getBF16Type(), scaled);
        }
        mlir::memref::StoreOp::create(rewriter, loc, staged, op.getAStage(),
                                      mlir::ValueRange{c0, aRow, aToken});
        rewriter.setInsertionPointAfter(loop);
        return;
    }
    auto loop = mlir::scf::ForOp::create(rewriter, loc, c0, c16, c1);
    rewriter.setInsertionPointToStart(loop.getBody());
    auto linear =
        indexAdd(rewriter, loc, tid,
                 indexMul(rewriter, loc, loop.getInductionVar(), c128));
    auto aWave = mlir::arith::DivUIOp::create(rewriter, loc, linear, c1024);
    auto aRem = mlir::arith::RemUIOp::create(rewriter, loc, linear, c1024);
    auto aRow = mlir::arith::DivUIOp::create(rewriter, loc, aRem, c32);
    auto aToken = mlir::arith::RemUIOp::create(rewriter, loc, aRem, c32);
    auto token = indexAdd(rewriter, loc, tokenBase, aToken);
    auto globalV = indexAdd(
        rewriter, loc, valueBase,
        indexAdd(rewriter, loc, indexMul(rewriter, loc, aWave, c32), aRow));
    auto vValue = mlir::memref::LoadOp::create(
        rewriter, loc, op.getSourceVNew(),
        mlir::ValueRange{c0, token, valueHead, globalV});
    mlir::Value staged = vValue;
    if (!precomputedVDecay) {
        auto gValue = mlir::memref::LoadOp::create(
            rewriter, loc, op.getSourceG(),
            mlir::ValueRange{c0, token, valueHead});
        auto diff =
            mlir::arith::SubFOp::create(rewriter, loc, op.getGLast(), gValue);
        auto decay = mlir::math::ExpOp::create(rewriter, loc, diff);
        auto vF32 = mlir::arith::ExtFOp::create(
            rewriter, loc, rewriter.getF32Type(), vValue,
            mlir::arith::FastMathFlagsAttr{});
        auto scaled = mlir::arith::MulFOp::create(rewriter, loc, vF32, decay);
        staged = mlir::arith::TruncFOp::create(rewriter, loc,
                                               rewriter.getBF16Type(), scaled);
    }
    mlir::memref::StoreOp::create(rewriter, loc, staged, op.getAStage(),
                                  mlir::ValueRange{aWave, aRow, aToken});
    rewriter.setInsertionPointAfter(loop);
}

void emitStageB(mlir::PatternRewriter &rewriter, mlir::Location loc,
                AMDGPUBlockDotBF16F32Op op, mlir::Value tokenHalf,
                int64_t colHalf, OperandStagingKind operandStaging) {
    if (operandStaging == OperandStagingKind::TypedVector) {
        emitTypedVectorStageB(rewriter, loc, op, tokenHalf, colHalf);
        return;
    }
    auto c0 = indexConstant(rewriter, loc, 0);
    auto c1 = indexConstant(rewriter, loc, 1);
    auto c8 = indexConstant(rewriter, loc, 8);
    auto c32 = indexConstant(rewriter, loc, 32);
    auto c64 = indexConstant(rewriter, loc, 64);
    auto c128 = indexConstant(rewriter, loc, kThreads);
    auto tid = toIndex(rewriter, loc, op.getThreadId());
    auto chunkStart = toIndex(rewriter, loc, op.getChunkStart());
    auto keyHead = toIndex(rewriter, loc, op.getKeyHead());
    auto kHalf = toIndex(rewriter, loc, op.getKHalf());
    auto tokenBase = indexAdd(rewriter, loc, chunkStart,
                              indexMul(rewriter, loc, tokenHalf, c32));
    auto kBase = indexAdd(rewriter, loc, indexMul(rewriter, loc, kHalf, c64),
                          indexConstant(rewriter, loc, colHalf * 32));
    auto loop = mlir::scf::ForOp::create(rewriter, loc, c0, c8, c1);
    rewriter.setInsertionPointToStart(loop.getBody());
    auto linear =
        indexAdd(rewriter, loc, tid,
                 indexMul(rewriter, loc, loop.getInductionVar(), c128));
    auto bRow = mlir::arith::DivUIOp::create(rewriter, loc, linear, c32);
    auto bToken = mlir::arith::RemUIOp::create(rewriter, loc, linear, c32);
    auto token = indexAdd(rewriter, loc, tokenBase, bToken);
    auto kColumn = indexAdd(rewriter, loc, kBase, bRow);
    auto value = mlir::memref::LoadOp::create(
        rewriter, loc, op.getSourceK(),
        mlir::ValueRange{c0, token, keyHead, kColumn});
    mlir::memref::StoreOp::create(rewriter, loc, value, op.getBStage(),
                                  mlir::ValueRange{bRow, bToken});
    rewriter.setInsertionPointAfter(loop);
}

mlir::Value joinColumns(mlir::PatternRewriter &rewriter, mlir::Location loc,
                        mlir::Value low, mlir::Value high) {
    auto resultType = mlir::VectorType::get({32}, rewriter.getF32Type());
    llvm::SmallVector<mlir::Value> values;
    values.reserve(32);
    for (int64_t index = 0; index < 16; ++index) {
        values.push_back(
            mlir::vector::ExtractOp::create(rewriter, loc, low, index));
    }
    for (int64_t index = 0; index < 16; ++index) {
        values.push_back(
            mlir::vector::ExtractOp::create(rewriter, loc, high, index));
    }
    return mlir::vector::FromElementsOp::create(rewriter, loc, resultType,
                                                values);
}

// Generic block-dot operand mode used by the Stage 6Z same-source audit.
// The high-level source owns the logical shared blocks and the K32 stage
// index; this helper owns only the final MFMA-B fragment materialization.
// Generic mode reconstructs B element-by-element.  gfx942 mode keeps the
// same source/stage coordinates but forms one typed BF16x8 LDS read.
mlir::Value emitGenericOperandBPair(
    mlir::PatternRewriter &rewriter, mlir::Location loc,
    AMDGPUBlockDotBF16F32Op op, mlir::Value accumulator, bool specialized,
    const LogicalBlockLayoutPlan *layoutPlan = nullptr) {
    auto bf16 = rewriter.getBF16Type();
    auto bf16x8 = mlir::VectorType::get({8}, bf16);
    auto bf16x4 = mlir::VectorType::get({4}, bf16);
    auto c2 = indexConstant(rewriter, loc, 2);
    auto c8 = indexConstant(rewriter, loc, 8);
    auto c32 = indexConstant(rewriter, loc, 32);
    auto c64 = indexConstant(rewriter, loc, 64);
    auto c128 = indexConstant(rewriter, loc, 128);
    mlir::Value tid;
    mlir::Value wave;
    mlir::Value lane;
    mlir::Value laneCol;
    mlir::Value laneGroup;
    mlir::Value rowHalf;
    mlir::Value valueHalf;
    mlir::Value kStage;
    if (layoutPlan) {
        tid = layoutPlan->tid;
        wave = layoutPlan->wave;
        lane = layoutPlan->lane;
        laneCol = layoutPlan->laneCol;
        laneGroup = layoutPlan->laneGroup;
        rowHalf = layoutPlan->rowHalf;
        valueHalf = layoutPlan->valueHalf;
        kStage = layoutPlan->kStage;
    } else {
        tid = toIndex(rewriter, loc, op.getThreadId());
        wave =
            mlir::arith::DivUIOp::create(rewriter, loc, tid, c64).getResult();
        lane =
            mlir::arith::RemUIOp::create(rewriter, loc, tid, c64).getResult();
        laneCol =
            mlir::arith::RemUIOp::create(rewriter, loc, lane, c32).getResult();
        laneGroup = mlir::arith::DivUIOp::create(rewriter, loc, lane, c32)
                        .getResult();
        rowHalf =
            mlir::arith::DivUIOp::create(rewriter, loc, wave, c2).getResult();
        valueHalf = mlir::arith::RemUIOp::create(rewriter, loc, wave, c2)
                        .getResult();
        kStage = toIndex(rewriter, loc, op.getKHalf());
    }
    auto qRow = indexAdd(
        rewriter, loc, indexMul(rewriter, loc, kStage, c64),
        indexAdd(rewriter, loc, indexMul(rewriter, loc, rowHalf, c32),
                 laneCol));

    // H is stored as [value, K-stage feature], while K is stored in the
    // source-half phase band.  Both use the same consumer-major B layout:
    // row is owned by laneCol and the contiguous vector is the token/feature
    // word consumed by MFMA32.
    mlir::Value bRow;
    if (genericOperandSourceRole(op) == "H") {
        bRow = indexAdd(
            rewriter, loc, indexConstant(rewriter, loc, 320),
            indexAdd(rewriter, loc, indexMul(rewriter, loc, valueHalf, c32),
                     laneCol));
    } else {
        auto sourceHalf = toIndex(rewriter, loc, op.getValueBase());
        bRow = indexAdd(
            rewriter, loc, indexConstant(rewriter, loc, 320),
            indexAdd(rewriter, loc, indexMul(rewriter, loc, sourceHalf, c128),
                     laneCol));
    }

    auto mfmaName = ir::intrinsics::MakeIntrinsicFuncName(
        "amdgpu", "rocdl_mfma_f32_32x32x8bf16_1k");
    auto firstWord = laneGroup;
    auto secondWord = indexAdd(rewriter, loc, laneGroup, c2);
    for (mlir::Value word : {firstWord, secondWord}) {
        auto elementOffset = indexMul(rewriter, loc, word, c8);
        if (layoutPlan) {
            // P1's typed-fragment arm loads exactly the four BF16 values
            // consumed by each MFMA call.  This is intentionally a direct
            // typed LDS fragment load, not a vector8 plus scalar slice; the
            // final ISA must show whether the target can preserve it without
            // reintroducing the old reconstruction chain.
            for (int64_t fragmentOffset : {int64_t{0}, int64_t{4}}) {
                auto fragmentIndex = indexAdd(
                    rewriter, loc, elementOffset,
                    indexConstant(rewriter, loc, fragmentOffset));
                auto aFrag = mlir::vector::LoadOp::create(
                    rewriter, loc, bf16x4, op.getAStage(),
                    mlir::ValueRange{qRow, fragmentIndex});
                mlir::Value bFrag;
                if (specialized) {
                    bFrag = mlir::vector::LoadOp::create(
                        rewriter, loc, bf16x4, op.getBStage(),
                        mlir::ValueRange{bRow, fragmentIndex});
                } else {
                    llvm::SmallVector<mlir::Value> values;
                    values.reserve(4);
                    for (int64_t element = 0; element < 4; ++element) {
                        values.push_back(mlir::memref::LoadOp::create(
                            rewriter, loc, op.getBStage(),
                            mlir::ValueRange{
                                bRow, indexAdd(
                                          rewriter, loc, fragmentIndex,
                                          indexConstant(rewriter, loc,
                                                        element))}));
                    }
                    bFrag = mlir::vector::FromElementsOp::create(
                        rewriter, loc, bf16x4, values);
                }
                auto call = mlir::func::CallOp::create(
                    rewriter, loc, mfmaName,
                    mlir::TypeRange{accumulator.getType()},
                    mlir::ValueRange{bFrag, aFrag, accumulator});
                call->setAttr("avelang.block_dot.mfma32",
                              rewriter.getUnitAttr());
                call->setAttr(
                    "avelang.block_dot.operand_role",
                    rewriter.getStringAttr(genericOperandRole(op)));
                call->setAttr("avelang.block_dot.b_operand_lowering",
                              rewriter.getStringAttr(
                                  specialized ? "typed_fragment_direct"
                                               : "generic_fragment_direct"));
                accumulator = call.getResult(0);
            }
            continue;
        }
        auto aRaw = mlir::vector::LoadOp::create(
            rewriter, loc, bf16x8, op.getAStage(),
            mlir::ValueRange{qRow, elementOffset});
        mlir::Value bRaw;
        if (specialized) {
            bRaw = mlir::vector::LoadOp::create(
                rewriter, loc, bf16x8, op.getBStage(),
                mlir::ValueRange{bRow, elementOffset});
        } else {
            llvm::SmallVector<mlir::Value> values;
            values.reserve(8);
            for (int64_t element = 0; element < 8; ++element) {
                values.push_back(mlir::memref::LoadOp::create(
                    rewriter, loc, op.getBStage(),
                    mlir::ValueRange{
                        bRow, indexAdd(rewriter, loc, elementOffset,
                                       indexConstant(rewriter, loc, element))}));
            }
            bRaw = mlir::vector::FromElementsOp::create(rewriter, loc, bf16x8,
                                                         values);
        }
        for (int64_t fragmentOffset : {int64_t{0}, int64_t{4}}) {
            auto aFrag = extractBF16x4(rewriter, loc, aRaw, fragmentOffset);
            auto bFrag = extractBF16x4(rewriter, loc, bRaw, fragmentOffset);
            auto call = mlir::func::CallOp::create(
                rewriter, loc, mfmaName, mlir::TypeRange{accumulator.getType()},
                mlir::ValueRange{bFrag, aFrag, accumulator});
            call->setAttr("avelang.block_dot.mfma32", rewriter.getUnitAttr());
            call->setAttr("avelang.block_dot.operand_role",
                          rewriter.getStringAttr(genericOperandRole(op)));
            call->setAttr("avelang.block_dot.b_operand_lowering",
                          rewriter.getStringAttr(specialized ? "typed_vector"
                                                              : "generic_scalar"));
            accumulator = call.getResult(0);
        }
    }
    return accumulator;
}

struct PlannedMfmaOperandRows {
    mlir::Value aRow;
    mlir::Value bRow;
    mlir::Value operandWord;
    mlir::Value kStage;
};

// Compute the logical rows once and carry them through the first-class
// operand op.  This is the important distinction from P1: the values are
// not immediately consumed by ordinary vector loads, so the target-specific
// consumer still owns the packed LDS read and MFMA operand materialization.
PlannedMfmaOperandRows planMfmaOperandRows(
    AMDGPUBlockDotBF16F32Op op, mlir::PatternRewriter &rewriter,
    mlir::Location loc, const LogicalBlockLayoutPlan *layoutPlan) {
    auto c2 = indexConstant(rewriter, loc, 2);
    auto c32 = indexConstant(rewriter, loc, 32);
    auto c64 = indexConstant(rewriter, loc, 64);
    auto c128 = indexConstant(rewriter, loc, 128);

    mlir::Value tid;
    mlir::Value wave;
    mlir::Value lane;
    mlir::Value laneCol;
    mlir::Value rowHalf;
    mlir::Value valueHalf;
    mlir::Value kStage;
    mlir::Value operandWord;
    if (layoutPlan) {
        tid = layoutPlan->tid;
        wave = layoutPlan->wave;
        lane = layoutPlan->lane;
        laneCol = layoutPlan->laneCol;
        rowHalf = layoutPlan->rowHalf;
        valueHalf = layoutPlan->valueHalf;
        kStage = layoutPlan->kStage;
        operandWord = layoutPlan->laneGroup;
    } else {
        tid = toIndex(rewriter, loc, op.getThreadId());
        wave = mlir::arith::DivUIOp::create(rewriter, loc, tid, c64);
        lane = mlir::arith::RemUIOp::create(rewriter, loc, tid, c64);
        laneCol = mlir::arith::RemUIOp::create(rewriter, loc, lane, c32);
        auto laneGroup = mlir::arith::DivUIOp::create(
            rewriter, loc, lane, c32);
        rowHalf = mlir::arith::DivUIOp::create(rewriter, loc, wave, c2);
        valueHalf = mlir::arith::RemUIOp::create(rewriter, loc, wave, c2);
        kStage = toIndex(rewriter, loc, op.getKHalf());
        operandWord = laneGroup;
    }

    const auto sourceRole = genericOperandSourceRole(op);
    if (sourceRole == "V" && useC22SchedulePreservingPhysical()) {
        // Preserve Z5B Phase-C exactly: value_base is the serial score/V
        // source half, A is the row-major score tile, and B is the rotating
        // V tile.  These are coordinates only; C22 emits neither a producer
        // nor a barrier here.
        auto sourceHalf = toIndex(rewriter, loc, op.getValueBase());
        auto token = indexAdd(
            rewriter, loc, indexMul(rewriter, loc, rowHalf, c32), laneCol);
        auto value = indexAdd(
            rewriter, loc, indexMul(rewriter, loc, valueHalf, c32), laneCol);
        auto scoreRow = indexAdd(
            rewriter, loc, indexConstant(rewriter, loc, 256),
            indexAdd(rewriter, loc,
                     indexMul(rewriter, loc, token, c2), sourceHalf));
        auto vRow = indexAdd(
            rewriter, loc, indexConstant(rewriter, loc, 384),
            indexAdd(rewriter, loc,
                     indexMul(rewriter, loc, value, c2), sourceHalf));
        return {scoreRow, vRow, operandWord, indexConstant(rewriter, loc, 0)};
    }
    if (sourceRole == "V" &&
        (useC18FullPhysicalRegion() || usesC19CompatibleFullPhysicalRegion())) {
        // C18 encodes k_half as (source_half << 1) | k32_stage.  The V
        // source is written once into the same physical score/V band that
        // the consumer reads; no generic fragment reconstruction is
        // allowed to reinterpret this block.
        auto sourceHalf = mlir::arith::ShRUIOp::create(
            rewriter, loc, kStage, indexConstant(rewriter, loc, 1));
        auto k32Stage = mlir::arith::AndIOp::create(
            rewriter, loc, kStage, indexConstant(rewriter, loc, 1));
        auto qRow = indexAdd(
            rewriter, loc,
            indexConstant(rewriter, loc,
                          usesC19CompatibleFullPhysicalRegion() ? 0 : 256),
            indexAdd(rewriter, loc,
                     indexMul(rewriter, loc,
                              indexAdd(rewriter, loc,
                                       indexMul(rewriter, loc, rowHalf, c32),
                                       laneCol),
                              indexConstant(rewriter, loc, 2)),
                     sourceHalf));
        auto vRow = indexAdd(
            rewriter, loc,
            indexConstant(rewriter, loc,
                          usesC19CompatibleFullPhysicalRegion() ? 128 : 384),
            indexAdd(rewriter, loc,
                     indexMul(rewriter, loc,
                              indexAdd(rewriter, loc,
                                       indexMul(rewriter, loc, valueHalf, c32),
                                       laneCol),
                              indexConstant(rewriter, loc, 2)),
                     sourceHalf));
        auto word = indexAdd(
            rewriter, loc,
            indexMul(rewriter, loc, k32Stage, indexConstant(rewriter, loc, 2)),
            operandWord);
        return {qRow, vRow, word, k32Stage};
    }

    // C21 rotates two Q K32 slots.  Source order still determines when a slot
    // is safe to overwrite; this expression only makes that ownership explicit
    // in the actual shared operand address rather than leaving it as metadata.
    auto qStage = useC21SelectedNativePipeline()
                      ? mlir::arith::AndIOp::create(
                            rewriter, loc, kStage,
                            indexConstant(rewriter, loc, 1))
                      : kStage;
    auto aRow = indexAdd(
        rewriter, loc, indexMul(rewriter, loc, qStage, c64),
        indexAdd(rewriter, loc, indexMul(rewriter, loc, rowHalf, c32),
                 laneCol));

    mlir::Value bRow;
    if (sourceRole == "H") {
        bRow = indexAdd(
            rewriter, loc,
            indexConstant(rewriter, loc,
                          usesC19CompatibleFullPhysicalRegion() ? 256 : 320),
            indexAdd(rewriter, loc, indexMul(rewriter, loc, valueHalf, c32),
                     laneCol));
    } else {
        auto sourceHalf = toIndex(rewriter, loc, op.getValueBase());
        bRow = indexAdd(
            rewriter, loc,
            indexConstant(rewriter, loc,
                          usesC19CompatibleFullPhysicalRegion() ? 256 : 320),
            usesC19CompatibleFullPhysicalRegion()
                ? laneCol
                : indexAdd(rewriter, loc,
                           indexMul(rewriter, loc, sourceHalf, c128),
                           laneCol));
    }
    return {aRow, bRow, operandWord, kStage};
}

mlir::Value emitFirstClassMfmaOperand(
    AMDGPUBlockDotBF16F32Op op, mlir::PatternRewriter &rewriter,
    mlir::Location loc, mlir::Value accumulator,
    const LogicalBlockLayoutPlan *layoutPlan,
    const c13::ChunkOPhysicalPlan *c17Plan = nullptr) {
    auto rows = planMfmaOperandRows(op, rewriter, loc, layoutPlan);
    auto planned = AMDGPUBlockDotMfmaOperandOp::create(
        rewriter, loc, accumulator.getType(),
        mlir::ValueRange{op.getAStage(), op.getBStage(), accumulator,
                         rows.aRow, rows.bRow, rows.operandWord, rows.kStage});
    auto setString = [&](llvm::StringRef name, llvm::StringRef value) {
        planned->setAttr(name, rewriter.getStringAttr(value));
    };
    planned->setAttr("avelang.block_dot.first_class_operand",
                    rewriter.getUnitAttr());
    planned->setAttr(
        "c18.mfma_callee",
        rewriter.getStringAttr(ir::intrinsics::MakeIntrinsicFuncName(
            "amdgpu", "rocdl_mfma_f32_32x32x8bf16_1k")));
    if (usesC19CompatibleFullPhysicalRegion())
        planned->setAttr(
            "c19.mfma_callee",
            rewriter.getStringAttr(ir::intrinsics::MakeIntrinsicFuncName(
                "amdgpu", "rocdl_mfma_f32_32x32x8bf16_1k")));
    if (useC21SelectedNativePipeline()) {
        planned->setAttr("c21.selected_native_pipeline", rewriter.getUnitAttr());
        planned->setAttr("c21.q_slot", rewriter.getStringAttr("k_stage_mod_2"));
    }
    setString("avelang.block_dot.operand_role", genericOperandRole(op));
    setString("avelang.block_dot.source_role",
              genericOperandSourceRole(op));
    setString("avelang.block_dot.transpose",
              genericOperandSourceRole(op) == "H" ? "rhs_transposed"
                                                    : "none");
    setString("avelang.block_dot.logical_shape", "32x32x32");
    setString("avelang.block_dot.physical_encoding",
              "gfx942_shared_b32_mfma32");
    setString("avelang.block_dot.fragment_mapping",
              "lane_group_word_pair_k32_order");
    setString("avelang.block_dot.target_mfma",
              "f32_32x32x8_bf16");
    if (c17Plan) {
        auto role = genericOperandSourceRole(op);
        annotateC17PhysicalPlan(planned, *c17Plan, role,
                                role == "H" ? "Q@H"
                                             : role == "K" ? "Q@K" : "score@V");
        planned->setAttr("c17.operand_lifetime",
                         rewriter.getStringAttr("producer_to_mfma"));
        if (useC18FullPhysicalRegion()) {
            FullPhysicalRegionPlan c18;
            c18.physical = *c17Plan;
            annotateC18PhysicalRegion(
                planned, c18, role,
                role == "H" ? "Q@H" : role == "K" ? "Q@K" : "score@V");
        }
    }
    if (usesC19CompatibleFullPhysicalRegion()) {
        auto role = genericOperandSourceRole(op);
        planned->setAttr("c19.full_physical_region",
                         rewriter.getUnitAttr());
        planned->setAttr("c19.plan_role", rewriter.getStringAttr(role));
        planned->setAttr(
            "c19.consumer",
            rewriter.getStringAttr(role == "H" ? "Q@H"
                                  : role == "K" ? "Q@K" : "score@V"));
        planned->setAttr("c19.owner",
                         rewriter.getStringAttr("FullPhysicalRegionPlan"));
        planned->setAttr("c19.legacy_owner", rewriter.getBoolAttr(false));
        if (useC21SelectedNativePipeline()) {
            planned->setAttr("c21.plan_owner",
                             rewriter.getStringAttr("ChunkOPipelinePlan"));
            planned->setAttr("c21.consumer_phase",
                             rewriter.getStringAttr("same_k32_superloop"));
        }
    }
    return planned.getResult();
}

mlir::LogicalResult lowerGenericOperandMode(
    AMDGPUBlockDotBF16F32Op op, mlir::PatternRewriter &rewriter,
    LoweringKind kind) {
    auto loc = op.getLoc();
    auto accumulator = materializeAccumulatorVector(rewriter, loc, op.getAccLow());
    if (!accumulator)
        return rewriter.notifyMatchFailure(
            op, "generic operand mode requires local/vector FP32 accumulator");

    if (useC22SchedulePreservingPhysical()) {
        const auto role = genericOperandSourceRole(op);
        auto physical = getC22StaticPhysicalPlan(role);
        if (!physical)
            return rewriter.notifyMatchFailure(
                op, "C22 requires a validated C13-C16 physical role");
        auto planned = emitFirstClassMfmaOperand(op, rewriter, loc,
                                                  accumulator, nullptr);
        auto plannedOp = mlir::dyn_cast_or_null<AMDGPUBlockDotMfmaOperandOp>(
            planned.getDefiningOp());
        if (!plannedOp)
            return rewriter.notifyMatchFailure(
                op, "C22 failed to preserve a first-class MFMA operand");
        plannedOp->setAttr("c22.z5b_schedule_preserving",
                           rewriter.getUnitAttr());
        plannedOp->setAttr("c14.static_physical", rewriter.getUnitAttr());
        plannedOp->setAttr("c22.physical_role", rewriter.getStringAttr(role));
        plannedOp->setAttr("c13.distributed",
                           physical->distributed.toAttr(rewriter.getContext()));
        plannedOp->setAttr("c13.shared",
                           physical->shared.toAttr(rewriter.getContext()));
        plannedOp->setAttr("c13.mfma",
                           physical->mfma.toAttr(rewriter.getContext()));
        plannedOp->setAttr("c13.dot",
                           physical->dot.toAttr(rewriter.getContext()));
        plannedOp->setAttr("c13.transform",
                           physical->transform.toAttr(rewriter.getContext()));
        auto result = joinColumns(rewriter, loc, planned, planned);
        if (auto *def = result.getDefiningOp()) {
            def->setAttr("avelang.block_dot.operand_mode",
                         rewriter.getStringAttr("c22_z5b_schedule_preserving"));
            def->setAttr("c22.z5b_schedule_preserving", rewriter.getUnitAttr());
            def->setAttr("c22.physical_role", rewriter.getStringAttr(role));
        }
        rewriter.replaceOp(op, result);
        return mlir::success();
    }

    if (isC15RealTile(op)) {
        auto physical = c13::ChunkOPhysicalPlan::makeC12T2048WG256();
        std::string error;
        if (!physical.verify(&error))
            return rewriter.notifyMatchFailure(
                op, ("C15 physical recipe verification failed: " + error).c_str());
        C14StaticPhysicalPlan c14Plan{
            physical.v.distributed, physical.v.shared, physical.mfma,
            physical.v.dot, physical.v.transform};
        if (!c14Plan.distributed.verify(&error) ||
            !c14Plan.shared.verifyMapping({{64, 64}}, 2, &error) ||
            !c14Plan.mfma.verify(&error) || !c14Plan.dot.verify(&error) ||
            !c14Plan.transform.verify(&error))
            return rewriter.notifyMatchFailure(
                op, ("C15 V recipe is not a valid C14 recipe: " + error).c_str());

        if (mlir::failed(emitC15RealVTile(op, rewriter, c14Plan)))
            return mlir::failure();

        // One public generic block-dot represents the two K32 halves of the
        // real 64-wide tile.  Each internal first-class operand retains the
        // C14 typed recipe until the post-outlining consumer pass.
        auto tid64 = c14I64(rewriter, loc, op.getThreadId());
        auto wave = mlir::arith::ShRUIOp::create(
            rewriter, loc, tid64, c14I64Constant(rewriter, loc, 6));
        auto lane = mlir::arith::AndIOp::create(
            rewriter, loc, tid64, c14I64Constant(rewriter, loc, 63));
        auto laneCol = mlir::arith::AndIOp::create(
            rewriter, loc, lane, c14I64Constant(rewriter, loc, 31));
        auto laneGroup = mlir::arith::ShRUIOp::create(
            rewriter, loc, lane, c14I64Constant(rewriter, loc, 5));
        auto rowBase = mlir::arith::AddIOp::create(
            rewriter, loc,
            mlir::arith::ShLIOp::create(
                rewriter, loc, wave, c14I64Constant(rewriter, loc, 5)),
            laneCol);
        auto aRow = c15Index(rewriter, loc, rowBase);
        auto operandWord = c15Index(rewriter, loc, laneGroup);
        const bool scoreVFull64 = op->hasAttr("c15.scorev_full64");
        auto accHigh = scoreVFull64
                           ? materializeAccumulatorVector(rewriter, loc,
                                                          op.getAccHigh())
                           : accumulator;
        if (!accHigh)
            return rewriter.notifyMatchFailure(
                op, "C15 score@V full64 requires a second FP32 accumulator");

        auto makeChain = [&](mlir::Value chain, int64_t outputHalf) {
            auto outputBase = c14I64Constant(rewriter, loc, outputHalf * 32);
            auto bRow = c15Index(
                rewriter, loc,
                mlir::arith::AddIOp::create(rewriter, loc, outputBase,
                                             laneCol));
            for (int64_t kStage = 0; kStage < 2; ++kStage) {
                auto planned = AMDGPUBlockDotMfmaOperandOp::create(
                    rewriter, loc, chain.getType(),
                    mlir::ValueRange{op.getAStage(), op.getBStage(), chain,
                                     aRow, bRow, operandWord,
                                     indexConstant(rewriter, loc, kStage)});
                planned->setAttr("avelang.block_dot.first_class_operand",
                                 rewriter.getUnitAttr());
                planned->setAttr("c14.static_physical", rewriter.getUnitAttr());
                planned->setAttr("c15.real_tile", rewriter.getStringAttr("V"));
                planned->setAttr("c15.k32_half",
                                 rewriter.getI64IntegerAttr(kStage));
                if (scoreVFull64) {
                    planned->setAttr("c15.scorev_full64", rewriter.getUnitAttr());
                    planned->setAttr("c15.scorev_output_half",
                                     rewriter.getI64IntegerAttr(outputHalf));
                }
                planned->setAttr(
                    "c14.mfma_callee",
                    rewriter.getStringAttr(
                        ir::intrinsics::MakeIntrinsicFuncName(
                            "amdgpu", "rocdl_mfma_f32_32x32x8bf16_1k")));
                planned->setAttr(
                    "c13.distributed",
                    c14Plan.distributed.toAttr(rewriter.getContext()));
                planned->setAttr("c13.shared",
                                 c14Plan.shared.toAttr(rewriter.getContext()));
                planned->setAttr("c13.mfma",
                                 c14Plan.mfma.toAttr(rewriter.getContext()));
                planned->setAttr("c13.dot",
                                 c14Plan.dot.toAttr(rewriter.getContext()));
                planned->setAttr("c13.transform",
                                 c14Plan.transform.toAttr(rewriter.getContext()));
                planned->setAttr("avelang.block_dot.operand_role",
                                 rewriter.getStringAttr("B"));
                planned->setAttr("avelang.block_dot.source_role",
                                 rewriter.getStringAttr("V"));
                planned->setAttr("avelang.block_dot.target_mfma",
                                 rewriter.getStringAttr("f32_32x32x8_bf16"));
                chain = planned.getResult();
            }
            return chain;
        };

        auto low = makeChain(accumulator, 0);
        auto high = scoreVFull64 ? makeChain(accHigh, 1) : accumulator;
        auto result = joinColumns(rewriter, loc, low, high);
        if (auto *def = result.getDefiningOp()) {
            def->setAttr("c15.real_tile_lowered", rewriter.getUnitAttr());
            def->setAttr("avelang.block_dot.operand_mode",
                         rewriter.getStringAttr(scoreVFull64
                                                    ? "c15_scorev_full64"
                                                    : "c15_real_v"));
        }
        rewriter.replaceOp(op, result);
        return mlir::success();
    }

    if (isC16RealTile(op)) {
        auto roleAttr = op->getAttrOfType<mlir::StringAttr>(
            "c16.real_tile_role");
        auto role = roleAttr ? roleAttr.getValue() : llvm::StringRef();
        auto physical = getC16StaticPhysicalPlan(role);
        if (!physical)
            return rewriter.notifyMatchFailure(
                op, "C16 Q/H/K role has no valid selected C13 physical plan");
        const bool qDualSecond =
            role == "Q" && isC16QDualConsumerOnly(op);
        if (!qDualSecond &&
            mlir::failed(emitC16RealQHKTile(op, rewriter, *physical, role)))
            return mlir::failure();

        const bool wg128QHFull64 =
            useC16WG128QHFull64() && (role == "Q" || role == "H");
        const bool wg128QKFull64 =
            useC16WG128QKFull64() && role == "K";
        if (wg128QHFull64 || wg128QKFull64) {
            // A WG128 CTA has two waves and two independent accumulator
            // halves.  Q/H use the native Q/H B-row recipe; K uses the
            // native [32,64] B operand, so its B row is the K coordinate and
            // does not carry the output-half offset.  The late consumer turns
            // the explicit output half into the corresponding virtual
            // two-by-two MFMA parent without changing the intrinsic.
            auto tid64 = c14I64(rewriter, loc, op.getThreadId());
            auto wave = mlir::arith::ShRUIOp::create(
                rewriter, loc, tid64, c14I64Constant(rewriter, loc, 6));
            auto lane = mlir::arith::AndIOp::create(
                rewriter, loc, tid64, c14I64Constant(rewriter, loc, 63));
            auto lane32 = mlir::arith::AndIOp::create(
                rewriter, loc, lane, c14I64Constant(rewriter, loc, 31));
            auto operandWord = c15Index(
                rewriter, loc,
                mlir::arith::ShRUIOp::create(
                    rewriter, loc, lane, c14I64Constant(rewriter, loc, 5)));
            auto aRow = c15Index(
                rewriter, loc,
                mlir::arith::AddIOp::create(
                    rewriter, loc,
                    mlir::arith::ShLIOp::create(
                        rewriter, loc, wave, c14I64Constant(rewriter, loc, 5)),
                    lane32));

            auto accLow = materializeAccumulatorVector(
                rewriter, loc, op.getAccLow());
            auto accHigh = materializeAccumulatorVector(
                rewriter, loc, op.getAccHigh());
            if (!accLow || !accHigh)
                return rewriter.notifyMatchFailure(
                    op, "WG128 full64 requires two FP32 accumulator vectors");

            auto makeHalf = [&](mlir::Value accumulator,
                                int64_t outputHalf) -> mlir::Value {
                mlir::Value bRow;
                if (wg128QKFull64) {
                    bRow = c15Index(rewriter, loc, lane32);
                } else {
                    bRow = c15Index(
                        rewriter, loc,
                        mlir::arith::AddIOp::create(
                            rewriter, loc,
                            c14I64Constant(rewriter, loc, outputHalf * 32),
                            lane32));
                }
                auto planned = AMDGPUBlockDotMfmaOperandOp::create(
                    rewriter, loc, accumulator.getType(),
                    mlir::ValueRange{op.getAStage(), op.getBStage(), accumulator,
                                     aRow, bRow, operandWord, wave});
                planned->setAttr("avelang.block_dot.first_class_operand",
                                 rewriter.getUnitAttr());
                planned->setAttr("c16.real_tile_role",
                                 rewriter.getStringAttr(role));
                planned->setAttr(
                    wg128QKFull64 ? "c16.wg128_qhk_full64"
                                  : "c16.wg128_qh_full64",
                    rewriter.getUnitAttr());
                planned->setAttr("c16.wg128_qh_output_half",
                                 rewriter.getI64IntegerAttr(outputHalf));
                planned->setAttr("c16.wg128_qh_source_wave",
                                 rewriter.getUnitAttr());
                planned->setAttr("c14.static_physical",
                                 rewriter.getUnitAttr());
                planned->setAttr(
                    "c14.mfma_callee",
                    rewriter.getStringAttr(
                        ir::intrinsics::MakeIntrinsicFuncName(
                            "amdgpu", "rocdl_mfma_f32_32x32x8bf16_1k")));
                planned->setAttr(
                    "avelang.block_dot.operand_role",
                    rewriter.getStringAttr(role == "Q" ? "A" : "B"));
                planned->setAttr("avelang.block_dot.source_role",
                                 rewriter.getStringAttr(role));
                return planned.getResult();
            };

            auto low = makeHalf(accLow, 0);
            auto high = makeHalf(accHigh, 1);
            auto result = joinColumns(rewriter, loc, low, high);
            if (auto *def = result.getDefiningOp()) {
                def->setAttr("c16.wg128_qh_full64_lowered",
                             rewriter.getUnitAttr());
                def->setAttr("avelang.block_dot.operand_mode",
                             rewriter.getStringAttr("c16_wg128_qh_full64"));
            }
            rewriter.replaceOp(op, result);
            return mlir::success();
        }

        auto tid64 = c14I64(rewriter, loc, op.getThreadId());
        auto wave = mlir::arith::ShRUIOp::create(
            rewriter, loc, tid64, c14I64Constant(rewriter, loc, 6));
        auto lane = mlir::arith::AndIOp::create(
            rewriter, loc, tid64, c14I64Constant(rewriter, loc, 63));
        auto lane32 = mlir::arith::AndIOp::create(
            rewriter, loc, lane, c14I64Constant(rewriter, loc, 31));
        auto waveRow = mlir::arith::ShRUIOp::create(
            rewriter, loc, wave, c14I64Constant(rewriter, loc, 1));
        auto waveCol = mlir::arith::AndIOp::create(
            rewriter, loc, wave, c14I64Constant(rewriter, loc, 1));
        auto aRow = c15Index(
            rewriter, loc,
            mlir::arith::AddIOp::create(
                rewriter, loc,
                mlir::arith::ShLIOp::create(
                    rewriter, loc, waveRow,
                    c14I64Constant(rewriter, loc, 5)),
                lane32));
        mlir::Value bRow;
        if (role == "K") {
            bRow = c15Index(rewriter, loc, lane32);
        } else {
            bRow = c15Index(
                rewriter, loc,
                mlir::arith::AddIOp::create(
                    rewriter, loc,
                    mlir::arith::ShLIOp::create(
                        rewriter, loc, waveCol,
                        c14I64Constant(rewriter, loc, 5)),
                    lane32));
        }
        auto operandWord = c15Index(
            rewriter, loc,
            mlir::arith::ShRUIOp::create(
                rewriter, loc, lane, c14I64Constant(rewriter, loc, 5)));
        // A selected Q/H/K source tile is one K32 block.  The lane's
        // operandWord/word+2 pair covers that full K32 block; kStage is
        // repurposed only as the static wave-column selector needed by the
        // two-by-two MFMA parent (0 or 1), not as an extra source stage.
        auto planned = AMDGPUBlockDotMfmaOperandOp::create(
            rewriter, loc, accumulator.getType(),
            mlir::ValueRange{op.getAStage(), op.getBStage(), accumulator,
                             aRow, bRow, operandWord,
                             c15Index(rewriter, loc, waveCol)});
        planned->setAttr("avelang.block_dot.first_class_operand",
                         rewriter.getUnitAttr());
        planned->setAttr("c16.real_tile_role",
                         rewriter.getStringAttr(role));
        if (role == "Q" && useC16QDualConsumer() && !qDualSecond)
            planned->setAttr("c16.q_dual_producer", rewriter.getUnitAttr());
        planned->setAttr("c14.static_physical", rewriter.getUnitAttr());
        planned->setAttr(
            "c14.mfma_callee",
            rewriter.getStringAttr(ir::intrinsics::MakeIntrinsicFuncName(
                "amdgpu", "rocdl_mfma_f32_32x32x8bf16_1k")));
        planned->setAttr("c13.distributed",
                         physical->distributed.toAttr(rewriter.getContext()));
        planned->setAttr("c13.shared",
                         physical->shared.toAttr(rewriter.getContext()));
        planned->setAttr("c13.mfma",
                         physical->mfma.toAttr(rewriter.getContext()));
        planned->setAttr("c13.dot",
                         physical->dot.toAttr(rewriter.getContext()));
        planned->setAttr("c13.transform",
                         physical->transform.toAttr(rewriter.getContext()));
        planned->setAttr(
            "avelang.block_dot.operand_role",
            rewriter.getStringAttr(role == "Q" ? "A" : "B"));
        planned->setAttr("avelang.block_dot.source_role",
                         rewriter.getStringAttr(role));
        accumulator = planned.getResult();
        auto result = joinColumns(rewriter, loc, accumulator, accumulator);
        if (auto *def = result.getDefiningOp()) {
            def->setAttr("c16.real_tile_lowered", rewriter.getUnitAttr());
            def->setAttr("avelang.block_dot.operand_mode",
                         rewriter.getStringAttr("c16_real_qhk"));
        }
        rewriter.replaceOp(op, result);
        return mlir::success();
    }

    auto updated = emitGenericOperandBPair(
        rewriter, loc, op, accumulator, kind == LoweringKind::Specialized);
    auto result = joinColumns(rewriter, loc, updated, updated);
    auto *def = result.getDefiningOp();
    if (def) {
        def->setAttr("avelang.block_dot.operand_mode",
                     rewriter.getStringAttr("generic_mfma_b"));
        def->setAttr("avelang.block_dot.b_operand_lowering",
                     rewriter.getStringAttr(kind == LoweringKind::Specialized
                                                ? "gfx942_typed_vector"
                                                : "generic_scalar"));
    }
    rewriter.replaceOp(op, result);
    return mlir::success();
}

struct C23PendingKPacket {
    mlir::Value active;
    mlir::Value packet;
};

// The C21 source order is Q -> H -> K0 -> K1.  Its former lowering treated
// each logical operation as an indivisible load/store/consumer phase.  This
// small compiler-internal materializer separates only a K packet's issue from
// its existing LDS commit: H issues K0 after its own load and before its LDS
// store, and K0 issues K1 before it commits K0.  The packet stays vector SSA;
// no new source op, shared layout, or accumulator schedule is introduced.
class C23LatePipelineMaterializer {
  public:
    bool enabledFor(AMDGPUBlockDotBF16F32Op op) const {
        return useC23LatePipelineMaterialization() &&
               op->hasAttr("c21.selected_native_pipeline");
    }

    AMDGPUBlockDotBF16F32Op
    nextK(AMDGPUBlockDotBF16F32Op op) const {
        for (auto *cursor = op->getNextNode(); cursor;
             cursor = cursor->getNextNode()) {
            auto candidate = mlir::dyn_cast<AMDGPUBlockDotBF16F32Op>(cursor);
            if (!candidate)
                continue;
            if (!candidate->hasAttr("c21.selected_native_pipeline"))
                return {};
            // The C21 superloop has exactly H -> K0 -> K1.  Do not cross a
            // different logical role or an iteration boundary in C23.
            return genericOperandSourceRole(candidate) == "K" ? candidate
                                                                : AMDGPUBlockDotBF16F32Op{};
        }
        return {};
    }

    mlir::LogicalResult prefetchK(AMDGPUBlockDotBF16F32Op op,
                                  mlir::PatternRewriter &rewriter) {
        if (pending_.contains(op.getOperation()))
            return mlir::success();
        auto loc = op.getLoc();
        auto source = op.getSourceK();
        auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
        if (!sourceType || sourceType.getRank() != 4)
            return rewriter.notifyMatchFailure(
                op, "C23 K prefetch requires rank-4 BF16 K source");

        auto c0 = indexConstant(rewriter, loc, 0);
        auto c1 = indexConstant(rewriter, loc, 1);
        auto c2 = indexConstant(rewriter, loc, 2);
        auto c3 = indexConstant(rewriter, loc, 3);
        auto c6 = indexConstant(rewriter, loc, 6);
        auto c8 = indexConstant(rewriter, loc, 8);
        auto c32 = indexConstant(rewriter, loc, 32);
        auto bf16 = rewriter.getBF16Type();
        auto bf16x8 = mlir::VectorType::get({8}, bf16);
        auto tid = toIndex(rewriter, loc, op.getThreadId());
        auto wave = mlir::arith::ShRUIOp::create(rewriter, loc, tid, c6);
        auto lane = mlir::arith::AndIOp::create(rewriter, loc, tid,
                                                indexConstant(rewriter, loc, 63));
        auto wavePair = mlir::arith::ShRUIOp::create(rewriter, loc, wave, c1);
        auto producerLinear = indexAdd(
            rewriter, loc, indexMul(rewriter, loc, wavePair,
                                    indexConstant(rewriter, loc, 64)),
            lane);
        auto packetRow = mlir::arith::ShRUIOp::create(rewriter, loc,
                                                       producerLinear, c2);
        auto packet = mlir::arith::AndIOp::create(rewriter, loc,
                                                   producerLinear, c3);
        auto feature = indexAdd(
            rewriter, loc,
            indexMul(rewriter, loc, toIndex(rewriter, loc, op.getKHalf()), c32),
            indexMul(rewriter, loc, packet, c8));
        auto token = indexAdd(
            rewriter, loc,
            indexAdd(rewriter, loc, toIndex(rewriter, loc, op.getChunkStart()),
                     indexMul(rewriter, loc, toIndex(rewriter, loc, op.getValueBase()),
                              c32)),
            packetRow);
        auto active = mlir::arith::CmpIOp::create(
            rewriter, loc, mlir::arith::CmpIPredicate::eq,
            mlir::arith::AndIOp::create(rewriter, loc, wave, c1), c0);
        auto guarded = mlir::scf::IfOp::create(
            rewriter, loc, mlir::TypeRange{bf16x8}, active,
            /*withElseRegion=*/true);
        rewriter.setInsertionPointToStart(&guarded.getThenRegion().front());
        auto load = mlir::vector::LoadOp::create(
            rewriter, loc, bf16x8, source,
            mlir::ValueRange{c0, token, toIndex(rewriter, loc, op.getKeyHead()),
                             feature});
        load->setAttr("c23.late_pipeline", rewriter.getStringAttr("K_issue"));
        load->setAttr("c23.packet", rewriter.getStringAttr("bf16x8"));
        mlir::scf::YieldOp::create(rewriter, loc, load.getResult());
        rewriter.setInsertionPointToStart(&guarded.getElseRegion().front());
        auto zero = mlir::arith::ConstantOp::create(
            rewriter, loc, bf16, rewriter.getFloatAttr(bf16, 0.0));
        auto emptyPacket = mlir::vector::SplatOp::create(rewriter, loc, bf16x8, zero);
        mlir::scf::YieldOp::create(rewriter, loc, emptyPacket.getResult());
        rewriter.setInsertionPointAfter(guarded);
        pending_.try_emplace(op.getOperation(),
                             C23PendingKPacket{active, guarded.getResult(0)});
        return mlir::success();
    }

    mlir::LogicalResult emitHWithNextK(
        AMDGPUBlockDotBF16F32Op op, mlir::PatternRewriter &rewriter,
        const FullPhysicalRegionPlan &plan) {
        auto target = nextK(op);
        if (!target)
            return rewriter.notifyMatchFailure(
                op, "C23 H stage has no immediately following K consumer");
        auto loc = op.getLoc();
        auto source = op.getSourceK();
        auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
        if (!sourceType || sourceType.getRank() != 5)
            return rewriter.notifyMatchFailure(
                op, "C23 H producer requires rank-5 BF16 H source");
        auto c0 = indexConstant(rewriter, loc, 0);
        auto c2 = indexConstant(rewriter, loc, 2);
        auto c3 = indexConstant(rewriter, loc, 3);
        auto c8 = indexConstant(rewriter, loc, 8);
        auto c32 = indexConstant(rewriter, loc, 32);
        auto c64 = indexConstant(rewriter, loc, 64);
        auto bf16x8 = mlir::VectorType::get({8}, rewriter.getBF16Type());
        auto tid = toIndex(rewriter, loc, op.getThreadId());
        auto packetRow = mlir::arith::ShRUIOp::create(rewriter, loc, tid, c2);
        auto packet = mlir::arith::AndIOp::create(rewriter, loc, tid, c3);
        auto feature = indexAdd(
            rewriter, loc,
            indexMul(rewriter, loc, toIndex(rewriter, loc, op.getKHalf()), c32),
            indexMul(rewriter, loc, packet, c8));
        auto chunk = mlir::arith::DivUIOp::create(
            rewriter, loc, toIndex(rewriter, loc, op.getChunkStart()), c64);
        auto globalValue = indexAdd(rewriter, loc,
                                    toIndex(rewriter, loc, op.getValueBase()),
                                    packetRow);
        auto hPacket = mlir::vector::LoadOp::create(
            rewriter, loc, bf16x8, source,
            mlir::ValueRange{c0, chunk, toIndex(rewriter, loc, op.getValueHead()),
                             globalValue, feature});
        hPacket->setAttr("c23.late_pipeline", rewriter.getStringAttr("H_issue"));
        hPacket->setAttr("c23.packet", rewriter.getStringAttr("bf16x8"));
        // The next K load follows the current H load. This lets the AMDGPU
        // waitcnt lowerer satisfy the H store while retaining the newer K
        // request outstanding for the following Q@H MFMA window.
        if (mlir::failed(prefetchK(target, rewriter)))
            return mlir::failure();
        auto dstRow = indexAdd(rewriter, loc,
                               indexConstant(rewriter, loc, plan.c19PhaseBase),
                               packetRow);
        auto store = mlir::vector::StoreOp::create(
            rewriter, loc, hPacket, op.getBStage(),
            mlir::ValueRange{dstRow, indexMul(rewriter, loc, packet, c8)});
        store->setAttr("c19.shared_region", rewriter.getStringAttr("H"));
        store->setAttr("c23.late_pipeline", rewriter.getStringAttr("H_commit"));
        mlir::gpu::BarrierOp::create(rewriter, loc);
        return mlir::success();
    }

    mlir::LogicalResult emitPendingKCommit(
        AMDGPUBlockDotBF16F32Op op, mlir::PatternRewriter &rewriter,
        const FullPhysicalRegionPlan &plan) {
        auto it = pending_.find(op.getOperation());
        if (it == pending_.end())
            return rewriter.notifyMatchFailure(
                op, "C23 K consumer has no prefetched packet");

        // prefetchK(next) may grow pending_ and invalidate DenseMap
        // iterators. Retain the current packet before issuing its successor.
        const C23PendingKPacket pending = it->second;
        pending_.erase(it);

        auto loc = op.getLoc();
        auto c128 = indexConstant(rewriter, loc, 128);
        auto c2 = indexConstant(rewriter, loc, 2);
        auto c3 = indexConstant(rewriter, loc, 3);
        auto c6 = indexConstant(rewriter, loc, 6);
        auto c1 = indexConstant(rewriter, loc, 1);
        auto tid = toIndex(rewriter, loc, op.getThreadId());
        auto wave = mlir::arith::ShRUIOp::create(rewriter, loc, tid, c6);
        auto lane = mlir::arith::AndIOp::create(rewriter, loc, tid,
                                                indexConstant(rewriter, loc, 63));
        auto wavePair = mlir::arith::ShRUIOp::create(rewriter, loc, wave, c1);
        auto producerLinear = indexAdd(
            rewriter, loc, indexMul(rewriter, loc, wavePair,
                                    indexConstant(rewriter, loc, 64)),
            lane);
        auto packetRow = mlir::arith::ShRUIOp::create(rewriter, loc,
                                                       producerLinear, c2);
        auto packet = mlir::arith::AndIOp::create(rewriter, loc,
                                                   producerLinear, c3);
        auto sourceHalf = toIndex(rewriter, loc, op.getValueBase());
        auto dstRow = indexAdd(
            rewriter, loc, indexConstant(rewriter, loc, plan.c19PhaseBase),
            indexAdd(rewriter, loc, indexMul(rewriter, loc, sourceHalf, c128),
                     packetRow));
        auto guarded = mlir::scf::IfOp::create(
            rewriter, loc, pending.active, /*withElseRegion=*/false);
        rewriter.setInsertionPointToStart(&guarded.getThenRegion().front());
        auto store = mlir::vector::StoreOp::create(
            rewriter, loc, pending.packet, op.getBStage(),
            mlir::ValueRange{dstRow,
                             indexMul(rewriter, loc, packet,
                                      indexConstant(rewriter, loc, 8))});
        store->setAttr("c19.shared_region", rewriter.getStringAttr("K"));
        store->setAttr("c23.late_pipeline", rewriter.getStringAttr("K_commit"));
        rewriter.setInsertionPointAfter(guarded);
        return mlir::success();
    }

  private:
    llvm::DenseMap<mlir::Operation *, C23PendingKPacket> pending_;
};

// C24 owns the lifetime of a pending packet at the function/region boundary.
// It deliberately creates only the first H -> K0 edge in C21.  The
// corresponding issue and commit are ordinary dialect operations with a
// direct SSA payload, which lets later passes verify dominance and erase the
// pair atomically after lowering.  C21's ownership, physical layout, K32
// order, and current MFMA sequence are left untouched.
struct C24PendingPacketPlan {
    AMDGPUBlockDotBF16F32Op currentH;
    AMDGPUBlockDotBF16F32Op nextK;
    AMDGPURegionPendingPacketIssueOp issue;
    AMDGPURegionPendingPacketCommitOp commit;
};

class C24RegionPendingPacketPlanner {
  public:
    mlir::LogicalResult materialize(mlir::func::FuncOp function,
                                    const FullPhysicalRegionPlan &plan) {
        if (!useC24RegionPendingPacketInfrastructure() &&
            !useC25CurrentReadyNextPending())
            return mlir::success();

        llvm::SmallVector<AMDGPUBlockDotBF16F32Op> candidates;
        function.walk([&](AMDGPUBlockDotBF16F32Op op) {
            if (op->hasAttr("c21.selected_native_pipeline") &&
                genericOperandSourceRole(op) == "H")
                candidates.push_back(op);
        });
        if (candidates.empty()) {
            function.emitError(
                "C24 pending-packet infrastructure requires a C21 H region");
            return mlir::failure();
        }

        // C21 is one source K32 loop.  Each H logically precedes K0.  Build
        // the one proof edge for every statically unrolled H region, but do
        // not speculate across K0 -> K1 or an iteration boundary.
        for (auto h : candidates) {
            auto k = nextK(h);
            if (!k) {
                h.emitError("C24 H region has no same-block next K consumer");
                return mlir::failure();
            }
            if (mlir::failed(materializeHToK(h, k, plan)))
                return mlir::failure();
        }

        if (mlir::failed(mlir::verify(function.getOperation()))) {
            function.emitError(
                "C24 region plan produced invalid SSA or dominance");
            return mlir::failure();
        }
        function->setAttr(
            useC25CurrentReadyNextPending()
                ? "avelang.stage6z.c25.current_ready_next_pending"
                : "avelang.stage6z.c24.region_pending_packet",
            mlir::StringAttr::get(
                function.getContext(),
                useC25CurrentReadyNextPending()
                    ? "h_ready_then_k_pending_region_slots"
                    : "h_to_k0_direct_ssa_issue_commit"));
        return mlir::success();
    }

  private:
    AMDGPUBlockDotBF16F32Op nextK(AMDGPUBlockDotBF16F32Op op) const {
        for (auto *cursor = op->getNextNode(); cursor;
             cursor = cursor->getNextNode()) {
            auto candidate = mlir::dyn_cast<AMDGPUBlockDotBF16F32Op>(cursor);
            if (!candidate)
                continue;
            if (!candidate->hasAttr("c21.selected_native_pipeline"))
                return {};
            return genericOperandSourceRole(candidate) == "K" ? candidate
                                                                : AMDGPUBlockDotBF16F32Op{};
        }
        return {};
    }

    mlir::LogicalResult materializeHToK(
        AMDGPUBlockDotBF16F32Op h, AMDGPUBlockDotBF16F32Op k,
        const FullPhysicalRegionPlan &plan) {
        auto hSourceType = mlir::dyn_cast<mlir::MemRefType>(h.getSourceK().getType());
        auto hDestinationType =
            mlir::dyn_cast<mlir::MemRefType>(h.getBStage().getType());
        auto sourceType = mlir::dyn_cast<mlir::MemRefType>(k.getSourceK().getType());
        auto destinationType =
            mlir::dyn_cast<mlir::MemRefType>(k.getBStage().getType());
        if (useC25CurrentReadyNextPending() &&
            (!hSourceType || hSourceType.getRank() != 5 ||
             !hSourceType.getElementType().isBF16() || !hDestinationType ||
             hDestinationType.getRank() != 2 ||
             !hDestinationType.getElementType().isBF16())) {
            h.emitError("C25 requires BF16 rank-5 H source and rank-2 shared destination");
            return mlir::failure();
        }
        if (!sourceType || sourceType.getRank() != 4 ||
            !sourceType.getElementType().isBF16() || !destinationType ||
            destinationType.getRank() != 2 ||
            !destinationType.getElementType().isBF16()) {
            k.emitError("C24 requires BF16 rank-4 K source and rank-2 shared destination");
            return mlir::failure();
        }
        if (k->hasAttr("avelang.c24.pending_packet.commit"))
            return mlir::success();

        mlir::PatternRewriter rewriter(k.getContext());
        auto loc = k.getLoc();
        // Every index and predicate below feeds the pending issue.  Anchor
        // their definitions before H first; constructing them with an
        // unpositioned PatternRewriter leaves unlinked SSA operands that the
        // region verifier correctly rejects.
        rewriter.setInsertionPoint(h);
        auto c0 = indexConstant(rewriter, loc, 0);
        auto c1 = indexConstant(rewriter, loc, 1);
        auto c2 = indexConstant(rewriter, loc, 2);
        auto c3 = indexConstant(rewriter, loc, 3);
        auto c6 = indexConstant(rewriter, loc, 6);
        auto c8 = indexConstant(rewriter, loc, 8);
        auto c32 = indexConstant(rewriter, loc, 32);
        auto c64 = indexConstant(rewriter, loc, 64);
        // The issue is deliberately before H. C25's proof edge is the first
        // serialized Q@K source half (K0), so its source-half is the fixed
        // constant zero. H's value_base is the output V64 block (0 or 64),
        // not a K source-half; using it here addresses beyond the current
        // BT64 chunk for the second V block. The later K SSA value cannot be
        // used at this insertion point because it does not dominate H.
        auto tid = toIndex(rewriter, loc, h.getThreadId());
        auto wave = mlir::arith::ShRUIOp::create(rewriter, loc, tid, c6);
        auto lane = mlir::arith::AndIOp::create(rewriter, loc, tid,
                                                indexConstant(rewriter, loc, 63));
        auto wavePair = mlir::arith::ShRUIOp::create(rewriter, loc, wave, c1);
        auto producerLinear = indexAdd(
            rewriter, loc, indexMul(rewriter, loc, wavePair,
                                    indexConstant(rewriter, loc, 64)),
            lane);
        auto packetRow = mlir::arith::ShRUIOp::create(rewriter, loc,
                                                       producerLinear, c2);
        auto packet = mlir::arith::AndIOp::create(rewriter, loc,
                                                   producerLinear, c3);
        auto feature = indexAdd(
            rewriter, loc,
            indexMul(rewriter, loc, toIndex(rewriter, loc, h.getKHalf()), c32),
            indexMul(rewriter, loc, packet, c8));
        auto token = indexAdd(rewriter, loc,
                              toIndex(rewriter, loc, h.getChunkStart()),
                              packetRow);
        auto active = mlir::arith::CmpIOp::create(
            rewriter, loc, mlir::arith::CmpIPredicate::eq,
            mlir::arith::AndIOp::create(rewriter, loc, wave, c1), c0);
        // C21 serializes score source halves, so K reuses the one C19 K
        // phase slot.  This is the same destination used by its ordinary K
        // producer, not a V64-block-indexed slot.
        auto dstRow = indexAdd(rewriter, loc,
                               indexConstant(rewriter, loc, plan.c19PhaseBase),
                               packetRow);
        auto packetType = mlir::VectorType::get({8}, rewriter.getBF16Type());

        if (useC25CurrentReadyNextPending()) {
            // Region-level prologue.  H is the current operand: publish it
            // before K issues, so the later H MFMA window can cover the
            // younger K packet instead of draining it in H's first wait.
            auto trueValue = mlir::arith::ConstantIntOp::create(
                rewriter, loc, 1, 1);
            auto hChunk = mlir::arith::DivUIOp::create(
                rewriter, loc, toIndex(rewriter, loc, h.getChunkStart()), c64);
            auto hPacketRow = mlir::arith::ShRUIOp::create(rewriter, loc, tid, c2);
            auto hPacket = mlir::arith::AndIOp::create(rewriter, loc, tid, c3);
            auto hFeature = indexAdd(
                rewriter, loc,
                indexMul(rewriter, loc, toIndex(rewriter, loc, h.getKHalf()), c32),
                indexMul(rewriter, loc, hPacket, c8));
            auto hValue = indexAdd(
                rewriter, loc, toIndex(rewriter, loc, h.getValueBase()), hPacketRow);
            auto hDstRow = indexAdd(
                rewriter, loc, indexConstant(rewriter, loc, plan.c19PhaseBase),
                hPacketRow);
            auto hIssue = AMDGPURegionPendingPacketIssueOp::create(
                rewriter, loc, packetType, h.getSourceK(), trueValue,
                mlir::ValueRange{c0, hChunk,
                                 toIndex(rewriter, loc, h.getValueHead()),
                                 hValue, hFeature});
            hIssue->setAttr("avelang.region_pending_packet.role",
                            rewriter.getStringAttr("H"));
            hIssue->setAttr("avelang.region_pending_packet.stage_state",
                            rewriter.getStringAttr("prologue_pending_current"));
            auto hCommit = AMDGPURegionPendingPacketCommitOp::create(
                rewriter, loc, hIssue.getPacket(), trueValue, h.getBStage(),
                mlir::ValueRange{hDstRow,
                                 indexMul(rewriter, loc, hPacket, c8)});
            hCommit->setAttr("avelang.region_pending_packet.role",
                             rewriter.getStringAttr("H"));
            hCommit->setAttr("avelang.region_pending_packet.stage_state",
                             rewriter.getStringAttr("current_ready"));
            hCommit->setAttr("c19.shared_region", rewriter.getStringAttr("H"));
            h->setAttr("avelang.c25.current_ready_packet.commit",
                       rewriter.getUnitAttr());
        }

        // In C24 this issue precedes the normal H producer.  C25's H packet
        // has already been made ready above, so this becomes the steady-state
        // next-packet issue immediately before the current H MFMA window.
        auto issue = AMDGPURegionPendingPacketIssueOp::create(
            rewriter, loc, packetType, k.getSourceK(), active,
            mlir::ValueRange{c0, token, toIndex(rewriter, loc, h.getKeyHead()),
                             feature});
        issue->setAttr("avelang.region_pending_packet.role",
                       rewriter.getStringAttr("K"));
        issue->setAttr("avelang.region_pending_packet.issue_point",
                       rewriter.getStringAttr("before_current_H_block_dot"));
        issue->setAttr("avelang.region_pending_packet.stage_id",
                       rewriter.getI64IntegerAttr(
                           useC25CurrentReadyNextPending() ? 1 : 0));
        if (useC25CurrentReadyNextPending()) {
            issue->setAttr("avelang.region_pending_packet.stage_state",
                           rewriter.getStringAttr("steady_pending_next"));
        }

        rewriter.setInsertionPoint(k);
        auto commit = AMDGPURegionPendingPacketCommitOp::create(
            rewriter, loc, issue.getPacket(), active, k.getBStage(),
            mlir::ValueRange{dstRow, indexMul(rewriter, loc, packet, c8)});
        commit->setAttr("avelang.region_pending_packet.role",
                        rewriter.getStringAttr("K"));
        commit->setAttr("avelang.region_pending_packet.commit_point",
                        rewriter.getStringAttr("before_next_K_block_dot"));
        commit->setAttr("avelang.region_pending_packet.first_consumer",
                        rewriter.getStringAttr("Q_at_K"));
        commit->setAttr("c19.shared_region", rewriter.getStringAttr("K"));
        if (useC25CurrentReadyNextPending()) {
            commit->setAttr("avelang.region_pending_packet.stage_state",
                            rewriter.getStringAttr("epilogue_next_ready"));
        }
        h->setAttr("avelang.c24.pending_packet.issue",
                   rewriter.getUnitAttr());
        k->setAttr("avelang.c24.pending_packet.commit",
                   rewriter.getUnitAttr());
        plans_.push_back({h, k, issue, commit});
        return mlir::success();
    }

    llvm::SmallVector<C24PendingPacketPlan> plans_;
};

// C19 is deliberately a separate producer builder.  It is not allowed to
// call the C18/P2 producer helper: all four logical roles must enter through
// the same FullPhysicalRegionPlan ownership boundary.  The physical recipes
// are the already validated C16/C18 equations; this helper only changes who
// materializes them and records the single shared arena/lifetime contract.
mlir::LogicalResult emitC19FullPhysicalProducer(
    AMDGPUBlockDotBF16F32Op op, mlir::PatternRewriter &rewriter,
    const FullPhysicalRegionPlan &plan,
    C23LatePipelineMaterializer *c23Materializer = nullptr) {
    auto loc = op.getLoc();
    auto source = op.getSourceK();
    auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
    auto role = genericOperandSourceRole(op);
    if (!sourceType || (role != "Q" && role != "H" && role != "K" &&
                        role != "V"))
        return rewriter.notifyMatchFailure(
            op, "C19 producer requires a plan-owned Q/H/K/V logical source");
    if (!op->hasAttr("c19.full_physical_region"))
        return rewriter.notifyMatchFailure(
            op, "C19 producer is missing FullPhysicalRegionPlan ownership");

    auto c0 = indexConstant(rewriter, loc, 0);
    auto c1 = indexConstant(rewriter, loc, 1);
    auto c2 = indexConstant(rewriter, loc, 2);
    auto c3 = indexConstant(rewriter, loc, 3);
    auto c6 = indexConstant(rewriter, loc, 6);
    auto c8 = indexConstant(rewriter, loc, 8);
    auto c32 = indexConstant(rewriter, loc, 32);
    auto c64 = indexConstant(rewriter, loc, 64);
    auto tid = toIndex(rewriter, loc, op.getThreadId());
    auto chunkStart = toIndex(rewriter, loc, op.getChunkStart());
    auto valueHead = toIndex(rewriter, loc, op.getValueHead());
    auto keyHead = toIndex(rewriter, loc, op.getKeyHead());
    auto valueBase = toIndex(rewriter, loc, op.getValueBase());
    auto kStage = toIndex(rewriter, loc, op.getKHalf());
    auto bf16x8 = mlir::VectorType::get({8}, rewriter.getBF16Type());
    auto f32x8 = mlir::VectorType::get({8}, rewriter.getF32Type());
    const bool c21Pipeline = op->hasAttr("c21.selected_native_pipeline");

    if (role == "Q") {
        if (!op->hasAttr("c19.q_producer_owner"))
            return mlir::success();
        if (sourceType.getRank() != 4 || sourceType.getShape()[2] != 4)
            return rewriter.notifyMatchFailure(
                op, "C19 Q producer requires BF16 [1,T,4,128]");
        if (c21Pipeline) {
            // A C21 logical Q owner lives inside the K32 source superloop.
            // It issues exactly the current stage's packet into one of two
            // physical slots.  The H/K consumers read that same slot before
            // the next source iteration can rotate over it.
            auto token = mlir::arith::ShRUIOp::create(rewriter, loc, tid, c2);
            auto packet = mlir::arith::AndIOp::create(rewriter, loc, tid, c3);
            auto feature = indexAdd(
                rewriter, loc, indexMul(rewriter, loc, kStage, c32),
                indexMul(rewriter, loc, packet, c8));
            auto stagedLoad = mlir::vector::LoadOp::create(
                rewriter, loc, bf16x8, source,
                mlir::ValueRange{c0, indexAdd(rewriter, loc, chunkStart, token),
                                 keyHead, feature});
            stagedLoad->setAttr("c21.producer", rewriter.getStringAttr("Q"));
            stagedLoad->setAttr("c21.packet", rewriter.getStringAttr("bf16x8"));
            auto qF32 = mlir::arith::ExtFOp::create(
                rewriter, loc, f32x8, stagedLoad.getResult(),
                mlir::arith::FastMathFlagsAttr{});
            auto scale = mlir::vector::SplatOp::create(rewriter, loc, f32x8,
                                                        op.getGLast());
            auto scaled = mlir::arith::MulFOp::create(rewriter, loc, qF32, scale);
            auto staged = mlir::arith::TruncFOp::create(rewriter, loc, bf16x8,
                                                        scaled);
            auto slot = mlir::arith::AndIOp::create(
                rewriter, loc, kStage, indexConstant(rewriter, loc, 1));
            auto store = mlir::vector::StoreOp::create(
                rewriter, loc, staged, op.getBStage(),
                mlir::ValueRange{
                    indexAdd(rewriter, loc, indexMul(rewriter, loc, slot, c64),
                             token),
                    indexMul(rewriter, loc, packet, c8)});
            store->setAttr("c21.shared_region",
                           rewriter.getStringAttr("Q_stage_slot"));
            store->setAttr("c21.slot_index", rewriter.getStringAttr("k_stage_mod_2"));
            mlir::gpu::BarrierOp::create(rewriter, loc);
            return mlir::success();
        }
        // Q is a single logical producer over four K32 stages.  Each stage is
        // a 64x32 block and is written into the plan-owned Q region.
        for (int64_t stage = 0; stage < 4; ++stage) {
            auto token = mlir::arith::ShRUIOp::create(
                rewriter, loc, tid, c2);
            auto packet = mlir::arith::AndIOp::create(
                rewriter, loc, tid, c3);
            auto feature = indexAdd(
                rewriter, loc,
                indexAdd(rewriter, loc,
                         indexConstant(rewriter, loc, stage * 32),
                         indexMul(rewriter, loc, packet, c8)),
                c0);
            auto stagedLoad = mlir::vector::LoadOp::create(
                rewriter, loc, bf16x8, source,
                mlir::ValueRange{c0, indexAdd(rewriter, loc, chunkStart, token),
                                 keyHead, feature});
            stagedLoad->setAttr("c19.producer", rewriter.getStringAttr("Q"));
            stagedLoad->setAttr("c19.packet", rewriter.getStringAttr("bf16x8"));
            mlir::Value staged = stagedLoad.getResult();
            // Z5B's Q producer stores BF16(round(q * scale)).  Keep that
            // ABI-visible rounding in the single C19 plan-owned producer;
            // the logical operation carries the constexpr scale in g_last
            // only for this Q-owner call.
            auto qF32 = mlir::arith::ExtFOp::create(
                rewriter, loc, f32x8, staged,
                mlir::arith::FastMathFlagsAttr{});
            auto scale = mlir::vector::SplatOp::create(
                rewriter, loc, f32x8, op.getGLast());
            auto scaled = mlir::arith::MulFOp::create(
                rewriter, loc, qF32, scale);
            staged = mlir::arith::TruncFOp::create(
                rewriter, loc, bf16x8, scaled);
            auto store = mlir::vector::StoreOp::create(
                rewriter, loc, staged, op.getBStage(),
                mlir::ValueRange{
                    indexAdd(rewriter, loc,
                             indexConstant(rewriter, loc, stage * 64), token),
                    indexMul(rewriter, loc, packet, c8)});
            store->setAttr("c19.shared_region",
                           rewriter.getStringAttr("Q"));
        }
        mlir::gpu::BarrierOp::create(rewriter, loc);
        return mlir::success();
    }

    if (role == "H") {
        if (c21Pipeline && useC25CurrentReadyNextPending() &&
            op->hasAttr("avelang.c25.current_ready_packet.commit")) {
            // C25's region planner has already issued, waited for and
            // published this current H packet.  Re-emitting the ordinary H
            // producer would duplicate global work and collapse the proof.
            return mlir::success();
        }
        if (c21Pipeline && c23Materializer && c23Materializer->enabledFor(op))
            return c23Materializer->emitHWithNextK(op, rewriter, plan);
        if (sourceType.getRank() != 5)
            return rewriter.notifyMatchFailure(
                op, "C19 H producer requires rank-5 BF16 H source");
        auto chunk = mlir::arith::DivUIOp::create(rewriter, loc, chunkStart, c64);
        auto packetRow = mlir::arith::ShRUIOp::create(rewriter, loc, tid, c2);
        auto packet = mlir::arith::AndIOp::create(rewriter, loc, tid, c3);
        auto feature = indexAdd(
            rewriter, loc, indexMul(rewriter, loc, kStage, c32),
            indexMul(rewriter, loc, packet, c8));
        auto value = indexAdd(rewriter, loc, valueBase, packetRow);
        auto staged = mlir::vector::LoadOp::create(
            rewriter, loc, bf16x8, source,
            mlir::ValueRange{c0, chunk, valueHead, value, feature});
        staged->setAttr("c19.producer", rewriter.getStringAttr("H"));
        staged->setAttr("c19.packet", rewriter.getStringAttr("bf16x8"));
        auto store = mlir::vector::StoreOp::create(
            rewriter, loc, staged, op.getBStage(),
            mlir::ValueRange{
                indexAdd(rewriter, loc,
                         indexConstant(rewriter, loc, plan.c19PhaseBase),
                         packetRow),
                indexMul(rewriter, loc, packet, c8)});
        store->setAttr("c19.shared_region", rewriter.getStringAttr("H"));
        mlir::gpu::BarrierOp::create(rewriter, loc);
        return mlir::success();
    }

    if (role == "K") {
        // C24 has already created a region-owned direct SSA issue/commit
        // pair before greedy lowering starts.  The late GPU-module commit
        // publishes this exact packet and its existing barrier before this
        // K consumer, so emitting the ordinary per-op producer here would
        // both duplicate the global load and collapse the pending window.
        if (c21Pipeline && useC24RegionPendingPacketInfrastructure() &&
            op->hasAttr("avelang.c24.pending_packet.commit"))
            return mlir::success();
        if (c21Pipeline && c23Materializer && c23Materializer->enabledFor(op))
            return c23Materializer->emitPendingKCommit(op, rewriter, plan);
        if (sourceType.getRank() != 4)
            return rewriter.notifyMatchFailure(
                op, "C19 K producer requires rank-4 BF16 K source");
        // thread_id is a linear work-item id.  The wave id is tid >> 6;
        // c64 is the wave width used for multiplication/division elsewhere,
        // not the shift amount.
        auto wave = mlir::arith::ShRUIOp::create(rewriter, loc, tid, c6);
        auto lane = mlir::arith::AndIOp::create(rewriter, loc, tid,
                                                indexConstant(rewriter, loc, 63));
        // Two adjacent waves form one producer pair.  The pair index is
        // wave >> 1; using wave >> 2 would make waves 0 and 2 write the same
        // K rows and leave the second half of the staged block undefined.
        auto wavePair = mlir::arith::ShRUIOp::create(rewriter, loc, wave, c1);
        auto producerLinear = indexAdd(
            rewriter, loc, indexMul(rewriter, loc, wavePair, c64), lane);
        auto packetRow = mlir::arith::ShRUIOp::create(
            rewriter, loc, producerLinear, c2);
        auto packet = mlir::arith::AndIOp::create(
            rewriter, loc, producerLinear, c3);
        auto feature = indexAdd(
            rewriter, loc, indexMul(rewriter, loc, kStage, c32),
            indexMul(rewriter, loc, packet, c8));
        auto token = indexAdd(
            rewriter, loc,
            indexAdd(rewriter, loc, chunkStart,
                     indexMul(rewriter, loc, valueBase, c32)),
            packetRow);
        auto active = mlir::arith::CmpIOp::create(
            rewriter, loc, mlir::arith::CmpIPredicate::eq,
            mlir::arith::AndIOp::create(
                rewriter, loc, wave, indexConstant(rewriter, loc, 1)), c0);
        auto guarded = mlir::scf::IfOp::create(rewriter, loc, active,
                                               /*withElseRegion=*/false);
        rewriter.setInsertionPointToStart(&guarded.getThenRegion().front());
        auto staged = mlir::vector::LoadOp::create(
            rewriter, loc, bf16x8, source,
            mlir::ValueRange{c0, token, keyHead, feature});
        staged->setAttr("c19.producer", rewriter.getStringAttr("K"));
        staged->setAttr("c19.packet", rewriter.getStringAttr("bf16x8"));
        auto store = mlir::vector::StoreOp::create(
            rewriter, loc, staged, op.getBStage(),
            mlir::ValueRange{
                indexAdd(rewriter, loc,
                         indexConstant(rewriter, loc, plan.c19PhaseBase),
                         packetRow),
                indexMul(rewriter, loc, packet, c8)});
        store->setAttr("c19.shared_region", rewriter.getStringAttr("K"));
        rewriter.setInsertionPointAfter(guarded);
        mlir::gpu::BarrierOp::create(rewriter, loc);
        return mlir::success();
    }

    if (sourceType.getRank() != 4 || sourceType.getShape()[2] != 8)
        return rewriter.notifyMatchFailure(
            op, "C19 V producer requires BF16 [1,T,8,128]");
    if (!op->hasAttr("c19.v_producer_owner"))
        return mlir::success();
    // Preserve the C18 V ownership exactly: one owner emits two repetitions
    // of 256 threads x BF16x8, covering all 64 tokens of the CTA-local
    // V-new tile. C19 changes ownership, not V producer coverage.
    auto c7 = indexConstant(rewriter, loc, 7);
    for (int64_t rep = 0; rep < 2; ++rep) {
        auto linear = indexAdd(
            rewriter, loc, tid, indexConstant(rewriter, loc, rep * 256));
        auto packet = mlir::arith::AndIOp::create(rewriter, loc, linear, c7);
        auto token = mlir::arith::ShRUIOp::create(rewriter, loc, linear, c3);
        auto valueOffset = indexMul(rewriter, loc, packet, c8);
        auto sourceValue = indexAdd(rewriter, loc, valueBase, valueOffset);
        auto staged = mlir::vector::LoadOp::create(
            rewriter, loc, bf16x8, source,
            mlir::ValueRange{c0, indexAdd(rewriter, loc, chunkStart, token),
                             valueHead, sourceValue});
        staged->setAttr("c19.producer", rewriter.getStringAttr("V"));
        staged->setAttr("c19.packet", rewriter.getStringAttr("bf16x8"));
        for (int64_t element = 0; element < 8; ++element) {
            auto row = indexAdd(
                rewriter, loc,
                indexConstant(rewriter, loc,
                              plan.scoreBase + plan.scoreVBase),
                indexAdd(
                    rewriter, loc,
                    indexMul(rewriter, loc,
                             indexAdd(rewriter, loc, valueOffset,
                                      indexConstant(rewriter, loc, element)),
                             c2),
                    mlir::arith::ShRUIOp::create(
                        rewriter, loc, token, indexConstant(rewriter, loc, 5))));
            auto value = mlir::vector::ExtractOp::create(rewriter, loc, staged,
                                                          element);
            auto store = mlir::memref::StoreOp::create(
                rewriter, loc, value, op.getBStage(),
                mlir::ValueRange{row, mlir::arith::AndIOp::create(
                                      rewriter, loc, token,
                                      indexConstant(rewriter, loc, 31))});
            store->setAttr("c19.shared_region", rewriter.getStringAttr("V"));
        }
    }
    mlir::gpu::BarrierOp::create(rewriter, loc);
    return mlir::success();
}

// Full-scope logical-block lowering.  The source presents an existing
// resident A block, a shared destination B block, and a global logical B
// source in the operation's target-independent block contract.  This helper
// owns the producer packet, its shared placement, and the existing MFMA-B
// consumer in one lowering boundary.  It deliberately uses the same B
// consumer helper as operand mode so K and H cannot silently acquire separate
// Qwen-specific planners.
mlir::LogicalResult emitFullScopeProducer(
    AMDGPUBlockDotBF16F32Op op, mlir::PatternRewriter &rewriter,
    bool specialized, const LogicalBlockLayoutPlan *layoutPlan = nullptr,
    const FullPhysicalRegionPlan *c18Plan = nullptr) {
    auto loc = op.getLoc();
    auto source = op.getSourceK();
    auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
    if (!sourceType)
        return rewriter.notifyMatchFailure(
            op, "full-scope block-dot source must be a memref");

    auto sourceRole = genericOperandSourceRole(op);
    const bool isH = sourceRole == "H";
    const bool isK = sourceRole == "K";
    const bool isV = sourceRole == "V";
    if (!isH && !isK && !isV)
        return rewriter.notifyMatchFailure(
            op, "full-scope block-dot source role must be K or H");

    // C18 makes V a first-class logical block-dot source.  The source passes
    // vn as both source slots so this branch can reuse the existing public
    // operation without introducing a Qwen-specific op.  The full region
    // plan owns the packet producer and its physical score/V shared band;
    // later V consumers are marked non-owners and do not reload global V.
    if (isV) {
        if (!c18Plan || !useC18FullPhysicalRegion())
            return rewriter.notifyMatchFailure(
                op, "V full-scope producer requires the C18 physical region");
        if (sourceType.getRank() != 4 ||
            sourceType.getShape()[2] != 8 ||
            sourceType.getShape()[3] != 128)
            return rewriter.notifyMatchFailure(
                op, "C18 V source must be BF16 [1,T,8,128]");
        if (!op->hasAttr("c18.v_producer_owner"))
            return mlir::success();

        auto bf16x8 = mlir::VectorType::get({8}, rewriter.getBF16Type());
        auto c0 = indexConstant(rewriter, loc, 0);
        auto c2 = indexConstant(rewriter, loc, 2);
        auto c3 = indexConstant(rewriter, loc, 3);
        auto c8 = indexConstant(rewriter, loc, 8);
        auto c32 = indexConstant(rewriter, loc, 32);
        auto tid = layoutPlan ? layoutPlan->tid
                              : toIndex(rewriter, loc, op.getThreadId());
        auto packetLinear = tid;
        auto oneWavePackets = indexConstant(rewriter, loc, 256);
        auto valueHead = toIndex(rewriter, loc, op.getValueHead());
        auto chunkStart = toIndex(rewriter, loc, op.getChunkStart());
        auto valueBase = toIndex(rewriter, loc, op.getValueBase());
        // 256 threads * 2 packets covers V[64,128] once.  The packet is
        // contiguous in the feature dimension, so the source-level vector
        // load remains a real BF16x8 producer rather than eight scalar loads.
        for (int64_t rep = 0; rep < 2; ++rep) {
            auto linear = indexAdd(
                rewriter, loc, packetLinear,
                indexMul(rewriter, loc, indexConstant(rewriter, loc, rep),
                         oneWavePackets));
            auto token = mlir::arith::ShRUIOp::create(
                rewriter, loc, linear, c3);
            auto packet = mlir::arith::AndIOp::create(
                rewriter, loc, linear, indexConstant(rewriter, loc, 7));
            auto valueOffset = indexMul(rewriter, loc, packet, c8);
            auto sourceValue = indexAdd(rewriter, loc, valueBase, valueOffset);
            auto staged = mlir::vector::LoadOp::create(
                rewriter, loc, bf16x8, source,
                mlir::ValueRange{c0, indexAdd(rewriter, loc, chunkStart, token),
                                 valueHead, sourceValue});
            staged->setAttr("c18.full_region.producer",
                            rewriter.getStringAttr("V_global_bf16x8_once"));
            for (int64_t element = 0; element < 8; ++element) {
                auto row = indexAdd(
                    rewriter, loc,
                    indexConstant(rewriter, loc, c18Plan->qCacheRows +
                                                       c18Plan->scoreVBase),
                    indexAdd(rewriter, loc,
                             indexMul(rewriter, loc,
                                      indexAdd(rewriter, loc, valueOffset,
                                               indexConstant(rewriter, loc,
                                                             element)),
                                      c2),
                             mlir::arith::ShRUIOp::create(
                                 rewriter, loc, token,
                                 indexConstant(rewriter, loc, 5))));
                auto value = mlir::vector::ExtractOp::create(
                    rewriter, loc, staged, element);
                auto store = mlir::memref::StoreOp::create(
                    rewriter, loc, value, op.getBStage(),
                    mlir::ValueRange{row, mlir::arith::AndIOp::create(
                                          rewriter, loc, token,
                                          indexConstant(rewriter, loc, 31))});
                store->setAttr("c18.full_region.shared_slot",
                               rewriter.getStringAttr("score_v"));
                store->setAttr("c18.full_region.packet",
                               rewriter.getStringAttr("bf16x8"));
            }
        }
        mlir::gpu::BarrierOp::create(rewriter, loc);
        return mlir::success();
    }

    auto bf16x8 = mlir::VectorType::get({8}, rewriter.getBF16Type());
    auto c0 = indexConstant(rewriter, loc, 0);
    auto c2 = indexConstant(rewriter, loc, 2);
    auto c4 = indexConstant(rewriter, loc, 4);
    auto c8 = indexConstant(rewriter, loc, 8);
    auto c32 = indexConstant(rewriter, loc, 32);
    auto c64 = indexConstant(rewriter, loc, 64);
    auto c128 = indexConstant(rewriter, loc, 128);
    mlir::Value tid;
    mlir::Value kStage;
    mlir::Value packetRow;
    mlir::Value packet;
    mlir::Value packetCol;
    mlir::Value feature;
    if (layoutPlan) {
        tid = layoutPlan->tid;
        kStage = layoutPlan->kStage;
        packetRow = layoutPlan->packetRow;
        packet = layoutPlan->packet;
        packetCol = layoutPlan->packetCol;
        feature = layoutPlan->feature;
    } else {
        tid = toIndex(rewriter, loc, op.getThreadId());
        kStage = toIndex(rewriter, loc, op.getKHalf());
        packetRow = mlir::arith::DivUIOp::create(rewriter, loc, tid, c4);
        packet = mlir::arith::RemUIOp::create(rewriter, loc, tid, c4);
        packetCol = indexMul(rewriter, loc, packet, c8);
        feature = indexAdd(
            rewriter, loc, indexMul(rewriter, loc, kStage, c32), packetCol);
    }

    // H is laid out [chunk, value-head, value, K].  Each thread owns one
    // contiguous BF16x8 packet in the K dimension, so the specialized arm
    // can keep one typed global load and one packed LDS store.  The generic
    // arm expands the exact same ownership into eight scalar loads/stores.
    if (isH) {
        if (sourceType.getRank() != 5)
            return rewriter.notifyMatchFailure(
                op, "full-scope H source must be rank-5");
        auto chunk = mlir::arith::DivUIOp::create(
            rewriter, loc, toIndex(rewriter, loc, op.getChunkStart()), c64);
        auto globalValue = indexAdd(rewriter, loc,
                                    toIndex(rewriter, loc, op.getValueBase()),
                                    packetRow);
        auto dstRow = indexAdd(rewriter, loc, indexConstant(rewriter, loc, 320),
                               packetRow);
        if (specialized) {
            auto staged = mlir::vector::LoadOp::create(
                rewriter, loc, bf16x8, source,
                mlir::ValueRange{c0, chunk, toIndex(rewriter, loc,
                                                   op.getValueHead()),
                                 globalValue, feature});
            staged->setAttr("avelang.block_dot.full_scope.producer",
                            rewriter.getStringAttr("global_bf16x8"));
            if (c18Plan)
                staged->setAttr("c18.full_region.producer",
                                rewriter.getStringAttr("H_global_bf16x8"));
            auto store = mlir::vector::StoreOp::create(
                rewriter, loc, staged, op.getBStage(),
                mlir::ValueRange{dstRow, packetCol});
            store->setAttr("avelang.block_dot.full_scope.layout",
                           rewriter.getStringAttr("consumer_shared_b32"));
            if (c18Plan)
                store->setAttr("c18.full_region.shared_slot",
                               rewriter.getStringAttr("H_phase"));
        } else {
            for (int64_t element = 0; element < 8; ++element) {
                auto sourceCol = indexAdd(
                    rewriter, loc, feature,
                    indexConstant(rewriter, loc, element));
                auto value = mlir::memref::LoadOp::create(
                    rewriter, loc, source,
                    mlir::ValueRange{c0, chunk, toIndex(rewriter, loc,
                                                       op.getValueHead()),
                                     globalValue, sourceCol});
                auto storeCol = indexAdd(
                    rewriter, loc, packetCol,
                    indexConstant(rewriter, loc, element));
                auto store = mlir::memref::StoreOp::create(
                    rewriter, loc, value, op.getBStage(),
                    mlir::ValueRange{dstRow, storeCol});
                store->setAttr("avelang.block_dot.full_scope.layout",
                               rewriter.getStringAttr("consumer_shared_b32"));
                if (c18Plan)
                    store->setAttr("c18.full_region.shared_slot",
                                   rewriter.getStringAttr("H_phase"));
            }
        }
        return mlir::success();
    }

    // K is laid out [token, key-head, K].  The first 128 threads own the
    // 32x32 block, again as one contiguous eight-element packet per thread.
    // The remaining wave participates in the barrier but does not issue a
    // duplicate global producer.
    if (sourceType.getRank() != 4)
        return rewriter.notifyMatchFailure(
            op, "full-scope K source must be rank-4");
    mlir::Value valueHalf;
    if (layoutPlan) {
        valueHalf = layoutPlan->valueHalf;
        packetRow = layoutPlan->packetRow;
        packet = layoutPlan->packet;
        packetCol = layoutPlan->packetCol;
        feature = layoutPlan->feature;
    } else {
        auto wave = mlir::arith::DivUIOp::create(rewriter, loc, tid, c64);
        auto lane = mlir::arith::RemUIOp::create(rewriter, loc, tid, c64);
        valueHalf = mlir::arith::RemUIOp::create(rewriter, loc, wave, c2);
        auto producerLinear = indexAdd(
            rewriter, loc,
            indexMul(rewriter, loc,
                     mlir::arith::DivUIOp::create(rewriter, loc, wave, c2),
                     c64),
            lane);
        packetRow =
            mlir::arith::DivUIOp::create(rewriter, loc, producerLinear, c4);
        packet =
            mlir::arith::RemUIOp::create(rewriter, loc, producerLinear, c4);
        packetCol = indexMul(rewriter, loc, packet, c8);
        feature = indexAdd(
            rewriter, loc, indexMul(rewriter, loc, kStage, c32), packetCol);
    }
    auto active = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::eq, valueHalf, c0);
    auto guarded = mlir::scf::IfOp::create(rewriter, loc, active,
                                           /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&guarded.getThenRegion().front());
    auto sourceHalf = toIndex(rewriter, loc, op.getValueBase());
    auto token = indexAdd(
        rewriter, loc,
        indexAdd(rewriter, loc, toIndex(rewriter, loc, op.getChunkStart()),
                 indexMul(rewriter, loc, sourceHalf, c32)),
        packetRow);
    auto dstRow = indexAdd(
        rewriter, loc, indexConstant(rewriter, loc, 320),
        indexMul(rewriter, loc, sourceHalf, c128));
    dstRow = indexAdd(rewriter, loc, dstRow, packetRow);
    auto keyHead = toIndex(rewriter, loc, op.getKeyHead());
    if (specialized) {
        auto staged = mlir::vector::LoadOp::create(
            rewriter, loc, bf16x8, source,
            mlir::ValueRange{c0, token, keyHead, feature});
        staged->setAttr("avelang.block_dot.full_scope.producer",
                        rewriter.getStringAttr("global_bf16x8"));
        if (c18Plan)
            staged->setAttr("c18.full_region.producer",
                            rewriter.getStringAttr("K_global_bf16x8"));
        auto store = mlir::vector::StoreOp::create(
            rewriter, loc, staged, op.getBStage(),
            mlir::ValueRange{dstRow, packetCol});
        store->setAttr("avelang.block_dot.full_scope.layout",
                       rewriter.getStringAttr("consumer_shared_b32"));
        if (c18Plan)
            store->setAttr("c18.full_region.shared_slot",
                           rewriter.getStringAttr("K_phase"));
    } else {
        for (int64_t element = 0; element < 8; ++element) {
            auto sourceCol = indexAdd(
                rewriter, loc, feature,
                indexConstant(rewriter, loc, element));
            auto value = mlir::memref::LoadOp::create(
                rewriter, loc, source,
                mlir::ValueRange{c0, token, keyHead, sourceCol});
            auto storeCol = indexAdd(
                rewriter, loc, packetCol,
                indexConstant(rewriter, loc, element));
            auto store = mlir::memref::StoreOp::create(
                rewriter, loc, value, op.getBStage(),
                mlir::ValueRange{dstRow, storeCol});
            store->setAttr("avelang.block_dot.full_scope.layout",
                           rewriter.getStringAttr("consumer_shared_b32"));
            if (c18Plan)
                store->setAttr("c18.full_region.shared_slot",
                               rewriter.getStringAttr("K_phase"));
        }
    }
    rewriter.setInsertionPointAfter(guarded);
    return mlir::success();
}

mlir::LogicalResult lowerFullScopeOperandMode(
    AMDGPUBlockDotBF16F32Op op, mlir::PatternRewriter &rewriter,
    LoweringKind kind, AccumulatorForwardingMap *forwardedAccumulators,
    const c13::ChunkOPhysicalPlan *c17Plan = nullptr,
    const FullPhysicalRegionPlan *c18Plan = nullptr,
    const FullPhysicalRegionPlan *c19Plan = nullptr,
    C23LatePipelineMaterializer *c23Materializer = nullptr) {
    auto loc = op.getLoc();
    if (c19Plan) {
        auto role = genericOperandSourceRole(op);
        if (role != "Q" && role != "H" && role != "K" && role != "V")
            return rewriter.notifyMatchFailure(
                op, "C19 full physical plan received an unknown source role");
        annotateC19PhysicalRegion(
            op, *c19Plan, role,
            role == "H" ? "Q@H" : role == "K" ? "Q@K"
                                               : role == "Q" ? "Q producer"
                                                              : "score@V");
        auto accumulator = materializeAccumulatorVector(
            rewriter, loc, op.getAccLow());
        if (!accumulator)
            return rewriter.notifyMatchFailure(
                op, "C19 full-scope operation requires FP32 accumulator");
        if (mlir::failed(emitC19FullPhysicalProducer(
                op, rewriter, *c19Plan, c23Materializer)))
            return mlir::failure();
        if (role == "Q") {
            auto result = joinColumns(rewriter, loc, accumulator, accumulator);
            if (auto *def = result.getDefiningOp()) {
                def->setAttr("c19.q_producer_only", rewriter.getUnitAttr());
                def->setAttr("c19.legacy_owner", rewriter.getBoolAttr(false));
            }
            rewriter.replaceOp(op, result);
            return mlir::success();
        }
        mlir::gpu::BarrierOp::create(rewriter, loc);
        auto layoutPlan = makeC17StaticAffineLayoutPlan(
            op, rewriter, loc, role == "H");
        mlir::Value updated;
        if (role == "K") {
            // K producer/consumer ownership is CTA-wide for staging, but the
            // score accumulator is owned only by value_half == 0.  Keep the
            // non-owner waves out of the MFMA path exactly as the validated
            // C18 plan does; otherwise they form a second private score
            // recurrence that is not committed to the source-level tile.
            auto active = mlir::arith::CmpIOp::create(
                rewriter, loc, mlir::arith::CmpIPredicate::eq,
                layoutPlan.valueHalf,
                indexConstant(rewriter, loc, 0));
            auto consumer = mlir::scf::IfOp::create(
                rewriter, loc, accumulator.getType(), active,
                /*withElseRegion=*/true);
            rewriter.setInsertionPointToStart(
                &consumer.getThenRegion().front());
            auto activeValue = emitFirstClassMfmaOperand(
                op, rewriter, loc, accumulator, &layoutPlan, nullptr);
            mlir::scf::YieldOp::create(rewriter, loc, activeValue);
            rewriter.setInsertionPointToStart(
                &consumer.getElseRegion().front());
            mlir::scf::YieldOp::create(rewriter, loc, accumulator);
            rewriter.setInsertionPointAfter(consumer);
            updated = consumer.getResult(0);
        } else {
            updated = emitFirstClassMfmaOperand(
                op, rewriter, loc, accumulator, &layoutPlan, nullptr);
        }
        mlir::gpu::BarrierOp::create(rewriter, loc);
        auto result = joinColumns(rewriter, loc, updated, updated);
        if (auto *def = result.getDefiningOp()) {
            def->setAttr("c19.full_physical_lowered", rewriter.getUnitAttr());
            def->setAttr("c19.legacy_owner", rewriter.getBoolAttr(false));
            if (useC21SelectedNativePipeline()) {
                def->setAttr("c21.selected_native_pipeline_lowered",
                             rewriter.getUnitAttr());
                def->setAttr("c21.q_slot",
                             rewriter.getStringAttr("k_stage_mod_2"));
            }
        }
        rewriter.replaceOp(op, result);
        return mlir::success();
    }
    if (c17Plan) {
        auto role = genericOperandSourceRole(op);
        if (role != "H" && role != "K" &&
            !(useC18FullPhysicalRegion() && role == "V"))
            return rewriter.notifyMatchFailure(
                op, "full physical plan received an unsupported logical role");
        annotateC17PhysicalPlan(op, *c17Plan, role,
                                role == "H" ? "Q@H"
                                             : role == "K" ? "Q@K" : "score@V");
        if (c18Plan)
            annotateC18PhysicalRegion(
                op, *c18Plan, role,
                role == "H" ? "Q@H" : role == "K" ? "Q@K" : "score@V");
    }
    const bool resetInEnclosingLoop =
        useAccumulatorReusePlan() &&
        hasEnclosingAccumulatorReset(op, op.getAccLow());
    mlir::Value accumulator;
    if (useAccumulatorReusePlan() && forwardedAccumulators &&
        !resetInEnclosingLoop) {
        auto it = forwardedAccumulators->find(op.getAccLow());
        if (it != forwardedAccumulators->end())
            accumulator = it->second;
    }
    if (!accumulator)
        accumulator = materializeAccumulatorVector(rewriter, loc, op.getAccLow());
    if (!accumulator)
        return rewriter.notifyMatchFailure(
            op, "full-scope block-dot requires local/vector FP32 accumulator");

    std::optional<LogicalBlockLayoutPlan> layoutPlan;
    if (c17Plan) {
        layoutPlan.emplace(makeC17StaticAffineLayoutPlan(
            op, rewriter, loc, genericOperandSourceRole(op) == "H"));
    } else if (useLogicalBlockLayoutPlan()) {
        const bool isH = genericOperandSourceRole(op) == "H";
        layoutPlan.emplace(
            makeLogicalBlockLayoutPlan(op, rewriter, loc, isH));
    }
    const auto *plannedLayout = layoutPlan ? &*layoutPlan : nullptr;
    const bool firstClassOperand = useFirstClassMfmaOperandPlan();
    if (firstClassOperand && kind != LoweringKind::Specialized)
        return rewriter.notifyMatchFailure(
            op, "first-class MFMA operand preservation requires specialized lowering");
    if (c17Plan && !firstClassOperand)
        return rewriter.notifyMatchFailure(
            op, "C17 full physical plan requires first-class MFMA operands");
    if (mlir::failed(emitFullScopeProducer(
        op, rewriter, kind == LoweringKind::Specialized, plannedLayout,
        c18Plan)))
        return mlir::failure();

    // The full-scope op owns the producer-to-consumer handoff.  The source
    // calls the logical K form from the whole CTA; only the original
    // value-half-zero waves consume the result.  That lets this barrier remain CTA-uniform while
    // preserving the original MFMA ownership.
    if (!(c18Plan && genericOperandSourceRole(op) == "V"))
        mlir::gpu::BarrierOp::create(rewriter, loc);
    mlir::Value updated;
    if (genericOperandSourceRole(op) == "K") {
        mlir::Value valueHalf;
        if (plannedLayout) {
            valueHalf = plannedLayout->valueHalf;
        } else {
            auto tid = toIndex(rewriter, loc, op.getThreadId());
            auto wave = mlir::arith::DivUIOp::create(
                rewriter, loc, tid, indexConstant(rewriter, loc, 64));
            valueHalf = mlir::arith::RemUIOp::create(
                rewriter, loc, wave, indexConstant(rewriter, loc, 2));
        }
        auto active = mlir::arith::CmpIOp::create(
            rewriter, loc, mlir::arith::CmpIPredicate::eq, valueHalf,
            indexConstant(rewriter, loc, 0));
        auto consumer = mlir::scf::IfOp::create(
            rewriter, loc, accumulator.getType(), active,
            /*withElseRegion=*/true);
        rewriter.setInsertionPointToStart(&consumer.getThenRegion().front());
        auto activeValue = firstClassOperand
                               ? emitFirstClassMfmaOperand(
                                     op, rewriter, loc, accumulator,
                                     plannedLayout, c17Plan)
                               : emitGenericOperandBPair(
                                     rewriter, loc, op, accumulator,
                                     kind == LoweringKind::Specialized,
                                     plannedLayout);
        mlir::scf::YieldOp::create(rewriter, loc, activeValue);
        rewriter.setInsertionPointToStart(&consumer.getElseRegion().front());
        mlir::scf::YieldOp::create(rewriter, loc, accumulator);
        rewriter.setInsertionPointAfter(consumer);
        updated = consumer.getResult(0);
    } else {
        updated = firstClassOperand
                      ? emitFirstClassMfmaOperand(op, rewriter, loc,
                                                  accumulator, plannedLayout,
                                                  c17Plan)
                      : emitGenericOperandBPair(
                            rewriter, loc, op, accumulator,
                            kind == LoweringKind::Specialized, plannedLayout);
    }
    if (!(c18Plan && genericOperandSourceRole(op) == "V"))
        mlir::gpu::BarrierOp::create(rewriter, loc);
    // The public block-dot contract is vector<32xf32>, but full-scope
    // consumers in this pipeline only ever read the low MFMA accumulator
    // tile.  P4 keeps that tile as vector SSA through the compiler-internal
    // lowering boundary instead of duplicating it into a dead high half and
    // immediately extracting the low half again.  This is a representation
    // optimization only: the source op, MFMA calls, ownership and arithmetic
    // remain unchanged, and other block-dot modes retain the public shape.
    mlir::Value result;
    if (useAccumulatorReusePlan()) {
        result = updated;
        result.getDefiningOp()->setAttr(
            "avelang.block_dot.p4_low_tile_ssa", rewriter.getUnitAttr());
    } else {
        result = joinColumns(rewriter, loc, updated, updated);
    }
    if (auto *def = result.getDefiningOp()) {
        def->setAttr("avelang.block_dot.operand_mode",
                     rewriter.getStringAttr("full_scope_lowered"));
        def->setAttr("avelang.block_dot.scope",
                     rewriter.getStringAttr("logical_block"));
        def->setAttr("avelang.block_dot.source_role",
                     rewriter.getStringAttr(genericOperandSourceRole(op)));
        def->setAttr("avelang.block_dot.rhs_producer",
                     rewriter.getStringAttr(kind == LoweringKind::Specialized
                                                ? "global_bf16x8"
                                                : "generic_scalar"));
        if (firstClassOperand)
            def->setAttr(
                "avelang.block_dot.operand_preservation",
                rewriter.getStringAttr(usePackedOperandReusePlan()
                                           ? "p3_packed_reuse"
                                           : "p2_first_class"));
        if (useAccumulatorReusePlan())
            def->setAttr("avelang.block_dot.accumulator_plan",
                         rewriter.getStringAttr(
                             resetInEnclosingLoop
                                 ? "p4_ssa_forwarding_reset_boundary"
                                 : "p4_ssa_forwarding"));
        if (c17Plan) {
            auto role = genericOperandSourceRole(op);
            annotateC17PhysicalPlan(def, *c17Plan, role,
                                    role == "H" ? "Q@H"
                                                 : role == "K" ? "Q@K" : "score@V");
            def->setAttr("c17.phase_boundary",
                         rewriter.getStringAttr("source_release_before_score"));
            if (c18Plan)
                annotateC18PhysicalRegion(
                    def, *c18Plan, role,
                    role == "H" ? "Q@H"
                                 : role == "K" ? "Q@K" : "score@V");
        }
    }
    if (useAccumulatorReusePlan())
        op.getResult().setType(mlir::cast<mlir::VectorType>(result.getType()));
    // The source commits the returned low accumulator to the same memref
    // before the next K-stage block dot.  Preserve that semantic commit, but
    // let the next lowering consume this SSA value directly.  Reset-bearing
    // loops are deliberately excluded above, so a new source_half starts from
    // its initialized accumulator rather than stale cached state.
    if (useAccumulatorReusePlan() && forwardedAccumulators &&
        !resetInEnclosingLoop)
        (*forwardedAccumulators)[op.getAccLow()] = updated;
    rewriter.replaceOp(op, result);
    return mlir::success();
}

mlir::Value scaleAndAccumulate(mlir::PatternRewriter &rewriter,
                               mlir::Location loc, mlir::Value persistent,
                               mlir::Value delta, mlir::Value gLast) {
    auto gLastExp = mlir::math::ExpOp::create(rewriter, loc, gLast);
    auto scale = mlir::vector::SplatOp::create(rewriter, loc,
                                               persistent.getType(), gLastExp);
    auto carried =
        mlir::arith::MulFOp::create(rewriter, loc, persistent, scale);
    return mlir::arith::AddFOp::create(rewriter, loc, carried, delta);
}

// The State-KV pred M/N-swap probe needs B[K,V] at fixed V but consecutive
// K.  Do not route this through an AveLang local memref: that route loses the
// vector value type at LLVM conversion.  The first-class op carries the
// ownership semantics until this late point, where it becomes vector SSA made
// directly from the four K-strided LDS elements.
class PredStateKVFragLoadLowering
    : public mlir::OpRewritePattern<AMDGPUQwenPredStateKVFragLoadOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(AMDGPUQwenPredStateKVFragLoadOp op,
                    mlir::PatternRewriter &rewriter) const override {
        auto stateType = mlir::dyn_cast<mlir::MemRefType>(
            op.getStateKV().getType());
        if (!stateType || stateType.getShape() !=
                              llvm::ArrayRef<int64_t>({2, 64, 32}) ||
            !stateType.getElementType().isBF16()) {
            return rewriter.notifyMatchFailure(
                op, "requires workgroup BF16 state_kv[2,64,32]");
        }
        auto loc = op.getLoc();
        auto bf16x4 = mlir::VectorType::get({4}, rewriter.getBF16Type());
        auto wave = toIndex(rewriter, loc, op.getWave());
        auto vLane = toIndex(rewriter, loc, op.getVLane());
        auto kVector = toIndex(rewriter, loc, op.getKVector());
        auto fragment = toIndex(rewriter, loc, op.getFragmentOffset());
        auto kBase = indexMul(rewriter, loc, kVector,
                              indexConstant(rewriter, loc, 8));
        kBase = indexAdd(rewriter, loc, kBase,
                         indexMul(rewriter, loc, fragment,
                                  indexConstant(rewriter, loc, 4)));
        llvm::SmallVector<mlir::Value> values;
        values.reserve(4);
        for (int64_t element = 0; element < 4; ++element) {
            auto k = indexAdd(rewriter, loc, kBase,
                              indexConstant(rewriter, loc, element));
            values.push_back(mlir::memref::LoadOp::create(
                rewriter, loc, op.getStateKV(),
                mlir::ValueRange{wave, k, vLane}));
        }
        auto fragmentValue = mlir::vector::FromElementsOp::create(
            rewriter, loc, bf16x4, values);
        fragmentValue->setAttr(
            "avelang.qwen.pred_state_kv_mn_swap.direct_b_fragment",
            rewriter.getUnitAttr());
        rewriter.replaceOp(op, fragmentValue.getResult());
        return mlir::success();
    }
};

// The first-class plan is consumed only after GPU outlining.  It deliberately
// bypasses the ordinary memref/vector load builder used by P1: the static
// strided layout is converted to one explicit addrspace(3) packed load, and
// only then is the existing MFMA32 intrinsic called.  This keeps the
// representation target-aware without introducing a Qwen-specific intrinsic.
class BlockDotMfmaOperandLowering
    : public mlir::OpRewritePattern<AMDGPUBlockDotMfmaOperandOp> {
  public:
    using mlir::OpRewritePattern<
        AMDGPUBlockDotMfmaOperandOp>::OpRewritePattern;

    mlir::Value emitPackedLdsLoad(mlir::PatternRewriter &rewriter,
                                  mlir::Location loc, mlir::Value source,
                                  mlir::Value row, mlir::Value col) const {
        auto memrefType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
        if (!memrefType || memrefType.getRank() != 2 ||
            !memrefType.getElementType().isBF16())
            return {};

        int64_t offset = 0;
        llvm::SmallVector<int64_t> strides;
        if (mlir::failed(
                memrefType.getStridesAndOffset(strides, offset)) ||
            strides.size() != 2 || mlir::ShapedType::isDynamic(offset) ||
            mlir::ShapedType::isDynamic(strides[0]) ||
            mlir::ShapedType::isDynamic(strides[1]))
            return {};

        mlir::Value indexOffset = mlir::arith::MulIOp::create(
            rewriter, loc, row,
            mlir::arith::ConstantIndexOp::create(rewriter, loc, strides[0]));
        indexOffset = mlir::arith::AddIOp::create(
            rewriter, loc, indexOffset,
            mlir::arith::MulIOp::create(
                rewriter, loc, col,
                mlir::arith::ConstantIndexOp::create(rewriter, loc,
                                                      strides[1])));
        if (offset != 0)
            indexOffset = mlir::arith::AddIOp::create(
                rewriter, loc, indexOffset,
                mlir::arith::ConstantIndexOp::create(rewriter, loc, offset));

        auto baseIndex = mlir::memref::ExtractAlignedPointerAsIndexOp::create(
            rewriter, loc, rewriter.getIndexType(), source);
        auto baseI64 = mlir::arith::IndexCastOp::create(
            rewriter, loc, rewriter.getI64Type(), baseIndex);
        auto ldsPtrType = mlir::LLVM::LLVMPointerType::get(
            rewriter.getContext(), /*addressSpace=*/3);
        auto ldsBase = mlir::LLVM::IntToPtrOp::create(
            rewriter, loc, ldsPtrType, baseI64,
            mlir::LLVM::DereferenceableAttr{});
        auto offsetI64 = mlir::arith::IndexCastOp::create(
            rewriter, loc, rewriter.getI64Type(), indexOffset);
        auto ldsPtr = mlir::LLVM::GEPOp::create(
            rewriter, loc, ldsPtrType, rewriter.getBF16Type(), ldsBase,
            mlir::ValueRange{offsetI64});
        auto bf16x4 = mlir::VectorType::get({4}, rewriter.getBF16Type());
        // Keep the packed 64-bit read as an integer through LLVM lowering.
        // The explicit bitcast is the one controlled distinction from P1's
        // ordinary vector<4xbf16> load; it prevents the target pipeline from
        // treating the operand as an unconstrained vector slice too early.
        auto load = mlir::LLVM::LoadOp::create(
            rewriter, loc, rewriter.getI64Type(), ldsPtr, /*alignment=*/8,
            /*volatile=*/true, /*nontemporal=*/false);
        load->setAttr("avelang.block_dot.first_class_lds_b64_i64",
                      rewriter.getUnitAttr());
        auto packed = mlir::LLVM::BitcastOp::create(
            rewriter, loc, bf16x4, load.getResult());
        return packed.getResult();
    }

    // C14 path: use the typed shared recipe rather than the old row-major
    // memref/GEP planner.  The only dynamic arithmetic is the fixed target
    // affine recipe (shift/mask/xor/add); non-power-of-two recipes fail
    // explicitly instead of falling back to generic layout lowering.
    mlir::Value emitC14PackedLdsLoad(
        mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value source,
        mlir::Value row, mlir::Value col,
        const C14StaticPhysicalPlan &plan, bool alreadyTransformed = false) const {
        auto memrefType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
        if (!memrefType || memrefType.getRank() != 2 ||
            !memrefType.getElementType().isBF16())
            return {};

        auto transformedRow = row;
        auto transformedCol = col;
        if (!alreadyTransformed &&
            plan.transform.permutation == std::array<int64_t, 2>{1, 0}) {
            transformedRow = col;
            transformedCol = row;
        }
        auto elementOffset = emitC14SharedElementOffset(
            rewriter, loc, plan, transformedRow, transformedCol);
        if (mlir::failed(elementOffset))
            return {};

        auto baseIndex = mlir::memref::ExtractAlignedPointerAsIndexOp::create(
            rewriter, loc, rewriter.getIndexType(), source);
        auto baseI64 = mlir::arith::IndexCastOp::create(
            rewriter, loc, rewriter.getI64Type(), baseIndex);
        auto ldsPtrType = mlir::LLVM::LLVMPointerType::get(
            rewriter.getContext(), /*addressSpace=*/3);
        auto ldsBase = mlir::LLVM::IntToPtrOp::create(
            rewriter, loc, ldsPtrType, baseI64,
            mlir::LLVM::DereferenceableAttr{});
        auto ldsPtr = mlir::LLVM::GEPOp::create(
            rewriter, loc, ldsPtrType, rewriter.getBF16Type(), ldsBase,
            mlir::ValueRange{*elementOffset});
        auto bf16x4 = mlir::VectorType::get({4}, rewriter.getBF16Type());
        auto load = mlir::LLVM::LoadOp::create(
            rewriter, loc, bf16x4, ldsPtr, /*alignment=*/8,
            /*volatile=*/false, /*nontemporal=*/false);
        load->setAttr("c14.static_encoding_consumed",
                      rewriter.getUnitAttr());
        load->setAttr("c14.shared_kind",
                      rewriter.getStringAttr(plan.shared.kind));
        load->setAttr("c14.transform_kind",
                      rewriter.getStringAttr(plan.transform.kind));
        load->setAttr("c14.dot_op_idx",
                      rewriter.getI64IntegerAttr(plan.dot.opIdx));
        load->setAttr("c14.mfma_shape",
                      mlir::DenseI64ArrayAttr::get(
                          rewriter.getContext(),
                          llvm::ArrayRef<int64_t>(plan.mfma.instrShape)));
        if (plan.transform.kind == "in_thread_transpose") {
            // The basis is consumed while validating this path and is also
            // retained on the load for machine-evidence attribution.  The
            // actual data movement is the fixed coordinate transform above;
            // no generic transpose or ds_bpermute operation is introduced.
            load->setAttr("c14.in_thread_basis_register_count",
                          rewriter.getI64IntegerAttr(
                              plan.transform.registerBasis.size() / 2));
            load->setAttr("c14.in_thread_basis_lane_count",
                          rewriter.getI64IntegerAttr(
                              plan.transform.laneBasis.size() / 2));
            load->setAttr("c14.in_thread_basis_warp_count",
                          rewriter.getI64IntegerAttr(
                              plan.transform.warpBasis.size() / 2));
    }
        return load.getResult();
    }

    // Native selected-T2048 dot fragments are addressed by a byte offset
    // emitted by the fixed gfx942 lane algebra.  Keep this as a typed BF16x4
    // LDS load so the Q/H/K consumer remains a real packed operand, while
    // avoiding the older logical row/column interpretation at this boundary.
    mlir::Value emitC16NativePackedLdsLoadAtByte(
        mlir::PatternRewriter &rewriter, mlir::Location loc,
        mlir::Value source, mlir::Value byteOffsetI64,
        llvm::StringRef role) const {
        auto memrefType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
        if (!memrefType || memrefType.getRank() != 2 ||
            !memrefType.getElementType().isBF16())
            return {};

        auto baseIndex = mlir::memref::ExtractAlignedPointerAsIndexOp::create(
            rewriter, loc, rewriter.getIndexType(), source);
        auto baseI64 = mlir::arith::IndexCastOp::create(
            rewriter, loc, rewriter.getI64Type(), baseIndex);
        auto ldsPtrType = mlir::LLVM::LLVMPointerType::get(
            rewriter.getContext(), /*addressSpace=*/3);
        auto ldsBase = mlir::LLVM::IntToPtrOp::create(
            rewriter, loc, ldsPtrType, baseI64,
            mlir::LLVM::DereferenceableAttr{});
        auto elementOffset = mlir::arith::ShRUIOp::create(
            rewriter, loc, byteOffsetI64,
            c14I64Constant(rewriter, loc, 1));
        auto ldsPtr = mlir::LLVM::GEPOp::create(
            rewriter, loc, ldsPtrType, rewriter.getBF16Type(), ldsBase,
            mlir::ValueRange{elementOffset});
        auto bf16x4 = mlir::VectorType::get({4}, rewriter.getBF16Type());
        auto load = mlir::LLVM::LoadOp::create(
            rewriter, loc, bf16x4, ldsPtr, /*alignment=*/8,
            /*volatile=*/false, /*nontemporal=*/false);
        load->setAttr("c16.native_dot_lds_load", rewriter.getUnitAttr());
        load->setAttr("c16.native_dot_role",
                      rewriter.getStringAttr(role));
        load->setAttr("c16.native_dot_load_width",
                      rewriter.getI64IntegerAttr(4));
        return load.getResult();
    }

    // The C16 numerical probe uses ordinary row-major identity tiles for the
    // operand opposite the real Q/H/K tile.  Recover the logical coordinates
    // represented by the native physical Q/K byte address, then read that
    // identity tile row-major.  This is the inverse of the selected shared
    // encoding, not a second physical-layout planner.
    mlir::Value emitC16IdentityOperandFromNativeByte(
        mlir::PatternRewriter &rewriter, mlir::Location loc,
        mlir::Value source, mlir::Value byteOffsetI64,
        bool qOperand) const {
        auto physicalOuter = mlir::arith::ShRUIOp::create(
            rewriter, loc, byteOffsetI64,
            c14I64Constant(rewriter, loc, 6));
        auto physicalInner = mlir::arith::AndIOp::create(
            rewriter, loc,
            mlir::arith::ShRUIOp::create(
                rewriter, loc, byteOffsetI64,
                c14I64Constant(rewriter, loc, 1)),
            c14I64Constant(rewriter, loc, 31));
        auto phase = mlir::arith::AndIOp::create(
            rewriter, loc,
            mlir::arith::ShRUIOp::create(
                rewriter, loc, physicalOuter,
                c14I64Constant(rewriter, loc, 1)),
            c14I64Constant(rewriter, loc, 7));
        auto logicalGroup = mlir::arith::XOrIOp::create(
            rewriter, loc,
            mlir::arith::ShRUIOp::create(
                rewriter, loc, physicalInner,
                c14I64Constant(rewriter, loc, 2)),
            phase);
        auto intra = mlir::arith::AndIOp::create(
            rewriter, loc, physicalInner,
            c14I64Constant(rewriter, loc, 3));
        auto grouped = mlir::arith::OrIOp::create(
            rewriter, loc,
            mlir::arith::ShLIOp::create(
                rewriter, loc, logicalGroup,
                c14I64Constant(rewriter, loc, 2)),
            intra);
        mlir::Value logicalRow =
            qOperand ? physicalOuter.getResult() : grouped.getResult();
        mlir::Value logicalCol =
            qOperand ? grouped.getResult() : physicalOuter.getResult();
        if (!qOperand) {
            auto sourceType =
                mlir::dyn_cast<mlir::MemRefType>(source.getType());
            if (!sourceType || sourceType.getRank() != 2 ||
                !sourceType.getElementType().isBF16())
                return {};
            auto row = mlir::arith::IndexCastOp::create(
                rewriter, loc, rewriter.getIndexType(), logicalRow);
            auto col = mlir::arith::IndexCastOp::create(
                rewriter, loc, rewriter.getIndexType(), logicalCol);
            llvm::SmallVector<mlir::Value> values;
            values.reserve(4);
            for (int64_t element = 0; element < 4; ++element) {
                auto elementRow = indexAdd(
                    rewriter, loc, row,
                    indexConstant(rewriter, loc, element));
                values.push_back(mlir::memref::LoadOp::create(
                    rewriter, loc, source,
                    mlir::ValueRange{elementRow, col}));
            }
            return mlir::vector::FromElementsOp::create(
                rewriter, loc,
                mlir::VectorType::get({4}, rewriter.getBF16Type()), values);
        }
        return emitPackedLdsLoad(
            rewriter, loc, source,
            mlir::arith::IndexCastOp::create(
                rewriter, loc, rewriter.getIndexType(), logicalRow),
            mlir::arith::IndexCastOp::create(
                rewriter, loc, rewriter.getIndexType(), logicalCol));
    }

    // The selected native #shared2 K producer is a direct transposed
    // packetization: a logical K[row, col] element occupies element offset
    // col * 32 + row.  Q@K's structured identity B operand must therefore
    // use this inverse, not the Q/#shared phase inverse above.  This keeps
    // the identity control on the same typed shared encoding as the real K
    // consumer and avoids hiding a second layout planner in the probe.
    mlir::Value emitC16KIdentityOperandFromNativeByte(
        mlir::PatternRewriter &rewriter, mlir::Location loc,
        mlir::Value source, mlir::Value byteOffsetI64) const {
        auto physicalCol = mlir::arith::ShRUIOp::create(
            rewriter, loc, byteOffsetI64,
            c14I64Constant(rewriter, loc, 6));
        auto physicalRow = mlir::arith::AndIOp::create(
            rewriter, loc,
            mlir::arith::ShRUIOp::create(
                rewriter, loc, byteOffsetI64,
                c14I64Constant(rewriter, loc, 1)),
            c14I64Constant(rewriter, loc, 31));
        auto row = mlir::arith::IndexCastOp::create(
            rewriter, loc, rewriter.getIndexType(), physicalRow);
        auto col = mlir::arith::IndexCastOp::create(
            rewriter, loc, rewriter.getIndexType(), physicalCol);
        auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
        if (!sourceType || sourceType.getRank() != 2 ||
            !sourceType.getElementType().isBF16())
            return {};
        llvm::SmallVector<mlir::Value> values;
        values.reserve(4);
        for (int64_t element = 0; element < 4; ++element) {
            auto elementRow = indexAdd(
                rewriter, loc, row, indexConstant(rewriter, loc, element));
            values.push_back(mlir::memref::LoadOp::create(
                rewriter, loc, source,
                mlir::ValueRange{elementRow, col}));
        }
        return mlir::vector::FromElementsOp::create(
            rewriter, loc,
            mlir::VectorType::get({4}, rewriter.getBF16Type()), values);
    }

    // P3 materializes the two adjacent 64-bit fragments as one 128-bit LDS
    // read.  The pair stays a single SSA value until the two explicit
    // vector-slice users are formed below; this is the compiler-internal
    // consumer-group boundary, not a new source operation.
    mlir::Value emitPackedLdsLoad128(mlir::PatternRewriter &rewriter,
                                     mlir::Location loc, mlir::Value source,
                                     mlir::Value row, mlir::Value col) const {
        auto memrefType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
        if (!memrefType || memrefType.getRank() != 2 ||
            !memrefType.getElementType().isBF16())
            return {};

        int64_t offset = 0;
        llvm::SmallVector<int64_t> strides;
        if (mlir::failed(
                memrefType.getStridesAndOffset(strides, offset)) ||
            strides.size() != 2 || mlir::ShapedType::isDynamic(offset) ||
            mlir::ShapedType::isDynamic(strides[0]) ||
            mlir::ShapedType::isDynamic(strides[1]))
            return {};

        mlir::Value indexOffset = mlir::arith::MulIOp::create(
            rewriter, loc, row,
            mlir::arith::ConstantIndexOp::create(rewriter, loc, strides[0]));
        indexOffset = mlir::arith::AddIOp::create(
            rewriter, loc, indexOffset,
            mlir::arith::MulIOp::create(
                rewriter, loc, col,
                mlir::arith::ConstantIndexOp::create(rewriter, loc,
                                                      strides[1])));
        if (offset != 0)
            indexOffset = mlir::arith::AddIOp::create(
                rewriter, loc, indexOffset,
                mlir::arith::ConstantIndexOp::create(rewriter, loc, offset));

        auto baseIndex = mlir::memref::ExtractAlignedPointerAsIndexOp::create(
            rewriter, loc, rewriter.getIndexType(), source);
        auto baseI64 = mlir::arith::IndexCastOp::create(
            rewriter, loc, rewriter.getI64Type(), baseIndex);
        auto ldsPtrType = mlir::LLVM::LLVMPointerType::get(
            rewriter.getContext(), /*addressSpace=*/3);
        auto ldsBase = mlir::LLVM::IntToPtrOp::create(
            rewriter, loc, ldsPtrType, baseI64,
            mlir::LLVM::DereferenceableAttr{});
        auto offsetI64 = mlir::arith::IndexCastOp::create(
            rewriter, loc, rewriter.getI64Type(), indexOffset);
        auto ldsPtr = mlir::LLVM::GEPOp::create(
            rewriter, loc, ldsPtrType, rewriter.getBF16Type(), ldsBase,
            mlir::ValueRange{offsetI64});
        auto bf16x8 = mlir::VectorType::get({8}, rewriter.getBF16Type());
        auto load = mlir::LLVM::LoadOp::create(
            rewriter, loc, bf16x8, ldsPtr, /*alignment=*/16,
            /*volatile=*/false, /*nontemporal=*/false);
        load->setAttr("avelang.block_dot.first_class_lds_b128_group",
                      rewriter.getUnitAttr());
        return load.getResult();
    }

    std::optional<std::pair<mlir::Value, mlir::Value>>
    splitPackedFragmentPair(mlir::PatternRewriter &rewriter,
                            mlir::Location loc, mlir::Value packed) const {
        llvm::SmallVector<int64_t> offsetsLow{0};
        llvm::SmallVector<int64_t> offsetsHigh{4};
        llvm::SmallVector<int64_t> sizes{4};
        llvm::SmallVector<int64_t> strides{1};
        auto low = mlir::vector::ExtractStridedSliceOp::create(
            rewriter, loc, packed, offsetsLow, sizes, strides);
        auto high = mlir::vector::ExtractStridedSliceOp::create(
            rewriter, loc, packed, offsetsHigh, sizes, strides);
        return std::make_pair(low.getResult(), high.getResult());
    }

    mlir::LogicalResult emitPackedConsumerGroup(
        AMDGPUBlockDotMfmaOperandOp op, mlir::PatternRewriter &rewriter,
        mlir::Location loc, mlir::Value &accumulator, mlir::Value word,
        const llvm::StringRef role) const {
        auto baseCol = indexMul(rewriter, loc, word,
                                indexConstant(rewriter, loc, 8));
        auto aPacked = emitPackedLdsLoad128(
            rewriter, loc, op.getAStage(), op.getARow(), baseCol);
        auto bPacked = emitPackedLdsLoad128(
            rewriter, loc, op.getBStage(), op.getBRow(), baseCol);
        if (!aPacked || !bPacked)
            return rewriter.notifyMatchFailure(
                op, "packed operand reuse requires static BF16 rank-2 LDS stages");
        auto aFragments = splitPackedFragmentPair(rewriter, loc, aPacked);
        auto bFragments = splitPackedFragmentPair(rewriter, loc, bPacked);
        if (!aFragments || !bFragments)
            return rewriter.notifyMatchFailure(
                op, "packed operand reuse failed to split a b128 fragment pair");

        auto mfmaName = ir::intrinsics::MakeIntrinsicFuncName(
            "amdgpu", "rocdl_mfma_f32_32x32x8bf16_1k");
        // Both MFMA calls consume slices of the same packed load.  Do not
        // move this into emitPackedLdsLoad: keeping the pair users adjacent
        // is what lets the late pipeline preserve the reuse relation.
        for (unsigned index : {0u, 1u}) {
            const mlir::Value bFragment =
                index == 0 ? bFragments->first : bFragments->second;
            const mlir::Value aFragment =
                index == 0 ? aFragments->first : aFragments->second;
            auto call = mlir::func::CallOp::create(
                rewriter, loc, mfmaName,
                mlir::TypeRange{accumulator.getType()},
                mlir::ValueRange{bFragment, aFragment, accumulator});
            call->setAttr("avelang.block_dot.mfma32", rewriter.getUnitAttr());
            call->setAttr("avelang.block_dot.first_class_consumer",
                          rewriter.getUnitAttr());
            call->setAttr("avelang.block_dot.consumer_group",
                          rewriter.getStringAttr("packed_b128_pair"));
            call->setAttr("avelang.block_dot.consumer_group_role",
                          rewriter.getStringAttr(role));
            if (auto operandRole = op->getAttrOfType<mlir::StringAttr>(
                    "avelang.block_dot.operand_role"))
                call->setAttr("avelang.block_dot.operand_role", operandRole);
            if (auto source = op->getAttrOfType<mlir::StringAttr>(
                    "avelang.block_dot.source_role"))
                call->setAttr("avelang.block_dot.source_role", source);
            accumulator = call.getResult(0);
        }
        return mlir::success();
    }

    mlir::LogicalResult
    matchAndRewrite(AMDGPUBlockDotMfmaOperandOp op,
                    mlir::PatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        mlir::Value accumulator = op.getAccumulator();
        // Internal clients may carry this affine coordinate as either an
        // index or a signless i64.  Normalize it before the C16/C14 affine
        // helpers so every add/mul remains type-homogeneous at MLIR level.
        auto word0 = c15Index(rewriter, loc, op.getOperandWord());
        auto word1 = indexAdd(
            rewriter, loc, word0, indexConstant(rewriter, loc, 2));
        auto c8 = indexConstant(rewriter, loc, 8);
        auto mfmaName = ir::intrinsics::MakeIntrinsicFuncName(
            "amdgpu", "rocdl_mfma_f32_32x32x8bf16_1k");

        if (op->hasAttr("c22.z5b_schedule_preserving")) {
            auto roleAttr = op->getAttrOfType<mlir::StringAttr>(
                "c22.physical_role");
            auto role = roleAttr ? roleAttr.getValue() : llvm::StringRef();
            if (role != "H" && role != "K" && role != "V")
                return rewriter.notifyMatchFailure(
                    op, "C22 consumer requires H, K, or V physical role");
            auto qPlan = getC22StaticPhysicalPlan("Q");
            auto bPlan = getC22StaticPhysicalPlan(role);
            if (!qPlan || !bPlan)
                return rewriter.notifyMatchFailure(
                    op, "C22 consumer is missing a verified physical plan");

            auto byteFromElement = [&](mlir::Value element) {
                return mlir::arith::ShLIOp::create(
                    rewriter, loc, element, c14I64Constant(rewriter, loc, 1))
                    .getResult();
            };
            auto addElementBase = [&](mlir::Value element, int64_t base) {
                return mlir::arith::AddIOp::create(
                    rewriter, loc, c14I64Constant(rewriter, loc, base), element)
                    .getResult();
            };
            auto emitC22Call = [&](mlir::Value aFrag, mlir::Value bFrag)
                -> mlir::LogicalResult {
                if (!aFrag || !bFrag)
                    return rewriter.notifyMatchFailure(
                        op, "C22 cannot form a packed static LDS operand");
                auto call = mlir::func::CallOp::create(
                    rewriter, loc, mfmaName,
                    mlir::TypeRange{accumulator.getType()},
                    mlir::ValueRange{bFrag, aFrag, accumulator});
                call->setAttr("c22.z5b_schedule_preserving_physical_consumer",
                              rewriter.getUnitAttr());
                call->setAttr("c22.physical_role", rewriter.getStringAttr(role));
                call->setAttr("c14.static_physical_consumed",
                              rewriter.getUnitAttr());
                call->setAttr("avelang.block_dot.mfma32", rewriter.getUnitAttr());
                accumulator = call.getResult(0);
                return mlir::success();
            };

            for (mlir::Value word : {word0, word1}) {
                auto logicalCol = indexMul(rewriter, loc, word, c8);
                for (int64_t fragmentOffset : {int64_t{0}, int64_t{4}}) {
                    auto col = indexAdd(rewriter, loc, logicalCol,
                                        indexConstant(rewriter, loc, fragmentOffset));
                    mlir::Value aFrag;
                    mlir::Value bFrag;
                    if (role == "V") {
                        // The score tile remains Z5B row-major.  Only the V
                        // operand crosses C15's rotating shared bridge.
                        aFrag = emitPackedLdsLoad(rewriter, loc, op.getAStage(),
                                                   op.getARow(), col);
                        auto bRowI64 = c14I64(rewriter, loc, op.getBRow());
                        auto relative = mlir::arith::SubIOp::create(
                            rewriter, loc, bRowI64,
                            c14I64Constant(rewriter, loc, 384));
                        auto value = mlir::arith::ShRUIOp::create(
                            rewriter, loc, relative,
                            c14I64Constant(rewriter, loc, 1));
                        auto sourceHalf = mlir::arith::AndIOp::create(
                            rewriter, loc, relative,
                            c14I64Constant(rewriter, loc, 1));
                        auto tokenBase = mlir::arith::ShLIOp::create(
                            rewriter, loc, sourceHalf,
                            c14I64Constant(rewriter, loc, 5));
                        auto token = mlir::arith::AddIOp::create(
                            rewriter, loc, tokenBase.getResult(),
                            c14I64(rewriter, loc, col));
                        auto vElement = emitC14SharedElementOffset(
                            rewriter, loc, *bPlan, token, value);
                        if (mlir::failed(vElement))
                            return rewriter.notifyMatchFailure(
                                op, "C22 V shared encoding rejected the operand");
                        bFrag = emitC16NativePackedLdsLoadAtByte(
                            rewriter, loc, op.getBStage(),
                            byteFromElement(addElementBase(*vElement, 384 * 32)),
                            role);
                    } else {
                        // Q is a single Z5B cache producer.  Decode only the
                        // current K32 stage for its C16 physical consumer;
                        // no producer, phase, or lifetime is moved.
                        auto aRowI64 = c14I64(rewriter, loc, op.getARow());
                        auto qStage = mlir::arith::ShRUIOp::create(
                            rewriter, loc, aRowI64,
                            c14I64Constant(rewriter, loc, 6));
                        auto qRow = mlir::arith::AndIOp::create(
                            rewriter, loc, aRowI64,
                            c14I64Constant(rewriter, loc, 63));
                        auto qElement = emitC16NativeSharedElementOffset(
                            rewriter, loc, *qPlan, qRow, col);
                        if (mlir::failed(qElement))
                            return rewriter.notifyMatchFailure(
                                op, "C22 Q shared encoding rejected the operand");
                        auto qBase = mlir::arith::ShLIOp::create(
                            rewriter, loc, qStage,
                            c14I64Constant(rewriter, loc, 11));
                        aFrag = emitC16NativePackedLdsLoadAtByte(
                            rewriter, loc, op.getAStage(),
                            byteFromElement(mlir::arith::AddIOp::create(
                                rewriter, loc, qBase.getResult(), *qElement)
                                                .getResult()), "Q");

                        auto bRowI64 = c14I64(rewriter, loc, op.getBRow());
                        if (role == "H") {
                            auto hRow = mlir::arith::SubIOp::create(
                                rewriter, loc, bRowI64,
                                c14I64Constant(rewriter, loc, 320));
                            auto hElement = emitC16NativeSharedElementOffset(
                                rewriter, loc, *bPlan, hRow, col);
                            if (mlir::failed(hElement))
                                return rewriter.notifyMatchFailure(
                                    op, "C22 H shared encoding rejected the operand");
                            bFrag = emitC16NativePackedLdsLoadAtByte(
                                rewriter, loc, op.getBStage(),
                                byteFromElement(addElementBase(*hElement, 320 * 32)),
                                role);
                        } else {
                            auto relative = mlir::arith::SubIOp::create(
                                rewriter, loc, bRowI64,
                                c14I64Constant(rewriter, loc, 320));
                            auto sourceHalf = mlir::arith::ShRUIOp::create(
                                rewriter, loc, relative,
                                c14I64Constant(rewriter, loc, 7));
                            auto kRow = mlir::arith::AndIOp::create(
                                rewriter, loc, relative,
                                c14I64Constant(rewriter, loc, 31));
                            // The score B operand is logical K[feature,
                            // token], while Z5B's source loop stores
                            // K[token, feature].  C16's shared2 recipe takes
                            // the logical order, yielding token*32+feature.
                            auto kElement = emitC16NativeKPacketElementOffset(
                                rewriter, loc, col, kRow);
                            if (mlir::failed(kElement))
                                return rewriter.notifyMatchFailure(
                                    op, "C22 K shared encoding rejected the operand");
                            auto kHalfBase = mlir::arith::ShLIOp::create(
                                rewriter, loc, sourceHalf,
                                c14I64Constant(rewriter, loc, 12));
                            auto kBase = mlir::arith::AddIOp::create(
                                rewriter, loc,
                                c14I64Constant(rewriter, loc, 320 * 32),
                                kHalfBase.getResult());
                            bFrag = emitC16NativePackedLdsLoadAtByte(
                                rewriter, loc, op.getBStage(),
                                byteFromElement(mlir::arith::AddIOp::create(
                                    rewriter, loc, kBase.getResult(), *kElement)
                                                    .getResult()), role);
                        }
                    }
                    if (mlir::failed(emitC22Call(aFrag, bFrag)))
                        return mlir::failure();
                }
            }
            rewriter.replaceOp(op, accumulator);
            return mlir::success();
        }

        if (op->hasAttr("c19.full_physical_region")) {
            auto roleAttr = op->getAttrOfType<mlir::StringAttr>(
                "c19.plan_role");
            auto role = roleAttr ? roleAttr.getValue() : llvm::StringRef();
            if (role != "H" && role != "K" && role != "V")
                return rewriter.notifyMatchFailure(
                    op, "C19 consumer requires H, K or V plan role");
            // V's score consumer uses the validated C18 single-word recipe:
            // operandWord plus the two packed fragments at offsets 0 and 4.
            // H/K retain the established two-word recipe because their
            // logical block consumes both word0 and word0+2.  Applying the
            // V rule to H/K would silently halve Q@H/Q@K.
            const unsigned wordCount = role == "V" ? 1u : 2u;
            for (unsigned wordIndex = 0; wordIndex < wordCount;
                 ++wordIndex) {
                mlir::Value word = wordIndex == 0 ? word0 : word1;
                auto col = indexMul(rewriter, loc, word, c8);
                for (int64_t fragmentOffset : {int64_t{0}, int64_t{4}}) {
                    auto fragmentCol = indexAdd(
                        rewriter, loc, col,
                        indexConstant(rewriter, loc, fragmentOffset));
                    auto aFrag = emitPackedLdsLoad(
                        rewriter, loc, op.getAStage(), op.getARow(),
                        fragmentCol);
                    auto bFrag = emitPackedLdsLoad(
                        rewriter, loc, op.getBStage(), op.getBRow(),
                        fragmentCol);
                    if (!aFrag || !bFrag)
                        return rewriter.notifyMatchFailure(
                            op, "C19 consumer cannot form packed LDS operands");
                    auto call = mlir::func::CallOp::create(
                        rewriter, loc, mfmaName,
                        mlir::TypeRange{accumulator.getType()},
                        mlir::ValueRange{bFrag, aFrag, accumulator});
                    call->setAttr("c19.full_physical_consumer",
                                  rewriter.getUnitAttr());
                    call->setAttr("c19.plan_role",
                                  rewriter.getStringAttr(role));
                    call->setAttr("c19.mfma32", rewriter.getUnitAttr());
                    accumulator = call.getResult(0);
                }
            }
            rewriter.replaceOp(op, accumulator);
            return mlir::success();
        }

        // C18 is the actual physical-region consumer.  Its row/word recipe
        // is already materialized by FullPhysicalRegionPlan, so lower the
        // two MFMA fragments from one packed LDS pair and bypass the old
        // generic operand reconstruction branches below.
        if (op->hasAttr("c18.full_physical_region") &&
            op->getAttrOfType<mlir::StringAttr>("c18.plan_role") &&
            op->getAttrOfType<mlir::StringAttr>("c18.plan_role").getValue() ==
                "V") {
            auto roleAttr = op->getAttrOfType<mlir::StringAttr>(
                "c18.plan_role");
            auto role = roleAttr ? roleAttr.getValue() : llvm::StringRef();
            if (role.empty())
                return rewriter.notifyMatchFailure(
                    op, "C18 operand is missing its physical plan role");
            // Keep the C18 physical row/word recipe, but use the same
            // vector<4xbf16> LDS load granularity as the proven phase_vec
            // consumer.  This isolates C18 ownership from a possible
            // vector<8> bitcast/order difference at the MFMA boundary.
            auto emitC18Fragment = [&](mlir::Value word,
                                       int64_t fragmentOffset)
                -> mlir::LogicalResult {
                auto col = indexAdd(
                    rewriter, loc,
                    indexMul(rewriter, loc, word,
                             indexConstant(rewriter, loc, 8)),
                    indexConstant(rewriter, loc, fragmentOffset));
                auto aFrag = emitPackedLdsLoad(
                    rewriter, loc, op.getAStage(), op.getARow(), col);
                auto bFrag = emitPackedLdsLoad(
                    rewriter, loc, op.getBStage(), op.getBRow(), col);
                if (!aFrag || !bFrag)
                    return rewriter.notifyMatchFailure(
                        op, "C18 physical consumer cannot form BF16x4 LDS load");
                auto call = mlir::func::CallOp::create(
                    rewriter, loc, mfmaName,
                    mlir::TypeRange{accumulator.getType()},
                    mlir::ValueRange{bFrag, aFrag, accumulator});
                call->setAttr("avelang.block_dot.mfma32",
                              rewriter.getUnitAttr());
                call->setAttr("avelang.block_dot.first_class_consumer",
                              rewriter.getUnitAttr());
                call->setAttr("avelang.block_dot.consumer_group_role",
                              rewriter.getStringAttr(role));
                accumulator = call.getResult(0);
                return mlir::success();
            };
            for (int64_t fragmentOffset : {int64_t{0}, int64_t{4}})
                if (mlir::failed(emitC18Fragment(word0, fragmentOffset)))
                    return mlir::failure();
            rewriter.replaceOp(op, accumulator);
            return mlir::success();
        }

        if (op->hasAttr("c16.real_tile_role")) {
            auto roleAttr = op->getAttrOfType<mlir::StringAttr>(
                "c16.real_tile_role");
            auto role = roleAttr ? roleAttr.getValue() : llvm::StringRef();
            auto plan = getC16StaticPhysicalPlan(role);
            if (!plan)
                return rewriter.notifyMatchFailure(
                    op, "C16 consumer is missing a valid Q/H/K physical plan");
            // These are the selected native WG256 lane equations recovered
            // from the T2048 LLVM/ISA.  They are a compact expression of the
            // C13 distributed/shared/dot plan, not a Qwen-specific address
            // table.  Q and K use the swizzled XOR stepping visible in the
            // native ds_read sequence; H uses the fixed-transpose row-major
            // stepping.
            // The internal consumer op intentionally carries the compact
            // planned coordinates rather than a second thread-id operand.
            // Reconstruct the original tid from the same finite encoding:
            // aRow supplies waveRow/laneLow, kStage supplies waveCol, and
            // operandWord supplies laneHigh.  This keeps producer and
            // consumer on one representation without adding a hidden runtime
            // planner input.
            mlir::Value tidI64;
            const bool wg128Full64 =
                op->hasAttr("c16.wg128_qh_full64") ||
                op->hasAttr("c16.wg128_qhk_full64");
            if (wg128Full64) {
                auto aRowI64 = c14I64(rewriter, loc, op.getARow());
                auto sourceWave = mlir::arith::ShRUIOp::create(
                    rewriter, loc, aRowI64, c14I64Constant(rewriter, loc, 5));
                auto laneLow = mlir::arith::AndIOp::create(
                    rewriter, loc, aRowI64,
                    c14I64Constant(rewriter, loc, 31));
                auto laneHigh = mlir::arith::ShLIOp::create(
                    rewriter, loc, c14I64(rewriter, loc, op.getOperandWord()),
                    c14I64Constant(rewriter, loc, 5));
                auto lane = mlir::arith::OrIOp::create(rewriter, loc, laneLow,
                                                       laneHigh);
                auto outputHalf = op->getAttrOfType<mlir::IntegerAttr>(
                    "c16.wg128_qh_output_half");
                if (!outputHalf || (outputHalf.getInt() != 0 &&
                                    outputHalf.getInt() != 1))
                    return rewriter.notifyMatchFailure(
                        op, "WG128 full64 Q/H/K operand is missing output-half 0/1");
                auto virtualWave = mlir::arith::OrIOp::create(
                    rewriter, loc,
                    mlir::arith::ShLIOp::create(
                        rewriter, loc, sourceWave,
                        c14I64Constant(rewriter, loc, 1)),
                    c14I64Constant(rewriter, loc, outputHalf.getInt()));
                tidI64 = mlir::arith::OrIOp::create(
                    rewriter, loc,
                    mlir::arith::ShLIOp::create(
                        rewriter, loc, virtualWave,
                        c14I64Constant(rewriter, loc, 6)),
                    lane);
            } else {
                auto aRowI64 = c14I64(rewriter, loc, op.getARow());
                auto laneLow = mlir::arith::AndIOp::create(
                    rewriter, loc, aRowI64,
                    c14I64Constant(rewriter, loc, 31));
                auto waveRow = mlir::arith::ShRUIOp::create(
                    rewriter, loc, aRowI64,
                    c14I64Constant(rewriter, loc, 5));
                auto waveCol = c14I64(rewriter, loc, op.getKStage());
                auto wave = mlir::arith::OrIOp::create(
                    rewriter, loc,
                    mlir::arith::ShLIOp::create(
                        rewriter, loc, waveRow,
                        c14I64Constant(rewriter, loc, 1)),
                    waveCol);
                auto laneHigh = mlir::arith::ShLIOp::create(
                    rewriter, loc, c14I64(rewriter, loc, op.getOperandWord()),
                    c14I64Constant(rewriter, loc, 5));
                auto lane = mlir::arith::OrIOp::create(
                    rewriter, loc, laneLow, laneHigh);
                tidI64 = mlir::arith::OrIOp::create(
                    rewriter, loc,
                    mlir::arith::ShLIOp::create(
                        rewriter, loc, wave,
                        c14I64Constant(rewriter, loc, 6)),
                    lane);
            }
            auto t113 = mlir::arith::AndIOp::create(
                rewriter, loc,
                mlir::arith::ShLIOp::create(
                    rewriter, loc, tidI64, c14I64Constant(rewriter, loc, 6)),
                c14I64Constant(rewriter, loc, 1984));
            auto t115 = mlir::arith::AndIOp::create(
                rewriter, loc,
                mlir::arith::ShLIOp::create(
                    rewriter, loc, tidI64, c14I64Constant(rewriter, loc, 2)),
                c14I64Constant(rewriter, loc, 56));
            auto t116 = mlir::arith::AndIOp::create(
                rewriter, loc,
                mlir::arith::ShRUIOp::create(
                    rewriter, loc, tidI64, c14I64Constant(rewriter, loc, 2)),
                c14I64Constant(rewriter, loc, 8));
            auto qInner = mlir::arith::XOrIOp::create(
                rewriter, loc, t115, t116);
            auto t117 = mlir::arith::AndIOp::create(
                rewriter, loc,
                mlir::arith::ShLIOp::create(
                    rewriter, loc, tidI64, c14I64Constant(rewriter, loc, 4)),
                c14I64Constant(rewriter, loc, 2048));
            auto t125 = mlir::arith::AndIOp::create(
                rewriter, loc,
                mlir::arith::ShLIOp::create(
                    rewriter, loc, tidI64, c14I64Constant(rewriter, loc, 5)),
                c14I64Constant(rewriter, loc, 2048));
            auto qBase = mlir::arith::OrIOp::create(
                rewriter, loc,
                mlir::arith::OrIOp::create(rewriter, loc, t113, t117),
                qInner);
            auto nativeHInner = mlir::arith::XOrIOp::create(
                rewriter, loc, t115, t116);
            auto hBase = mlir::arith::OrIOp::create(
                rewriter, loc,
                mlir::arith::OrIOp::create(rewriter, loc, t113,
                                           nativeHInner),
                t125);
            auto kBase = mlir::arith::OrIOp::create(
                rewriter, loc,
                mlir::arith::OrIOp::create(rewriter, loc, t113, t116),
                t125);

            // Four packed vectors are the native MFMA32 K fragments.  The
            // ordinary side remains the probe's row-major identity stage;
            // only the real Q/H/K operand is switched to the recovered
            // native shared consumer address in this step.
            unsigned fragment = 0;
            for (; fragment < 4; ++fragment) {
                mlir::Value aFrag;
                mlir::Value bFrag;
                auto step = c14I64Constant(
                    rewriter, loc, static_cast<int64_t>(fragment) * 16);
                auto qByte = mlir::arith::XOrIOp::create(
                    rewriter, loc, qBase, step);
                if (role == "Q") {
                    aFrag = emitC16NativePackedLdsLoadAtByte(
                        rewriter, loc, op.getAStage(), qByte, role);
                    // The packed-H probe uses a compact [64,32] B stage.  In
                    // addition to the opt-in process gate, recognize that
                    // structural contract here so the late first-class
                    // matcher cannot silently fall back to the legacy scalar
                    // gather when the launcher and compiler pass have
                    // different environment snapshots.  The ordinary
                    // [64,64] Q/H probe remains byte-for-byte untouched.
                    auto bType = mlir::dyn_cast<mlir::MemRefType>(
                        op.getBStage().getType());
                    const bool compactHShape =
                        bType && bType.getRank() == 2 &&
                        bType.getDimSize(0) == 64 &&
                        bType.getDimSize(1) == 32;
                    if (useC16WG128QHPackedH() || compactHShape) {
                        // The selected native H/#shared1 consumer uses
                        // t126 xor t127 for the inner address bits.  The
                        // experimental packed-H producer uses the inverse
                        // native pbase packet map, so this is a direct
                        // producer-to-consumer physical contract rather
                        // than a self-consistent compact-layout shortcut.
                        auto hByte = mlir::arith::XOrIOp::create(
                            rewriter, loc, hBase, step);
                        bFrag = emitC16NativePackedLdsLoadAtByte(
                            rewriter, loc, op.getBStage(), hByte, "H");
                    } else {
                        auto kByte = mlir::arith::AddIOp::create(
                            rewriter, loc, kBase, step);
                        bFrag = emitC16KIdentityOperandFromNativeByte(
                            rewriter, loc, op.getBStage(), kByte);
                    }
                } else if (role == "H") {
                    aFrag = emitC16IdentityOperandFromNativeByte(
                        rewriter, loc, op.getAStage(), qByte,
                        /*qOperand=*/true);
                    // Native H/shared1 uses the XOR step sequence visible in
                    // the selected LLVM (%127, xor16, xor32, xor48).
                    auto hByte = mlir::arith::XOrIOp::create(
                        rewriter, loc, hBase, step);
                    bFrag = emitC16NativePackedLdsLoadAtByte(
                        rewriter, loc, op.getBStage(), hByte, role);
                } else {
                    aFrag = emitC16IdentityOperandFromNativeByte(
                        rewriter, loc, op.getAStage(), qByte,
                        /*qOperand=*/true);
                    // Native K/shared2 uses the selected LLVM address chain:
                    // shared2 + t113 + t116 + t125, followed by the
                    // contiguous +16/+32/+48 packet offsets.  It does not
                    // reuse Q's qInner (t115 xor t116); that value is specific
                    // to the Q/H shared encodings.
                    auto kByte = mlir::arith::AddIOp::create(
                        rewriter, loc, kBase, step);
                    bFrag = emitC16NativePackedLdsLoadAtByte(
                        rewriter, loc, op.getBStage(), kByte, role);
                }
                if (!aFrag || !bFrag)
                    return rewriter.notifyMatchFailure(
                        op, "C16 Q/H/K consumer cannot form packed operands");
                auto call = mlir::func::CallOp::create(
                    rewriter, loc, mfmaName,
                    mlir::TypeRange{accumulator.getType()},
                    mlir::ValueRange{bFrag, aFrag, accumulator});
                call->setAttr("c16.real_tile_consumer",
                              rewriter.getUnitAttr());
                call->setAttr("c16.role", rewriter.getStringAttr(role));
                call->setAttr("c14.static_physical_consumed",
                              rewriter.getUnitAttr());
                call->setAttr(
                    "c14.dot_op_idx",
                    rewriter.getI64IntegerAttr(plan->dot.opIdx));
                accumulator = call.getResult(0);
            }
            rewriter.replaceOp(op, accumulator);
            return mlir::success();
        }

        if (op->hasAttr("c14.static_physical")) {
            auto plan = getC14StaticPhysicalPlan(op);
            if (!plan)
                return rewriter.notifyMatchFailure(
                    op, "C14 static physical op is missing a valid typed recipe");
            // C14 uses the existing MFMA consumer, but its B operand address is
            // generated from the typed SharedEncoding/StaticTransform recipe.
            // There is intentionally no generic ownership decomposition here.
            const bool c15 = op->hasAttr("c15.real_tile");
            auto kBase = c15
                             ? indexMul(rewriter, loc, op.getKStage(),
                                        indexConstant(rewriter, loc, 32))
                             : indexConstant(rewriter, loc, 0);
            for (mlir::Value word : {word0, word1}) {
                auto baseCol = indexAdd(
                    rewriter, loc, kBase,
                    indexMul(rewriter, loc, word, c8));
                for (int64_t fragmentOffset : {int64_t{0}, int64_t{4}}) {
                    auto col = indexAdd(
                        rewriter, loc, baseCol,
                        indexConstant(rewriter, loc, fragmentOffset));
                    auto aFrag = emitPackedLdsLoad(
                        rewriter, loc, op.getAStage(), op.getARow(), col);
                    auto bFrag = emitC14PackedLdsLoad(
                        rewriter, loc, op.getBStage(), op.getBRow(), col,
                        *plan, c15);
                    if (!aFrag || !bFrag)
                        return rewriter.notifyMatchFailure(
                            op, "C14 static recipe cannot form packed BF16 LDS operands");
                    auto call = mlir::func::CallOp::create(
                        rewriter, loc, mfmaName,
                        mlir::TypeRange{accumulator.getType()},
                        mlir::ValueRange{bFrag, aFrag, accumulator});
                    call->setAttr("c14.static_physical_consumed",
                                  rewriter.getUnitAttr());
                    call->setAttr("c14.dot_op_idx",
                                  rewriter.getI64IntegerAttr(plan->dot.opIdx));
                    call->setAttr("c14.mfma_target",
                                  rewriter.getStringAttr(plan->mfma.target));
                    call->setAttr("c14.mfma_instr_shape",
                                  mlir::DenseI64ArrayAttr::get(
                                      rewriter.getContext(),
                                      llvm::ArrayRef<int64_t>(
                                          plan->mfma.instrShape)));
                    accumulator = call.getResult(0);
                }
            }
            rewriter.replaceOp(op, accumulator);
            return mlir::success();
        }

        if (usePackedOperandReusePlan()) {
            for (mlir::Value word : {word0, word1}) {
                if (mlir::failed(emitPackedConsumerGroup(
                        op, rewriter, loc, accumulator, word,
                        op->getAttrOfType<mlir::StringAttr>(
                            "avelang.block_dot.source_role")
                            ? op->getAttrOfType<mlir::StringAttr>(
                                  "avelang.block_dot.source_role")
                                  .getValue()
                            : llvm::StringRef("unknown"))))
                    return mlir::failure();
            }
            rewriter.replaceOp(op, accumulator);
            return mlir::success();
        }

        for (mlir::Value word : {word0, word1}) {
            auto baseCol = indexMul(rewriter, loc, word, c8);
            for (int64_t fragmentOffset : {int64_t{0}, int64_t{4}}) {
                auto col = indexAdd(
                    rewriter, loc, baseCol,
                    indexConstant(rewriter, loc, fragmentOffset));
                auto aFrag = emitPackedLdsLoad(
                    rewriter, loc, op.getAStage(), op.getARow(), col);
                auto bFrag = emitPackedLdsLoad(
                    rewriter, loc, op.getBStage(), op.getBRow(), col);
                if (!aFrag || !bFrag)
                    return rewriter.notifyMatchFailure(
                        op, "first-class operand requires static BF16 rank-2 LDS stages");
                auto call = mlir::func::CallOp::create(
                    rewriter, loc, mfmaName,
                    mlir::TypeRange{accumulator.getType()},
                    mlir::ValueRange{bFrag, aFrag, accumulator});
                call->setAttr("avelang.block_dot.mfma32",
                              rewriter.getUnitAttr());
                call->setAttr("avelang.block_dot.first_class_consumer",
                              rewriter.getUnitAttr());
                if (auto role = op->getAttrOfType<mlir::StringAttr>(
                        "avelang.block_dot.operand_role"))
                    call->setAttr("avelang.block_dot.operand_role", role);
                if (auto source = op->getAttrOfType<mlir::StringAttr>(
                        "avelang.block_dot.source_role"))
                    call->setAttr("avelang.block_dot.source_role", source);
                accumulator = call.getResult(0);
            }
        }
        rewriter.replaceOp(op, accumulator);
        return mlir::success();
    }
};

class BlockDotLowering
    : public mlir::OpRewritePattern<AMDGPUBlockDotBF16F32Op> {
  public:
    BlockDotLowering(mlir::MLIRContext *context, LoweringKind kind,
                     OperandStagingKind operandStaging,
                     const c13::ChunkOPhysicalPlan *c17Plan = nullptr,
                     const FullPhysicalRegionPlan *c18Plan = nullptr,
                     const FullPhysicalRegionPlan *c19Plan = nullptr)
        : OpRewritePattern(context), kind_(kind),
          operandStaging_(operandStaging), c17Plan_(c17Plan),
          c18Plan_(c18Plan), c19Plan_(c19Plan) {}

    mlir::LogicalResult
    matchAndRewrite(AMDGPUBlockDotBF16F32Op op,
                    mlir::PatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        if (isFullScopeOperandMode(op))
            return lowerFullScopeOperandMode(op, rewriter, kind_,
                                             &forwardedAccumulators_, c17Plan_,
                                             c18Plan_, c19Plan_,
                                             &c23Materializer_);
        if (isGenericOperandMode(op))
            return lowerGenericOperandMode(op, rewriter, kind_);
        auto c32 = indexConstant(rewriter, loc, 32);
        auto c64 = indexConstant(rewriter, loc, 64);
        auto tid = toIndex(rewriter, loc, op.getThreadId());
        auto wave = mlir::arith::DivUIOp::create(rewriter, loc, tid, c64);
        auto lane = mlir::arith::RemUIOp::create(rewriter, loc, tid, c64);
        auto laneCol = mlir::arith::RemUIOp::create(rewriter, loc, lane, c32);
        auto laneGroup = mlir::arith::DivUIOp::create(rewriter, loc, lane, c32);
        auto vec16 = mlir::VectorType::get({16}, rewriter.getF32Type());
        const bool cooperativeBv32 = isCooperativeBv32(op);
        const bool precomputedVDecay = isPrecomputedVDecay(op);
        const bool stagedVDecay = isStagedVDecay(op);
        const bool preloadedK = isPreloadedK(op);
        const bool stateKV = isStateKV(op);
        const bool jointV4LdsRetile = preloadedK && usesJointV4LdsRetile();
        if (stateKV && !jointV4LdsRetile) {
            return rewriter.notifyMatchFailure(
                op, "state-KV requires the existing R4 preloaded-K retile");
        }
        const bool requiresCooperativeBv32 =
            operandStaging_ == OperandStagingKind::TypedVector ||
            operandStaging_ == OperandStagingKind::PersistentTypedBlock ||
            operandStaging_ == OperandStagingKind::PersistentTypedLdsLayout;
        if (requiresCooperativeBv32 &&
            !cooperativeBv32) {
            return rewriter.notifyMatchFailure(
                op, "typed operand staging requires cooperative BV32");
        }
        const bool persistentTypedBlock =
            operandStaging_ == OperandStagingKind::PersistentTypedBlock ||
            operandStaging_ == OperandStagingKind::PersistentTypedLdsLayout;
        const bool packedTokenMajor =
            operandStaging_ == OperandStagingKind::PersistentTypedLdsLayout;
        if (persistentTypedBlock &&
            kind_ != LoweringKind::Specialized) {
            return rewriter.notifyMatchFailure(
                op, "persistent typed block staging requires specialized lowering");
        }

        llvm::SmallVector<AMDGPUQwenK64CoreIssueOp> deferredCoreIssues;
        for (mlir::Operation *cursor = op->getNextNode(); cursor;
             cursor = cursor->getNextNode()) {
            auto issue = mlir::dyn_cast<AMDGPUQwenK64CoreIssueOp>(cursor);
            if (!issue || !isCoreLastUseDeferredIssue(issue)) {
                break;
            }
            deferredCoreIssues.push_back(issue);
        }
        auto moveDeferredIssuesAfter = [&](int64_t consumerGroup,
                                           mlir::Operation *anchor) {
            for (auto issue : deferredCoreIssues) {
                auto target = issue->getAttrOfType<mlir::IntegerAttr>(
                    "avelang.qwen.core_staggered.issue_after_consumer_group");
                if (!target || target.getInt() != consumerGroup) {
                    continue;
                }
                issue->moveAfter(anchor);
                issue->setAttr("avelang.qwen.core_staggered.actual_issue_boundary",
                               rewriter.getStringAttr("update_mfma_group_last_use"));
                anchor = issue.getOperation();
            }
        };

        mlir::Value low = makeZeroVector(rewriter, loc, vec16);
        mlir::Value high = makeZeroVector(rewriter, loc, vec16);
        if (persistentTypedBlock) {
            // Keep the full V32xT64 operand CTA-local across both uniform
            // k_half loop iterations. K64xT64 is refilled once per owning
            // half. The source's K32 accumulation order is preserved.
            if (stagedVDecay && !precomputedVDecay) {
                return rewriter.notifyMatchFailure(
                    op, "staged V-decay must carry the precomputed-V-decay semantic");
            }
            auto aBlock = stagedVDecay
                              ? op.getAStage()
                              : (packedTokenMajor
                                     ? makeWorkgroupScratch(rewriter, loc, op.getAStage(),
                                                            {64, 32})
                                     : makeWorkgroupScratch(rewriter, loc, op.getAStage(),
                                                            {1, 32, 64}));
            auto bBlock = stagedVDecay
                              ? op.getBStage()
                              : makeWorkgroupScratch(rewriter, loc, op.getBStage(),
                                                     {64, 64});
            if (!aBlock || !bBlock) {
                return rewriter.notifyMatchFailure(
                    op, "persistent typed block scratch allocation failed");
            }
            auto c0 = indexConstant(rewriter, loc, 0);
            auto kHalf = toIndex(rewriter, loc, op.getKHalf());
            auto isFirstKHalf = mlir::arith::CmpIOp::create(
                rewriter, loc, mlir::arith::CmpIPredicate::eq, kHalf, c0);
            if (!stagedVDecay) {
                auto stageV = mlir::scf::IfOp::create(
                    rewriter, loc, isFirstKHalf, /*withElseRegion=*/false);
                rewriter.setInsertionPointToStart(&stageV.getThenRegion().front());
                emitPersistentTypedBlockStageA(rewriter, loc, op, aBlock,
                                               precomputedVDecay, packedTokenMajor);
                rewriter.setInsertionPointAfter(stageV);
            }

            if (!preloadedK) {
                emitPersistentTypedBlockStageB(rewriter, loc, op, bBlock,
                                               packedTokenMajor);
            }
            mlir::gpu::BarrierOp::create(rewriter, loc);
            int64_t consumerGroup = 0;
            for (int64_t token = 0; token < 2; ++token) {
                    auto tokenHalf = indexConstant(rewriter, loc, token);
                    for (int64_t col = 0; col < 2; ++col) {
                        if (col == 0) {
                            low = jointV4LdsRetile
                                      ? emitPersistentRetiledMfmaForOwnership(
                                            rewriter, loc, op, aBlock, bBlock,
                                            wave, laneCol, laneGroup, tokenHalf,
                                            col, low, stateKV)
                                      : packedTokenMajor
                                      ? emitPersistentPackedMfmaForOwnership(
                                            rewriter, loc, op, aBlock, bBlock,
                                            wave, laneCol, laneGroup, tokenHalf,
                                            col, low)
                                      : emitPersistentMfmaForOwnership(
                                            rewriter, loc, op, aBlock, bBlock,
                                            wave, laneCol, laneGroup, tokenHalf,
                                            col, low, preloadedK);
                            if (!deferredCoreIssues.empty()) {
                                moveDeferredIssuesAfter(
                                    consumerGroup, low.getDefiningOp());
                            }
                        } else {
                            high = jointV4LdsRetile
                                       ? emitPersistentRetiledMfmaForOwnership(
                                             rewriter, loc, op, aBlock, bBlock,
                                             wave, laneCol, laneGroup, tokenHalf,
                                             col, high, stateKV)
                                       : packedTokenMajor
                                       ? emitPersistentPackedMfmaForOwnership(
                                             rewriter, loc, op, aBlock, bBlock,
                                             wave, laneCol, laneGroup, tokenHalf,
                                             col, high)
                                       : emitPersistentMfmaForOwnership(
                                             rewriter, loc, op, aBlock, bBlock,
                                             wave, laneCol, laneGroup, tokenHalf,
                                             col, high, preloadedK);
                            if (!deferredCoreIssues.empty()) {
                                moveDeferredIssuesAfter(
                                    consumerGroup, high.getDefiningOp());
                            }
                        }
                        ++consumerGroup;
                }
            }
            mlir::gpu::BarrierOp::create(rewriter, loc);
        } else if (kind_ == LoweringKind::Generic) {
            // Exact direct-K64 source schedule: each output K32 tile stages
            // V-decay independently for both token halves.
            for (int64_t col = 0; col < 2; ++col) {
                auto accumulator = makeZeroVector(rewriter, loc, vec16);
                for (int64_t token = 0; token < 2; ++token) {
                    auto tokenHalf = indexConstant(rewriter, loc, token);
                    emitStageA(rewriter, loc, op, tokenHalf, cooperativeBv32,
                               precomputedVDecay, operandStaging_);
                    emitStageB(rewriter, loc, op, tokenHalf, col,
                               operandStaging_);
                    mlir::gpu::BarrierOp::create(rewriter, loc);
                    accumulator = emitMfmaForOwnership(
                        rewriter, loc, op, wave, laneCol, laneGroup,
                        accumulator, cooperativeBv32);
                    mlir::gpu::BarrierOp::create(rewriter, loc);
                }
                if (col == 0)
                    low = accumulator;
                else
                    high = accumulator;
            }
        } else {
            // Keep the V-decay block live only across its two immediate K32
            // consumers. K is still staged and consumed one tile at a time.
            // BV32 ownership only changes which wave consumes each K64 half.
            for (int64_t token = 0; token < 2; ++token) {
                auto tokenHalf = indexConstant(rewriter, loc, token);
                emitStageA(rewriter, loc, op, tokenHalf, cooperativeBv32,
                           precomputedVDecay, operandStaging_);
                for (int64_t col = 0; col < 2; ++col) {
                    emitStageB(rewriter, loc, op, tokenHalf, col,
                               operandStaging_);
                    mlir::gpu::BarrierOp::create(rewriter, loc);
                    if (col == 0) {
                        low = emitMfmaForOwnership(rewriter, loc, op, wave,
                                                   laneCol, laneGroup, low,
                                                   cooperativeBv32);
                    } else {
                        high = emitMfmaForOwnership(rewriter, loc, op, wave,
                                                    laneCol, laneGroup, high,
                                                    cooperativeBv32);
                    }
                    mlir::gpu::BarrierOp::create(rewriter, loc);
                }
            }
        }

        auto accLow =
            materializeAccumulatorVector(rewriter, loc, op.getAccLow());
        auto accHigh =
            materializeAccumulatorVector(rewriter, loc, op.getAccHigh());
        if (!accLow || !accHigh) {
            return rewriter.notifyMatchFailure(
                op, "persistent accumulator was not F32 vector/local [16]");
        }
        low = scaleAndAccumulate(rewriter, loc, accLow, low, op.getGLast());
        high = scaleAndAccumulate(rewriter, loc, accHigh, high, op.getGLast());

        auto result = joinColumns(rewriter, loc, low, high);
        auto attrName = kind_ == LoweringKind::Generic
                            ? "avelang.block_dot.generic"
                            : "avelang.block_dot.gfx942_specialized";
        if (auto *def = result.getDefiningOp()) {
            def->setAttr(attrName, rewriter.getUnitAttr());
        }
        rewriter.replaceOp(op, result);
        return mlir::success();
    }

  private:
    LoweringKind kind_;
    OperandStagingKind operandStaging_;
    const c13::ChunkOPhysicalPlan *c17Plan_ = nullptr;
    const FullPhysicalRegionPlan *c18Plan_ = nullptr;
    const FullPhysicalRegionPlan *c19Plan_ = nullptr;
    mutable AccumulatorForwardingMap forwardedAccumulators_;
    mutable C23LatePipelineMaterializer c23Materializer_;
};

class LowerQwenBlockDotPass
    : public mlir::PassWrapper<LowerQwenBlockDotPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerQwenBlockDotPass)

    llvm::StringRef getArgument() const final { return "lower-qwen-block-dot"; }
    llvm::StringRef getDescription() const final {
        return "Lower Qwen direct-K64 block dot with generic or gfx942 "
               "scheduling";
    }

    void runOnOperation() override {
        bool hasBlockDot = false;
        bool hasPredStateKVFragment = false;
        getOperation().walk(
            [&](AMDGPUBlockDotBF16F32Op) { hasBlockDot = true; });
        getOperation().walk([&](AMDGPUQwenPredStateKVFragLoadOp) {
            hasPredStateKVFragment = true;
        });
        if (!hasBlockDot && !hasPredStateKVFragment) {
            return;
        }
        auto mode = llvm::sys::Process::GetEnv("AVELANG_BLOCK_DOT_LOWERING")
                        .value_or("generic");
        auto operandMode =
            llvm::sys::Process::GetEnv("AVELANG_BLOCK_DOT_OPERAND_LOWERING")
                .value_or("scalar");
        // C23 mutates cross-operation state from a greedy RewritePattern.
        // It is retained below solely as historical evidence, but is blocked
        // before any rewrite can run.  C24 replaces it with a real region SSA
        // graph and a dedicated late materializer.
        if (useC23LatePipelineMaterialization()) {
            getOperation().emitError(
                "C23 late pipeline materialization is retired: cross-op packet "
                "ownership inside greedy rewriting is invalid; use the C24 "
                "region pending-packet infrastructure");
            signalPassFailure();
            return;
        }
        LoweringKind kind;
        if (mode == "generic") {
            kind = LoweringKind::Generic;
        } else if (mode == "specialized") {
            kind = LoweringKind::Specialized;
        } else {
            getOperation().emitError() << "AVELANG_BLOCK_DOT_LOWERING must be "
                                          "generic or specialized, got "
                                       << mode;
            signalPassFailure();
            return;
        }

        std::optional<c13::ChunkOPhysicalPlan> c17Plan;
        std::optional<FullPhysicalRegionPlan> c18Plan;
        std::optional<FullPhysicalRegionPlan> c19Plan;
        std::optional<ChunkOPipelinePlan> c21PipelinePlan;
        if (useC17FullPhysicalPlan()) {
            if (kind != LoweringKind::Specialized) {
                getOperation().emitError()
                    << "C17 full physical plan requires specialized block-dot lowering";
                signalPassFailure();
                return;
            }
            if (!useFirstClassMfmaOperandPlan()) {
                getOperation().emitError()
                    << "C17 full physical plan requires p2_first_class or later "
                       "first-class operand preservation";
                signalPassFailure();
                return;
            }
            c17Plan.emplace(c13::ChunkOPhysicalPlan::makeC12T2048WG256());
            std::string error;
            if (!c17Plan->verify(&error)) {
                getOperation().emitError()
                    << "C17 ChunkOPhysicalPlan verification failed: " << error;
                signalPassFailure();
                return;
            }
            getOperation()->setAttr(
                "avelang.stage6z.full_physical_plan",
                mlir::StringAttr::get(&getContext(),
                                      "gfx942_bt64_bv64_joint_c17"));
            getOperation()->setAttr(
                "avelang.stage6z.physical_target",
                mlir::StringAttr::get(&getContext(),
                                      "gfx942_wave64_wg256_mfma32"));
            getOperation()->setAttr(
                "avelang.stage6z.shared_lifetime",
                mlir::StringAttr::get(
                    &getContext(), "source_Q_H_K_then_score_V"));
            getOperation()->setAttr(
                "avelang.stage6z.q_dual_consumer",
                mlir::StringAttr::get(&getContext(), "single_Q_to_QH_QK"));
        }
        if (useC18FullPhysicalRegion()) {
            if (kind != LoweringKind::Specialized) {
                getOperation().emitError()
                    << "C18 full physical region requires specialized block-dot lowering";
                signalPassFailure();
                return;
            }
            if (!useFirstClassMfmaOperandPlan()) {
                getOperation().emitError()
                    << "C18 full physical region requires first-class MFMA operands";
                signalPassFailure();
                return;
            }
            c18Plan.emplace();
            c18Plan->physical =
                c13::ChunkOPhysicalPlan::makeC12T2048WG256();
            std::string error;
            if (!c18Plan->physical.verify(&error)) {
                getOperation().emitError()
                    << "C18 FullPhysicalRegionPlan verification failed: "
                    << error;
                signalPassFailure();
                return;
            }
            getOperation()->setAttr(
                "avelang.stage6z.full_physical_region",
                mlir::StringAttr::get(
                    &getContext(), "gfx942_bt64_bv64_full_region_c18"));
            getOperation()->setAttr(
                "avelang.stage6z.full_region_owner",
                mlir::StringAttr::get(
                    &getContext(), "FullPhysicalRegionPlan"));
            getOperation()->setAttr(
                "avelang.stage6z.v_source_consumer_owner",
                mlir::BoolAttr::get(&getContext(), true));

            // Mark exactly one V logical block-dot as the physical producer
            // owner.  Every V consumer still carries the same plan, but only
            // this one emits the single global BF16x8 producer and barrier.
            bool vOwnerAssigned = false;
            getOperation().walk([&](AMDGPUBlockDotBF16F32Op op) {
                if (!isFullScopeOperandMode(op) ||
                    genericOperandSourceRole(op) != "V")
                    return;
                op->setAttr("c18.full_physical_region",
                            mlir::UnitAttr::get(&getContext()));
                op->setAttr("c18.plan_role",
                            mlir::StringAttr::get(&getContext(), "V"));
                op->setAttr("c18.consumer",
                            mlir::StringAttr::get(&getContext(), "score@V"));
                if (!vOwnerAssigned) {
                    op->setAttr("c18.v_producer_owner",
                                mlir::UnitAttr::get(&getContext()));
                    vOwnerAssigned = true;
                }
            });
            if (!vOwnerAssigned) {
                getOperation().emitError()
                    << "C18 source has no V logical block-dot owner";
                signalPassFailure();
                return;
            }

            // C18 intentionally removes the handwritten source MFMA in
            // Phase-C, so the ordinary source generator no longer leaves an
            // intrinsic declaration for the late physical consumers.  Seed
            // the same private declaration that the embedded AMDGPU
            // intrinsic library will replace/link later in the pipeline.
            auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
            const auto mfmaName = ir::intrinsics::MakeIntrinsicFuncName(
                "amdgpu", "rocdl_mfma_f32_32x32x8bf16_1k");
            if (module && !module.lookupSymbol<mlir::func::FuncOp>(mfmaName)) {
                mlir::OpBuilder builder(&getContext());
                builder.setInsertionPointToStart(module.getBody());
                auto bf16x4 = mlir::VectorType::get({4},
                                                    builder.getBF16Type());
                auto f32x16 = mlir::VectorType::get({16},
                                                    builder.getF32Type());
                auto type = builder.getFunctionType(
                    mlir::TypeRange{bf16x4, bf16x4, f32x16},
                    mlir::TypeRange{f32x16});
                auto declaration = mlir::func::FuncOp::create(
                    builder, builder.getUnknownLoc(), mfmaName, type);
                declaration.setPrivate();
            }
        }
        if (usesC19CompatibleFullPhysicalRegion()) {
            const bool c21Pipeline = useC21SelectedNativePipeline();
            if (kind != LoweringKind::Specialized) {
                getOperation().emitError()
                    << (c21Pipeline ? "C21 selected-native pipeline"
                                    : "C19 full physical region")
                    << " requires specialized block-dot lowering";
                signalPassFailure();
                return;
            }
            if (!useFirstClassMfmaOperandPlan()) {
                getOperation().emitError()
                    << (c21Pipeline ? "C21 selected-native pipeline"
                                    : "C19 full physical region")
                    << " requires first-class MFMA operands";
                signalPassFailure();
                return;
            }
            std::string error;
            if (c21Pipeline) {
                c21PipelinePlan.emplace();
                c21PipelinePlan->fullRegion.physical =
                    c13::ChunkOPhysicalPlan::makeC12T2048WG256();
                if (!c21PipelinePlan->verify(&error)) {
                    getOperation().emitError()
                        << "C21 ChunkOPipelinePlan verification failed: " << error;
                    signalPassFailure();
                    return;
                }
                c19Plan.emplace(c21PipelinePlan->fullRegion);
            } else {
                c19Plan.emplace();
                c19Plan->physical =
                    c13::ChunkOPhysicalPlan::makeC12T2048WG256();
            }
            if (!c19Plan->physical.verify(&error)) {
                getOperation().emitError()
                    << (c21Pipeline ? "C21 ChunkOPipelinePlan"
                                    : "C19 FullPhysicalRegionPlan")
                    << " verification failed: " << error;
                signalPassFailure();
                return;
            }
            getOperation()->setAttr(
                "avelang.stage6z.full_physical_region",
                mlir::StringAttr::get(
                    &getContext(), c21Pipeline
                                       ? "gfx942_bt64_bv64_selected_native_c21"
                                       : "gfx942_bt64_bv64_full_region_c19"));
            getOperation()->setAttr(
                "avelang.stage6z.full_region_owner",
                mlir::StringAttr::get(
                    &getContext(), c21Pipeline ? "ChunkOPipelinePlan"
                                                : "FullPhysicalRegionPlan"));
            getOperation()->setAttr(
                "avelang.stage6z.c19.legacy_owners_forbidden",
                mlir::BoolAttr::get(&getContext(), true));
            getOperation()->setAttr(
                "avelang.stage6z.c19.planned_shared_bytes",
                mlir::IntegerAttr::get(mlir::IntegerType::get(&getContext(), 64),
                                       24576));
            if (c21Pipeline) {
                getOperation()->setAttr(
                    "avelang.stage6z.c21.pipeline",
                    mlir::StringAttr::get(
                        &getContext(), "wg256_wave64_stage2_qh_qk_superloop"));
                getOperation()->setAttr(
                    "avelang.stage6z.c21.q_slot_count",
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(&getContext(), 64), 2));
            }

            bool qOwnerAssigned = false;
            bool vOwnerAssigned = false;
            bool hasH = false;
            bool hasK = false;
            getOperation().walk([&](AMDGPUBlockDotBF16F32Op op) {
                if (!isFullScopeOperandMode(op))
                    return;
                if (op->hasAttr("c18.full_physical_region") ||
                    op->hasAttr("c17.full_physical_plan")) {
                    getOperation().emitError()
                        << "C19 encountered a legacy physical owner attribute";
                    signalPassFailure();
                    return;
                }
                auto role = genericOperandSourceRole(op);
                if (role != "Q" && role != "H" && role != "K" && role != "V") {
                    getOperation().emitError()
                        << "C19 logical block-dot has no plan-owned Q/H/K/V role";
                    signalPassFailure();
                    return;
                }
                annotateC19PhysicalRegion(
                    op, *c19Plan, role,
                    role == "H" ? "Q@H" : role == "K" ? "Q@K"
                                                       : role == "Q" ? "Q producer"
                                                                      : "score@V");
                if (c21Pipeline) {
                    annotateC21SelectedNativePipeline(
                        op, *c21PipelinePlan, role,
                        role == "H" ? "Q@H" : role == "K" ? "Q@K"
                                                           : role == "Q" ? "Q producer"
                                                                          : "score@V");
                }
                if (role == "Q") {
                    if (qOwnerAssigned) {
                        getOperation().emitError()
                            << "C19 requires exactly one Q producer owner";
                        signalPassFailure();
                    }
                    op->setAttr("c19.q_producer_owner",
                                mlir::UnitAttr::get(&getContext()));
                    qOwnerAssigned = true;
                } else if (role == "V") {
                    if (vOwnerAssigned) {
                        getOperation().emitError()
                            << "C19 requires exactly one V producer owner";
                        signalPassFailure();
                    }
                    op->setAttr("c19.v_producer_owner",
                                mlir::UnitAttr::get(&getContext()));
                    vOwnerAssigned = true;
                } else if (role == "H") {
                    hasH = true;
                } else if (role == "K") {
                    hasK = true;
                }
            });
            if (!qOwnerAssigned || !vOwnerAssigned || !hasH || !hasK) {
                getOperation().emitError()
                    << "C19 requires compiler-owned Q/H/K/V producers and consumers";
                signalPassFailure();
                return;
            }
            auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
            const auto mfmaName = ir::intrinsics::MakeIntrinsicFuncName(
                "amdgpu", "rocdl_mfma_f32_32x32x8bf16_1k");
            if (module && !module.lookupSymbol<mlir::func::FuncOp>(mfmaName)) {
                mlir::OpBuilder builder(&getContext());
                builder.setInsertionPointToStart(module.getBody());
                auto bf16x4 = mlir::VectorType::get({4}, builder.getBF16Type());
                auto f32x16 = mlir::VectorType::get({16}, builder.getF32Type());
                auto type = builder.getFunctionType(
                    mlir::TypeRange{bf16x4, bf16x4, f32x16},
                    mlir::TypeRange{f32x16});
                auto declaration = mlir::func::FuncOp::create(
                    builder, builder.getUnknownLoc(), mfmaName, type);
                declaration.setPrivate();
            }

            // C24 needs the C21 ownership annotations on the complete H/K
            // region before it can form a direct SSA issue/commit edge.  It
            // cannot run in the earlier C18 setup block: C21 owns c19Plan and
            // does not enter that legacy branch.
            if (useC24RegionPendingPacketInfrastructure() ||
                useC25CurrentReadyNextPending()) {
                if (!c21Pipeline) {
                    getOperation().emitError(
                        "C24/C25 region pending-packet infrastructure requires "
                        "frozen C21 selected-native ownership");
                    signalPassFailure();
                    return;
                }
                C24RegionPendingPacketPlanner planner;
                if (mlir::failed(planner.materialize(getOperation(), *c19Plan))) {
                    signalPassFailure();
                    return;
                }
            }
        }
        OperandStagingKind operandStaging;
        if (operandMode == "scalar") {
            operandStaging = OperandStagingKind::Scalar;
        } else if (operandMode == "typed_vector") {
            operandStaging = OperandStagingKind::TypedVector;
        } else if (operandMode == "persistent_typed_block") {
            operandStaging = OperandStagingKind::PersistentTypedBlock;
        } else if (operandMode == "persistent_typed_lds_layout") {
            operandStaging = OperandStagingKind::PersistentTypedLdsLayout;
        } else {
            getOperation().emitError()
                << "AVELANG_BLOCK_DOT_OPERAND_LOWERING must be scalar, "
                   "typed_vector, persistent_typed_block, or "
                   "persistent_typed_lds_layout, got "
                << operandMode;
            signalPassFailure();
            return;
        }
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<BlockDotLowering>(&getContext(), kind, operandStaging,
                                       c17Plan ? &*c17Plan :
                                                 (c18Plan ? &c18Plan->physical
                                                          : nullptr),
                                       c18Plan ? &*c18Plan : nullptr,
                                       c19Plan ? &*c19Plan : nullptr);
        patterns.add<PredStateKVFragLoadLowering>(&getContext());
        if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                     std::move(patterns)))) {
            signalPassFailure();
            return;
        }
        bool remaining = false;
        getOperation().walk([&](AMDGPUBlockDotBF16F32Op op) {
            op.emitError("block_dot_bf16_f32 survived late lowering");
            remaining = true;
        });
        getOperation().walk([&](AMDGPUQwenPredStateKVFragLoadOp op) {
            op.emitError("pred state-KV BF16x4 fragment survived late lowering");
            remaining = true;
        });
        if (remaining) {
            signalPassFailure();
            return;
        }
        getOperation()->setAttr("avelang.block_dot.lowering",
                                mlir::StringAttr::get(&getContext(), mode));
        getOperation()->setAttr(
            "avelang.block_dot.operand_lowering",
            mlir::StringAttr::get(&getContext(), operandMode));
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerQwenBlockDotPass() {
    return std::make_unique<LowerQwenBlockDotPass>();
}

class LowerQwenBlockDotMfmaOperandPass
    : public mlir::PassWrapper<LowerQwenBlockDotMfmaOperandPass,
                               mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
        LowerQwenBlockDotMfmaOperandPass)

    llvm::StringRef getArgument() const final {
        return "lower-qwen-block-dot-mfma-operand";
    }

    llvm::StringRef getDescription() const final {
        return "Materialize the first-class block-dot MFMA operand plan";
    }

    void runOnOperation() override {
        bool hasPlan = false;
        getOperation().walk([&](AMDGPUBlockDotMfmaOperandOp) {
            hasPlan = true;
        });
        if (!hasPlan)
            return;

        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<BlockDotMfmaOperandLowering>(&getContext());
        if (mlir::failed(mlir::applyPatternsGreedily(
                getOperation(), std::move(patterns)))) {
            signalPassFailure();
            return;
        }
        bool remaining = false;
        getOperation().walk([&](AMDGPUBlockDotMfmaOperandOp op) {
            op.emitError("first-class block-dot MFMA operand survived late lowering");
            remaining = true;
        });
        if (remaining)
            signalPassFailure();
    }
};

std::unique_ptr<mlir::Pass> createLowerQwenBlockDotMfmaOperandPass() {
    return std::make_unique<LowerQwenBlockDotMfmaOperandPass>();
}

} // namespace causalflow::avelang::dialect
