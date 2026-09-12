#include "bounded_packet_schedule_pass.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/LLVMIR/ROCDLDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Process.h>

#include <optional>

namespace causalflow::avelang::dialect {
namespace {

bool useBoundedConsumerPointSchedule() {
    auto mode = llvm::sys::Process::GetEnv(
        "AVELANG_STAGE6Z_PACKET_SCHEDULING");
    return mode && *mode == "bounded_consumer_point";
}

bool boundedPacketScheduleDebugEnabled() {
    return llvm::sys::Process::GetEnv(
               "AVELANG_STAGE6Z_PACKET_SCHEDULING_DEBUG") ==
           std::optional<std::string>("1");
}

bool isPrivateScalarMemref(mlir::Value value) {
    auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
    if (!type || type.getRank() != 1 || type.getShape() != llvm::ArrayRef<int64_t>({1}))
        return false;
    auto space = mlir::dyn_cast_or_null<mlir::IntegerAttr>(type.getMemorySpace());
    return space && space.getInt() == 5;
}

bool isCloneablePrivateAddressOp(mlir::Operation *op) {
    if (op->getNumRegions() != 0)
        return false;
    if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op))
        return isPrivateScalarMemref(load.getMemRef());
    if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op))
        return isPrivateScalarMemref(store.getMemRef());
    return mlir::isMemoryEffectFree(op);
}

bool hasOnlyInternalAddressUses(llvm::ArrayRef<mlir::Operation *> operations,
                                mlir::Operation *packetLoad) {
    llvm::SmallPtrSet<mlir::Operation *, 16> set(operations.begin(),
                                                  operations.end());
    for (mlir::Operation *op : operations) {
        // The raw packet is intentionally consumed after the old producer
        // prefix by the existing bitcast/shared-store sequence.  Only its
        // address recipe must be closed before the fission point.
        if (op == packetLoad)
            continue;
        for (mlir::Value result : op->getResults()) {
            for (mlir::Operation *user : result.getUsers()) {
                if (!set.contains(user))
                    return false;
            }
        }
    }
    return true;
}

bool containsMfma32(mlir::scf::ForOp loop) {
    bool found = false;
    loop.walk([&](mlir::func::CallOp call) {
        found |= call.getCallee().contains("mfma_f32_32x32x8bf16");
    });
    return found;
}

mlir::Value createZeroPacket(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Type type) {
    auto vector = mlir::dyn_cast<mlir::VectorType>(type);
    auto integer = vector ? mlir::dyn_cast<mlir::IntegerType>(vector.getElementType())
                          : nullptr;
    if (!integer)
        return {};
    auto zero = builder.getIntegerAttr(integer, 0);
    auto dense = mlir::DenseElementsAttr::get(vector, zero);
    return mlir::arith::ConstantOp::create(builder, loc, vector, dense);
}

// A packet is eligible only when all address construction between the nested
// lane predicate and the raw load is private scalar bookkeeping.  That makes
// cloning the issue path and deleting the old producer exact: it cannot move
// a workgroup/global side effect, a barrier, or a second logical packet.
bool collectPrivatePacketPrefix(mlir::ROCDL::RawBufferLoadOp load,
                                llvm::SmallVectorImpl<mlir::Operation *> &prefix) {
    mlir::Block *block = load->getBlock();
    for (mlir::Operation &candidate : block->without_terminator()) {
        prefix.push_back(&candidate);
        if (&candidate == load)
            break;
    }
    if (prefix.empty() || prefix.back() != load)
        return false;
    for (mlir::Operation *op : prefix) {
        if (op != load && !isCloneablePrivateAddressOp(op))
            return false;
    }
    return hasOnlyInternalAddressUses(prefix, load);
}

// The outer stage guard may be defined just after the current consumer loop.
// It is legal to clone only a pure arith.cmpi whose operands already dominate
// the consumer loop.  This is deliberately narrower than a general code
// motion pass and prevents pulling loop-carried or memory-dependent guards.
mlir::Value cloneStageCondition(mlir::PatternRewriter &rewriter,
                                mlir::scf::ForOp consumerLoop,
                                mlir::scf::IfOp stageIf,
                                mlir::IRMapping &mapping) {
    auto cmp = mlir::dyn_cast_or_null<mlir::arith::CmpIOp>(
        stageIf.getCondition().getDefiningOp());
    if (!cmp || cmp->getBlock() != consumerLoop->getBlock())
        return {};
    for (mlir::Value operand : cmp->getOperands()) {
        if (auto *def = operand.getDefiningOp(); def &&
            def->getBlock() == consumerLoop->getBlock() &&
            !def->isBeforeInBlock(consumerLoop))
            return {};
    }
    auto clone = mlir::cast<mlir::arith::CmpIOp>(rewriter.clone(*cmp, mapping));
    mapping.map(cmp.getResult(), clone.getResult());
    return clone.getResult();
}

bool scheduleOneBoundedPacket(mlir::ROCDL::RawBufferLoadOp raw,
                              mlir::PatternRewriter &rewriter) {
    auto reject = [&](llvm::StringRef reason) {
        if (boundedPacketScheduleDebugEnabled())
            llvm::errs() << "[bounded-packet-schedule] reject " << reason
                         << " at " << raw.getLoc() << "\n";
        return false;
    };
    auto packetType = mlir::dyn_cast<mlir::VectorType>(raw.getType());
    if (!packetType || packetType.getShape() != llvm::ArrayRef<int64_t>({4}) ||
        !packetType.getElementType().isInteger(32))
        return reject("not a v4i32 packet");

    auto laneIf = raw->getParentOfType<mlir::scf::IfOp>();
    // getParentOfType on an operation includes that operation itself.  The
    // stage guard therefore has to be recovered from the lane-if's owning
    // operation, not by calling getParentOfType on laneIf again.
    auto stageIf = laneIf ? mlir::dyn_cast_or_null<mlir::scf::IfOp>(
                              laneIf->getParentOp())
                          : mlir::scf::IfOp{};
    if (!laneIf || !stageIf || laneIf->getBlock() != &stageIf.getThenRegion().front())
        return reject("missing nested lane/stage if shape");
    auto stageLoop = stageIf->getParentOfType<mlir::scf::ForOp>();
    if (!stageLoop || stageIf->getBlock() != stageLoop.getBody())
        return reject("stage if is not directly in an scf.for body");

    // scf.if's condition is materialized as the immediately preceding op in
    // the parent block.  The independent consumer loop is immediately before
    // that pure condition, not immediately before the if itself.
    auto *stageConditionDef = stageIf.getCondition().getDefiningOp();
    auto consumerLoop = stageConditionDef
                            ? mlir::dyn_cast_or_null<mlir::scf::ForOp>(
                                  stageConditionDef->getPrevNode())
                            : mlir::scf::ForOp{};
    auto releaseBarrier = mlir::dyn_cast_or_null<mlir::gpu::BarrierOp>(stageIf->getNextNode());
    if (!consumerLoop || !releaseBarrier || !containsMfma32(consumerLoop)) {
        return reject("not directly after an MFMA loop and before a barrier");
    }

    llvm::SmallVector<mlir::Operation *> outerPrefix;
    for (mlir::Operation &op : stageIf.getThenRegion().front().without_terminator()) {
        if (&op == laneIf)
            break;
        if (!isCloneablePrivateAddressOp(&op))
            return reject("outer guard has non-private address work");
        outerPrefix.push_back(&op);
    }
    if (outerPrefix.empty())
        return reject("outer guard has no cloneable prefix");

    llvm::SmallVector<mlir::Operation *> packetPrefix;
    if (!collectPrivatePacketPrefix(raw, packetPrefix))
        return reject("packet address prefix is not private and closed");

    rewriter.setInsertionPoint(consumerLoop);
    mlir::IRMapping mapping;
    mlir::Value earlyCondition = cloneStageCondition(rewriter, consumerLoop,
                                                      stageIf, mapping);
    if (!earlyCondition)
        return reject("stage guard is not a cloneable dominating arith.cmpi");
    auto earlyStage = mlir::scf::IfOp::create(
        rewriter, raw.getLoc(), packetType, earlyCondition,
        /*withElseRegion=*/true);

    rewriter.setInsertionPointToStart(&earlyStage.getThenRegion().front());
    for (mlir::Operation *op : outerPrefix)
        rewriter.clone(*op, mapping);
    mlir::Value earlyLaneCondition = mapping.lookupOrDefault(laneIf.getCondition());
    if (earlyLaneCondition == laneIf.getCondition())
        return reject("lane guard was not materialized in early issue region");
    auto earlyLane = mlir::scf::IfOp::create(
        rewriter, raw.getLoc(), packetType, earlyLaneCondition,
        /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&earlyLane.getThenRegion().front());
    for (mlir::Operation *op : packetPrefix) {
        auto *clone = rewriter.clone(*op, mapping);
        if (op == raw)
            clone->setAttr("avelang.bounded_packet_schedule",
                           rewriter.getStringAttr("issue_early"));
    }
    mlir::Value earlyPacket = mapping.lookup(raw.getResult());
    if (!earlyPacket)
        return reject("raw packet did not clone");
    mlir::scf::YieldOp::create(rewriter, raw.getLoc(), earlyPacket);
    rewriter.setInsertionPointToStart(&earlyLane.getElseRegion().front());
    auto zero = createZeroPacket(rewriter, raw.getLoc(), packetType);
    if (!zero)
        return reject("cannot materialize lane-false packet value");
    mlir::scf::YieldOp::create(rewriter, raw.getLoc(), zero);

    rewriter.setInsertionPointToEnd(&earlyStage.getThenRegion().front());
    mlir::scf::YieldOp::create(rewriter, raw.getLoc(), earlyLane.getResult(0));
    rewriter.setInsertionPointToStart(&earlyStage.getElseRegion().front());
    auto outerZero = createZeroPacket(rewriter, raw.getLoc(), packetType);
    if (!outerZero)
        return reject("cannot materialize stage-false packet value");
    mlir::scf::YieldOp::create(rewriter, raw.getLoc(), outerZero);

    raw.getResult().replaceAllUsesWith(earlyStage.getResult(0));
    for (mlir::Operation *op : llvm::reverse(packetPrefix))
        rewriter.eraseOp(op);
    stageLoop->setAttr("avelang.bounded_packet_schedule",
                       rewriter.getStringAttr("consumer_point"));
    return true;
}

class BoundedPacketSchedulePass
    : public mlir::PassWrapper<BoundedPacketSchedulePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BoundedPacketSchedulePass)

    llvm::StringRef getArgument() const final {
        return "bounded-packet-schedule";
    }

    llvm::StringRef getDescription() const final {
        return "Issue bounded raw-buffer packets before independent MFMA work";
    }

    void runOnOperation() override {
        if (!useBoundedConsumerPointSchedule())
            return;
        llvm::SmallVector<mlir::ROCDL::RawBufferLoadOp> candidates;
        getOperation().walk([&](mlir::ROCDL::RawBufferLoadOp op) {
            candidates.push_back(op);
        });
        // This pass is registered on every function before intrinsic linking.
        // Most helper functions do not contain a raw packet and must remain
        // inert rather than turning the opt-in schedule into a module-wide
        // structural requirement.
        if (candidates.empty())
            return;
        mlir::PatternRewriter rewriter(&getContext());
        unsigned transformed = 0;
        for (auto raw : candidates) {
            if (!raw || scheduleOneBoundedPacket(raw, rewriter))
                ++transformed;
        }
        // The opt-in pass runs before intrinsic linking for every generated
        // GPU function.  A raw-buffer micro-kernel or unrelated helper may
        // have packet loads without the bounded producer/consumer shape; it
        // is intentionally outside this transform's domain.  The Z9S driver
        // verifies the one required transformation through its MLIR marker
        // and final ISA evidence.  More than one match would make the
        // schedule ambiguous and remains a hard failure.
        if (transformed == 0)
            return;
        if (transformed != 1) {
            getOperation().emitError()
                << "bounded packet schedule requires exactly one structurally "
                   "safe packet, found "
                << transformed;
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createBoundedPacketSchedulePass() {
    return std::make_unique<BoundedPacketSchedulePass>();
}

} // namespace causalflow::avelang::dialect
