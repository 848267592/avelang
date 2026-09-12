#include "qwen_kfrag_producer_consumer_rewrite_pass.h"

#include "Dialect/AveLang/IR/AveLangOps.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/raw_ostream.h>

#include <optional>

namespace causalflow::avelang::dialect {

namespace {

constexpr int64_t kSourceTokens = 64;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kThreads = 128;

mlir::Value toIndex(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value value) {
    if (value.getType().isIndex()) {
        return value;
    }
    return mlir::arith::IndexCastOp::create(builder, loc,
                                            builder.getIndexType(), value);
}

mlir::Value peelIndexCasts(mlir::Value value) {
    while (auto cast = mlir::dyn_cast_or_null<mlir::arith::IndexCastOp>(
               value.getDefiningOp())) {
        value = cast.getIn();
    }
    return value;
}

mlir::Value materializeScalarAt(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value value,
                                int64_t *clonedScalarReloads) {
    value = peelIndexCasts(value);
    if (auto load = mlir::dyn_cast_or_null<mlir::memref::LoadOp>(
            value.getDefiningOp())) {
        // Scalar lexical values are spilled to private memrefs and reloaded
        // in each branch. Clone that reload at the staging point instead of
        // using a value defined later in a consumer branch.
        ++*clonedScalarReloads;
        return mlir::memref::LoadOp::create(builder, loc, load.getMemRef(),
                                            load.getIndices());
    }
    return value;
}

unsigned enclosingForDepth(mlir::Operation *op) {
    unsigned depth = 0;
    for (auto *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
        if (mlir::isa<mlir::scf::ForOp>(parent)) {
            ++depth;
        }
    }
    return depth;
}

bool hasStaticShape(mlir::Value value, llvm::ArrayRef<int64_t> shape,
                    mlir::Type elementType) {
    auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
    return type && type.getShape() == shape &&
           type.getElementType() == elementType;
}

bool hasQwenSourceShape(mlir::Value value, mlir::Type elementType) {
    auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
    auto shape = type ? type.getShape() : llvm::ArrayRef<int64_t>();
    return type && shape.size() == 4 && shape[0] == 1 && shape[1] >= 64 &&
           shape[2] == 4 && shape[3] == 128 &&
           type.getElementType() == elementType;
}

bool isMfma16BConsumer(AMDGPUQwenUpdateKFragLoadOp op) {
    if (!op.getResult().hasOneUse()) {
        return false;
    }
    auto *user = *op.getResult().getUsers().begin();
    auto call = mlir::dyn_cast<mlir::func::CallOp>(user);
    if (!call || call.getNumOperands() < 2 ||
        call.getOperand(1) != op.getResult()) {
        return false;
    }
    return call.getCallee().contains("mfma_f32_16x16x16bf16_1k");
}

mlir::scf::ForOp findProducerLoop(AMDGPUQwenUpdateKFragLoadOp op,
                                  mlir::memref::StoreOp &producerStore) {
    for (auto *user : op.getSharedK().getUsers()) {
        auto store = mlir::dyn_cast<mlir::memref::StoreOp>(user);
        if (!store) {
            continue;
        }
        auto loop = store->getParentOfType<mlir::scf::ForOp>();
        if (!loop) {
            continue;
        }
        if (producerStore) {
            return {};
        }
        producerStore = store;
    }
    return producerStore ? producerStore->getParentOfType<mlir::scf::ForOp>()
                         : mlir::scf::ForOp{};
}

bool validateProducerLoop(mlir::scf::ForOp loop,
                          mlir::memref::StoreOp producerStore,
                          AMDGPUQwenUpdateKFragLoadOp op) {
    auto lower = mlir::getConstantIntValue(loop.getLowerBound());
    auto upper = mlir::getConstantIntValue(loop.getUpperBound());
    auto step = mlir::getConstantIntValue(loop.getStep());
    if (!lower || !upper || !step || *lower != 0 || *upper != 64 ||
        *step != 1 || producerStore.getIndices().size() != 2) {
        return false;
    }

    int targetStoreCount = 0;
    int sourceLoadCount = 0;
    loop.walk([&](mlir::Operation *nested) {
        if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(nested)) {
            if (store.getMemRef() == op.getSharedK()) {
                ++targetStoreCount;
            }
            return;
        }
        if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(nested)) {
            // Function-argument lowering may introduce a view/reinterpret cast
            // between the persistent source operand and this producer load.
            // Keep the guard structural rather than requiring SSA identity.
            if (hasQwenSourceShape(
                    load.getMemRef(),
                    mlir::BFloat16Type::get(loop.getContext()))) {
                ++sourceLoadCount;
            }
        }
    });
    return targetStoreCount == 1 && sourceLoadCount == 1;
}

void eraseDeadSharedChain(mlir::Value value) {
    while (auto *def = value.getDefiningOp()) {
        if (!value.use_empty()) {
            return;
        }
        mlir::Value next;
        if (auto view = mlir::dyn_cast<mlir::memref::ViewOp>(def)) {
            next = view.getSource();
        } else if (auto cast =
                       mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(def)) {
            next = cast.getSource();
        } else if (!mlir::isa<mlir::memref::AllocaOp>(def)) {
            return;
        }
        def->erase();
        if (!next) {
            return;
        }
        value = next;
    }
}

class QwenKFragProducerConsumerRewritePass
    : public mlir::PassWrapper<QwenKFragProducerConsumerRewritePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
        QwenKFragProducerConsumerRewritePass)

    llvm::StringRef getArgument() const final {
        return "qwen-kfrag-producer-consumer-rewrite";
    }

    llvm::StringRef getDescription() const final {
        return "Rewrite guarded Qwen BF16 update K-fragment staging";
    }

    void runOnOperation() override {
        auto module = getOperation();
        const bool debug =
            llvm::sys::Process::GetEnv("AVELANG_QWEN_KFRAG_DEBUG").has_value();
        const bool useLateBFrag =
            llvm::sys::Process::GetEnv("AVELANG_QWEN_KFRAG_LATE_BLOAD") ==
            std::optional<std::string>("1");
        // Keep source-K address formation opaque until after GPU outlining.
        // This is intentionally independent of the terminal B-fragment
        // experiment above: it changes only the producer-side global address
        // lifetime, while leaving the four update MFMA16 consumer loads on
        // their existing path.
        const bool useLateAddress =
            llvm::sys::Process::GetEnv("AVELANG_QWEN_KFRAG_LATE_ADDRESS") ==
            std::optional<std::string>("1");
        llvm::SmallVector<AMDGPUQwenUpdateKFragLoadOp> pending;
        module.walk(
            [&](AMDGPUQwenUpdateKFragLoadOp op) { pending.push_back(op); });
        if (pending.empty()) {
            return;
        }

        if (debug) {
            llvm::errs() << "[qwen-kfrag] persistent_ops_seen="
                         << pending.size() << "\n";
        }

        llvm::DenseSet<mlir::Operation *> handledConsumers;
        int64_t rewriteCount = 0;
        int64_t matchedProducerLoops = 0;
        int64_t erasedBroadProducerStores = 0;
        int64_t compactProducerLoops = 0;
        int64_t clonedScalarReloads = 0;
        int64_t lateAddressStageLoads = 0;
        for (auto seed : pending) {
            if (!seed || handledConsumers.contains(seed.getOperation())) {
                continue;
            }
            auto bf16 = mlir::BFloat16Type::get(module.getContext());
            if (!hasStaticShape(seed.getSharedK(), {kHeadDim, kSourceTokens},
                                bf16) ||
                !hasQwenSourceShape(seed.getSourceK(), bf16) ||
                !isMfma16BConsumer(seed)) {
                seed.emitError("Qwen K-fragment rewrite guard failed");
                signalPassFailure();
                return;
            }

            mlir::memref::StoreOp producerStore;
            auto producerLoop = findProducerLoop(seed, producerStore);
            if (!producerLoop ||
                !validateProducerLoop(producerLoop, producerStore, seed)) {
                seed.emitError(
                    "failed to match exact [128,64] K producer loop");
                signalPassFailure();
                return;
            }
            ++matchedProducerLoops;

            llvm::SmallVector<AMDGPUQwenUpdateKFragLoadOp> consumers;
            for (auto candidate : pending) {
                if (candidate && candidate.getSharedK() == seed.getSharedK()) {
                    mlir::memref::StoreOp candidateStore;
                    if (findProducerLoop(candidate, candidateStore) ==
                        producerLoop) {
                        consumers.push_back(candidate);
                    }
                }
            }
            if (consumers.size() != 4) {
                seed.emitError(
                    "expected exactly four MFMA16 fragment consumers");
                signalPassFailure();
                return;
            }
            auto oldShared = seed.getSharedK();
            auto directOuterFor =
                producerLoop.getOperation()->getParentOfType<mlir::scf::ForOp>();
            auto directOuterUpper =
                directOuterFor
                    ? mlir::getConstantIntValue(directOuterFor.getUpperBound())
                    : std::optional<int64_t>{};
            if (debug) {
                llvm::errs()
                    << "[qwen-kfrag] match consumers=" << consumers.size()
                    << " producer_for_depth="
                    << enclosingForDepth(producerLoop.getOperation())
                    << " stage_alloc_for_depth="
                    << enclosingForDepth(producerLoop.getOperation())
                    << " direct_outer_for_upper="
                    << (directOuterUpper ? *directOuterUpper : -1)
                    << " hoist_compact_alloca=0"
                    << " old_shared_def="
                    << oldShared.getDefiningOp()->getName().getStringRef()
                    << "\n";
            }
            // Diagnostic full-width tile: preserve the original token
            // coordinate system while testing the dedicated producer/
            // consumer lowering.  This isolates fragment lowering from any
            // compact-tile offset issue; shrink the physical tile only after
            // this full-width path is semantically proven.
            int64_t compactTokens = 64;
            for (auto consumer : consumers) {
                // The memref/index type converter can materialize separate,
                // equivalent casts for each helper call.  The shared producer
                // and the exact producer loop are the stable identity here.
                if (!isMfma16BConsumer(consumer)) {
                    consumer.emitError(
                        "fragment consumer is not an MFMA16 B operand");
                    signalPassFailure();
                    return;
                }
            }

            mlir::OpBuilder builder(producerLoop);
            auto loc = producerLoop.getLoc();
            auto sharedType =
                mlir::cast<mlir::MemRefType>(seed.getSharedK().getType());
            auto compactType = mlir::MemRefType::get(
                {kHeadDim, compactTokens}, bf16,
                mlir::MemRefLayoutAttrInterface(), sharedType.getMemorySpace());
            auto compact = mlir::memref::AllocaOp::create(
                builder, loc, compactType, mlir::ValueRange{},
                mlir::IntegerAttr{});
            // Keep the workgroup base as a scalar index. The late lowering
            // turns this plus fixed Qwen coordinates into one 8-byte LDS load.
            auto compactBase = mlir::memref::ExtractAlignedPointerAsIndexOp::create(
                builder, loc, builder.getIndexType(), compact);

            auto c0 = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
            auto c1 = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
            auto cTokens = mlir::arith::ConstantIndexOp::create(
                builder, loc, compactTokens);
            auto stageLoop =
                mlir::scf::ForOp::create(builder, loc, c0, cTokens, c1);
            builder.setInsertionPointToStart(stageLoop.getBody());
            auto iv = stageLoop.getInductionVar();
            auto tid =
                toIndex(builder, loc,
                        materializeScalarAt(builder, loc, seed.getThreadId(),
                                            &clonedScalarReloads));
            auto keyHead =
                toIndex(builder, loc,
                        materializeScalarAt(builder, loc, seed.getKeyHead(),
                                            &clonedScalarReloads));
            auto window = toIndex(
                builder, loc,
                materializeScalarAt(builder, loc, seed.getTokenWindowBase(),
                                    &clonedScalarReloads));
            auto cThreads =
                mlir::arith::ConstantIndexOp::create(builder, loc, kThreads);
            auto linear = mlir::arith::AddIOp::create(
                builder, loc, tid,
                mlir::arith::MulIOp::create(builder, loc, iv, cThreads));
            auto kk = mlir::arith::DivUIOp::create(builder, loc, linear,
                                                   cTokens);
            auto tokenLocal =
                mlir::arith::RemUIOp::create(builder, loc, linear, cTokens);
            auto sourceToken =
                mlir::arith::AddIOp::create(builder, loc, window, tokenLocal);
            mlir::Value value;
            if (useLateAddress) {
                auto lateStageLoad = AMDGPUQwenKFragStageLoadOp::create(
                    builder, loc, bf16, seed.getSourceK(), sourceToken,
                    keyHead, kk);
                value = lateStageLoad.getResult();
                ++lateAddressStageLoads;
            } else {
                value = mlir::memref::LoadOp::create(
                    builder, loc, seed.getSourceK(),
                    mlir::ValueRange{c0, sourceToken, keyHead, kk});
            }
            mlir::memref::StoreOp::create(builder, loc, value, compact,
                                          mlir::ValueRange{kk, tokenLocal});

            for (auto consumer : consumers) {
                handledConsumers.insert(consumer.getOperation());
                mlir::OpBuilder loadBuilder(consumer);
                auto kColumn = toIndex(loadBuilder, consumer.getLoc(),
                                       consumer.getKColumn());
                auto fragmentBase = toIndex(loadBuilder, consumer.getLoc(),
                                            consumer.getTokenFragmentBase());
                auto consumerWindow = toIndex(loadBuilder, consumer.getLoc(),
                                              consumer.getTokenWindowBase());
                auto localBase =
                    mlir::arith::SubIOp::create(loadBuilder, consumer.getLoc(),
                                                fragmentBase, consumerWindow);
                if (useLateBFrag) {
                    auto late = AMDGPUQwenUpdateKFragLDSLoadOp::create(
                        loadBuilder, consumer.getLoc(),
                        consumer.getResult().getType(), compactBase, kColumn,
                        localBase);
                    consumer.replaceAllUsesWith(late.getResult());
                } else {
                    auto load = mlir::vector::LoadOp::create(
                        loadBuilder, consumer.getLoc(),
                        consumer.getResult().getType(), compact,
                        mlir::ValueRange{kColumn, localBase});
                    consumer.replaceAllUsesWith(load.getResult());
                }
                consumer.erase();
            }

            producerLoop.erase();
            ++erasedBroadProducerStores;
            ++compactProducerLoops;
            eraseDeadSharedChain(oldShared);
            ++rewriteCount;
        }

        bool hasUnrewritten = false;
        module.walk([&](AMDGPUQwenUpdateKFragLoadOp op) {
            op.emitError("persistent Qwen K-fragment op survived rewrite pass");
            hasUnrewritten = true;
        });
        if (hasUnrewritten) {
            signalPassFailure();
            return;
        }
        module->setAttr(
            "avelang.qwen_kfrag_rewrite_count",
            mlir::IntegerAttr::get(
                mlir::IntegerType::get(module.getContext(), 64), rewriteCount));
        module.emitRemark()
            << "Qwen K-fragment producer-consumer rewrites: " << rewriteCount;
        if (debug) {
            llvm::errs()
                << "[qwen-kfrag] rewritten=" << rewriteCount
                << " producer_loops=" << matchedProducerLoops
                << " broad_producer_stores_erased=" << erasedBroadProducerStores
                << " compact_producer_loops=" << compactProducerLoops
                << " cloned_scalar_reloads=" << clonedScalarReloads
                << " late_bfrag=" << (useLateBFrag ? 1 : 0)
                << " late_address=" << (useLateAddress ? 1 : 0)
                << " late_address_stage_loads=" << lateAddressStageLoads
                << " unrewritten=0\n";
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createQwenKFragProducerConsumerRewritePass() {
    return std::make_unique<QwenKFragProducerConsumerRewritePass>();
}

} // namespace causalflow::avelang::dialect
