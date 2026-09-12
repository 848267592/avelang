#include "lower_qwen_gdn_recurrence_step_pass.h"

#include "Dialect/AveLang/IR/AveLangOps.h"
#include "IR/Intrinsics/intrinsic_support.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/raw_ostream.h>

namespace causalflow::avelang::dialect {
namespace {

constexpr int64_t kThreads = 128;
constexpr int64_t kBT = 64;
constexpr int64_t kBV = 32;

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

mlir::Value materializeStateVector(mlir::PatternRewriter &rewriter,
                                   mlir::Location loc, mlir::Value state) {
    if (mlir::isa<mlir::VectorType>(state.getType())) {
        return state;
    }
    auto type = mlir::dyn_cast<mlir::MemRefType>(state.getType());
    if (!type || type.getRank() != 1 ||
        type.getShape() != llvm::ArrayRef<int64_t>({16}) ||
        !type.getElementType().isF32()) {
        return {};
    }
    auto zero = indexConstant(rewriter, loc, 0);
    auto vectorType = mlir::VectorType::get({16}, rewriter.getF32Type());
    return mlir::vector::LoadOp::create(rewriter, loc, vectorType, state,
                                        mlir::ValueRange{zero});
}

mlir::Value extractBF16x4(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::Value vector8, int64_t offset) {
    auto resultType = mlir::VectorType::get({4}, rewriter.getBF16Type());
    llvm::SmallVector<mlir::Value> values;
    values.reserve(4);
    for (int64_t i = 0; i < 4; ++i) {
        values.push_back(mlir::vector::ExtractOp::create(rewriter, loc, vector8,
                                                         offset + i));
    }
    return mlir::vector::FromElementsOp::create(rewriter, loc, resultType,
                                                values);
}

mlir::Value emitMfmaFromPacked8(mlir::PatternRewriter &rewriter,
                                mlir::Location loc, mlir::Value aRaw,
                                mlir::Value bRaw, mlir::Value accumulator,
                                llvm::StringRef phase,
                                bool preserveSourceOperandOrder) {
    auto mfmaName = ir::intrinsics::MakeIntrinsicFuncName(
        "amdgpu", "rocdl_mfma_f32_32x32x8bf16_1k");
    for (int64_t fragmentOffset : {int64_t{0}, int64_t{4}}) {
        auto aFrag = extractBF16x4(rewriter, loc, aRaw, fragmentOffset);
        auto bFrag = extractBF16x4(rewriter, loc, bRaw, fragmentOffset);
        // The C0 block-dot update lowering uses the intrinsic's B/A matrix
        // convention. Pred is a direct replacement of B0's public
        // mfma(state, W, acc) source operation and must retain that order.
        llvm::SmallVector<mlir::Value> operands;
        if (preserveSourceOperandOrder) {
            operands = {aFrag, bFrag, accumulator};
        } else {
            operands = {bFrag, aFrag, accumulator};
        }
        auto call = mlir::func::CallOp::create(
            rewriter, loc, mfmaName, mlir::TypeRange{accumulator.getType()},
            operands);
        call->setAttr("avelang.qwen_gdn.stream32.mfma", rewriter.getUnitAttr());
        call->setAttr("avelang.qwen_gdn.stream32.phase",
                      mlir::StringAttr::get(rewriter.getContext(), phase));
        accumulator = call.getResult(0);
    }
    return accumulator;
}

void emitAuditIf(mlir::PatternRewriter &rewriter, mlir::Location loc,
                 mlir::Value emitAudit, llvm::function_ref<void()> body) {
    auto gate = mlir::scf::IfOp::create(rewriter, loc, emitAudit,
                                        /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&gate.getThenRegion().front());
    body();
    rewriter.setInsertionPointAfter(gate);
}

mlir::Value joinState(mlir::PatternRewriter &rewriter, mlir::Location loc,
                      mlir::Value low, mlir::Value high) {
    auto resultType = mlir::VectorType::get({32}, rewriter.getF32Type());
    llvm::SmallVector<mlir::Value> values;
    values.reserve(32);
    for (int64_t i = 0; i < 16; ++i) {
        values.push_back(
            mlir::vector::ExtractOp::create(rewriter, loc, low, i));
    }
    for (int64_t i = 0; i < 16; ++i) {
        values.push_back(
            mlir::vector::ExtractOp::create(rewriter, loc, high, i));
    }
    return mlir::vector::FromElementsOp::create(rewriter, loc, resultType,
                                                values);
}

mlir::Value scaleAndAdd(mlir::PatternRewriter &rewriter, mlir::Location loc,
                        mlir::Value state, mlir::Value delta,
                        mlir::Value gLastExp) {
    auto scale =
        mlir::vector::SplatOp::create(rewriter, loc, state.getType(), gLastExp);
    auto carried = mlir::arith::MulFOp::create(rewriter, loc, state, scale);
    return mlir::arith::AddFOp::create(rewriter, loc, carried, delta);
}

class Stream32RecurrenceStepLowering
    : public mlir::OpRewritePattern<AMDGPUQwenGdnRecurrenceStepBF16F32Op> {
  public:
    using mlir::OpRewritePattern<
        AMDGPUQwenGdnRecurrenceStepBF16F32Op>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(AMDGPUQwenGdnRecurrenceStepBF16F32Op op,
                    mlir::PatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto c0 = indexConstant(rewriter, loc, 0);
        auto c1 = indexConstant(rewriter, loc, 1);
        auto c2 = indexConstant(rewriter, loc, 2);
        auto c4 = indexConstant(rewriter, loc, 4);
        auto c8 = indexConstant(rewriter, loc, 8);
        auto c32 = indexConstant(rewriter, loc, kBV);
        auto c64 = indexConstant(rewriter, loc, kBT);
        auto c128 = indexConstant(rewriter, loc, kThreads);
        auto tid = toIndex(rewriter, loc, op.getThreadId());
        auto chunkIndex = toIndex(rewriter, loc, op.getChunkIndex());
        auto valueHead = toIndex(rewriter, loc, op.getValueHead());
        auto keyHead = toIndex(rewriter, loc, op.getKeyHead());
        auto valueBase = toIndex(rewriter, loc, op.getValueBase());
        auto wave = mlir::arith::DivUIOp::create(rewriter, loc, tid, c64);
        auto lane = mlir::arith::RemUIOp::create(rewriter, loc, tid, c64);
        auto laneRow = mlir::arith::RemUIOp::create(rewriter, loc, lane, c32);
        auto laneGroup = mlir::arith::DivUIOp::create(rewriter, loc, lane, c32);
        auto chunkStart = indexMul(rewriter, loc, chunkIndex, c64);
        auto finalToken = indexAdd(rewriter, loc, chunkStart,
                                   indexConstant(rewriter, loc, kBT - 1));
        auto gLast = mlir::memref::LoadOp::create(
            rewriter, loc, op.getSourceG(),
            mlir::ValueRange{c0, finalToken, valueHead});
        auto gLastExp = mlir::math::ExpOp::create(rewriter, loc, gLast);
        auto stateLow = materializeStateVector(rewriter, loc, op.getStateLow());
        auto stateHigh =
            materializeStateVector(rewriter, loc, op.getStateHigh());
        if (!stateLow || !stateHigh) {
            return rewriter.notifyMatchFailure(
                op, "persistent state was not vector/local FP32 [16]");
        }
        auto vec16 = mlir::cast<mlir::VectorType>(stateLow.getType());
        auto bf16x8 = mlir::VectorType::get({8}, rewriter.getBF16Type());

        // B0's persistent state is the only feedback carrier. H and stateStage
        // are BF16 snapshots for this chunk and are staged exactly once.
        for (int64_t accI = 0; accI < 16; ++accI) {
            auto localK = indexAdd(
                rewriter, loc,
                indexConstant(rewriter, loc, (accI >> 2) * 8 + (accI & 3)),
                indexMul(rewriter, loc, laneGroup, c4));
            auto globalV = indexAdd(rewriter, loc, valueBase, laneRow);
            auto low =
                mlir::vector::ExtractOp::create(rewriter, loc, stateLow, accI);
            auto high =
                mlir::vector::ExtractOp::create(rewriter, loc, stateHigh, accI);
            auto wave0 = mlir::arith::CmpIOp::create(
                rewriter, loc, mlir::arith::CmpIPredicate::eq, wave, c0);
            auto writeWave0 = mlir::scf::IfOp::create(rewriter, loc, wave0,
                                                      /*withElseRegion=*/false);
            rewriter.setInsertionPointToStart(
                &writeWave0.getThenRegion().front());
            mlir::memref::StoreOp::create(
                rewriter, loc,
                mlir::arith::TruncFOp::create(rewriter, loc,
                                              rewriter.getBF16Type(), low),
                op.getOutputH(),
                mlir::ValueRange{c0, chunkIndex, valueHead, globalV, localK});
            mlir::memref::StoreOp::create(
                rewriter, loc,
                mlir::arith::TruncFOp::create(rewriter, loc,
                                              rewriter.getBF16Type(), high),
                op.getOutputH(),
                mlir::ValueRange{c0, chunkIndex, valueHead, globalV,
                                 indexAdd(rewriter, loc, localK, c32)});
            mlir::memref::StoreOp::create(
                rewriter, loc,
                mlir::arith::TruncFOp::create(rewriter, loc,
                                              rewriter.getBF16Type(), low),
                op.getStateStage(), mlir::ValueRange{c0, laneRow, localK});
            mlir::memref::StoreOp::create(
                rewriter, loc,
                mlir::arith::TruncFOp::create(rewriter, loc,
                                              rewriter.getBF16Type(), high),
                op.getStateStage(),
                mlir::ValueRange{c0, laneRow,
                                 indexAdd(rewriter, loc, localK, c32)});
            rewriter.setInsertionPointAfter(writeWave0);

            auto wave1 = mlir::arith::CmpIOp::create(
                rewriter, loc, mlir::arith::CmpIPredicate::eq, wave, c1);
            auto writeWave1 = mlir::scf::IfOp::create(rewriter, loc, wave1,
                                                      /*withElseRegion=*/false);
            rewriter.setInsertionPointToStart(
                &writeWave1.getThenRegion().front());
            mlir::memref::StoreOp::create(
                rewriter, loc,
                mlir::arith::TruncFOp::create(rewriter, loc,
                                              rewriter.getBF16Type(), low),
                op.getOutputH(),
                mlir::ValueRange{c0, chunkIndex, valueHead, globalV,
                                 indexAdd(rewriter, loc, localK, c64)});
            mlir::memref::StoreOp::create(
                rewriter, loc,
                mlir::arith::TruncFOp::create(rewriter, loc,
                                              rewriter.getBF16Type(), high),
                op.getOutputH(),
                mlir::ValueRange{c0, chunkIndex, valueHead, globalV,
                                 indexAdd(rewriter, loc, localK,
                                          indexConstant(rewriter, loc, 96))});
            mlir::memref::StoreOp::create(
                rewriter, loc,
                mlir::arith::TruncFOp::create(rewriter, loc,
                                              rewriter.getBF16Type(), low),
                op.getStateStage(), mlir::ValueRange{c1, laneRow, localK});
            mlir::memref::StoreOp::create(
                rewriter, loc,
                mlir::arith::TruncFOp::create(rewriter, loc,
                                              rewriter.getBF16Type(), high),
                op.getStateStage(),
                mlir::ValueRange{c1, laneRow,
                                 indexAdd(rewriter, loc, localK, c32)});
            rewriter.setInsertionPointAfter(writeWave1);
        }
        mlir::gpu::BarrierOp::create(rewriter, loc);

        mlir::Value updateLow = makeZeroVector(rewriter, loc, vec16);
        mlir::Value updateHigh = makeZeroVector(rewriter, loc, vec16);

        // stream_t32: only this token-half's pred accumulator, partial planes
        // and V-decay tile are live. phaseStage is W here and K after C.
        for (int64_t tokenHalf = 0; tokenHalf < 2; ++tokenHalf) {
            auto tokenBase = indexConstant(rewriter, loc, tokenHalf * 32);
            auto stageW = mlir::scf::ForOp::create(rewriter, loc, c0, c32, c1);
            rewriter.setInsertionPointToStart(stageW.getBody());
            auto linear = indexAdd(
                rewriter, loc, tid,
                indexMul(rewriter, loc, stageW.getInductionVar(), c128));
            auto kHalf = mlir::arith::DivUIOp::create(
                rewriter, loc, linear, indexConstant(rewriter, loc, 32 * 64));
            auto rem = mlir::arith::RemUIOp::create(
                rewriter, loc, linear, indexConstant(rewriter, loc, 32 * 64));
            auto tokenOff =
                mlir::arith::DivUIOp::create(rewriter, loc, rem, c64);
            auto localK = mlir::arith::RemUIOp::create(rewriter, loc, rem, c64);
            auto sourceToken = indexAdd(
                rewriter, loc, indexAdd(rewriter, loc, chunkStart, tokenBase),
                tokenOff);
            auto sourceK = indexAdd(
                rewriter, loc, indexMul(rewriter, loc, kHalf, c64), localK);
            auto wValue = mlir::memref::LoadOp::create(
                rewriter, loc, op.getSourceW(),
                mlir::ValueRange{c0, sourceToken, valueHead, sourceK});
            auto wRow = indexAdd(rewriter, loc,
                                 indexMul(rewriter, loc, kHalf, c32), tokenOff);
            mlir::memref::StoreOp::create(rewriter, loc, wValue,
                                          op.getPhaseStage(),
                                          mlir::ValueRange{wRow, localK});
            rewriter.setInsertionPointAfter(stageW);
            mlir::gpu::BarrierOp::create(rewriter, loc);

            mlir::Value predAcc = makeZeroVector(rewriter, loc, vec16);
            auto phaseWRow = indexAdd(
                rewriter, loc, indexMul(rewriter, loc, wave, c32), laneRow);
            for (int64_t kPack = 0; kPack < 4; ++kPack) {
                auto word = indexAdd(rewriter, loc,
                                     indexConstant(rewriter, loc, kPack * 2),
                                     laneGroup);
                auto elementOffset = indexMul(rewriter, loc, word, c8);
                auto stateRaw = mlir::vector::LoadOp::create(
                    rewriter, loc, bf16x8, op.getStateStage(),
                    mlir::ValueRange{wave, laneRow, elementOffset});
                auto wRaw = mlir::vector::LoadOp::create(
                    rewriter, loc, bf16x8, op.getPhaseStage(),
                    mlir::ValueRange{phaseWRow, elementOffset});
                predAcc = emitMfmaFromPacked8(rewriter, loc, stateRaw, wRaw,
                                              predAcc, "pred", true);
            }
            for (int64_t accI = 0; accI < 16; ++accI) {
                auto outCol = indexAdd(
                    rewriter, loc,
                    indexConstant(rewriter, loc, (accI >> 2) * 8 + (accI & 3)),
                    indexMul(rewriter, loc, laneGroup, c4));
                auto value = mlir::vector::ExtractOp::create(rewriter, loc,
                                                             predAcc, accI);
                mlir::memref::StoreOp::create(
                    rewriter, loc, value, op.getPredPartial(),
                    mlir::ValueRange{wave, laneRow, outCol});
            }
            mlir::gpu::BarrierOp::create(rewriter, loc);

            auto correct = mlir::scf::ForOp::create(
                rewriter, loc, c0, indexConstant(rewriter, loc, 8), c1);
            rewriter.setInsertionPointToStart(correct.getBody());
            auto outputLinear = indexAdd(
                rewriter, loc, tid,
                indexMul(rewriter, loc, correct.getInductionVar(), c128));
            auto outputToken =
                mlir::arith::DivUIOp::create(rewriter, loc, outputLinear, c32);
            auto localV =
                mlir::arith::RemUIOp::create(rewriter, loc, outputLinear, c32);
            auto token = indexAdd(
                rewriter, loc, indexAdd(rewriter, loc, chunkStart, tokenBase),
                outputToken);
            auto globalV = indexAdd(rewriter, loc, valueBase, localV);
            auto partial0 = mlir::memref::LoadOp::create(
                rewriter, loc, op.getPredPartial(),
                mlir::ValueRange{c0, outputToken, localV});
            auto partial1 = mlir::memref::LoadOp::create(
                rewriter, loc, op.getPredPartial(),
                mlir::ValueRange{c1, outputToken, localV});
            auto pred =
                mlir::arith::AddFOp::create(rewriter, loc, partial0, partial1);
            auto uBf16 = mlir::memref::LoadOp::create(
                rewriter, loc, op.getSourceU(),
                mlir::ValueRange{c0, token, valueHead, globalV});
            auto uF32 = mlir::arith::ExtFOp::create(
                rewriter, loc, rewriter.getF32Type(), uBf16,
                mlir::arith::FastMathFlagsAttr{});
            auto corrected =
                mlir::arith::SubFOp::create(rewriter, loc, uF32, pred);
            auto vNew = mlir::arith::TruncFOp::create(
                rewriter, loc, rewriter.getBF16Type(), corrected);
            auto vNewF32 = mlir::arith::ExtFOp::create(
                rewriter, loc, rewriter.getF32Type(), vNew,
                mlir::arith::FastMathFlagsAttr{});
            auto gValue = mlir::memref::LoadOp::create(
                rewriter, loc, op.getSourceG(),
                mlir::ValueRange{c0, token, valueHead});
            auto decay = mlir::math::ExpOp::create(
                rewriter, loc,
                mlir::arith::SubFOp::create(rewriter, loc, gLast, gValue));
            auto vDecay = mlir::arith::TruncFOp::create(
                rewriter, loc, rewriter.getBF16Type(),
                mlir::arith::MulFOp::create(rewriter, loc, vNewF32, decay));
            mlir::memref::StoreOp::create(
                rewriter, loc, vNew, op.getOutputVNew(),
                mlir::ValueRange{c0, token, valueHead, globalV});
            mlir::memref::StoreOp::create(
                rewriter, loc, vDecay, op.getVdecayStage(),
                mlir::ValueRange{c0, localV, outputToken});
            emitAuditIf(rewriter, loc, op.getEmitAudit(), [&] {
                mlir::memref::StoreOp::create(
                    rewriter, loc, pred, op.getAuditPredF32(),
                    mlir::ValueRange{c0, token, valueHead, globalV});
                mlir::memref::StoreOp::create(
                    rewriter, loc,
                    mlir::arith::TruncFOp::create(rewriter, loc,
                                                  rewriter.getBF16Type(), pred),
                    op.getAuditPredBF16(),
                    mlir::ValueRange{c0, token, valueHead, globalV});
                mlir::memref::StoreOp::create(
                    rewriter, loc, vDecay, op.getAuditVDecay(),
                    mlir::ValueRange{c0, token, valueHead, globalV});
            });
            rewriter.setInsertionPointAfter(correct);

            // The K-stage barrier below dominates every V-decay consumer and
            // is reached by every work-item after correction. It therefore
            // simultaneously closes V-decay production and opens the K phase;
            // adding a separate barrier here would only split the same phase.
            // The typed BF16x8 K producer and the consumer-friendly
            // [K,token] LDS view are the C0 persistent-block mechanics
            // restricted to this T32.
            for (int64_t kHalfLiteral = 0; kHalfLiteral < 2; ++kHalfLiteral) {
                auto kHalfValue = indexConstant(rewriter, loc, kHalfLiteral);
                auto bToken =
                    mlir::arith::DivUIOp::create(rewriter, loc, tid, c4);
                auto rowGroup =
                    mlir::arith::RemUIOp::create(rewriter, loc, tid, c4);
                auto rowBase = indexMul(rewriter, loc, rowGroup, c8);
                for (int64_t colHalf = 0; colHalf < 2; ++colHalf) {
                    auto kColumn =
                        indexAdd(rewriter, loc,
                                 indexAdd(rewriter, loc,
                                          indexConstant(rewriter, loc,
                                                        kHalfLiteral * 64 +
                                                            colHalf * 32),
                                          rowBase),
                                 c0);
                    auto staged = mlir::vector::LoadOp::create(
                        rewriter, loc, bf16x8, op.getSourceK(),
                        mlir::ValueRange{
                            c0,
                            indexAdd(
                                rewriter, loc,
                                indexAdd(rewriter, loc, chunkStart, tokenBase),
                                bToken),
                            keyHead, kColumn});
                    for (int64_t element = 0; element < 8; ++element) {
                        auto value = mlir::vector::ExtractOp::create(
                            rewriter, loc, staged, element);
                        auto row = indexAdd(
                            rewriter, loc,
                            indexAdd(rewriter, loc,
                                     indexConstant(rewriter, loc, colHalf * 32),
                                     rowBase),
                            indexConstant(rewriter, loc, element));
                        mlir::memref::StoreOp::create(
                            rewriter, loc, value, op.getPhaseStage(),
                            mlir::ValueRange{row, bToken});
                    }
                }
                mlir::gpu::BarrierOp::create(rewriter, loc);

                auto active = mlir::arith::CmpIOp::create(
                    rewriter, loc, mlir::arith::CmpIPredicate::eq, wave,
                    kHalfValue);
                for (int64_t colHalf = 0; colHalf < 2; ++colHalf) {
                    auto current = colHalf == 0 ? updateLow : updateHigh;
                    auto guarded = mlir::scf::IfOp::create(
                        rewriter, loc, mlir::TypeRange{vec16}, active,
                        /*withElseRegion=*/true);
                    rewriter.setInsertionPointToStart(
                        &guarded.getThenRegion().front());
                    auto updated = current;
                    auto kRow = indexAdd(
                        rewriter, loc,
                        indexConstant(rewriter, loc, colHalf * 32), laneRow);
                    llvm::SmallVector<mlir::Value> words{
                        laneGroup.getResult(),
                        indexAdd(rewriter, loc, laneGroup, c2),
                    };
                    for (mlir::Value word : words) {
                        auto elementOffset = indexMul(rewriter, loc, word, c8);
                        auto aRaw = mlir::vector::LoadOp::create(
                            rewriter, loc, bf16x8, op.getVdecayStage(),
                            mlir::ValueRange{c0, laneRow, elementOffset});
                        auto bRaw = mlir::vector::LoadOp::create(
                            rewriter, loc, bf16x8, op.getPhaseStage(),
                            mlir::ValueRange{kRow, elementOffset});
                        updated = emitMfmaFromPacked8(rewriter, loc, aRaw, bRaw,
                                                      updated, "update", false);
                    }
                    mlir::scf::YieldOp::create(rewriter, loc, updated);
                    rewriter.setInsertionPointToStart(
                        &guarded.getElseRegion().front());
                    mlir::scf::YieldOp::create(rewriter, loc, current);
                    rewriter.setInsertionPointAfter(guarded);
                    if (colHalf == 0) {
                        updateLow = guarded.getResult(0);
                    } else {
                        updateHigh = guarded.getResult(0);
                    }
                }
                mlir::gpu::BarrierOp::create(rewriter, loc);
            }
        }

        auto nextLow =
            scaleAndAdd(rewriter, loc, stateLow, updateLow, gLastExp);
        auto nextHigh =
            scaleAndAdd(rewriter, loc, stateHigh, updateHigh, gLastExp);
        emitAuditIf(rewriter, loc, op.getEmitAudit(), [&] {
            for (int64_t accI = 0; accI < 16; ++accI) {
                auto localK = indexAdd(
                    rewriter, loc,
                    indexConstant(rewriter, loc, (accI >> 2) * 8 + (accI & 3)),
                    indexMul(rewriter, loc, laneGroup, c4));
                auto globalV = indexAdd(rewriter, loc, valueBase, laneRow);
                auto low = mlir::vector::ExtractOp::create(rewriter, loc,
                                                           nextLow, accI);
                auto high = mlir::vector::ExtractOp::create(rewriter, loc,
                                                            nextHigh, accI);
                auto wave0 = mlir::arith::CmpIOp::create(
                    rewriter, loc, mlir::arith::CmpIPredicate::eq, wave, c0);
                auto writeWave0 = mlir::scf::IfOp::create(
                    rewriter, loc, wave0, /*withElseRegion=*/false);
                rewriter.setInsertionPointToStart(
                    &writeWave0.getThenRegion().front());
                mlir::memref::StoreOp::create(
                    rewriter, loc, low, op.getAuditStateAfter(),
                    mlir::ValueRange{c0, chunkIndex, valueHead, globalV,
                                     localK});
                mlir::memref::StoreOp::create(
                    rewriter, loc, high, op.getAuditStateAfter(),
                    mlir::ValueRange{c0, chunkIndex, valueHead, globalV,
                                     indexAdd(rewriter, loc, localK, c32)});
                rewriter.setInsertionPointAfter(writeWave0);
                auto wave1 = mlir::arith::CmpIOp::create(
                    rewriter, loc, mlir::arith::CmpIPredicate::eq, wave, c1);
                auto writeWave1 = mlir::scf::IfOp::create(
                    rewriter, loc, wave1, /*withElseRegion=*/false);
                rewriter.setInsertionPointToStart(
                    &writeWave1.getThenRegion().front());
                mlir::memref::StoreOp::create(
                    rewriter, loc, low, op.getAuditStateAfter(),
                    mlir::ValueRange{c0, chunkIndex, valueHead, globalV,
                                     indexAdd(rewriter, loc, localK, c64)});
                mlir::memref::StoreOp::create(
                    rewriter, loc, high, op.getAuditStateAfter(),
                    mlir::ValueRange{
                        c0, chunkIndex, valueHead, globalV,
                        indexAdd(rewriter, loc, localK,
                                 indexConstant(rewriter, loc, 96))});
                rewriter.setInsertionPointAfter(writeWave1);
            }
        });

        auto result = joinState(rewriter, loc, nextLow, nextHigh);
        if (auto *def = result.getDefiningOp()) {
            def->setAttr("avelang.qwen_gdn.recurrence_step.stream_t32",
                         rewriter.getUnitAttr());
        }
        rewriter.replaceOp(op, result);
        return mlir::success();
    }
};

class LowerQwenGdnRecurrenceStepPass
    : public mlir::PassWrapper<LowerQwenGdnRecurrenceStepPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerQwenGdnRecurrenceStepPass)

    llvm::StringRef getArgument() const final {
        return "lower-qwen-gdn-recurrence-step";
    }

    llvm::StringRef getDescription() const final {
        return "Lower experimental Qwen Direct-K64 stream32 recurrence step";
    }

    void runOnOperation() override {
        bool hasStep = false;
        getOperation().walk(
            [&](AMDGPUQwenGdnRecurrenceStepBF16F32Op) { hasStep = true; });
        if (!hasStep) {
            return;
        }
        auto mode = llvm::sys::Process::GetEnv(
                        "AVELANG_QWEN_GDN_RECURRENCE_STEP_LOWERING")
                        .value_or("stream_t32");
        if (mode != "stream_t32") {
            getOperation().emitError() << "AVELANG_QWEN_GDN_RECURRENCE_STEP_"
                                          "LOWERING must be stream_t32, got "
                                       << mode;
            signalPassFailure();
            return;
        }
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<Stream32RecurrenceStepLowering>(&getContext());
        if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                     std::move(patterns)))) {
            signalPassFailure();
            return;
        }
        bool remaining = false;
        getOperation().walk([&](AMDGPUQwenGdnRecurrenceStepBF16F32Op op) {
            op.emitError("qwen_gdn recurrence step survived late lowering");
            remaining = true;
        });
        if (remaining) {
            signalPassFailure();
            return;
        }
        getOperation()->setAttr("avelang.qwen_gdn.recurrence_step.lowering",
                                mlir::StringAttr::get(&getContext(), mode));
        llvm::errs() << "[qwen-gdn-recurrence-step] mode=" << mode
                     << " lowered before block-dot/intrinsic linking\n";
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerQwenGdnRecurrenceStepPass() {
    return std::make_unique<LowerQwenGdnRecurrenceStepPass>();
}

} // namespace causalflow::avelang::dialect
