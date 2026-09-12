#include "lower_qwen_kfrag_lds_pass.h"

#include "Dialect/AveLang/IR/AveLangOps.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace causalflow::avelang::dialect {
namespace {

class QwenKFragLDSLoadLowering
    : public mlir::OpRewritePattern<AMDGPUQwenUpdateKFragLDSLoadOp> {
  public:
    using mlir::OpRewritePattern<AMDGPUQwenUpdateKFragLDSLoadOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(AMDGPUQwenUpdateKFragLDSLoadOp op,
                    mlir::PatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto i64 = rewriter.getI64Type();
        // compact[kk, token] is BF16 with row stride 64. Keep the late
        // lowering in element units, matching the generic vector.load's GEP
        // semantics exactly. The dedicated op still avoids the upstream
        // memref/vector indexing chain; this only removes an unsafe
        // byte-address/inttoptr reinterpretation from the experiment.
        auto base = mlir::arith::IndexCastOp::create(
            rewriter, loc, i64, op.getSharedBase());
        auto k = mlir::arith::IndexCastOp::create(
            rewriter, loc, i64, op.getKColumn());
        auto token = mlir::arith::IndexCastOp::create(
            rewriter, loc, i64, op.getTokenFragmentLocalBase());
        auto c64 = mlir::arith::ConstantIntOp::create(rewriter, loc, 64, 64);
        auto elements = mlir::arith::AddIOp::create(
            rewriter, loc,
            mlir::arith::MulIOp::create(rewriter, loc, k, c64), token);
        auto ldsPtrType = mlir::LLVM::LLVMPointerType::get(
            rewriter.getContext(), /*addressSpace=*/3);
        auto ldsBasePtr = mlir::LLVM::IntToPtrOp::create(
            rewriter, loc, ldsPtrType, base,
            mlir::LLVM::DereferenceableAttr{});
        auto ldsPtr = mlir::LLVM::GEPOp::create(
            rewriter, loc, ldsPtrType, rewriter.getBF16Type(), ldsBasePtr,
            mlir::ValueRange{elements});
        auto resultType = mlir::cast<mlir::VectorType>(op.getResult().getType());
        auto load = mlir::LLVM::LoadOp::create(
            rewriter, loc, resultType, ldsPtr,
            /*alignment=*/2, /*volatile=*/false, /*nontemporal=*/false);
        load->setAttr("avelang.qwen_kfrag.direct_lds_b64",
                      rewriter.getUnitAttr());
        rewriter.replaceOp(op, load.getResult());
        return mlir::success();
    }
};

// Expand the producer-side source-K load only after GPU outlining.  The
// generated operation is deliberately the same scalar BF16 memref.load used
// by the generic branch.  The experimental variable is its construction
// point: this prevents the full 64-bit global-address tuple from becoming a
// long-lived value while pred MFMA work is scheduled.
class QwenKFragStageLoadLowering
    : public mlir::OpRewritePattern<AMDGPUQwenKFragStageLoadOp> {
  public:
    using mlir::OpRewritePattern<AMDGPUQwenKFragStageLoadOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(AMDGPUQwenKFragStageLoadOp op,
                    mlir::PatternRewriter &rewriter) const override {
        auto zero = mlir::arith::ConstantIndexOp::create(rewriter, op.getLoc(),
                                                          0);
        auto load = mlir::memref::LoadOp::create(
            rewriter, op.getLoc(), op.getSourceK(),
            mlir::ValueRange{zero, op.getSourceToken(), op.getKeyHead(),
                             op.getKColumn()});
        load->setAttr("avelang.qwen_kfrag.late_address",
                      rewriter.getUnitAttr());
        rewriter.replaceOp(op, load.getResult());
        return mlir::success();
    }
};

class LowerQwenKFragLDSPass
    : public mlir::PassWrapper<LowerQwenKFragLDSPass,
                               mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerQwenKFragLDSPass)

    llvm::StringRef getArgument() const final {
        return "lower-qwen-kfrag-lds";
    }

    llvm::StringRef getDescription() const final {
        return "Lower persistent Qwen K fragments and delayed stage loads";
    }

    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<QwenKFragLDSLoadLowering, QwenKFragStageLoadLowering>(
            &getContext());
        if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                     std::move(patterns)))) {
            signalPassFailure();
        }
        bool remaining = false;
        getOperation().walk([&](AMDGPUQwenUpdateKFragLDSLoadOp op) {
            op.emitError("persistent Qwen B-fragment survived late lowering");
            remaining = true;
        });
        getOperation().walk([&](AMDGPUQwenKFragStageLoadOp op) {
            op.emitError("persistent Qwen source-K stage load survived late lowering");
            remaining = true;
        });
        if (remaining) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerQwenKFragLDSPass() {
    return std::make_unique<LowerQwenKFragLDSPass>();
}

} // namespace causalflow::avelang::dialect
