#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace causalflow::avelang::dialect {

// Lowers the experimental direct-K64 block_dot_bf16_f32 op after
// AveLang-to-memref conversion and before intrinsic implementation linking.
// The lowering selection is controlled only by
// AVELANG_BLOCK_DOT_LOWERING=generic|specialized.
std::unique_ptr<mlir::Pass> createLowerQwenBlockDotPass();

// Materializes the internal block-dot MFMA operand plan after GPU outlining,
// so its packed shared/fragment identity survives the ordinary block-dot
// lowering boundary without adding a source-facing operation.
std::unique_ptr<mlir::Pass> createLowerQwenBlockDotMfmaOperandPass();

} // namespace causalflow::avelang::dialect
