#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace causalflow::avelang::dialect {

// Lowers the experimental compiler-owned BT64 stream32 Qwen recurrence step
// after AveLang-to-memref conversion and before block-dot/intrinsic linking.
std::unique_ptr<mlir::Pass> createLowerQwenGdnRecurrenceStepPass();

} // namespace causalflow::avelang::dialect
