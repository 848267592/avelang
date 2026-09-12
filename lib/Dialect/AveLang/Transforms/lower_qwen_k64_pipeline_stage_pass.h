#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace causalflow::avelang::dialect {

/// Lowers the experimental opaque Qwen K64 pipeline-stage token after GPU
/// outlining. The pass selects an immediate or distributed packet placement
/// through AVELANG_QWEN_K64_PIPELINE_LOWERING.
std::unique_ptr<mlir::Pass> createLowerQwenK64PipelineStagePass();

} // namespace causalflow::avelang::dialect
