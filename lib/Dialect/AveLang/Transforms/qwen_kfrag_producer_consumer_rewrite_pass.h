#pragma once

#include <memory>
#include <mlir/Pass/Pass.h>

namespace causalflow::avelang::dialect {

/// Rewrite the opt-in Qwen update K-fragment producer/consumer pair from a
/// broad [128,64] shared tile to a compact [128,16] physical staging tile.
std::unique_ptr<mlir::Pass> createQwenKFragProducerConsumerRewritePass();

} // namespace causalflow::avelang::dialect
