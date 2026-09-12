#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace causalflow::avelang::dialect {

/// Lower the guarded Qwen update B-fragment op after GPU outlining. This is
/// intentionally AMDGPU-specific and emits a single addrspace(3) vector load.
std::unique_ptr<mlir::Pass> createLowerQwenKFragLDSPass();

} // namespace causalflow::avelang::dialect
