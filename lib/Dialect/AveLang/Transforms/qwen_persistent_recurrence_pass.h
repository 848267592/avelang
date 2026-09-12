#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace causalflow::avelang::dialect {

// Forms a first-class region op around the exact B0 device-side recurrence
// source body. The frontend emits begin/end delimiters so the formation is
// explicit and restricted to experimental R0 kernels.
std::unique_ptr<mlir::Pass> createFormQwenPersistentRecurrencePass();

// Validates and attaches one whole-recurrence target plan before any
// recurrence or block-dot lowering. R1's joint_v1 plan is intentionally a
// single plan for pred, the BF16 boundary, update, and feedback.
std::unique_ptr<mlir::Pass> createPlanQwenPersistentRecurrencePass();

// Lowers the semantic region using the intentionally non-optimizing
// `legacy_b0` mode. It inlines the validated B0 body before block-dot
// lowering; future schedule planning can replace this boundary without
// changing high-level recurrence semantics.
std::unique_ptr<mlir::Pass> createLowerQwenPersistentRecurrencePass();

} // namespace causalflow::avelang::dialect
