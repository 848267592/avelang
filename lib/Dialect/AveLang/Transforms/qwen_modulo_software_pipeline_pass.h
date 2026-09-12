#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace causalflow::avelang::dialect {

/// Rewrites an annotated recurrence loop into a modulo software pipeline.
///
/// The pass deliberately knows only the producer/commit stage-token contract;
/// the recurrence arithmetic between those operations remains opaque.  For a
/// distance-one schedule it generates a token prologue, a steady scf.for with
/// loop-carried packet recipes, and a peeled epilogue.  This is the AveLang
/// counterpart of Triton's schedule/expander split, not a Qwen ISA sequence.
std::unique_ptr<mlir::Pass> createQwenModuloSoftwarePipelinePass();

} // namespace causalflow::avelang::dialect
