# IR Evidence

No separate MLIR/LLVM dump was required for this high-level-only Stage 4 gate.
The W and U HSACOs contain MFMA16, have zero private segment/spills, and the
rocprof data captures the final machine-resource behavior.
