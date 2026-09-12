# IR Evidence

No separate MLIR/LLVM dump was required for this high-level-only Stage 4 gate.
The retained HSACO, disassembly, rocprof counters, and zero-spill code-object
metadata prove the KKT kernel lowered to MFMA16 without scalar fallback.
