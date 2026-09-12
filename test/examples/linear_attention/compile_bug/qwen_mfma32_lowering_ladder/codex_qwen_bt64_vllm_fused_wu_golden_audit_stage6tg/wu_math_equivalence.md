# W/U Math Equivalence

F1 consumes FP32 `a_solved`, materializes a BF16 main coefficient and a BF16 residual coefficient, and runs two MFMA16 passes per W/U contribution. Actual `wy_fast.py` consumes BF16 `A`, forms BF16 operands, and has a single `tl.dot` per 64-column block with no residual dot. The public outputs are tolerance-equivalent, but the intermediate precision contracts are not bitwise equivalent.
