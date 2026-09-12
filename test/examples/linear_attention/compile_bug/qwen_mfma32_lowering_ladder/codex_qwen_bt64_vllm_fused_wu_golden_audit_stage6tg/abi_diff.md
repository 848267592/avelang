# ABI Difference

The argument order and kernarg size differ. The decisive incompatibility is the solved matrix: native vLLM consumes BF16 `A`, F1 consumes FP32 `a_solved`. Same logical layout is not direct ABI compatibility.
