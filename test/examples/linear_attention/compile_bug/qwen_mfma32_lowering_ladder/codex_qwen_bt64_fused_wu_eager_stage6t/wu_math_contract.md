# W/U Math Contract

`W[t,:]=(a_solved[t,:]*beta[:]*exp(g[:])) @ K[:]`，`U[t,:]=(a_solved[t,:]*beta[:]) @ V[:]`。F0 与 F1 均先以 BF16 primary tile 和 BF16 residual tile 执行 MFMA，并在 FP32 accumulator 中累积。唯一差异是 F0 最后写 FP32，F1 最后 numeric-convert 后写 BF16。
