# LDS Plan

共享存储为 `coeff[64,16]`、`operand0[16,16]`、`operand1[16,16]`，共 3072 B。每个 source 16-token tile 重用同一 LDS region；F0/F1 profiler 均报告 LDS block 3072 B。
