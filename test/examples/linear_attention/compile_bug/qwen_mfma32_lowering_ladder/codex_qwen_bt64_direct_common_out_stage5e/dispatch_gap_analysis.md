# Stage 5E rocprof Dispatch-Gap 分析

已执行 T=2048 direct-common full graph 和 T=2048/8192 tail 的 targeted
rocprof。下游 W/U/asm/chunk-o/cast 的动态 VALU/SALU/MFMA/VMEM/LDS counts 在
A/B 间一致。

rocprof counter collection 下，T=2048 tail 的 dispatch gap 中位数约为：

- v18：`59.65--61.39 us`；
- hierarchical v1：`90.67--94.26 us`。

这个 30 us 级差异确实出现在工具 trace，但不能作为原生 graph gap 结论：counter
collection 在每个 dispatch 间引入了远大于正常图的工具开销；A/B 还是两个独立
进程，并且 solve 之前本应相同的 cumsum/KKT trace 也漂移了 2.8--3.2 us。
因此标记为“观测到差异，但被 profiler 扰动污染”，不能据此修改 runtime 或 kernel。

