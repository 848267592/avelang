# Stage 6T Eager Public API Contract

所有权威 correctness 和性能结论来自一次完整 public API eager 调用。输入在计时前创建；public wrapper 内的 allocation、cast、dispatch、返回对象均处于计时区间。每个样本由 HIP event 包围并在当前 stream 同步。wall-clock 使用同一次调用补充记录。

禁止的 capture/replay 机制未被使用；内部 kernel body 和 rocprof trace 只用于资源诊断。
