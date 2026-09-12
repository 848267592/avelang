# Stage 5F 唯一建议

唯一建议选择 **C：继续 whole-graph cache/dispatch counter 审计**，但下一轮应使用
能在单进程内交替 A/B、减少 rocprof dispatch instrumentation 扰动的方法，并优先
寻找更细的 TCC/TCP/CU 分布或硬件 trace。不要修改 W/U、asm、compiler，不要加
dummy warmup，不做 solve->W/U fusion。

理由：direct common pointer 已被否定；cache counters 的总量近乎一致，但
warm/perturb 的稳定行为说明瞬态状态仍值得审计。现有 `amd-smi` telemetry 太粗，
所以 D 不是下一步主动作。预期可恢复收益仍为 `N/A`；64 us 只是现象上界，不是
可承诺收益。

