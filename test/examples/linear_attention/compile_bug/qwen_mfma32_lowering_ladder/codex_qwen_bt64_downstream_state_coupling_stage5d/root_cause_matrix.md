# Stage 5D Root-Cause Matrix

| candidate | evidence | opposing evidence | classification | confidence |
|:--|:--|:--|:--|:--|
| solved 数值 | canonical bitwise 固定后仍慢 64.6 us | 无 | 排除主因 | high |
| downstream pointer/alignment | canonical data_ptr 相同；可见低位均对齐 | direct same solve-out 未跑 | 不是已证实主因 | medium-high |
| allocator pool | fixed-buffer full 改变收益传递 | canonical fixed tail 仍有差距 | 次要/未关闭 | medium |
| L2/cache residency | warm tail 和 512 MiB perturb 均把差压到 <0.3 us | reduction prime 不消除；无 cache counter | 支持推断 | medium |
| 其它 cache | 同上 | 无分层 counter | 未区分 | low-medium |
| dispatch gap | 无内部 event tail 仍复现 | 无 timeline | 未测 | N/A |
| predecessor resource state | v18 后快；v1/no-solve/dummy 后慢 | 无精确硬件状态 | 直接操作性根因 | high |
| clock/power ramp | 长 workload 可归一化，短 predecessor 不行 | 无 telemetry，prime 结果复杂 | 支持假设 | medium-low |
| per-stage event | 对 full A/B 差值扰动 14.2 us | 单 tail event 仍复现 64.3 us | 非主因 | high |
| rocprof instrumentation | 尚未运行 | N/A | 未测 | N/A |
| hidden dispatch | frozen fixed graph 均 8 dispatch | 无新 timeline | 反对 | high |
| downstream code object | 同 symbol/constexpr/进程 JIT cache，asm hash 相同 | 新 binary hash capture 未跑 | 反对 | high |
| unresolved interaction | cache/clock/solve-store pointer 尚未完全分离 | 多项控制已缩小范围 | 仍存在 | high |

Measured fact：v1 predecessor 后的首个连续 tail 在 T=2048 慢约 64 us；一致
warm/perturb predecessor 后差距消失。Supported inference：瞬态 cache/执行状态是
主要机制。Unresolved hypothesis：具体是 solve output store 的 cache-set、GPU
frequency/power ramp，还是二者共同作用。
