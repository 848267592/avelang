# Stage 5E 跨 Allocation 分析

- 每个 `(operation,T)` 使用 5 个 caller-owned output allocation。
- 共执行 80 个 allocation session，观察到 35 个不同的实际虚拟地址；ROCm
  caching allocator 合法复用了其余地址。
- 决定性的 T=2048 tail 五个 session 差距为
  `64.035/63.314/64.175/64.336/63.254 us`。
- 每个 session 内 A/B 的 shape、stride、storage offset、dtype、size、alignment
  和 exact `data_ptr` 完全相同。
- contract 样例指针为 `0x7f78e9600000`，大小 4 MiB，对 16/64/128/256 B、
  4 KiB 和 64 KiB 均整除。

因此约 64 us 差距不依赖某一个偶然 common pointer。该实验排除了“两个 solve
写入不同物理 allocation/虚拟地址”作为主要解释，但没有排除前驱 kernel 对缓存、
频率或队列执行状态的影响。

