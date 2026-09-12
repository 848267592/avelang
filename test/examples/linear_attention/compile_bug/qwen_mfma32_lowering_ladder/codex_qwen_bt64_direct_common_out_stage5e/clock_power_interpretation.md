# Stage 5E 时钟/功耗观测解释

使用 `amd-smi metric -g 0 --json` 对 T=8192 A-only、B-only、ABBA、warm 和
perturb 循环进行只读采样。A-only/B-only none 的 XCP gfx clock 中位数分别约
`2008/1988 MHz`，socket power 中位数 `187/185 W`；二者完整范围高度重叠，
且采样包含进入/退出循环时的低频点。warm 的对应 clock 为 `1994/1982 MHz`，
也没有稳定分叉。

perturb 两边都观察到个别 PPT violation 状态；这是大工作集控制自身的背景，不能
归因到 solve。`amd-smi` 的亚秒级采样无法解析微秒级单个 kernel，也没有证据证明
64 us penalty 来自时钟或功耗。结论是：clock/power 仍是候选，但本轮 telemetry
只排除了明显、持续的 A/B 频率或功耗分叉。

