# Stage 5E 根因决策

分类为 **CASE B**。

`v18` 与 `hierarchical_fp32_v1` 已直接写同一个 caller-provided pointer，solve
与 W 之间没有 copy/fill/allocation/额外 dispatch，但 T=2048 tail penalty 仍为
`64.035 us`，与 Stage 5D 的 `64.255 us` 只差 `0.220 us`。五个 allocation、
ABAB/BABA/ABBA 均复现。

因此 solve-store output pointer/address interaction 被拒绝为主因。当前最准确的
根因类别仍是 **predecessor-induced transient downstream execution state**。
warm/perturb 可压平差距，但本轮 TCC/TCP counters 和粗粒度 telemetry 尚不能把
它唯一细分为 cache、clock 或 queue/runtime state。

