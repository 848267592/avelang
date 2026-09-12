# Stage 5E Final Summary

已完成真正 direct-common-out：v18/v1 直接写同一 caller-owned pointer，无
copy-to-canonical。Correctness 与 53 项合并回归通过。

决定性结果是 T=2048 tail `0.267237/0.331152 ms`，v1 penalty
`64.035 us`；Stage 5D 为 `64.255 us`。因此 pointer effect 被拒绝，分类
CASE B。warm 和 512 MiB perturb 将差距压平，但 TCC/TCP counters 近乎一致，
粗粒度 telemetry 也没有稳定 clock/power 分叉，确切硬件机制仍未分离。

唯一下一步是更低扰动的 whole-graph cache/dispatch counter 审计。无需修改 W/U、
asm、compiler 或 production。

