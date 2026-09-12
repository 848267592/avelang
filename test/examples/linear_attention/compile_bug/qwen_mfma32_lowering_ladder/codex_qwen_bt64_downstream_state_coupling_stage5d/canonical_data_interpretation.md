# Canonical Data Control

canonical tensor 是 v18 输出的 bitwise 固定副本。A/B 各自执行 solve，但丢弃
结果，下游读取同一个 canonical data_ptr。

| T | after v18 ms | after v1 ms | v1 tail penalty |
|--:|--:|--:|--:|
| 2048 | 0.267518 | 0.332073 | 64.555 us |
| 8192 | 0.991954 | 1.005834 | 13.880 us |

T=2048 无 solve predecessor 为 0.334737 ms，小 dummy predecessor 为 0.343390
ms，均接近 v1 而不是 v18。故微小 solve 数值差不是 tail 差距主因；较长 v18
predecessor 产生了某种可被下游利用的执行状态。

真实 solved A/B 在 T=2048 的 max/mean abs 为 `2.98e-8/3.81e-10`，1048576
个元素中 317998 个 bitwise 不同；两边无 NaN、Inf 或 subnormal。完整分布在
`solved_data_statistics.csv`。
