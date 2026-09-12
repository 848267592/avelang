# Stage 5E Cache/Execution-State 控制

T=2048 direct-common tail 的 v1-v18 penalty：

| 控制 | penalty us | 5-session 范围 us |
|:--|--:|--:|
| none | 62.613 | [61.972, 63.374] |
| warm 一次相同 tail | 0.021 | [-0.021, 0.080] |
| 相同 512 MiB perturbation | -0.020 | [-0.080, 0.020] |
| reduction prime | 80.280 | [79.839, 81.841] |

T=8192 的对应值为 `10.556/-0.060/-0.261/6.670 us`。warm 和大工作集
perturbation 继续压平差距，而 reduction prime 没有。这支持“前驱诱发的瞬态
执行状态”类别，但不能宣称 512 MiB 操作精确清空了某一级 cache。

本轮 TCC/TCP counter 中，下游各 kernel 的 `TCP_TOTAL_CACHE_ACCESSES_sum`
A/B 完全一致；TCC hit/miss/read-request 最大相对差为 U 的 `TCC_HIT_sum`
`-0.232%`，其余大多小于 `0.12%`。counter 没有给出与 64 us 相称的稳定缓存
流量差异，所以具体机制仍未定位到某一级缓存。

