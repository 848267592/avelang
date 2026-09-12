# Stage 5E 与 Stage 5C 对照

不同 harness 的绝对时间不能混合比较，下面只比较各自 A/B 差值。

| 指标，T=2048 | v18 ms | hierarchical v1 ms | v1 收益 |
|:--|--:|--:|--:|
| Stage 5C public full | 0.474385 | 0.455577 | 18.808 us |
| Stage 5E direct-common solve+tail | 0.398271 | 0.361016 | 37.296 us |
| Stage 5E direct-common full | 0.450670 | 0.403620 | 46.370 us |

时间列是各实现 session median 的中位数，收益列是 paired session delta 的中位数，
因此两者不要求严格相减。Stage 5E 的固定预分配、单 event full harness 传递出更多 solve 收益，但这不能
倒推出 production 收益，因为 Stage 5C 是另一套 public harness。更重要的因果
比较是 Stage 5D 与 Stage 5E 的 tail：`64.255 us` 对 `64.035 us`，只减少
`0.220 us`。因此 direct common output pointer 没有修复 downstream coupling。
