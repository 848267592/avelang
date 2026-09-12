# Stage 6V 实验复盘

## 一句话复盘

这轮证明了一个很具体的事实：**C0 的四段 lane-group predicated MFMA16 可以在
Avelang source 层折叠成一次 wave-uniform MFMA16，动态 MFMA 也确实从
1024/CTA 降到了 256/CTA。** 但这并不自动等于整图更快，因为 selection chain
把 VGPR 提高了，W/U 又只占全图的一部分。最后完整 U1 图的收益小到不能通过
T=2048 的稳定性 gate，所以没有晋级。

## 为什么想到这个实验

Stage 6U 已经把 C0 的主矩阵从残差版本收敛成 BF16 solved 的 fused W/U。它仍有
一个明显的 source pattern：四个 `lane_group` 分支各自调用同一个 MFMA16 intrinsic。
静态 ISA 和 PMC 表明这对应 1024 MFMA/CTA，而 native vLLM 的同逻辑只有 128。
Stage 6V 不试图一次追平 native；只验证最小的 4 倍 collapse 是否真的能发生。

## 实际改动和第一次失败

第一次把四个 fragment 写入 `a_operand/b_operand`，再在 if 后调用一次 MFMA。数值
错误。原因不是 MFMA layout，而是 Avelang 的 statement `if` 会生成带隔离 symbol
scope 的 `scf.if`；分支赋值不会自动合流成 if 后可用的 SSA value。因此所有 lane 都
读取到默认 group-0 operand。

第二次改为 Python 条件表达式。Avelang 已有 `IfExp -> arith.select` lowering，所以
每一个 operand 都变成真正的 per-lane select，MFMA 保持在 select 之后的单一位置。
这次数值通过。

## 得到什么

| 指标 | C0 | V0 | 含义 |
|:--|--:|--:|:--|
| 静态 MFMA16 | 64 | 16 | 成功四倍 collapse |
| 动态 MFMA/CTA | 1024 | 256 | 硬件计数符合预期 |
| scratch/spill | 0/0 | 0/0 | 没有资源灾难 |
| VGPR | 68 | 100 | select 有明显寄存器代价 |
| profiler trace | 41.642 us | 38.598 us | isolated 有正收益 |
| T=2048 full aggregate | 0.364962 ms | 0.361397 ms | 只有约 3.6 us |

V0 相比 C0 有极少量 BF16 最低位差，最大 `9.7656e-04`，低于冻结 `1e-3` W/U
gate 和完整图阈值；V1 的输出/final-state 均继续通过原有 vLLM public contract。

## 为什么不继续

更严格的 T=2048 nine-session ABBA 确认里，V1 aggregate 只快 `1.722 us`，paired
bootstrap 95% CI 是 `[-2.003, 43.375] us`，包含零。这里继续改 C0 很容易把测量
噪声当优化收益。正确决定是保留实验和证据，停止这条小优化线。

完整数据、命令、HSACO 和 rocprof CSV 在
`codex_qwen_bt64_predicate_collapse_stage6v/`；正式结论见
`qwen_gfx942_bt64_predicate_collapse_stage6v_report.md`。
