# Direct-K64 Full Recurrence P0-P2 后的决策（历史 pre-fix）

> 此文件记录修复前 P2 的 stop condition。该 stop 已由 BF16 V-new boundary
> repair 解除；最新状态见
> [post-fix 决策](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_next_decision_after_direct_k64_update_bf16_vnew_fix.md)。

## 当前结论

停止 Direct-K64 full recurrence 的性能路线。P0 修复并证实了 nonzero-W pred
MFMA32 lane/fragment mapping；P1 证实单 BT64 chunk 的 pred、V-new、V-decay 和
C0 update 组合正确，且 HSACO/MIR 显示 scratch 与 spill 都为零。

但 P2 feedback ladder 在 `T=2048` 的 chunk 23 首次出现
`pred_f32=5.206061e-05 > 5e-05`，因此不能宣称全递归 correct。该错误发生在
feedback 后，而非 P0 原始 one-hot mapping；最终 state 误差只有 `2.279105e-04`
且没有非有限值，但这不允许跳过冻结的严格 pred gate。

## 已完成

| 门槛 | 状态 |
|:--|:--|
| P0 非零 W one-hot / scan / sparse / permutation | 通过 |
| P0 4096-row empirical lane mapping | 通过，unresolved=0 |
| P1 T=64 fused composition | 六个用例全部通过 |
| P1 scratch / MIR spill 审计 | private segment=0，VGPR/SGPR/MIR spill=0 |
| P2 T=128 | 通过 |
| P2 T=512 | 通过 |
| P2 T=2048 | chunk 23 严格 pred gate 失败，停止 |

完整证据见 [P0-P2 中文报告](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_direct_k64_full_recurrence_p0_p2_correctness_report.md)。

## P3 已完成

P3 已在同一 P1 kernel、同一 `T=2048/chunk=23` 的 W/U/K/g 下完成三臂 state
source decomposition：

- A actual state 复现 `pred_f32=5.206061e-05` trajectory error；
- B reference state 恢复到 `1.862645e-09`；
- C `BF16(actual)` 与 A pred bit-exact；
- actual/reference BF16 state 有 `68,018 / 131,072` 个 bitwise 不同元素。

因此 pred mapping/lowering 不是当前根因，错误种子在此前 update/state path。完整
证据见 [P3 报告](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_direct_k64_feedback_source_decomposition_p3_report.md)。

## 唯一允许的下一步

做一个没有性能计时的 update seed decomposition，从 chunk 0 开始按顺序比较：BF16
V-new、BF16 V-decay、消费相同 BF16 V-decay/K 的 MFMA delta、FP32 state scale、
最终 state add。必须报告第一处错误中间量；在这之前不做 phase-lifetime A/B、不跑
benchmark、不改 RA、LDS layout、MFMA geometry、CTA ownership 或 production 路径。
