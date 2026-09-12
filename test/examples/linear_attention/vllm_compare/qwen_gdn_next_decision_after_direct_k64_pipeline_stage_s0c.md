# Qwen Direct-K64 S0-C 后的决策

## 决策

**关闭 current-S0 K64 pipeline-stage 路线。** 不实施 S1，不把它接入 B0/full
recurrence，也不继续做 store-width、LDS-layout、barrier 或 RA 枚举。

S0-C 的 Triton mapping audit 已经完成且映射唯一：4096 个 native K64 BF16 element
覆盖 rotating LDS 的 4096 个位置，并由 4096 个 MFMA B-operand word 完整、无重复地
消费。然而该 mapping 也严格证明，它不能投影到当前 S0 的 frozen source packet ABI：

```text
current S0 packet: fixed token x K[8 contiguous]
native packet:     fixed K x token[8 contiguous]
```

每个 native packet 需要八个不同 S0 lanes 的值。只允许 lane-local repack、禁止 cross-lane
exchange、禁止新 swizzle 且保持 C0 consumer 不变时，不存在合法的
`triton_packet_commit` lowering。因此没有创建虚假的 A/B、没有跑不具同源意义的性能数据。

S0 已经证明 early-load placement 可行但性能中性；S0-C 又证明最后的 scalar commit
不能被单独替换为 native packet commit。当前阶段的合理结论是：native Triton 受益于
load ownership、transpose、rotating LDS 和 dot fragment consumer 的整体协同，不能把它
当成一个 isolated store-width compiler patch。

完整证据见：

- `compile_bug/qwen_mfma32_lowering_ladder/qwen_direct_k64_pipeline_stage_s0c_report.md`
- `rocprof_outputs/qwen_direct_k64_pipeline_stage_s0c/audit/`
