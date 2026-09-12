# Qwen Direct-K64 BV32 B1 后续决策

## 决策

关闭 `compiler-owned stream_t32` recurrence-step 路线，不创建第二个 token-size、LDS
layout、barrier 或 RA candidate。B1 完整正确且显著降低寄存器/LDS，但 full-sequence
body 在 T=512/1024/2048/8192 全部慢于 B0：T=2048 为 `0.908771 ms` 对 `0.678028 ms`，
即 `1.340x` 慢；slope 从 `21.778` 恶化到 `28.655 us/chunk`。

## 已确认

- B1 与 B0、P2 host microscope 在 T=64/128/512/2048 的 H、raw pred、pred BF16、
  V-new、V-decay、state-after 与 final-state 均 byte-exact。
- B1 final code object：VGPR `332`、AGPR `76`、LDS `26,624 B`、scratch/spill `0`；
  B0 body 是 VGPR `460`、AGPR `204`、LDS `36,864 B`。
- B1 精确 LTO MIR 在 greedy、virtregrewriter 与 prolog/epilog 均无 spill。
- B1/B0 MFMA 与 VMEM 动态总数相同；B1 甚至减少 VALU/LDS，但仍稳定变慢。

## 含义

这不是一个 RA 或 LDS capacity 问题，而是这个细粒度 `P -> C -> U` schedule 在完整
recurrence 内的吞吐回退。当前证据不足以把回退唯一归因到某条机器指令，但足以否定
“仅通过 token32 phase slicing 即可回收 B0 长序列 gap”的假设。

不将 `qwen_gdn_recurrence_step_bf16_f32` 整理为正式 compiler PR；它保留在实验树作为
negative evidence。B0 仍是 Avelang-native full recurrence correctness baseline，
current-vLLM external HSACO bridge 仍是 matched-ABI 的较快 control。

完整证据见：

- `compile_bug/qwen_mfma32_lowering_ladder/qwen_direct_k64_bv32_stream32_b1_report.md`
