# Qwen Direct-K64 S0 后的决策

## 决策

关闭 S0 的扩展链：**不实施 S1、S2、S3，也不把 S0 接回 B0 full recurrence。**

S0 的 compiler-owned K64 stage 已达到机器结构目标：same-source 分叉保留到 ISA，
next-K vector load 在 current MFMA 前发射，LDS overwrite 在 current consumer 后发生，
并且保持同一份 MFMA/VMEM/LDS 工作、`scratch=0` 与 `spill=0`。但 15 个
fresh-process session 的 latency 无统计上可区分的收益：`0.024917 ms` 对
`0.024917 ms`，paired 95% CI 跨零。

这说明“避免 B2 local-array spill”是必要条件但不是充分条件。一个 K64 tile 的单 bank
lookahead 没有足够计算窗口隐藏它的 global load；把它机械放大成两个 K half 或完整
W/K/V pipeline 会违背本实验的逐级性能 gate。

## 保留的成果

- `amdgpu_qwen_k64_pipeline_stage_load/commit` 提供了 opaque、compiler-owned 的
  distributed BF16x8 packet stage。
- late pass 在 GPU outlining 后才 materialize vector global loads 和 LDS commit，避开
  source `k_next[64]` 的长 live range。
- 该表示在小 repro 中与 immediate arm byte-exact，保留 `global_load_dwordx4`，且
  code object 为 `private=0`、VGPR/SGPR spill 均为零。

它是一个有效的 experimental compiler capability，不是性能 baseline。

## 不做的事情

- 不恢复 B2 local-array lookahead；
- 不改 allocator/RA；
- 不改 MFMA geometry、BV32 ownership、C0 consumer 或 LDS swizzle；
- 不做 S1 K0+K1、S2 W、S3 V/U/g 或 B3 full-recurrence 接入；
- 不把中性 isolated 结果宣传为 Triton-matched pipeline 加速。

## 未来重启的前置证据

只有在新的 Triton machine audit 能证明以下三点时，才值得另立实验：

1. native 有比 S0 明显更宽的 next-tile outstanding-load window；
2. 该窗口有确定的 waitcnt、lane packet transpose 和 typed dot-fragment 对应关系；
3. 这些额外机制能在一个新的最小 same-work repro 中形成正收益，再考虑接入
   full recurrence。

否则应把精力放在已有正确且已验证收益的图级路径，而不是继续放大这个局部 lookahead。

详细证据见
`compile_bug/qwen_mfma32_lowering_ladder/qwen_direct_k64_pipeline_stage_s0_report.md`。
