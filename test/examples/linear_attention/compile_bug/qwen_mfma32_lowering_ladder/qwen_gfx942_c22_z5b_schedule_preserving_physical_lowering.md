# Qwen gfx942 C22: Z5B Schedule-Preserving Native Physical Lowering

## 结论

**`STOP_C22_SCHEDULE_LAYOUT_INCOMPATIBLE`。** C22 没有进入性能阶段，也不
晋级。失败不是 C13-C16 静态 physical-layout 基础设施的失败，而是将其 selected
native fragment ownership 直接嵌入 Z5B 的既有 phase schedule 时发生了可复现的
lane-to-fragment 数值错配。

在 T=64、`V-new=0`、caller-owned 输出预填 NaN 的最小完整 chunk-o 测试中：

| 输入 | finite | 对 Z5B BF16 byte-exact | max abs |
|:--|:--:|:--:|--:|
| random Q/H/K，zero-V | 是 | 否 | `0.0054473876953125` |
| structured Q/H/K，zero-V | 是 | 否 | `52.0` |

zero-V 排除了 score@V / V producer 对这个首错的影响；错误已经位于 Q/H/K
MFMA bridge。按预注册规则，任何一个 T=64 correctness failure 都禁止 T=2048、
4096、8192、16384 的性能、PMC、slope 或 native 排名测试。因此这些字段均是
`N/A_correctness_gate_failed`，没有用旧 Z5B 或 C13-C16 的数值补写。

## 目标与冻结

C22 从 Z5B 分叉，保留 WG256、BT64/BV64/BK32、2 CTA/chunk-head、16 KiB dedicated
Q cache、Q global producer once、A -> B half0 -> B half1 -> C 的 source order、
K32 reduction order、BF16 ABI、caller-owned output 和 phase-separated
`inter_acc` / `score_acc` lifetime。没有启用 C19/C21 的 full-region owner、
superloop、pipeline、bank rotation 或 allocator/RA 路线。

唯一候选通过
`AVELANG_STAGE6Z_SCHEDULE_PRESERVING_PHYSICAL=c22` 启用。它复用同一个
`al.amdgpu.block_dot_bf16_f32_operand`，没有新增 Qwen 专用 op：

- Q：保留 global -> dedicated Q cache，只把 cache consumer 改为 C16 Q shared/dot
  recipe。
- H：仍在 Z5B Phase A 的原 producer/barrier 位置写 shared；consumer 改为 C16
  `shared1` recipe。
- K：仍按 source-half 0 再 source-half 1、每 half 四个 K32 stage；producer/consumer
  尝试 C16 `shared2` feature-token recipe。
- V：仍在 Phase C score commit 后写入；consumer 尝试 C15 rotating shared recipe。

`static_physical_layout_test` 的 8/8 与
`c14_static_physical_codegen_test` 的 1/1 都通过，证明 C13/C14 属性、映射验证和
target codegen 没有被 C22 前端接入破坏。

## 为什么不能组合

C16 real-tile closure是正确的：它的 `[64,32]` Q/H tile 由四个 wave 以 native
blocked2 ownership 共同拥有，mapping 中的 wave 维度随 logical row 变化。Z5B 的
已有 MFMA schedule 则让两组 value-half wave 复用同一组 Q logical row，并在
phase-separated accumulator lifetime 下形成各自的 output fragment。

两者的 shared element mapping 都可以单独正确，但它们的 **MFMA operand fragment
ownership 并不相同**。C22 保持 Z5B `word0/word1` consumer iteration，同时仅把
LDS address 替换为 native physical address，无法保持 native C16 所需的
wave/lane/register fragment选择。若改成 C16 的完整 ownership，则会改变 Z5B的
consumer schedule、phase topology 或 accumulator live range，违反 C22 的硬冻结。

因此这不是“再调一个 XOR 或 packet width”能够解决的错误。它明确证明：

```text
C13-C16 physical bridge 正确
&& Z5B schedule 正确
!= 两者可以 schedule-preserving 地直接组合
```

## 实现与证据

新增实验源：
`vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_c22_z5b_schedule_preserving_physical.py`。
它的 source loop/barrier 骨架与 Z5B一致；C22只改 Q/K/V shared physical coordinate
与 late `AMDGPUBlockDotMfmaOperandOp` consumer。前端还增加了一个 C22-only V
operand admission，使 Phase-C 的 BF16 V-new 使用同一个 existing block-dot op，而
没有获得 C18/C19 full-region ownership。

关键修复尝试也记录在源码中：C16 K 的 logical coordinate 是
`[feature, token]`，不是 Z5B producer loop 的 `[token, feature]`。将 K shared2
地址修正为 `token * 32 + feature` 后，T=64 仍保持完全相同的错误幅度。这排除了
简单 K row/column 反置是唯一根因，进一步支持 fragment ownership incompatibility。

本轮构建成功，运行时绑定也已确认同步到 source-tree Python path；此前的旧 binding
导致 C22 V frontend gate 未生效，这一环境问题已修正，最终失败发生在真实 GPU
执行的数值比较，而不是未加载新编译器。

## 未执行项目

下列项目是有意未执行，不是遗漏：

- C22 T=2048/T8192 PMC 与 resource capture；
- 7-session fresh-process formal body benchmarks；
- T4096/T8192/T16384 long-text tests 与 slope fit；
- native selected kernel comparison；
- X2 integration、selector 或 production 变更；
- C23、packet sweep、barrier sweep、pipeline/superloop、RA tuning。

如果未来需要再次研究 native physical layout，必须将它作为一个**新的 schedule
candidate**，先显式说明并验证 ownership/lifetime 改变；不能把它伪装成 Z5B
schedule-preserving bridge。Z5B 继续是唯一 isolated Stage 6Z performance baseline。

机器可读 closure 在同目录的 `stage6z_c22_*.json`。所有 `N/A` 都明确标记了
correctness gate，避免把不可比的历史性能带入 C22。
