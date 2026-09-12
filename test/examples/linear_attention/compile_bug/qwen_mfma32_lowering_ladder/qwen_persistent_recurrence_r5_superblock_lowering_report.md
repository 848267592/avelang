# Qwen Persistent Recurrence R5 Full-Recurrence Superblock Lowering

## 结论

**R5 correctness pass、机器图确实改变，但 performance No-Go。**

`gfx942_bt64_bv32_joint_v5` 保持了完整 nonzero-W recurrence 语义，且
在 MLIR、LLVM、MIR、ISA 与 HSACO 上都不是 R4 的别名。不过，它没有减少
R4 的动态机器工作，反而在五个独立 fresh-process session 中对 R4 稳定变慢。
R4 继续是 native full-recurrence baseline；R5 仅保留为一个反例和 compiler
能力证据，不接入 production selector。

R5 否定的不是“pred/update 联合规划”这个长期方向，而是更窄的假设：

```text
把 next K stage issue 从 pred 后提前到 pred 前
```

在当前 `stage_load -> i64 token -> tail commit` 实现中，不能自动变成有用的
async global-to-LDS pipeline。stage token 背后仍是由寄存器承载的 BF16x8 payload；
提前 issue 只会延长 packet 到 tail commit 的 live range，未建立 Triton 风格的
独立 copy/wait 机制或更宽的有效隐藏窗口。

## 范围与冻结项

本轮没有修改 production dispatch、external HSACO、allocator/RA、MFMA geometry、
BV32 ownership、K fragment layout、LDS swizzle、lifetime marker、prefetch 双缓冲或
数学定义。R5 继承 R4 的：

- gfx942、`BT=64`、`BV=32`、`WG=128`、32 CTA、two-wave cooperative ownership；
- BF16 `K/W/U/H/V-new`，FP32 `g/state/final_state`；
- P0 的 nonzero-W pred mapping；
- BF16 V-new round-trip、FP32 loop-carried feedback；
- Direct-K64 `v_mfma_f32_32x32x8_bf16`；
- R4 的 LDS-mediated K retile；
- one-chunk-ahead bank reuse 与 update 后 single-bank tail commit。

唯一新增候选为 `gfx942_bt64_bv32_joint_v5`。

## 实现

新增 recurrence plan/encoding：

```text
schedule             gfx942_bt64_bv32_joint_v5
shared encoding      joint_v5_superblock_bank
dot operand encoding superblock_lds_mediated_retile_dot
superblock           next_wk_issue_pred_vnew_update_tail_commit
phase lifetime       pred -> bf16_vnew -> vdecay -> update -> fp32_feedback
```

planner 在同一个 persistent recurrence op 内将 next chunk 的 W0/W1 和 K0/K1
stage issue 放在当前 pred 前；当前 pred、corrected、BF16 V-new、V-decay、K0/K1
update 和 FP32 feedback 保持原顺序。stage payload 仍只在当前 consumer 完成后，
由同一 shared bank 的 tail commit 写入下一 chunk 所需 LDS 区域。

实现位于：

- `lib/Dialect/AveLang/Transforms/qwen_recurrence_schedule_plan.h`
- `lib/Dialect/AveLang/Transforms/qwen_persistent_recurrence_pass.cc`
- `lib/Dialect/AveLang/Transforms/lower_qwen_k64_pipeline_stage_pass.cc`
- `lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc`
- `test/examples/linear_attention/vllm_compare/bench_qwen_gdn_persistent_recurrence_r5.py`

首次实现只移动 stage op，编译器立即给出 SSA dominance 错误：next-K stage 的 index
recipe 仍留在原来的 pred 后位置。修复是把 stage op 与其同 block 的直接 index/load
defs 作为一个不可拆分的 issue bundle 移动。这是正常的 IR 正确性修复，不改变计算、
LDS 分配或 buffer lifetime。

## 机器图证据

R5 post-planner MLIR 中，loop 内的 stage 顺序为：

```text
next W0 stage
next W1 stage
next K0 stage
next K1 stage
current pred
corrected -> BF16 V-new -> V-decay
current K0/K1 update
next W/K tail commit
```

对应的 `stage_load` 带有 `avelang.qwen.joint_v5.superblock_issue =
"next_w_before_pred"` 或 `"next_k_before_pred"`。最后四个 stage commit 仍位于
update 后。`avelang.qwen.joint_v5.stage = "next_k_after_pred"` 是复用旧分类时保留
的标签文字；实际位置和 `superblock_issue` 属性才是本轮的权威证据。

完整工件位于：

`codex_qwen_persistent_recurrence_r5_superblock_lowering/machine/`

其中包括 planner 前后 MLIR、block-dot lowering 后 MLIR、pre/post-opt LLVM、
exact-LTO pre/post-greedy MIR、ISA、HSACO 与 linker replay。HSACO SHA256 已变化：

| kernel | SHA256 |
|:--|:--|
| R4 `joint_v4` | `1f80bc215d7373c763d7761db083217af9d9cd9d5830d91f5d83d4bac21b70a4` |
| R5 `joint_v5` | `bf726da9e356f1d38e23f1fe71c43cb04a4b48934580d74b25291a1916784028` |

因此 R5 不是环境变量或文件名层面的假分支：高层 schedule 变动一直保留到了最终
code object。

## 正确性

对 T=64/128/512/2048 的完整 nonzero-W recurrence，R5 全部 finite 并通过冻结门槛。
R5 与 B0、P2 host microscope 的导出 `H`、raw pred、BF16 pred、BF16 V-new、BF16
V-decay、逐 chunk state 与 final state 都 byte-exact。下表只列 R5 对独立
device-contract 的最大绝对误差；它反映既有 BF16/FP32 contract rounding，而非 R5
引入的新分叉。

| T | H | V-new | V-decay | state-after | final state | pass |
|--:|--:|--:|--:|--:|--:|:--|
| 64 | 0 | 0 | 0 | `3.73e-09` | `3.73e-09` | yes |
| 128 | 0 | 0 | 0 | `3.73e-09` | `3.73e-09` | yes |
| 512 | `3.05e-05` | `7.63e-06` | `7.63e-06` | `4.38e-07` | `4.38e-07` | yes |
| 2048 | `2.44e-04` | `2.44e-04` | `2.44e-04` | `4.85e-05` | `4.84e-05` | yes |

该结果证明提前 stage 没有破坏 nonzero-W pred mapping、BF16 boundary 或 FP32 feedback；
它**不**证明提前 stage 有性能价值。

## LTO/MIR 与资源

R5 exact-LTO replay 覆盖 pre/post-greedy、virtregrewriter 与 prologepilog。所有阶段
的 `SI_SPILL_AV32_SAVE` 与 `SI_SPILL_AV64_SAVE` 均为零，因而没有 scratch/spill
cliff。T=2048 ROCprof PMC 的动态资源与 R4 相同：

| metric | R4 | R5 |
|:--|--:|--:|
| workgroup / grid | 128 / 4096 | 128 / 4096 |
| VGPR / AccVGPR / SGPR | 128 / 192 / 112 | 128 / 192 / 112 |
| LDS block | 53,248 B | 53,248 B |
| scratch | 0 B | 0 B |
| MFMA | 65,536 | 65,536 |
| VMEM | 202,240 | 202,240 |
| VALU | 2,294,144 | 2,294,144 |
| SALU | 163,072 | 163,072 |
| LDS instructions | 381,952 | 381,952 |
| occupancy | about 0.647 | about 0.647 |

这排除了“R5 因 RA spill 而慢”的解释，也表明提前 source-level stage 没有减少实际
动态 global/LDS/ALU 工作。

静态 ISA 仍不同，进一步说明变的是 instruction placement/code shape，而不是输出：

| static ISA class | R4 | R5 |
|:--|--:|--:|
| global load/store | 56 / 192 | 73 / 224 |
| DS read/write | 160 / 108 | 182 / 120 |
| `s_barrier` / `s_waitcnt` | 11 / 103 | 13 / 125 |
| `v_mfma` | 40 | 48 |
| `ds_bpermute` | 0 | 0 |

这些是反汇编的静态出现次数，不应直接等同于 loop 的动态工作量；动态 PMC 才是上表
的性能判断依据。静态代码变大而动态计数不降，也与“现有 token 不是可延迟提交的
硬件 async copy”这一解释一致。

## 正式 Body Benchmark

口径：预分配、当前 HIP stream、无 Graph capture、warmup=10、repeat=50、每长度
5 个 fresh-process session，八个 arm 采用 rotating palindromic order。下表是
session median 的中位数，属于 recurrence body diagnostic，不是 Eager public API
排名。

| T | chunks | R4 ms | R5 ms | R5-R4 | direct Triton ms | external bridge ms |
|--:|--:|--:|--:|--:|--:|--:|
| 512 | 8 | `0.132477` | `0.134840` | `+2.363 us` | `0.074711` | `0.041342` |
| 1024 | 16 | `0.267577` | `0.273526` | `+5.949 us` | `0.104055` | `0.068622` |
| 2048 | 32 | `0.500084` | `0.509197` | `+9.113 us` | `0.157094` | `0.118516` |
| 8192 | 128 | `2.007603` | `2.037427` | `+29.824 us` | `0.476789` | `0.422968` |

四个长度上、五个 session 的每个 R5-R4 paired difference 都大于零。线性拟合结果：

| arm | intercept ms | slope us/chunk |
|:--|--:|--:|
| R4 | `0.009124` | `15.604582` |
| R5 | `0.010759` | `15.825837` |
| direct Triton | `0.049547` | `3.339452` |
| external bridge | `0.016905` | `3.172983` |

R5 相对 R4 的 slope 增加 `0.221255 us/chunk`（约 `1.42%`）。T=2048 时 R5 是
direct Triton 的 `3.24x`、external bridge 的 `4.30x`。因此它既未降低长期 slope，
也没有缩小 native gap。

## 根因判断

本轮同时满足三件事：

1. 数学与 ABI bytes 保持不变；
2. planner 到 HSACO 的机器图确实不同；
3. 资源没有 cliff，但动态工作也没有下降，时间稳定回退。

因此最准确的结论是：**当前 AveLang 的 opaque K64 stage 表示不足以表达有效的
next-chunk async global-to-LDS pipeline。** 它的 payload 仍需先作为正常 SSA
register value 存活，直到 tail commit。将 next K issue 移至 pred 前，并没有让硬件
在 pred MFMA 时异步完成 LDS 写入和 wait；反而扩大了 packet 的有效存活区间，并改变了
静态机器代码形状。

这不是 RA 的失败，亦不是 K retile、MFMA geometry、BF16 boundary 或 nonzero-W
mapping 的失败。R5 不能作为“再调 waitcnt/再挪一条 load”继续枚举的起点。

## 决策

`gfx942_bt64_bv32_joint_v5 = no_go`。R4 保持 native baseline，R5 不晋级，也不进入
Eager public API 对比。

若未来重新开启这条 compiler 路线，下一步应是独立、最小、same-schedule 的功能实验：
让 stage 表示能承载 **non-register-resident global-to-LDS copy + completion/wait**，
并证明数据在 pred 区间中实际在 LDS 中前进，而不是先成为长寿命 vector SSA payload。
在这种 primitive 存在前，不应继续对 joint_v5 做 waitcnt、load placement、RA 或
fragment 微调。

## 复现工件

- correctness: `codex_qwen_persistent_recurrence_r5_superblock_lowering/correctness/all_lengths.json`
- machine IR/ISA/LTO: `codex_qwen_persistent_recurrence_r5_superblock_lowering/machine/`
- exact LTO replay: `machine/exact_lto/summary.json`
- T=2048 PMC: `machine/rocprof_t2048/`
- formal body samples: `benchmark/body_benchmark_r4_r5.json`

构建、正确性、exact-LTO replay 和 benchmark 的命令输出分别保存在上述目录的
`*.stdout`、link replay 和 JSON 工件中。
