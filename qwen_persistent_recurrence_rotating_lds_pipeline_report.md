# Qwen persistent recurrence：private packet ring → direct-LDS 软件流水实验

日期：2026-07-30。基线为 `gfx942_bt64_bv32_joint_v4`（R4）；本轮只改变
distance-one modulo pipeline 的 packet 传输媒介。没有尝试 distance=2，也没有改动
R4 已验证的数学、BV32 ownership、typed BF16x8 producer 或 LDS-mediated K retile。

## 结论

完成了一个可运行的 **direct-LDS、无 private packet ring** 候选：保留原有
prologue、四个轻量 `i64` `iter_args` 的 steady state、distance=1 和 epilogue；下一
packet 的全局 load 直接写入其 W/K LDS bank，逻辑 token 只保留 scheduler 的迭代边。
它在 T=64/128/512/2048 的 nonzero-W correctness 全部通过，且相对旧 private-ring
候选彻底消除了 scratch 与其 VMEM 放大。

但它**不是可安全宣称的物理双槽 role-rotating LDS**：真实的 W/K bank role swap 在
本轮被验证为 No-Go。`AMDGPUQwenGdnRecurrenceStepBF16F32Op` 把单一物理
`phaseStage` 隐藏在 opaque recurrence op 内；仅重映射外层 packet SSA/bank 无法切换
该内部 stage，T=128 的第一个 steady chunk 已出现错误（`pred` 最大绝对误差
`0.013713`）。因此最终候选保留已验证的 W/K bank identity，并在完整 current core
retire 后以 barrier 再写入 next packet。这样是正确的无-private 传输原型，但没有制造
不存在的双缓冲重叠。

长序列依然慢于 R4：private/VMEM 已不是瓶颈；机器证据指向更高的 AccVGPR、额外
LDS/VALU 动态工作和更多静态 wait/barrier。没有继续试 distance=2。

## 实现与流水结构

修改位置：

- `lib/Dialect/AveLang/Transforms/qwen_modulo_software_pipeline_pass.cc`
- `lib/Dialect/AveLang/Transforms/lower_qwen_k64_pipeline_stage_pass.cc`
- `test/examples/linear_attention/vllm_compare/repro_qwen_gdn_persistent_recurrence_software_pipeline.py`

调度器仍是通用的 loop expansion：`cloneCore` 带有前/后 clone hook，生成
prologue、`scf.for iter_args` steady state 与 epilogue，而不是写死 Qwen ISA。其物理边
由两个通用标记连接：

1. `direct_lds_commit`：每一个 stage-1 global packet producer 立刻使用既有 typed
   BF16x8/K-retile lowering 写入 LDS；
2. `logical_lds_consumer`：steady/epilogue 中的 i64 token 消费边在 late lowering 被
   擦除，绝不 materialize 成 payload storage。

结构为 `prologue(issue+consume) -> steady(iter_args,direct_lds,distance=1) -> epilogue`。
稳态中四个 next W/K stage 位于完整 current core（W local load + pred MFMA、BF16
V-new/V-decay、K local load + update MFMA、FP32 feedback）之后；首个 W producer 前
插入 workgroup barrier。每个 packet 的 producer stage 为 1，consumer 为下一次迭代的
stage 0，逻辑距离为 1；四个 `iter_args` 仅为 token carrier。

`post_software_pipeline_scheduler.mlir` 可见：prologue direct commits 在约 641--647
行、steady `scf.for` 的四个 `iter_args` 在 650 行、其 logical consumer 在 651--654
行、next direct commits 在 989--995 行、epilogue logical consume 在 999--1002 行。late
pass 对 software-pipeline modulo stage 到 legacy private ring 的路径直接报错，因此该
候选不可能静默回退到 `stagePacketStorage`。

## 旧 1024 B scratch 的归因与消除

旧候选的 exact-LTO MIR 在函数开头有四个 variable-sized frame object（`fi#0..fi#3`）
和一个 `fi#4 size=4`；其 late pass 的 `stagePacketStorage` 恰好为四个 packet stage
group 各创建一个 `memref<4xvector<8xbf16>, private>`。旧 HSACO metadata 的
`private_segment_fixed_size=16` B/lane，而 ROCprof 运行时显示 `Scratch_Size=1024` B，
即一个 64-lane wave 的 16 B/lane。旧 ISA 有成组 `scratch_store_dwordx4` 与
`scratch_load_dwordx4`。

这给出了 ring、frame 与 scratch 的直接因果链，而非仅凭计时猜测。新候选的 exact-LTO
MIR 没有 `Frame Objects`、没有 spill；scheduler 后及后续 MLIR 中不存在
`memref<4xvector<8xbf16>, private>`；ISA 没有 `scratch_load/store`；HSACO 的
`private_segment_fixed_size=0`，PMC `Scratch_Size=0`。

| T=2048，Qwen kernel | 旧 private-ring SWP | 本轮 direct-LDS | R4 |
|---|---:|---:|---:|
| private segment（B/lane） | 16 | 0 | 0 |
| PMC Scratch（B） | 1024 | 0 | 0 |
| LDS allocation（B） | 53248 | 53248 | 53248 |
| PMC VGPR / AccVGPR / SGPR | 128 / 248 / 112 | 128 / 248 / 112 | 128 / 192 / 112 |
| PMC `SQ_INSTS_VMEM` | 264640 | 201152 | 202240 |
| PMC `SQ_INSTS_LDS` | — | 385024 | 381952 |
| PMC `SQ_INSTS_VALU` | — | 2353408 | 2294144 |
| PMC MFMA | — | 65536 | 65536 |
| OccupancyPercent | — | 0.634（五次 0.6336--0.6402） | 0.643（五次 0.6429--0.6445） |

因此，删除 ring 后 VMEM 比旧候选少 63488（约 24%），并且比 R4 少 1088；所担心的
VMEM 增量已完全消失。LDS allocation 不变，说明此次不是以额外 LDS 空间换取结果。

## ISA、wait/barrier 与剩余代价

下列为 HSACO 静态计数（包含 prologue/steady/epilogue 代码体，不等同于动态计数）：

| ISA 项 | 旧 private-ring SWP | direct-LDS | R4 |
|---|---:|---:|---:|
| `v_mfma` | 128 | 128 | 48 |
| `global_load` | 122 | 122 | 73 |
| `global_store` | 512 | 512 | 224 |
| `ds_read` / `ds_write` | 504 / 288 | 504 / 288 | 182 / 120 |
| `s_waitcnt` | 328 | 289 | 130 |
| `s_barrier` | 32 | 32 | 13 |
| scratch load/store | 存在 | 0 / 0 | 0 / 0 |

direct-LDS 相对旧候选已减掉 39 个静态 wait，并移除了 scratch 指令；但相对 R4 仍多
159 个静态 wait 与 19 个 barrier。ISA 中 next-packet block 的模式是：先完成 current
core 的 `s_waitcnt lgkmcnt(0); s_barrier`，随后 `global_load` 与 typed LDS `ds_write`，
再以 `s_waitcnt lgkmcnt(0); s_barrier` 才进入 next local-load/MFMA。这证明它不是单纯
提前 SSA load，但也说明单 bank identity 迫使 producer 位于 core retire 之后，未能把
global-load latency 覆盖到 current MFMA 上。

PMC 同时显示动态 MFMA 与 R4 完全相等；direct-LDS 的 VMEM 更低、SALU 略低
（159808 vs 163072），却多 3072 LDS 指令、59264 VALU 指令并使用多 56 个 AccVGPR。
结合较低 occupancy，这比“VMEM 仍然增加”更能解释长序列退化。

## Correctness

以下每项均为 nonzero-W，并且相对 P2 host microscope 的 `h`、`pred`、`v_new`、
`v_decay`、state/final-state byte-equal：

| T | chunks | 结果 |
|---:|---:|---|
| 64 | 1 | 通过 |
| 128 | 2 | 通过 |
| 512 | 8 | 通过 |
| 2048 | 32 | 通过 |

device-contract reference 仍存在原 R4 已有的 BF16/FP32 容差内舍入差异；所有项通过。

## Fresh-process body benchmark

条件：每个 implementation 独立 Python process；HIP event；不 graph capture；每个点
5 warmup、20 repeat、2 session，报告为两个 session median 的 median（ms）。

首次批量 parent run 中，T=1024 与 T=2048 各有一个 software-pipeline worker 在退出前收到
`SIGSEGV`；随后相同 T 的单 worker 均成功，重跑的完整两 session 也全部成功，以下是后者
的正式数值。该信号没有伴随错误输出，且 correctness/PMC 路径未复现；因此它不改变本轮
吞吐结论，但应在后续双槽 ABI 改造前作为独立的 process-lifetime 稳定性项继续追踪。

| T | chunks | direct-LDS | R4 | Triton | direct-LDS / R4 |
|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 0.131425 | 0.135431 | 0.065648 | 0.9704（快 3.0%） |
| 1024 | 16 | 0.277673 | 0.266346 | 0.093239 | 1.0425（慢 4.3%） |
| 2048 | 32 | 0.524730 | 0.498180 | 0.146508 | 1.0533（慢 5.3%） |
| 8192 | 128 | 2.174872 | 2.005670 | 0.441036 | 1.0844（慢 8.4%） |

增量 slope（ms/chunk）也显示 steady-state 缺口，而非固定 prologue 成本：

| 区间 | direct-LDS | R4 | 额外 direct-LDS 成本 |
|---|---:|---:|---:|
| 8 → 16 chunks | 0.018281 | 0.016364 | 0.001917 |
| 16 → 32 chunks | 0.015441 | 0.014490 | 0.000951 |
| 32 → 128 chunks | 0.017189 | 0.015703 | 0.001486 |

这与机器级 wait/barrier、AccVGPR/occupancy 和 LDS/VALU 增量一致。旧 private-ring
候选在历史同配置下为 0.146878 / 0.300817 / 0.575895 / 2.371473 ms；无-private 版本
分别改善约 10.5% / 7.7% / 8.9% / 8.3%，但仍未跨过 R4 的长序列吞吐。

## 真正双槽 rotating LDS 的 No-Go 与后续边界

试验性 role swap 曾把 next W/K 交替写到对方 R4 bank，并以 slot `iter_arg` 选择当前
消费者。这在 scheduler 的外层 token SSA 看似合法，但 recurrence-step lowering 不使用
该 token 来选择内部 `phaseStage`。它固定把 W local load、后续 K staging/update 绑定到
同一 opaque `phaseStage`；所以外层 bank remap 与真实 consumer 不一致，T=128 在 chunk 1
即失真。额外 barrier 只能消除审计 race，不能恢复语义。

要实现题设意义的“current/next 物理 LDS 双槽”，下一步必须扩展
`AMDGPUQwenGdnRecurrenceStepBF16F32Op` 及其 late lowering：显式携带/选择 current 与
next phase-stage，并重新验证 LDS 容量和 occupancy。那是 recurrence op ABI/lowering 的
独立变更，不是本轮 scheduler-only 单变量实验可以安全假设的内容。基于本轮结果，不应
继续做 distance=2 或再调 waitcnt；应先完成该 op 的 bank identity 暴露与 alias/effect
建模，再重新进入通用 scheduler。

## 可复查工件

- 当前候选完整 MLIR/LLVM/LTO MIR/ISA/HSACO/PMC：
  `test/examples/linear_attention/vllm_compare/swp_direct_lds_artifacts_t2048/`
- 其中 `mlir/post_software_pipeline_scheduler.mlir`、
  `exact_lto_mir/kernel_section_19.mir`、
  `software_pipeline_direct_lds_t2048.isa.s` 与
  `pmc_csv/0364d3a007f9/682049_counter_collection.csv` 分别对应 scheduler、exact-LTO、
  ISA 和 direct-LDS PMC。
- 四项 fresh-process JSON：该目录的 `benchmarks/fresh_process_t{512,1024,2048,8192}.json`。
- 同口径 R4 HSACO/ISA/PMC：
  `test/examples/linear_attention/vllm_compare/r4_direct_lds_baseline_t2048/`。
- 四项 correctness JSON：`swp_direct_lds_t64/`、`swp_direct_lds_simple_t128/`、
  `swp_direct_lds_t512/`、`swp_direct_lds_t2048/`。
