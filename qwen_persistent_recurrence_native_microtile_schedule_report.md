# Qwen persistent recurrence：原生参数化 microtile schedule 第一轮报告

## 结论

本轮以 `gfx942_bt64_bv32_joint_v4_tail_issue`（以下称 R4-tail）为基线，接入的是 first-class persistent-recurrence op 与现有 joint planner 的原生 microtile 计划，而不是 `joint_v6`、isolated producer/consumer 实验或既有 software-pipeline expansion。

36 个参数组合均由同一 lowering 路径生成或在结构合法性阶段拒绝。18 个 distance=0 计划全部满足 Scratch=0、无 spill、MFMA/LDS/barrier 不变的资源门槛；9 个 `tail,d1` 因“tail issue”与“跨 core VGPR 驻留”在计划定义上矛盾而提前拒绝；9 个 `lastuse,d1` 通过跨 wave LDS lifetime 证明，但都使 HSA VGPR 从 228 增加，因而被资源门槛拒绝。

经过双 session、fresh-process 复测，最快候选与 R4-tail 的差异在噪声量级，未得到可复现的吞吐提升。`w1,k1,lastuse,d0` 的 2048→8192 slope 为 0.01072395 ms/chunk，相对 R4-tail 的 0.01073887 ms/chunk 仅低 0.14%；三项 top-3 的 PMC 计数与 R4-tail 完全一致。因此本轮的正确结论是：**原生参数化框架、生命周期证明和筛选链路已具备，但在不跨 core 保留 packet 的约束下，微分组没有制造机器级重叠；d1 的唯一真实重叠代价是额外 VGPR live range。** 不继续扩大 software pipeline，也不创建 `joint_v6`。

## 实现

新增 `QwenMicrotileSchedulePlan`，每一个计划统一携带：

| 字段 | 含义 / 当前取值 |
| --- | --- |
| `wPacketsPerGroup`, `kPacketsPerGroup` | W/K packet group 大小，搜索集合各为 `{1,2,4}` |
| `currentConsumerOrder` | `w0,w1 -> pred -> bf16_vnew_vdecay -> k0,k1 -> update -> fp32_feedback`；不拆分 pred/update core |
| `ldsRegionLastUse` | `w0,w1,k0,k1:single_core_exit` |
| `placement` | `tail` 或 `lastuse` |
| `vgprResidentMicroGroups` | `0` 或 `1` |
| `sameRegionCommit` | `after_existing_core_exit_barrier` |
| wait/barrier/state 边界 | 继承 R4-tail 的 wait、core-exit workgroup barrier 和 FP32 feedback 边界；不新增逐 group barrier |

实现位置：

- `lib/Dialect/AveLang/Transforms/qwen_recurrence_schedule_plan.h`：计划数据模型与 placement 枚举。
- `lib/Dialect/AveLang/Transforms/qwen_persistent_recurrence_pass.cc`：解析 `gfx942_bt64_bv32_microtile_experimental_w{1|2|4}_k{1|2|4}_{tail|lastuse}_d{0|1}`、构造计划、跨 wave legality 及在 first-class recurrence loop 内重排 stage/commit。
- `lib/Dialect/AveLang/Transforms/lower_qwen_k64_pipeline_stage_pass.cc`：按 packet range 发出 group load/commit，同时保留原有物理 LDS packet offset；不是只添加 metadata。
- `lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc`：将实验计划接入 R4 已验证的 K retile、BV32 ownership 与 typed BF16x8 producer。
- `test/examples/linear_attention/vllm_compare/repro_qwen_gdn_persistent_recurrence_microtile.py`、`search_qwen_gdn_persistent_recurrence_microtiles.py`、`bench_qwen_gdn_persistent_recurrence_microtile_top3.py`：统一的 full-nonzero-W candidate、筛选、P2 与 fresh-process harness。

实现没有分裂 loop 或复制 pred/update core：planner MLIR 中每个 top-3 均只有一个 `amdgpu_block_dot_bf16_f32`，没有 private packet memref，也没有 `software_pipeline_expanded` 标记。此处的计划改变只移动/分段 next-packet 的 stage/commit；recurrent state 仍在同一 loop 的 `FP32 feedback` 边界回写。

## LDS lifetime 与合法性

跨 wave 证明是保守的：所有物理 LDS 子区域 `w0,w1,k0,k1` 的 last-use 都是**完整 current core 退出**，因此所有 same-region LDS commit 都保留在既有 core-exit barrier 之后。没有把覆盖推入 W/K consumer group 之间，也没有新增逐 group 的 workgroup barrier。

`d1` 只允许把下个 W0 的 group-0 global issue 放在 core 前；LDS commit 依然在尾部既有 barrier 后。这维持 single LDS allocation（53,248 B）且不引入 private ring，但让该 packet 的 VGPR 定义跨越 current core。`tail,d1` 则违反“tail issue”的声明：若要 d1，就必须在 core 前 issue，故 9 个组合不进入 lowering。

## 全部 36 个计划与筛选

下表每一格是一个完整计划：`P`=通过编译与资源门槛并进入 T=2048/8192 粗筛；`S`=结构拒绝（tail+d1）；`V`=资源拒绝（lastuse+d1）。每个候选均为单 recurrence loop、单 pred/update core、无 private ring、无第二套完整 LDS。

| W group | K group | tail,d0 | tail,d1 | lastuse,d0 | lastuse,d1 |
| ---: | ---: | --- | --- | --- | --- |
| 1 | 1 | P | S | P | V |
| 1 | 2 | P | S | P | V |
| 1 | 4 | P | S | P | V |
| 2 | 1 | P | S | P | V |
| 2 | 2 | P | S | P | V |
| 2 | 4 | P | S | P | V |
| 4 | 1 | P | S | P | V |
| 4 | 2 | P | S | P | V |
| 4 | 4 | P | S | P | V |

`S` 的统一原因是 `tail+d1 conflicts with the declared issue position; a resident group requires a pre-core issue`。`V` 的统一根因是 `vgpr_no_regression`：所有均为 Scratch/private=0、AGPR=32、LDS=53,248 B、MFMA=48、barrier=12、无 SGPR/VGPR spill，但 VGPR 增至至少 236（随 group 大小最高 256），并且 waitcnt 为 137--139（d0 为 136）。这说明新增的成本是跨 core 的 packet live range，而非 LDS 覆盖、spill 或 barrier 线性增加。

18 个 `d0` 计划的 HSA/ISA 筛选资源完全一致：AGPR=32、VGPR=228、SGPR=43、private=0、LDS=53,248 B、VGPR/SGPR spill=0、MFMA=48、global-load=73、LDS 指令=302、waitcnt=136、barrier=12。这就是它们可以进入粗基准而机器计数又没有区分度的原因。

## top-3 正确性、代码生成与 PMC

粗筛按 T=2048/8192 的单 session slope 选出了以下 3 项；所有项随后完成 T=64/128/512/2048 full nonzero-W P2。相对于 P2 host microscope，`h`、pred、BF16 V-new/V-decay、每 chunk state 及 final state 均 byte-equal；device contract 比较也在既有容差内通过。

| plan | P2 T=64/128/512/2048 | Scratch/LDS | HSA AGPR/VGPR | PMC VGPR/AccVGPR/SGPR | MFMA | VALU | VMEM | LDS | barrier | slope (ms/chunk) |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R4-tail | 已有基线通过 | 0 / 53,248 | 32 / 228 | 128 / 160 / 112 | 65,536 | 2,239,872 | 202,240 | 381,952 | 12 | 0.01073887 |
| `w1,k2,tail,d0` | 全部通过 | 0 / 53,248 | 32 / 228 | 128 / 160 / 112 | 65,536 | 2,239,872 | 202,240 | 381,952 | 12 | 0.01074983 |
| `w4,k2,lastuse,d0` | 全部通过 | 0 / 53,248 | 32 / 228 | 128 / 160 / 112 | 65,536 | 2,239,872 | 202,240 | 381,952 | 12 | 0.01074075 |
| `w1,k1,lastuse,d0` | 全部通过 | 0 / 53,248 | 32 / 228 | 128 / 160 / 112 | 65,536 | 2,239,872 | 202,240 | 381,952 | 12 | 0.01072395 |

PMC 为 T=2048，`OccupancyPercent` 约 0.636（R4-tail 约 0.64）。机器审计还显示三项 top-3 都是 MFMA=48、LDS=302、waitcnt=136、barrier=12；无 stack frame（MIR `stack: []`），HSA metadata 报告 private segment=0、VGPR/SGPR spill=0。三个 candidate 的 ISA/MIR 的静态资源与动态计数相同，因而目前没有可归因到 microtile 参数的机器级性能改变。

导出物（每个 top-3 均有 MLIR、pre/post LLVM、HSACO、ISA、exact-LTO final-isel MIR）：

- `test/examples/linear_attention/vllm_compare/microtile_search_artifacts/<plan>/mlir/`
- `test/examples/linear_attention/vllm_compare/microtile_search_artifacts/<plan>/exact_lto_final_isel.mir`
- `test/examples/linear_attention/vllm_compare/microtile_search_artifacts/<plan>/kernel.isa.s`
- `test/examples/linear_attention/vllm_compare/microtile_top3_artifacts/pmc/<plan>/counters_counter_collection.csv`

其中 MIR 从编译捕获的 `postopt_llvm.ll` 用 AMD LLVM 22、`-march=amdgcn -mcpu=gfx942 -O3 -stop-after=finalize-isel` 再生；因此它是当前捕获 LTO 后 LLVM 输入的 final-isel 审计产物。

## fresh-process 性能复测

复测协议：每实现、每长度独立进程与独立 mode cache；HIP event 计时；无 graph capture；排除编译和分配；5 warmup + 20 repeat，2 个 session。结果为 session median 的中位数。

| 实现 | T=2048 ms | 相对 R4-tail | T=8192 ms | 相对 R4-tail | 2048→8192 slope ms/chunk |
| --- | ---: | ---: | ---: | ---: | ---: |
| R4-tail | 0.379163 | 1.000000 | 1.410095 | 1.000000 | 0.01073887 |
| `w1,k2,tail,d0` | 0.377641 | 0.995985 | 1.409624 | 0.999666 | 0.01074983 |
| `w4,k2,lastuse,d0` | 0.379013 | 0.999604 | 1.410125 | 1.000021 | 0.01074075 |
| `w1,k1,lastuse,d0` | 0.380315 | 1.003038 | 1.409814 | 0.999801 | 0.01072395 |

T=2048 的小幅差异与 session p10/p90 离散重叠；更长 T 下差异最大 0.033%。因此不能将表中任一微小差值报告为优化收益。

原始搜索记录位于 `test/examples/linear_attention/vllm_compare/microtile_search_artifacts/search.json`，稳定复测记录位于 `test/examples/linear_attention/vllm_compare/microtile_top3_artifacts/fresh_process_r4_tail_vs_top3.json`，P2 记录位于 `microtile_top3_artifacts/correctness/`。

## 下一步判断

此框架现在能以声明式计划表达 packet group、issue placement、LDS last-use/commit 与距离，并在不安全时拒绝计划。当前不继续 `distance=2`、不扩大 software pipeline。若未来继续 native 路线，必须先提出一个不增加 VGPR live range 且能让 issue 与不同 consumer phase 真正交错的合法 schedule；否则 R4-tail 是应保留的性能基线。
