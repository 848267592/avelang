# Qwen persistent recurrence：R4 tail-issue 因果控制

## 结论

**tail-issue control 没有复现 direct-LDS 的性能回退。** 相反，在保持 R4
的数学、BV32 ownership、typed producer、LDS-mediated K retile、MFMA 与单一
recurrence loop 不变时，仅把 next W/K 的 issue 和同-bank LDS commit 放到当前
core 完整 retire 后的尾部，得到比 R4 更低的 steady-state slope。

因此不关闭「改变 issue 位置」这一路；应暂停当前的 *software-pipeline
expansion* 实现。direct-LDS 的回退并非由 tail issue、本次 LDS placement、私有
scratch 或 VMEM 增长造成，而是 scheduler 为表达 prologue / steady / epilogue
复制了 opaque recurrence core，扩大了 MFMA accumulator 的机器 live range 与
静态代码体积。

本轮没有新增 `joint_v6`，没有启用 software-pipeline scheduler，也没有使用
private packet ring。

## 严格控制的实现

新增 lowering mode `gfx942_bt64_bv32_joint_v4_tail_issue`，但它复用原有
`gfx942_bt64_bv32_joint_v4` recurrence plan 和 K-retile consumer。其唯一
调度动作位于 `scheduleJointV4TailIssueControl`：

1. 验证四条完整的 `stage-load -> token local store -> token local load -> LDS
   commit` 依赖路径；
2. 保留各自的标量地址依赖在原位置；
3. 移除旧的尾 commit 后 barrier，并在尾部插入所需 barrier；
4. 在 current W local load、pred MFMA、BF16 V-new/V-decay、current K local
   load、update MFMA 与 FP32 state feedback 全部完成后，以
   `W0 -> commit, W1 -> commit, K0 -> commit, K1 -> commit` 的配对顺序移动
   这四条完整路径。

这样 commit 仍消费同一迭代新产生的 token，不改变 producer、bank、LDS layout
或计算图。planner snapshot 记录为 `schedule = gfx942_bt64_bv32_joint_v4` 和
`avelang.qwen.joint_v4.tail_issue_control =
"r4_next_wk_global_to_lds_after_full_current_core"`。tail control 的
`post_software_pipeline_scheduler.mlir` 中没有 scheduler 的
`software_pipeline_expanded` 标记或 `scf.for iter_args`。

实现位置：

- `lib/Dialect/AveLang/Transforms/qwen_persistent_recurrence_pass.cc`
- `lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc`
- `test/examples/linear_attention/vllm_compare/repro_qwen_gdn_persistent_recurrence_r4_tail_issue.py`
- `test/examples/linear_attention/vllm_compare/run_qwen_gdn_persistent_recurrence_r4_tail_issue_body.py`
- `test/examples/linear_attention/vllm_compare/bench_qwen_gdn_persistent_recurrence_tail_issue_control.py`

## 正确性

所有测试为 full nonzero-W recurrence。相对 R4/P2 host microscope 的 `h`、pred、
BF16 producer、`v_new`、`v_decay`、state-after 和 final-state 均 byte equal；
device contract 的非零差异均在既有浮点容差内。

| T | chunks | 结果 |
| ---: | ---: | --- |
| 64 | 1 | PASS；P2 byte equal |
| 128 | 2 | PASS；P2 byte equal |
| 512 | 8 | PASS；P2 byte equal |
| 2048 | 32 | PASS；P2 byte equal |

原始结果在 `tail_issue_t64/`、`tail_issue_t128/` 和
`tail_issue_correctness_long/`。

## Fresh-process body benchmark

条件：每个实现独立进程和 source-mode cache、HIP event、5 次 warm-up、每 session
20 次、两 session 的中位数再取平均；不使用 graph capture，编译和分配不计时。

| T | chunks | R4 (ms) | tail-issue (ms) | direct-LDS (ms) | tail / R4 | direct / R4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 16 | 0.267858 | 0.211354 | 0.276541 | 0.789x | 1.032x |
| 2048 | 32 | 0.498671 | 0.378132 | 0.524790 | 0.758x | 1.052x |
| 8192 | 128 | 2.005213 | 1.409824 | 2.179387 | 0.703x | 1.087x |

每 chunk slope（ms/chunk）：

| 区间 | R4 | tail-issue | direct-LDS |
| --- | ---: | ---: | ---: |
| 1024 -> 2048 | 0.014426 | 0.010424 | 0.015516 |
| 2048 -> 8192 | 0.015693 | 0.010747 | 0.017235 |

tail-issue 在大 T 的约 10.7 us/chunk，R4 为约 15.7 us/chunk；它与 direct-LDS
约 17.2 us/chunk 的回退方向相反。因此「发射挪到 current core 之后」不是
direct-LDS 回退的充分原因。

## T=2048 机器证据

PMC 是五次 collection；下表列资源行与动态计数的代表范围/不变量。

| 项目 | R4 | tail-issue | direct-LDS scheduler |
| --- | ---: | ---: | ---: |
| LDS / block | 53,248 B | 53,248 B | 53,248 B |
| private scratch | 0 B | 0 B | 0 B |
| VGPR | 128 | 128 | 128 |
| PMC AccVGPR | 192 | 160 | 248 |
| SGPR | 112 | 112 | 112 |
| MFMA | 65,536 | 65,536 | 65,536 |
| LDS instructions | 381,952 | 381,952 | 385,024 |
| VMEM | 202,240 | 202,240 | 201,152 |
| VALU | 2,294,144 | 2,239,872 | 2,353,408 |
| occupancy | about 0.64 | about 0.64 | about 0.64 |

这排除了 private packet ring / private frame 与新增 VMEM：严格 control 与 R4
同为 zero scratch 和 202,240 VMEM；direct-LDS 的 VMEM 反而略少，仍更慢。

HSACO metadata 也没有 spill：R4 为 `agpr=32, vgpr=276, sgpr=43`，tail-issue 为
`agpr=32, vgpr=228, sgpr=43`，direct-LDS 为 `agpr=64, vgpr=272, sgpr=50`，三者
`private_segment_fixed_size=0`。PMC 中 direct 的 248 AccVGPR 不是 1 KiB private
packet frame 的副作用。

静态 ISA 计数进一步定位差别：

| 指令类别 | R4 | tail-issue | direct-LDS |
| --- | ---: | ---: | ---: |
| `v_mfma` | 48 | 48 | 128 |
| `global_load` | 73 | 73 | 122 |
| `global_store` | 224 | 224 | 512 |
| `ds_read` | 182 | 182 | 504 |
| `ds_write` | 120 | 120 | 288 |
| `s_waitcnt` | 130 | 136 | 289 |
| `s_barrier` | 13 | 12 | 32 |
| scratch load/store | 0 / 0 | 0 / 0 | 0 / 0 |

tail control 保持了 R4 的 MFMA、VMEM、LDS 静态主体；仅有尾部位置相应的
wait/barrier 变化。direct-LDS 则把这些类别大约扩展为三份代码体，既不是单纯
提前一个 SSA load，也不是单一尾部 barrier 的成本。

## scheduler 额外 live-range 的定位

direct-LDS 的 scheduler snapshot 有三个
`ave.gpu.amdgpu_block_dot_bf16_f32`（prologue、含四个 `i64` `iter_args` 的
steady、epilogue），并带有
`software_pipeline_expanded` 与
`prologue(issue+consume)->steady(iter_args,direct_lds,distance=1)->epilogue`
标记。tail-issue snapshot 只有原来的一个 block-dot 和一个 recurrence loop。

因此已定位的结构性原因是：当前通用 scheduler 不是在同一 recurrence core 内
作 modulo schedule，而是为边界与 steady state 克隆包含 opaque block-dot / MFMA
accumulator 的完整 core。寄存器分配由此同时面对多份 accumulator-bearing code
region；AGPR metadata 从 32 增至 64，PMC AccVGPR 从 160 增至 248，且静态
`v_mfma`、LDS、wait/barrier 也随三段 body 增长。四个 `i64` iter_args 本身是轻量
token，不足以解释 AccVGPR；核心问题是它们所伴随的 core expansion。

## 决策与后续

本实验不满足「tail-issue 复现 direct-LDS 回退」的前提，故不关闭调度位置路线，
也不开始 R4-vs-Triton 的替代审计。当前 software-pipeline 的性能路线应冻结在
这版 expansion：下一步若继续，应使 scheduler 保持 **单一 recurrence core**，
以底层 loop scheduling / rotation 表达 prologue、steady、epilogue，而不能克隆
opaque block-dot 的完整计算体；在此之前不进行 distance=2。

## 可复核产物

`test/examples/linear_attention/vllm_compare/tail_issue_artifacts_t2048/` 包含：

- `mlir/`：planner、scheduler 前后、LLVM snapshots；
- `tail_issue_t2048.isa.s`、HSACO 与 metadata notes；
- `pmc_csv/0364d3a007f9/698671_counter_collection.csv`；
- `benchmark/fresh_process_r4_tail_direct_lds.json`；
- `correctness/tail_issue_correctness.json`（T=64 归档；其余原始 JSON 如上）；
- rocprof profile。
