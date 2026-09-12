# Qwen persistent recurrence：core 内真实 last-use staggered issue 实验报告

## 结论

已完成一个 first-class persistent-recurrence 的 **core 内 last-use-driven staggered issue** 原型。它不是完整 core 前的 d1、不是 distance=2、不是 software-pipeline expansion，也没有第二套 LDS、private packet ring 或逐 group barrier。

原型确实让 next W/K 的 global packet load 落在 current pred/update 的 MFMA consumer group 之间；next packet 是 SSA `vector<8xbf16>`，只在原有 core-tail 的安全边界写回原 LDS bank。所有 18 个 `W group × K group × {immediate, delay1}` 组合均通过 T=64 正确性和静态资源 gate；代表性的 w1/w2/w4 最快项完成 T=64/128/512/2048 P2 正确性。

性能结论是否定的：即使最快的 `w4,k4,delay1` 没有 scratch、spill、额外 LDS、额外 VMEM、额外 barrier 或 occupancy cliff，T=2048 仍为 R4-tail 的 **1.311×**，1024→8192 slope 为 **0.01540900 ms/chunk**，R4-tail 为 **0.01065782 ms/chunk**。因此应关闭这一版“完整 packet 在 VGPR 保留到 tail commit”的路线；不要继续扩大该搜索、不要 distance=2，也不要把它误报成仅 VGPR 门槛造成的拒绝。

## 窄审计：真正的 consumer / last-use 边界

R4-tail 的 persistent core 共有 48 条静态 MFMA：两个 W half 各有四个 pred consumer group，每组 2 条 MFMA；两个 K half 各有四个 update consumer group，每组在 block-dot lowering 后为 4 条 MFMA。

| packet region | consumer group | 最后一次实际消费 |
| --- | --- | --- |
| W0 | pred half 0, group 0..3 | 对应 group 的第 2 条 pred MFMA |
| W1 | pred half 1, group 0..3 | 对应 group 的第 2 条 pred MFMA |
| K0 | update half 0, group 0..3 | 对应 group 的第 4 条 lowered update MFMA |
| K1 | update half 1, group 0..3 | 对应 group 的第 4 条 lowered update MFMA |

旧 `lastuse,d1` 的命名并不准确：它把 W0 的 next load 放在完整 pred 与 BF16 V-new/V-decay 后、update 前。因而它跨越的是后续 32 条 update MFMA（不是先前报告的 48 条），并未对应 W micro-group 的真实 last-use。

Triton 的选定 ISA 也给出了相同的可行模式：在 `0x499c/0x49a8` 的 global loads 之后仍有 `0x49c4`、`0x4a1c` 的 MFMA，随后在 `0x4a34` 才出现 `s_waitcnt vmcnt(13)`。该二进制没有与 AveLang packet 一一对应的源级调试信息，故它只用于确认“load 位于 MFMA group 之间、wait 晚于 issue”的调度形态，而不把它错误归因于某一 R4 packet。

## 实现

新增的计划模型 `QwenCoreLastUseSchedulePlan` 表示 W/K packets-per-group、issue placement、consumer order、真实 last-use 与固定 tail commit：

- `lib/Dialect/AveLang/Transforms/qwen_recurrence_schedule_plan.h`
- `lib/Dialect/AveLang/Transforms/qwen_persistent_recurrence_pass.cc`

计划名为：

```
gfx942_bt64_bv32_core_lastuse_experimental_w{1|2|4}_k{1|2|4}_{immediate|delay1}
```

新的 `ave.gpu.amdgpu_qwen_k64_core_issue` / `...core_commit` 明确表达 global vector packet 到 LDS tail commit 的 SSA 边：

- `lib/Dialect/AveLang/IR/AveLangOps.td`
- `lib/Dialect/AveLang/IR/AveLangOps.cc`
- `lib/Dialect/AveLang/Transforms/lower_qwen_k64_pipeline_stage_pass.cc`

planner 静态展开 persistent op 内原有的固定 consumer loops，但仍保留一个 recurrence loop 与一套 pred/update core。W issue 插在 pred consumer group 的实际末尾；`lower_qwen_block_dot_pass.cc` 将 deferred K issue 移到 opaque block-dot 展开后的第 4 条 update MFMA 后，因而 K 不是停在 block-dot op 出口这个伪边界。tail 中仍使用原有 barrier 后的同-bank LDS vector store。

例如 `w1,k1,immediate` 的 lowering MLIR 中，W0 packet 0..3 的 `vector.load` 分别带有 `issue_after_consumer_group = 0..3` 和 `last_use_boundary = "pred_mfma_group_last_use"`；K0/K1 的 load 带有 `actual_issue_boundary = "update_mfma_group_last_use"`。`w4,k4,delay1` 因每 half 只有一个 packet group，延迟一组会钳制到 group 3；这正是该参数在该粒度下与 immediate 几乎同义的原因。

## ISA 证明与安全性

`w4,k4,delay1` 的 ISA 在 pred MFMA 后出现 W packet global loads，例如 `0x4ce0..0x5094` 的 pred MFMA 区间之后/之间有 packet loads；K packet loads 位于 update MFMA cluster 后，例如 `0x50c8..0x50e8`。这与 planner MLIR 中的 group boundary 属性相符，而不是只改变 metadata。

所有 18 个编译出的 ISA 都保留 48 MFMA、12 `s_barrier`；每个 HSA metadata 都是 private segment=0、VGPR/SGPR spill=0、LDS=53,248 B。没有 private memref，且 core commit 均在既有 tail barrier 后，故不存在把 still-read LDS region 提前覆盖给其他 wave 的情形。

| 参数族 | HSA AGPR | HSA VGPR | static waitcnt | Scratch/spill/MFMA/barrier |
| --- | ---: | ---: | ---: | --- |
| k1 immediate | 39 | 295 | 141 | 0 / 0 / 48 / 12 |
| k1 delay1 | 42 | 298 | 141 | 0 / 0 / 48 / 12 |
| k2 immediate | 41 | 297 | 138 | 0 / 0 / 48 / 12 |
| k2 delay1 | 42 | 298 | 137 | 0 / 0 / 48 / 12 |
| k4 immediate/delay1 | 41 | 297 | 131 | 0 / 0 / 48 / 12 |
| R4-tail | 32 | 228 | 136 | 0 / 0 / 48 / 12 |

W group `{1,2,4}` 不改变这张静态资源表；它改变的是这些 packet vector 值实际开始 live 的 consumer boundary。`w1` 的 ISA 确实把不同 packet loads 放在连续 pred/update MFMA group 之间，`w4` 则把四个 packet 合并到该 half 的最终 group 边界。

## 正确性

- 全部 18 个计划：T=64 full nonzero-W P2 host microscope 通过。
- `w1,k1,immediate`：T=64/128/512/2048 通过，host microscope 的 h、pred、BF16 V-new/V-decay、state-after 与 final state 均 byte-equal。
- 性能筛选中每个 W 粒度最快的 `w1,k4,delay1`、`w2,k4,delay1`、`w4,k4,delay1`：均完成 T=64/128/512/2048，且同样 byte-equal。

## T=2048 PMC：resource-cliff gate 不应退化成零增长 gate

R4-tail 与最快的 `w4,k4,delay1` 的 T=2048 PMC 如下。二者 MFMA、VMEM、LDS、scratch 和 occupancy 均没有负向变化；因此这次实验满足“无 scratch、无 spill、不跨 occupancy cliff、无额外 barrier 后必须实际测性能”的 gate，而不是因 VGPR 增长被事前拒绝。

| 指标 | R4-tail | w4,k4,delay1 |
| --- | ---: | ---: |
| PMC VGPR / AccVGPR / SGPR | 128 / 160 / 112 | 128 / 192 / 112 |
| OccupancyPercent | 0.638181 | 0.640984 |
| SQ_INSTS_MFMA | 65,536 | 65,536 |
| SQ_INSTS_VMEM | 202,240 | 202,240 |
| SQ_INSTS_LDS | 381,952 | 381,952 |
| SQ_INSTS_VALU | 2,239,872 | 2,290,240 |
| SQ_INSTS_SALU | 163,072 | 163,136 |

`w1,k1,immediate` 的 PMC 更差一些（AccVGPR=200、LDS=390,144、VALU=2,297,280、SALU=187,776），说明更小、较早的 groups 还会额外暴露 staging/address costs；但即使 k4 把 LDS 动态计数恢复到 R4 相同，性能仍然显著回退。

exact-LTO final-isel replay 对 `w1,k1,immediate` 与 `w4,k4,delay1` 都没有 `SI_SPILL`，final frame 也没有 stack object。这排除了旧 software-pipeline private ring 的 1 KiB scratch 根因。

## Fresh-process 性能

协议：每实现/长度独立 Python 进程，独立 source-mode cache，HIP event，5 warmup、20 repeat、2 sessions；没有 graph capture，也没有把编译或分配计入时间。

首先对 18 个计划做 T=2048 单 session 筛选。最快五项仍全部慢于 R4-tail：

| plan | T=2048 ms | 相对 R4-tail |
| --- | ---: | ---: |
| R4-tail | 0.379784 | 1.00000 |
| w4,k4,delay1 | 0.494955 | 1.30326 |
| w4,k4,immediate | 0.495536 | 1.30478 |
| w1,k4,delay1 | 0.497038 | 1.30874 |
| w2,k4,delay1 | 0.498480 | 1.31254 |
| w1,k1,immediate | 0.516468 | 1.35990 |

随后对每个 W 粒度的代表最快项复测长序列：

| 实现 | T=1024 ms | T=2048 ms | T=8192 ms | 1024→8192 slope ms/chunk |
| --- | ---: | ---: | ---: | ---: |
| R4-tail | 0.216121 | 0.378343 | 1.409797 | 0.01065782 |
| w1,k4,delay1 | 0.265755 | 0.496268 | 1.996811 | 0.01545585 |
| w2,k4,delay1 | 0.268038 | 0.497490 | 2.002469 | 0.01548599 |
| w4,k4,delay1 | 0.267298 | 0.495848 | 1.993105 | 0.01540900 |

三条 long-run slope 都比 R4-tail 高约 44–45%。故不是一个短 T 的固定开销现象。

## 因果判断与停止条件

这次合法的 safe-tail 版本缩短了各 packet 从其**自身最后一次消费**到 issue 的语义距离，但不能缩短其从 issue 到 tail commit 的物理 VGPR lifetime：已发射的 W/K packet 必须同时活到最后的统一 LDS commit。早 W0 group 仍横跨后面的 W1、BF16、K update 和 feedback；所有 vector packet 汇集在 tail 才能释放。HSA 的 295–298 VGPR / 39–42 AGPR，以及 PMC AccVGPR 的 160→192/200，是这一 live-range 事实的机器证据。

PMC 没显示 occupancy cliff，因此不能把失败归结为原来的 `vgpr_no_regression` 规则；正确 gate 应继续是 **no scratch/no spill/no occupancy cliff/no extra barrier + benchmark**。本次正是通过这个 gate 后，实测证明它仍不盈利。

因此当前停止条件是：

1. 不继续 full-core d1、distance=2、software-pipeline expansion、第二套 LDS 或 core 内原位覆盖；
2. 不再扩大此 W/K/issue 搜索；
3. 保留 R4-tail 作为性能基线；
4. 若重新开启路线，必须先有不让全部 next packet 同时 live 到 tail 的新表示/commit 合法性证明。那属于新的机制，不应被伪装为本轮参数微调。

## 可复现入口

- 全量 fresh-process bench：`test/examples/linear_attention/vllm_compare/bench_qwen_gdn_persistent_recurrence_core_lastuse.py`
- full candidate correctness：`test/examples/linear_attention/vllm_compare/repro_qwen_gdn_persistent_recurrence_microtile.py --plan <上述计划>`
- exact-LTO replay：`test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/replay_qwen_v29_lto_mir.py`

执行时生成的本轮临时 MLIR/HSACO/ISA/MIR/PMC captures 位于容器 `/tmp/qwen_core_lastuse_*`；报告中的数值、命令和源内 benchmark harness 足以重新生成它们。
