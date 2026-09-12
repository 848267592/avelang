# Qwen persistent recurrence：去 private packet-ring 的 LDS 实验

日期：2026-07-30
基线：`gfx942_bt64_bv32_joint_v4`（R4）
候选：`gfx942_bt64_bv32_software_pipeline` 的 distance=1 通用调度器，后端为 direct-LDS packet commit。

## 结论

本轮完成了一个可编译、可运行的 no-private-memory distance=1 候选：四个 next W/K packet 不再进入 private `memref` ring，而是由 late lowering 直接执行 global BF16x8 load 并写入既有的 LDS W/K bank。它保留了 scheduler 生成的 prologue、带四个轻量 `i64` iter-arg 的 steady-state 和 epilogue，T=64/128/512/2048 均通过 nonzero-W correctness。

private packet-ring 确实是旧候选的 1024 B Scratch 和额外 VMEM 的来源；移除后 T=2048 PMC 的 Scratch 变为 0，VMEM 从 264,640 降至 201,152（甚至略低于 R4 的 202,240）。不过长序列仍慢于 R4：T=1024/2048/8192 分别慢 4.25%/5.33%/8.44%。因此本轮结论是：**private ring 不是当前长序列回归的剩余主因；剩余成本是单一 `phaseStage` 强制的 retire/barrier 边界和更高的 AccVGPR，而不是应继续尝试 distance=2。** 本轮没有做 distance=2。

需要严格区分两件事：本次交付的 direct-LDS 候选是正确、无 private ring 的完整 scheduler 接入；但它并不是语义上可成立的“W/K 物理角色每轮互换”的真正双槽旋转版本。后者已做过单变量尝试并被 P2 正确性否决，原因和下一步接口要求见“物理双缓冲 No-Go”。不能把当前同 bank identity 的 direct-LDS 版本误报为已实现的双缓冲。

## 实现

改动集中在：

- `lib/Dialect/AveLang/Transforms/qwen_modulo_software_pipeline_pass.cc`
- `lib/Dialect/AveLang/Transforms/lower_qwen_k64_pipeline_stage_pass.cc`
- `test/examples/linear_attention/vllm_compare/repro_qwen_gdn_persistent_recurrence_software_pipeline.py`

调度器仍以通用 `scf.for` 克隆/映射实现，而非固定 Qwen ISA 列表：它识别四个已有 W/K producer stage，保留 R4 的 BF16x8 typed producer、BV32 ownership 与 LDS-mediated K retile，并生成下列控制结构。

```text
prologue:  issue P0 -> R4 LDS commit -> consume P0，并产生 P1 token
steady:    scf.for(... iter_args(Pi.w0, Pi.w1, Pi.k0, Pi.k1))
             logical consume Pi（late lowering 不再物化 payload ring）
             barrier
             current W local load -> pred MFMA -> V-new/V-decay
             current K local load -> update MFMA -> FP32 state feedback/audit
             barrier（完整 current core retire 后）
             issue Pi+1 -> direct LDS commit
             yield Pi+1 的四个 i64 token
epilogue:  consume最后一个 token，执行最后 current core，不再 issue 无用 packet
```

调度距离仍为 1：`Pi+1` 的 producer 位于第 i 次 steady 迭代的 tail，而 token 在第 i+1 次迭代入口作为 `iter_arg` 消费。`post_software_pipeline_scheduler.mlir` 可见 prologue 的四个 `direct_lds_commit`、steady `scf.for` 的四个 `iter_args`、四个 `logical_lds_consumer` 以及 epilogue drain；logical commit 仅表达 scheduler dependence，late lowering 将其删除，不再建 private payload。

late lowering 对标记为 `direct_lds_commit` 的 stage 直接调用现有 `emitPacketLoads` 与 `emitPacketCommit`；若本模式落回旧的 private packet-ring 路径则报错。旧的泛用途 ring helper 仍留在源文件中服务非本模式路径，但最终候选的 post-lowering MLIR、LLVM、MIR 和 ISA 都没有 packet ring。

为避免 reference 对比在同一 Python 进程意外匹配 candidate lowering，correctness harness 在 candidate 已 dispatch 且 synchronize 后显式切回 R4 环境再启动 reference。

## private frame / VMEM 归因

旧 distance=1 候选的 `post_joint_v1_stage_lowering.mlir` 有四个彼此独立的：

```mlir
memref.alloca() : memref<4xvector<8xbf16>, #gpu.address_space<private>>
```

每个均标有 `avelang.qwen.software_pipeline.packet_ring = "per_lane_distance_1"`，各自有四次 store 和四次 load。exact-LTO 的最终 MIR 与之对应地有四个 variable-sized frame object（另有一个 4-B object）；旧 ISA 还包含 32 条 `scratch_load` 和 32 条 `scratch_store`。这三层共同确认 Scratch 是四个 packet ring，而非 spill：旧 MIR `sgpr_spill_count`/`vgpr_spill_count` 都为 0。

| T=2048 指标 | 旧 private-ring d=1 | 本轮 direct-LDS | R4 |
| --- | ---: | ---: | ---: |
| PMC Scratch_Size | 1024 B | 0 B | 0 B |
| HSA private_segment_fixed_size | 16 B | 0 B | 0 B |
| 动态 `SQ_INSTS_VMEM` | 264,640 | 201,152 | 202,240 |
| `scratch_load` / `scratch_store`（静态 ISA） | 32 / 32 | 0 / 0 | 0 / 0 |
| workgroup LDS | 53,248 B | 53,248 B | 53,248 B |

因此本次改动消除了旧候选 63,488 次 VMEM 的主增量；direct-LDS 比 R4 少 1,088 次 VMEM。注意 code-object 的 `private_segment_fixed_size=16` 是静态 metadata，旧 PMC 的 1024 B 是运行时 Scratch 分配，二者不是同一计量口径；MIR 的四个 variable-sized frame object 将它们与 packet ring 对齐。

## 机器审计

本轮 T=2048 exact-LTO replay 的最终 kernel section 是 `kernel_section_19.mir`：`NoVRegs`，无 SGPR/VGPR spill；direct-LDS HSA metadata 为 `private_segment_fixed_size: 0`、`group_segment_fixed_size: 53248`、`vgpr_count: 272`、`agpr_count: 64`、`sgpr_count: 50`。PMC dispatch resource 字段为 LDS=53248、Scratch=0、VGPR=128、AccVGPR=248、SGPR=112。

直接与旧 private-ring 候选比较，静态 MFMA/global/LDS 数不变（128 / 122 / 504 / 288），barrier 仍为 32；`s_waitcnt` 从 328 降至 289。也就是说，移除 private ring 没有偷改 R4 数学或 K layout，且确实删掉 scratch traffic；但并没有删掉由 shared ownership 产生的跨 workgroup 同步。PMC（每次 T=2048 kernel dispatch）记录：MFMA=65,536，LDS=385,024，SALU=159,808，VALU=2,353,408，VMEM=201,152，OccupancyPercent 约 0.635。

最终 ISA 的交错也符合这一点：current MFMA/LDS read 段之后出现 `s_waitcnt lgkmcnt(0); s_barrier`，随后才是 next packet 的 `global_load_*`、`ds_write_b16` 和再次同步；不存在原先的 `scratch_load/store`。所以这不是把 private memory 换名：global-to-LDS packet 写入真实发生了。但为了保证 opaque recurrence 的完整读集已退休，next packet 被排在 current core 尾部，无法与该 core 的关键 MFMA 区间重叠。这正解释了 VMEM 已恢复但长序列没有获得吞吐收益。

## 物理双缓冲 No-Go

曾尝试严格按单变量方案复用既有 W/K bank：W 被 pred 消费后写入 next K，K 被 update 消费后写入 next W，并在下一轮翻转逻辑角色。这个候选在 T=128 的 P2 比较从 chunk 1 起失败（`pred_f32` 最大绝对差约 0.013713），故未保留。

原因不是 SSA token 或 waitcnt。`AMDGPUQwenGdnRecurrenceStepBF16F32Op` 仍拥有一个未暴露给外部 packet token 的实体 `phaseStage`；其 lower pass 以 `op.getPhaseStage()` 先作为 W local stage，再重用为 K stage。外部 `AMDGPUBlockDot`/stage token 改映射并不会改变这个 opaque op 内部使用的物理 LDS identity。因此 scheduler-only 的 role swap 会让“外层 packet 指向的 bank”和 recurrence step 实际读取的 `phaseStage` 脱节。

真正的 rotating LDS double buffer 需要扩展 recurrence op 及其 lowering：把 current/next phase stage（或可选择的 phase-stage index）显式化，并将该选择一路传给 W/K local-load、pred/update 和反馈审计；之后还必须做 LDS 容量/occupancy gate。现有注释估计增设第二 W/K pair 会把 53,248 B 推至 85 KiB 以上，不能在 scheduler 内静默分配。因此当前结果是对 scheduler-only 双缓冲方案的明确 No-Go，而不是继续挪单条 load 或调 waitcnt。

## 正确性

所有用例为 full nonzero-W recurrence。P2 host microscope 的 `h`、pred、V-new、V-decay、state 和 final-state 均 byte-equal；device contract 仅有既有 BF16/FP32 容差内舍入差异，全部 `pass=true`。

| T | chunks | 结果文件 |
| ---: | ---: | --- |
| 64 | 1 | `swp_direct_lds_t64/software_pipeline_correctness.json` |
| 128 | 2 | `swp_direct_lds_simple_t128/software_pipeline_correctness.json` |
| 512 | 8 | `swp_direct_lds_t512/software_pipeline_correctness.json` |
| 2048 | 32 | `swp_direct_lds_t2048/software_pipeline_correctness.json` |

构建也已完成：`ninja -C build-software-pipeline python/_avelang_bindings.cpython-312-x86_64-linux-gnu.so`。

## fresh-process body benchmark

每个 implementation/session 使用独立 Python worker、HIP event 计时、warmup=5、repeat=20、session=2、无 graph capture；四份原始 JSON 已放入 `swp_direct_lds_artifacts_t2048/benchmark/`。数值为两个 session median 的中位数。

| T | chunks | direct-LDS ms | R4 ms | 相对 R4 | Triton ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 8 | 0.131425 | 0.135431 | 0.9704x（快 3.05%） | 0.065648 |
| 1024 | 16 | 0.277673 | 0.266346 | 1.0425x（慢 4.25%） | 0.093239 |
| 2048 | 32 | 0.524730 | 0.498180 | 1.0533x（慢 5.33%） | 0.146508 |
| 8192 | 128 | 2.174872 | 2.005670 | 1.0844x（慢 8.44%） | 0.441036 |

从 T=2048 到 8192，候选的增量斜率约 17.19 us/chunk，R4 为约 15.70 us/chunk；额外约 1.49 us/chunk 与“current core 完整 retire + producer LDS 写入 + 下一轮 barrier”的串行边界一致。private VMEM 已清除而 slope 仍扩大，进一步排除了 packet ring 是剩余性能问题的解释。

## 工件

完整 T=2048 导出位于 `test/examples/linear_attention/vllm_compare/swp_direct_lds_artifacts_t2048/`：

- `mlir/`：planner、scheduler、late lowering、LLVM 前后快照；
- `exact_lto_mir/`：exact link argv、bitcode、最终 MIR section 与 replay summary；
- `software_pipeline_direct_lds_t2048.isa.s` 与 `hsaco_notes.txt`；
- `hsaco/`；
- `pmc_csv/0364d3a007f9/682049_counter_collection.csv`；
- `benchmark/`：四个 fresh-process benchmark JSON。

这些工件与旧 private-ring 的 `swp_final_artifacts_t2048/` 并存，未覆盖 R4 或此前候选。
