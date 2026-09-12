# Qwen persistent recurrence：lastuse,d1 W 粒度复核

## 结论

指定的三个跨完整 core 的 distance-one 候选均已完成 P2、fresh-process、PMC、final-isel MIR 和 ISA 审计：

- `w1,k1,lastuse,d1`：长序列 slope 比 R4-tail 高 **0.92%**；T=8192 慢 **1.07%**。
- `w2,k1,lastuse,d1`：slope 高 **2.50%**；T=8192 慢 **2.09%**。
- `w4,k1,lastuse,d1`：slope 高 **2.91%**；T=8192 慢 **2.34%**。

因此可以正式排除“把 next W0 group-0 放到完整 current core 之前”的 d1 方向。小 W packet 没有带来可复现收益；W group 越大，长序列退化越明显。下一步若继续 native 路线，issue 点必须进入 core 内部、放在更晚的合法 last-use 之后，而不能再放在 core 前；不扩大搜索空间，也不做 distance=2。

## 范围与正确性

固定 `K=1`，仅运行：

| 计划 | W packets/group | d1 的跨 core 寄存器 packet |
| --- | ---: | --- |
| `gfx942_bt64_bv32_microtile_experimental_w1_k1_lastuse_d1` | 1 | W0 packet 0 |
| `gfx942_bt64_bv32_microtile_experimental_w2_k1_lastuse_d1` | 2 | W0 packet group 0（两个 packet） |
| `gfx942_bt64_bv32_microtile_experimental_w4_k1_lastuse_d1` | 4 | W0 packet group 0（四个 packet） |

所有三个计划在 T=64/128/512/2048 full nonzero-W P2 都通过。与 P2 host microscope 对比，`h`、pred、BF16 V-new/V-decay、每 chunk state 和 final state 全部 byte-equal；device contract 对比也全部满足原有容差。原始记录在：

`test/examples/linear_attention/vllm_compare/microtile_d1_w_granularity_artifacts/correctness/`

## d1 的实际机器形态

结构化 planner MLIR 中，W0 group-0 `amdgpu_qwen_k64_pipeline_stage_load` 位于唯一 `amdgpu_block_dot_bf16_f32` 之前；其同 region commit 位于该 core 之后的既有 workgroup barrier 边界。K 仍位于 tail，LDS commit 没有提前到 core 内，也没有新 barrier、private ring 或第二套 LDS。

所以依赖路径为：

```text
next-W0 global load -> [完整 current pred/update core] -> vmcnt/wait -> same-region LDS commit
                           48 × v_mfma
```

三项 d1 都跨过同一完整 core 的 **48 条 MFMA**。这个数字按有序 planner 依赖路径和最终 ISA 的每 core `v_mfma` 数审计；不能只依 objdump 的文本行号判断，因为不同 machine basic block 的打印顺序不等于运行路径。

| 指标 | R4-tail | w1,d1 | w2,d1 | w4,d1 |
| --- | ---: | ---: | ---: | ---: |
| HSA code-object VGPR count | 228 | 236 | 240 | 256 |
| HSA AGPR / SGPR | 32 / 43 | 32 / 43 | 32 / 43 | 32 / 43 |
| private scratch / LDS | 0 / 53,248 B | 0 / 53,248 B | 0 / 53,248 B | 0 / 53,248 B |
| HSA spill | 0 | 0 | 0 | 0 |
| final-isel MIR stack | `[]` | `[]` | `[]` | `[]` |
| PMC VGPR / AccVGPR / SGPR | 128 / 160 / 112 | 128 / 160 / 112 | 128 / 160 / 112 | 128 / 160 / 112 |
| PMC occupancy | 0.63417 | 0.63693 | 0.63396 | 0.63648 |
| static global-load / MFMA / LDS / barrier | 73 / 48 / 302 / 12 | 相同 | 相同 | 相同 |
| static `s_waitcnt` | 136 | 137 | 136 | 139 |

HSA note 的逻辑 VGPR count 确实随跨 core packet 增长；但 runtime PMC 的资源档位为 128 VGPR、160 AccVGPR、112 SGPR，三个 d1 与 R4-tail 相同，occupancy 也都约为 0.634--0.637。因此这不是一次 occupancy cliff 或 spill 造成的回退。

PMC 的动态计数也相同：MFMA=65,536、VALU=2,239,872、SALU=163,072、VMEM=202,240、LDS=381,952。换言之，d1 没有减少任何动态内存或矩阵指令；它只把 packet 的 live range 拉长。

## register-class copy 审计

final-isel MIR 的通用 `COPY` 数为 R4-tail 2,460，w1/w2/w4 分别为 2,462 / 2,464 / 2,468，即 +2 / +4 / +8。这与跨 core 的 W packet 宽度线性对应，说明 allocator 的虚拟 copy/live-range 压力确实增加。

但它没有变成额外的最终 ISA AGPR↔VGPR copy：四项均为 `v_accvgpr_read_b32=64`、`v_accvgpr_write_b32=64`、`v_mov=94`、`v_cndmask=196`。因此可归因的是较长的 VGPR live range/调度压力，而不是新发射的 register-class move、spill、额外 LDS 或 barrier。

## fresh-process 性能

协议：R4-tail 与每个 candidate 分别独立进程和 mode cache，HIP event 计时、无 graph capture、排除编译/分配；每个长度 5 warmup + 20 repeat，2 session，取 session median 的中位数。

| 实现 | T=1024 ms | T=2048 ms | T=8192 ms | T=8192 / R4-tail | 1024→8192 slope ms/chunk |
| --- | ---: | ---: | ---: | ---: | ---: |
| R4-tail | 0.213898 | 0.379193 | 1.410786 | 1.000000 | 0.01068650 |
| w1,k1,lastuse,d1 | 0.217964 | 0.379554 | 1.425918 | 1.010726 | 0.01078531 |
| w2,k1,lastuse,d1 | 0.213527 | 0.385092 | 1.440290 | 1.020913 | 0.01095324 |
| w4,k1,lastuse,d1 | 0.212135 | 0.383490 | 1.443805 | 1.023405 | 0.01099705 |

T=1024 的 w2/w4 小幅波动不具备长序列意义；T=2048 和 T=8192，以及端点 slope 都一致指向退化。故此处不是“VGPR 增加但 occupancy 不变且性能仍提升”的情形，不能据此把 `vgpr_no_regression` 改成仅 resource-cliff gate。当前零增长 gate 对这批完整-core d1 是保守但有效的早筛；未来若有一个更晚的 core 内 issue 计划，可改为同时记录逻辑 VGPR、runtime resource tier、occupancy 与实测 slope 的两级 gate，而不能只看其中一个。

## 导出物

- 性能：`test/examples/linear_attention/vllm_compare/microtile_d1_w_granularity_artifacts/fresh_process_r4_tail_vs_d1_w_granularity.json`
- PMC：`test/examples/linear_attention/vllm_compare/microtile_d1_w_granularity_artifacts/pmc/`
- 计划 MLIR、LLVM、HSACO、ISA：`test/examples/linear_attention/vllm_compare/microtile_search_artifacts/gfx942_bt64_bv32_microtile_experimental_w{1,2,4}_k1_lastuse_d1/`
- exact-LTO final-isel MIR：各 candidate 目录下的 `exact_lto_final_isel.mir`；R4-tail 对照在 `tail_issue_artifacts_t2048/exact_lto_final_isel.mir`。

MIR 由捕获的 `postopt_llvm.ll` 使用 AMD LLVM 22 的 `-march=amdgcn -mcpu=gfx942 -O3 -stop-after=finalize-isel` 再生，用于与代码对象/ISA 的一致性审计。
