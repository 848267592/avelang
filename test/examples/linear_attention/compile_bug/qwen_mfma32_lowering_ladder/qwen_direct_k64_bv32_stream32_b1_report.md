# Qwen Direct-K64 BV32 B1：Compiler-Owned Stream32 Full Recurrence

## 结论

**NO-GO：关闭 B1 `stream_t32` 路线，不晋级为 native performance baseline，
不继续尝试 token16/token64、layout、barrier 或 RA sweep。**

B1 成功实现了预期的 compiler-owned recurrence-step abstraction，并通过了所有
结构、correctness 和 resource gate：它将 final code-object 的 VGPR 从 B0 body 的
`460` 降到 `332`，AGPR 从 `204` 降到 `76`，LDS 从 `36,864 B` 降到 `26,624 B`，
没有 scratch 或 MIR spill。

但这没有转化为 full-sequence recurrence body 吞吐。相同 fresh-process、HIP-event、
preallocated、no-Graph 协议下，B1 在四个长度都慢于 B0；T=2048 慢 `34.0%`，其
per-chunk slope 反而从 `21.778` 升到 `28.655 us/chunk`。因此它没有满足任一
性能晋级条件，不能整理为正式 compiler PR。

本实验是 experimental-only：没有改 production selector、external current-vLLM
HSACO、allocator/RA、MFMA geometry、BV32 ownership、C0 block-dot lowering、LDS
layout 或 full-v29 nonzero-W 路线。

## 1. 前置证据与范围

B1 只以以下 B0 源码和 benchmark 为基线：

- `vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_sequence_b0.py`
- `vllm_compare/bench_qwen_gdn_direct_k64_bv32_full_sequence_b0.py`

开始前复查的结论决定了本轮只允许一个 stream32 candidate：

| 既有证据 | 本轮如何使用 |
|:--|:--|
| v22/v23 distributed V-new/V-decay | 只借鉴“producer 后立即 consumer”的调度原则；不复制 BT16、MFMA16 或旧 ABI。 |
| full v29 compact-K / exact LTO MIR | 已证明 full live-set composition 会触发资源问题；B1 不重启 broad/compact-K。 |
| `end_lifetime` 与 hard shared boundary | 均为负结果；不再用 frontend marker 或整块 pred/update boundary。 |
| pred serialization | 有限正收益，但不是解决 full composition 的局部控制杆。 |
| Direct-K64、block-dot specialized、typed BF16x8、C0 persistent block | B1 的 update 端逐项复用，不回退到 scalar/view/MFMA16。 |
| C0.5/C0.5S/D0-P | 局部 LDS layout/swizzle 路线已停止；B1 不创建新 swizzle 或 transpose。 |
| P0/P1/P2/P3/U0 | 复用已修复的 nonzero-W pred mapping 与 `BF16(V-new) -> FP32 -> BF16(V-decay)` ABI 边界。 |
| B0 | B0 是唯一 full-sequence Avelang-native correctness baseline 和唯一对照 A。 |

相关历史报告包括：

- `qwen_direct_k64_bv32_full_sequence_b0_report.md`
- `qwen_v29_full_mir_and_pred_streaming_report.md`
- `qwen_phase_boundary_real_pred_report.md`
- `qwen_direct_k64_update_bf16_vnew_boundary_fix_report.md`

## 2. 实现：一个 opt-in late recurrence-step op

新增的 experimental op 为：

```python
al.amdgpu.qwen_gdn_recurrence_step_bf16_f32(...)
```

相关实现文件：

- `lib/Dialect/AveLang/IR/AveLangOps.td`
- `lib/Dialect/AveLang/IR/AveLangOps.cc`
- `lib/IR/Intrinsics/amdgpu_module.cc`
- `lib/Dialect/AveLang/Transforms/lower_qwen_gdn_recurrence_step_pass.{h,cc}`
- `lib/Target/GPU/lower_to_llvm.cc`
- `vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_sequence_b1_stream32.py`

外层仍是 B0 的 device-side chunk loop。Python source 只建立 B0 相同的 persistent
`h_lo/h_hi`、相同 grid/WG/ABI 与四个 shared allocation，然后对每个 BT64 chunk 调用
一个 op；BT64 内部的 stream32 phase 由 late lowering 展开。环境变量
`AVELANG_QWEN_GDN_RECURRENCE_STEP_LOWERING=stream_t32` 才启用该 pass，默认路径
未变。

编译快照给出明确的 lowering 边界：

| 层次 | 观察 |
|:--|:--|
| `pre_kfrag_branch.mlir` | 恰好一个 `ave.gpu.amdgpu_qwen_gdn_recurrence_step_bf16_f32`。 |
| `post_late_lowering.mlir` | dedicated op 为零；出现 48 个带 `avelang.qwen_gdn.stream32.mfma` 标记的 MFMA call。 |
| `post_recurrence_step_lowering.mlir` / `post_block_dot_lowering.mlir` | op 不再存在；stream32 的 pred/update phase tag 继续存在。 |
| `final_mlir.mlir` | 无 `builtin.unrealized_conversion_cast`。 |

这证明 B1 不是“在高层写了一个 helper，随后自动回退到 B0” ：分叉真正发生在
AveLang-to-memref 之后、LLVM lowering 之前，并保留到 LLVM/MIR/ISA。

## 3. 精确 phase 图

冻结参数：BF16 K/W/U/H/V-new、FP32 g/initial/final state、`BT=64`、`BV=32`、
`WG=128`、32 CTA、two-wave cooperative ownership、pred/update 都为
`v_mfma_f32_32x32x8_bf16`。K32 accumulation order、state/output layout 与 B0 相同。

```text
for chunk in device:
  snapshot persistent FP32 state -> BF16 H and stateStage
  update_acc = 0

  for token32 in [0:32, 32:64]:
    P: wave0 K[0:64] pred partial; wave1 K[64:128] pred partial
       -> predPartial[2, 32, 32] FP32
    C: all 128 threads sum partials
       corrected = FP32(Ubf16) - pred
       v_new = BF16(corrected) -> required global output
       v_decay = BF16(FP32(v_new) * exp(g_last - g_t))
       -> vdecayStage[1, 32, 32] BF16
    U: current T32 only, Direct-K64 C0 persistent typed block
       wave0 consumes K[0:64]; wave1 consumes K[64:128]
       -> same persistent update_acc

  state = old_state * exp(g_last) + update_acc
  FP32 state feedback -> next chunk; write H/V-new/final-state contracts
```

pred MFMA 在 lowering 中保持 B0 的公开 source operand 顺序 `mfma(state, W, acc)`；
update 仍采用已经验证的 C0 block-dot B/A intrinsic convention。初版曾误把 pred 套用
update convention，T=64 首错为 token/K 转置；修复 operand order 后，B1 与 B0/P2
逐字节一致。这是本轮唯一的 correctness 修复，不是性能调参。

### 3.1 LDS physical reuse / lifetime

| physical allocation | 类型与大小 | pred-W | pred partial | V-decay | update-K | lifetime |
|:--|:--|:--:|:--:|:--:|:--:|:--|
| `stateStage[2,32,64]` | BF16, 8 KiB | read | - | - | - | chunk snapshot 到两次 pred 完成 |
| `phaseStage[64,64]` | BF16, 8 KiB | write/read W | - | - | write/read K | token32 内 W 完成后，在 barrier 后复用为 K |
| `predPartial[2,32,32]` | FP32, 8 KiB | - | write/read | - | - | 当前 token32 的 P 到 C |
| `vdecayStage[1,32,32]` | BF16, 2 KiB | - | - | write/read | read | 当前 token32 的 C 到 U |
| **合计** | **26,624 B** | | | | | 小于 B0 36,864 B |

初版有 16 个 static `s_barrier`，比 B0 多 3，不满足预注册上限。审计发现 C 结束后的
barrier 被紧随其后的 K-stage barrier 支配：两者之间没有 V-decay consumer，且所有
work-item 都先写完 V-decay、再参与 K staging。因此删除 C 后的独立 barrier，令 K-stage
barrier 同时关闭 V-decay producer phase 并开启 K phase。重新跑 T=64/128/512/2048 后
仍与 B0/P2 byte-exact；最终 static barrier 为 14，符合 `B0 + 2` 上限。

这不是新的 barrier sweep，只是由同一 producer/consumer hazard 证明的冗余同步合并。

## 4. 结构、MIR 与资源 gate

### 4.1 满足的 machine-structure 条件

- B1 final machine graph 与 B0 不同：LDS、VGPR/AGPR、barrier 和 ISA 指令族均不同。
- `predAcc` 在 `tokenHalf` lowering loop 内重新创建；两个 token32 不共享或并行持有
  pred accumulator。没有 token0/token32 pred fragment live overlap。
- `phaseStage` 是唯一 W/K physical allocation；没有 W/K full-block 双 allocation。
- `vdecayStage` 是 `[1,32,32]`，不存在 BT64 V-decay storage。
- B1 source/lowering 只对 `outputVNew` 做 store；update 只从 CTA-local
  `vdecayStage` load，故不存在 global V-new/V-decay reload。
- B0/B1 static MFMA 均为 48；T=2048 PMC 动态 MFMA 均为 65,536。
- MLIR 无 unrealized cast；code object 和 exact LTO MIR 均为 scratch/spill 0。

### 4.2 Final code-object resources

这里的 VGPR/AGPR/SGPR 来自 final HSACO metadata，不是 rocprof 的
`Accum_VGPR_Count` 代理。

| T=2048 benchmark body | B0 | B1 stream32 | B1 - B0 |
|:--|--:|--:|--:|
| VGPR | 460 | **332** | -128 |
| AGPR | 204 | **76** | -128 |
| SGPR | 38 | 40 | +2 |
| LDS group segment | 36,864 B | **26,624 B** | -10,240 B |
| private segment / scratch | 0 B | 0 B | 0 |
| VGPR spill count | 0 | 0 | 0 |
| SGPR spill count | 0 | 0 | 0 |
| static `s_barrier` | 13 | 14 | +1 |
| static MFMA32 | 48 | 48 | 0 |

exact-LTO replay的 pre/post greedy、virtregrewriter 与 prolog/epilog 皆没有
`SI_SPILL_AV32/AV64_SAVE` 或 reload。B1 满足 `VGPR < 384`、LDS 小于 B0、barrier
不超过 B0+2 的 resource GO 条件。

MIR 未输出一个可直接作为物理 peak 的 LiveIntervals 数字，故不将 lexical vreg 数量
误称为 peak。作为同一 pre-greedy section 的 composition proxy：

| pre-greedy MIR unique vreg declarations | B0 | B1 |
|:--|--:|--:|
| `vreg_64_align2` | 434 | 256 |
| `vgpr_32` | 927 | 899 |
| 全部 virtual-register declarations | 1,423 | 1,217 |

这个 proxy 与 final `VGPR=460 -> 332` 的方向一致：stream32 的确缩短/切分了部分
64-bit fragment live family；但它本身不能证明吞吐会改善。

### 4.3 同定义 ISA 文本计数

下表使用同一 grep classifier（global load/store、`ds_read`/`ds_write`、`s_`、非 MFMA
`v_`）；它是静态文本代理，不能取代 PMC 动态指令数。

| static body ISA proxy | B0 | B1 |
|:--|--:|--:|
| global loads | 115 | 115 |
| global stores | 130 | 130 |
| LDS reads | 70 | 64 |
| LDS writes | 216 | 216 |
| DS total | 286 | 280 |
| SALU | 608 | 616 |
| non-MFMA VALU | 2,063 | 1,488 |
| MFMA32 | 48 | 48 |

## 5. Correctness gate

测试使用同一 nonzero-W BF16 K/W/U、FP32 g 和 FP32 initial state。B1 分别比较
device-contract reference、B0 full sequence 和 P2 host microscope；后两者覆盖每个
chunk 的 pred/V-new/V-decay/state feedback。

| T | chunks | B1 vs B0 | B1 vs P2 | finite | 结论 |
|--:|--:|:--:|:--:|:--:|:--:|
| 64 | 1 | H/pred/V-new/V-decay/state/final 全 byte-exact | 全 byte-exact | 是 | pass |
| 128 | 2 | 全 byte-exact | 全 byte-exact | 是 | pass |
| 512 | 8 | 全 byte-exact | 全 byte-exact | 是 | pass |
| 2048 | 32 | 全 byte-exact | 全 byte-exact | 是 | pass |

相对独立 device-contract reference 的误差仍在既有阈值内。T=2048 最大值为：H/V-new
`2.44140625e-4`、raw pred `2.19490612e-5`、final state `4.03299928e-5`；没有放宽
BF16、pred 或 state 门槛。

## 6. T=2048 PMC

PMC 是单次 audit-elided B1 body dispatch，B0 值来自同一既有 B0 protocol。HIP event
formal timing才是下一节的性能依据；不把 profiler trace 当作 latency。

| dynamic metric | B0 | B1 | change |
|:--|--:|--:|--:|
| MFMA | 65,536 | 65,536 | 0 |
| VMEM | 315,904 | 315,904 | 0 |
| VALU | 2,865,536 | 1,988,800 | -876,736 |
| SALU | 161,216 | 166,208 | +4,992 |
| LDS instructions | 495,616 | 483,328 | -12,288 |
| LDS block | 36,864 B | 26,624 B | -10,240 B |
| scratch | 0 B | 0 B | 0 |
| occupancy percent | 0.6464 | 0.6493 | nearly unchanged |
| rocprof `Accum_VGPR_Count` | 336 | 208 | metadata only |

最重要的事实是：B1 降低了 VALU/LDS/physical registers，却保持完全相同的 MFMA 和
VMEM。故性能回退不是以减少数学或 global traffic 换来的；它是在更细粒度 phase
串行化下发生的。

## 7. 正式 full-sequence body benchmark

协议严格复用 B0：preallocated、current HIP stream、no CUDA Graph、warmup=5、
repeat=20、5 fresh-process sessions。四臂在同一个 worker 中使用 rotate 的
palindromic order `[A,B,C,D,D,C,B,A]`；每实现每 session 有 40 个 HIP-event 样本。
结果为 session medians 的中位数。这是 recurrence body 诊断，不是 Eager public API
排名。

| T | chunks | B0 ms | B1 ms | Triton direct ms | external bridge ms | B1 / B0 | B1 / Triton |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 512 | 8 | 0.157494 | 0.178225 | 0.072107 | 0.040140 | 1.132x | 2.472x |
| 1024 | 16 | 0.357432 | 0.471020 | 0.100970 | 0.067019 | 1.318x | 4.665x |
| 2048 | 32 | 0.678028 | 0.908771 | 0.152567 | 0.117635 | **1.340x** | 5.957x |
| 8192 | 128 | 2.779951 | 3.647381 | 0.463849 | 0.419243 | 1.312x | 7.863x |

配对 B1-B0 session-median difference 的 20,000 次 cluster bootstrap：

| T | median B1-B0 | 95% CI | 判断 |
|--:|--:|--:|:--|
| 512 | +20.810 us | [+20.154, +21.067] us | 稳定回退 |
| 1024 | +113.529 us | [+112.852, +113.809] us | 稳定回退 |
| 2048 | +231.344 us | [+230.727, +231.829] us | 稳定回退 |
| 8192 | +867.649 us | [+865.583, +871.460] us | 稳定回退 |

拟合 `latency = intercept + slope * chunks`：

| arm | intercept ms | slope us/chunk |
|:--|--:|--:|
| B0 | -0.008584 | 21.778488 |
| B1 stream32 | -0.016778 | **28.654932** |
| direct Triton | 0.047762 | 3.252429 |
| external current-vLLM bridge | 0.016042 | 3.151469 |

B1 相比 B0 的 slope **恶化 31.6%**，而不是预期至少下降 20%；T=2048 不仅未快 20%，
反而慢 34.0%。因此四个性能 gate 全部失败。

## 8. 为什么资源改善仍然更慢

可以确认的事实：

1. B1 减少了 physical VGPR/AGPR/LDS，也没有 spill；因此不是 resource cliff 导致的
   回退。
2. MFMA 和 VMEM 动态总数完全相同；因此不是少算/多算，或 global V-new reload。
3. B1 的 VALU/LDS 动态数更低，却在每个 chunk 增加约 `6.88 us` 的拟合 slope。
4. B1 把一个 BT64 内较宽的 producer/consumer region 改为两个强制顺序的
   `P -> C -> U` token32 region。它降低 live set，但也降低了同 CTA 内可重排/可隐藏的
   工作窗口，并额外保留一个 barrier。

合理但仍属推断的解释是：该 stream32 schedule 的数据依赖和同步边界让 MFMA、LDS 与
地址计算的调度/latency hiding 更差，抵消并超过了资源节省。PMC 没有提供一个能把这
`6.88 us/chunk` 唯一归因到某一条指令的因果证据，所以报告不将其宣称为确定的硬件根因。

## 9. 与 v23 的关系

不能把 B1 与 v23 的历史 chunk_gdr 数字直接当作同合同横向排名：v23 使用不同的
distributed V-new/V-decay 调度、BT16/MFMA16 时代的路径和旧 ABI。B1 的唯一可信
performance 对照是同 ABI、同 BF16 boundary、同 Direct-K64/BV32 ownership、同
device-side chunk loop 的 B0，以及同 harness 的 Triton/bridge diagnostic controls。

## 10. 决策

```text
stream32_structure_pass       = true
stream32_correctness_pass     = true
stream32_resource_pass        = true
stream32_performance_promoted = false
stream32_route_closed         = true
formal_compiler_pr            = false
```

保留 dedicated recurrence-step op、late pass、B1 source 和完整工件作为一条高价值的
negative compiler scheduling evidence：**减少 live set/寄存器并不足以保证 full recurrence
吞吐提升**。但不应把该 op 推为默认 API 或正式 compiler PR，因为它只服务于这一个
experimental schedule，且完整 body 明确退化。

下一步不得在本路线继续 token16/token64、LDS layout、barrier 或 RA sweep。恢复 B0
作为唯一 Avelang-native full recurrence correctness baseline；external current-vLLM
HSACO bridge 仍是当前较快的 matched-ABI recurrence control。

## 11. 工件与复现

主要工件：

```text
rocprof_outputs/qwen_direct_k64_bv32_full_sequence_b1_stream32/
  t64_b1_only_structure_v2/                         # op/lowering snapshots
  correctness_after_barrier_merge/                   # B0/P2/reference matrix
  body_t2048_resource_after_barrier_merge/           # HSACO, ISA, exact LTO MIR
  profile_t2048/                                     # B1 PMC CSV
  formal_body_benchmark_clean/results.json           # 5-session four-arm timing
```

关键命令：

```bash
cd /workspace/project/avelang
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:./test/examples/linear_attention/vllm_compare
export PYTHONDONTWRITEBYTECODE=1

# Correctness first.
python3 test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_sequence_b1_stream32.py \
  --T 64 128 512 2048 --json

# Only after all gates: diagnostic body benchmark.
python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_direct_k64_bv32_full_sequence_b1_stream32.py \
  --T 512 1024 2048 8192 --warmup 5 --repeat 20 --sessions 5 --json
```
