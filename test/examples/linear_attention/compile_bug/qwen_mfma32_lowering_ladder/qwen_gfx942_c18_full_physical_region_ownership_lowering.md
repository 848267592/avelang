# Qwen gfx942 Stage 6Z C18：Full Physical Region Ownership Lowering

## 1. 实验定位

本报告记录 Direct-K64、BT64、BV64、BK32、WG256 chunk-o 路线中的
C18-FPR（Full Physical Region Ownership Lowering）实验。

本轮不是新的性能调参，也不是 Z5B、C17、BDV2-P2 的重新测量。C18 的
唯一目标是验证：能否由编译器内部的统一 FullPhysicalRegionPlan 同时拥有
Q/H/K/V 的 producer、shared placement、consumer feeding 和 lifetime，
从而真正脱离旧的 P2 机器图。

本轮明确冻结：

- gfx942 / wave64；
- BT64、BV64、BK32；
- WG256、2 CTA/chunk-head；
- MFMA32 数学和 K32 accumulation order；
- BF16 Q/K/H/V-new、FP32 g/state；
- causal mask、输出布局、caller-owned output；
- 不修改 allocator/RA；
- 不修改 production selector；
- 不调用 external HSACO；
- 不实现 X2 full graph；
- 不运行正式 Eager public API 排名；
- 不做新的 latency benchmark。

因此，本报告中的 latency 与 PMC 只用于机器图诊断，不能当成正式性能排名。

## 2. 先给结论

C18 得到的是一个有价值但未完成的结果：

1. C18 的 compiler-internal FullPhysicalRegionPlan 已被真正创建并进入
   block-dot lowering。
2. C18 的 V-new 路径已经由 compiler-owned source plan 接管，旧的
   手写 Phase-C V producer、phase_vec 和手写 V MFMA 循环已经删除。
3. C18 与 P2 的 post-block MLIR、pre-opt LLVM、post-opt LLVM、exact-LTO
   HSACO 和 final ISA 均不同。因此本轮没有触发
   STOP_C18_MACHINE_OWNERSHIP_NOT_ESTABLISHED。
4. C18 在所有要求的长度和 edge case 上与 Z5B BF16 byte-exact，且 finite、
   caller-owned output、NaN-prefill 检查全部通过。
5. 但是，C18 仍不是完整的 Full Physical Region：
   - Q 仍主要由 Z5B 的 dedicated Q cache source producer 提供；
   - H/K 的 consumer feeding 仍复用旧 P2 lowering；
   - C18 的 shared lifetime 计划目标是约 24576 B，但实际 source/LLVM/HSACO
     仍为 32768 B；
   - 因此 Q/H/K/V 全链路的 ownership、shared allocation 和 lifetime 尚未
     由同一计划统一兑现。
6. C18 的 T=2048 diagnostic PMC 相比 P2 有收益：
   - VMEM：448 -> 416 / CTA；
   - VALU：8474 -> 8010 / CTA；
   - MFMA：保持 160 / CTA；
   - 但 LDS：592 -> 800 / CTA；
   - SALU：780 -> 776 / CTA；
   - C18 trace median：43.745 us；
   - P2 trace median：44.947 us。
7. 由于本轮只允许 C18 机器图和 correctness 诊断，不能据此宣布性能晋级。
   正式结论是：

   **C18 machine-distinct gate = PASS；V compiler-owned gate = PASS；
   full physical-region completeness gate = FAIL；C18 = NO-GO，
   不接入生产路径，不运行正式 full benchmark。**

这不是“C18 没有改变机器图”，而是“机器图已经改变，但统一完整 ownership
模型仍不完整”。

## 3. 已冻结的前序事实

### 3.1 C17 结果

C17 的历史结论保持不变：

| 项目 | C17 结果 |
|:--|--:|
| full correctness | T=64/128/512/1024/2048/4096/8192/16384 通过 |
| dynamic MFMA / CTA | 160 |
| dynamic VMEM / CTA | 448 |
| dynamic LDS / CTA | 592 |
| dynamic VALU / CTA | 8474 |
| dynamic SALU / CTA | 780 |
| code-object VGPR / AGPR | 132 / 48 |
| LDS | 32768 B |
| scratch / spill | 0 / 0 |
| exact HSACO | f612f23968d1e9b9e205b1daf75a70cef2d9016d56f44df24fef6d3ae20c7947 |

C17 的关键限制是：虽然 source 中出现了 V 相关计划属性，但
V_source_level_consumer_owned_by_c17=false，Phase-C V 仍然是手写路径。
C18 的第一项任务就是补上这个缺口。

### 3.2 P2 结果

C18 以同一大 op 和同一 compiler path 的 P2 机器图为对照。P2 fresh
T=2048 PMC 为：

| 指标 | P2 / CTA |
|:--|--:|
| MFMA | 160 |
| VMEM | 448 |
| LDS | 592 |
| VALU | 8474 |
| SALU | 780 |

P2 exact code object：

- VGPR = 132；
- AGPR = 48；
- SGPR = 30；
- LDS = 32768 B；
- private segment = 0；
- 无 AV32/AV64 spill save/reload。

### 3.3 Z5B 参考

Z5B 是冻结的 isolated reference，不是 C18 的直接 source parent：

| 指标 | Z5B / CTA |
|:--|--:|
| MFMA | 160 |
| VMEM | 672 |
| LDS | 672 |
| VALU | 7072 |
| SALU | 768 |
| code-object VGPR / AGPR | 104 / 32 |
| LDS | 32768 B |
| scratch / spill | 0 / 0 |

Z5B 仍然是历史 isolated baseline。C18 不得把 P2/C17 的局部 machine
收益写成 Z5B 的正式替代。

## 4. C18 的设计

### 4.1 compiler-internal plan

C18 在
lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc
中新增了 compiler-internal：

FullPhysicalRegionPlan

该结构用于描述：

- physical chunk-o plan；
- Q cache region；
- score-V shared slot；
- phase shared region；
- Q dual-consumer 属性；
- V consumer-owned 属性；
- shared region reuse 属性。

它不是新的 public Qwen op，也没有把完整 Qwen schedule 硬编码到
source-facing intrinsic 中。

### 4.2 C18 source contract

C18 source 文件：

test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_c18_full_physical_region.py

source 仍然使用同一个
al.amdgpu.block_dot_bf16_f32_logical logical operation 表达 block-dot
consumer。环境变量：

AVELANG_STAGE6Z_FULL_PHYSICAL_REGION=c18

用于选择实验性的 compiler lowering。

source 中保留的冻结内容：

- Q dedicated full cache；
- H/K logical block-dot；
- BF16 V-new input；
- FP32 accumulator；
- phase-separated accumulator 顺序；
- BF16 output；
- 既有 causal 和 K32 数学顺序。

本轮最后清理掉了旧的未使用 phase_vec = al.view(...)，因此 source 中不再
残留 Phase-C 的手写 phase vector。

### 4.3 C18 V source path

C18 的 V logical op 使用 vn 作为两端 source operand：

resolved_args[2] == resolved_args[3]

compiler 以这个条件识别 C18 V source，而不是通过新的 Qwen 专用 public op
识别。

C18 V producer 的 source-level 逻辑由 compiler lowering 生成：

- 256 个线程；
- 2 个 packet repetition；
- 每个 packet 是 BF16x8 global load；
- 由 token、packet、value offset 计算 score-V shared slot；
- 写入 score_v shared region；
- 单次 producer barrier；
- 后续 score@V consumer 直接消费该 region。

这条路径的关键证据是：V 的 producer、slot 和 consumer 都带有
FullPhysicalRegionPlan 相关属性，而不是由 source 手写 Phase-C producer
和手写 MFMA。

## 5. current ownership 与 target ownership

机器可读版本：
stage6z_c18_current_vs_target_ownership.json

### 5.1 对照表

| region | 当前 P2/C17 组织 | C18 target | C18 实际状态 |
|:--|:--|:--|:--|
| Q global producer | Z5B dedicated Q cache source loop | FullPhysicalRegionPlan Q producer | 未完全接管 |
| Q shared placement | Q cache +旧 phase consumer | unified Q physical region | partial |
| H producer | 旧 P2 full-scope producer | plan-owned H producer | 带 C18 属性但仍复用旧公式 |
| K producer | 旧 P2 full-scope producer | plan-owned K producer | 带 C18 属性但仍复用旧公式 |
| V-new producer | source Phase-C 手写 | plan-owned V producer | 已接管 |
| Q@H consumer | P2 consumer path | plan-owned Q@H | partial |
| Q@K consumer | P2 consumer path | plan-owned Q@K | partial |
| score@V consumer | source/manual Phase-C | plan-owned score@V | 已接管 |
| shared lifetime | P2 32 KiB allocation | plan-derived reusable regions | 未完全兑现 |
| final release | old phase release | plan-owned release | partial |

所以 C18 不是空的 annotation-only pass，但也不是 full plan 已经控制所有
physical regions。

### 5.2 必须保留的负面结论

不能因为 function attr 出现：

avelang.stage6z.full_physical_region =
"gfx942_bt64_bv64_full_region_c18"

就宣称 Q/H/K/V 全部已经由 FullPhysicalRegionPlan 实际生成。

本轮证据只支持：

- plan created；
- V producer/consumer ownership real；
- H/K plan metadata present；
- Q/H/K old physical path not fully replaced；
- full shared lifetime not realized。

## 6. P2 convergence audit

机器可读版本：
stage6z_c18_p2_convergence_point.json

### 6.1 P2 的 convergence

P2 与更早的 C17/P2 线路在 pre-opt LLVM 层仍表现为等价或高度收敛。
这就是此前“只添加 plan 属性但最终机器图不变”的根源。

本轮没有重新对 C17 做无关的全量重审；C18 只执行了必要的 P2 对照与
post-lowering 到 exact-LTO 的 convergence 检查。

### 6.2 C18 的首次 divergence

C18 在 post-block-dot lowering 就出现了结构性差异。随后差异保留到：

- post-bounded-packet-schedule MLIR；
- post-kfrag-load-lowering MLIR；
- post-kfrag-rewrite MLIR；
- pre-opt LLVM；
- post-opt LLVM；
- exact-LTO MIR；
- final ISA；
- HSACO。

因此没有触发：

STOP_C18_MACHINE_OWNERSHIP_NOT_ESTABLISHED

### 6.3 机器 hash 证据

#### C18

| artifact | SHA256 |
|:--|:--|
| source current | 67885841ac969b65655b269e52ff24771d8bf67862e60af4f0c5785db005fb14 |
| post-block MLIR | f472a8cbcf79ef14cf5bf443d75dbbf8f38496041072f96567ffccc75329df3e |
| pre-opt LLVM | db3bd7daeb2af293a1e4b0eeec1491d829622d5bc9dfea91998d29140d057905 |
| post-opt LLVM | 89bd3764e3b0e96b9459d487430be94d43365f7f930decd17acd3e82598039f1 |
| exact HSACO | a4af6a4c7b972400ec5f6ce7a9a732e54b7916edf10ce9d157f078c0c32c94fe |
| final ISA | 939a3a42dbe0c0807a500f01d8064b5e7316058ce8c27a63b932bcce3e17e2cd |

#### P2

| artifact | SHA256 |
|:--|:--|
| post-block MLIR | da6cd14458b0f93b965a49e13580ac7e93824aee82d002d017724ceda7b709c1 |
| pre-opt LLVM | e633d0938887dfdb09ba300125f2c070b8168328deaaa5f2e64542ead508ebf9 |
| post-opt LLVM | 8e2122d3ac09a80f12996558f48862c33bc7266d548a2fa70cb299b30ccf1152 |
| exact HSACO | d17483b933a7abf68b34b0e193aa4e3e5d9f93f48a12cfa8d84b285c9091af5a |
| final ISA | 9e45a2dea7c35d69c628c8c8bd1646ae32d469d16c93eb56c01e35671e6b2c96 |

hash 全部不同，且差异不是只来自 source filename 或 metadata。C18
post-block MLIR 中出现了新的 V plan producer、score-V slot 和 C18
consumer attrs，LLVM/MIR/ISA 也产生了对应不同的 load、LDS 和 accumulator
安排。

## 7. post-block MLIR 证据

机器产物：

codex_qwen_gfx942_c18_full_physical_region_t2048/ir/post_block_dot_lowering.mlir

关键 function attrs：

~~~text
avelang.stage6z.full_physical_region =
  "gfx942_bt64_bv64_full_region_c18"
avelang.stage6z.full_region_owner =
  "FullPhysicalRegionPlan"
avelang.stage6z.v_source_consumer_owned = true
~~~

关键 C18 V attrs：

~~~text
c18.full_region.producer = "V_global_bf16x8_once"
c18.full_region.shared_slot = "score_v"
c18.consumer = "score@V"
c18.plan_role = "V"
c18.producer_owner = "FullPhysicalRegionPlan"
c18.full_region = "gfx942_bt64_bv64_full_region_c18"
~~~

post-block MLIR 计数：

| 项目 | 数量 |
|:--|--:|
| c18.full_region.producer = V_global_bf16x8_once | 2 |
| c18.full_region.shared_slot = score_v | 16 |
| c18.consumer = score@V | 2 |
| c18.plan_role = V | 2 |
| c18.producer_owner = FullPhysicalRegionPlan | 6 |
| logical block-dot MFMA operand op | 3 |
| vector.load | 7 |
| vector.store | 5 |
| memref.store | 48 |

这些 lexical 计数只用于证明 IR 图发生了何种变化，不能直接当成动态
instruction count。

## 8. C18 full physical plan 与实际落地的差距

机器可读版本：
stage6z_c18_full_physical_region.json

### 8.1 已实现的部分

- FullPhysicalRegionPlan 能够在 specialized + first-class block-dot path
  中创建；
- C18 V source role 能够被 compiler 正确识别；
- V 的 global producer、score-V shared slot 和 score@V consumer 由同一
  plan 属性连接；
- Q/H/K/V 的 logical block-dot 仍使用统一 operation；
- C18 V global source shape、dtype 和 rank verifier 已补齐；
- C18 V 的 Phase-C source 手写 producer 已删除；
- C18 V 不再 global reload；
- full correctness 不需要放宽误差。

### 8.2 未实现的部分

- Q 的 producer 仍由 source-level Z5B dedicated Q cache loop 承担；
- H/K 的 physical producer 与 consumer 仍保留 P2 的 legacy formula；
- H/K 的 typed shared/dot operand 没有全部由统一 C18 plan 重写；
- score-V shared region 没有与所有旧 phase region 做完整 lifetime coalescing；
- target 24576 B shared plan 未落成实际 24576 B allocation；
- 因而 C18 不是“Q/H/K/V 全部由 FullPhysicalRegionPlan 生成”的完整候选。

这个结果说明：属性和 plan object 已经足以切入 lowering，但要让整个 region
真正成为 compiler-owned，还需要把旧的 P2 producer/consumer lowering 从
Q/H/K 路径中移出。那属于下一条完整 ownership implementation，不应在本轮
偷偷补做第二个候选。

## 9. V full-region ownership 结果

机器可读版本：
stage6z_c18_v_full_region_ownership.json

### 9.1 source-level 变化

C17/P2 原先的 Phase-C source 做了以下工作：

- source 直接构造 phase_vec；
- source 写入 V phase shared region；
- source 手写 V MFMA operand 和 accumulator 循环。

C18 source 删除这些内容，改为三个 logical V block-dot calls，共同使用
vn 作为 source operand。compiler 通过 C18 V plan 生成 producer 和 consumer。

因此 V 这条链确实满足：

global V-new -> plan-owned shared score_v -> score@V consumer

### 9.2 V 的机器证据

C18 V logical lowering：

- V producer global vector loads：2 个 lowered vector load region；
- score-V shared slot：16；
- score@V consumers：2；
- V global reload：不存在；
- V source-level logical producer pass：1；
- old source phase_vec：已删除；
- old source handwritten V MFMA：已删除。

这里的“2 个 lowered vector load region”是静态 IR/ISA producer
region 证据，不能直接理解成每 CTA 只有两条动态 VMEM 指令。动态数量仍以 PMC
为准。

### 9.3 V 结论

V ownership gate = PASS。

这项局部成功足以证明：AveLang compiler-internal lowering 可以接管一个
source-level V logical block，并把 producer、shared slot、consumer 连接起来。
但它还不足以证明整个 Q/H/K/V full physical region 已经统一。

## 10. shared lifetime 结果

机器可读版本：
stage6z_c18_shared_lifetime_machine.json

### 10.1 计划值与实际值

| 项目 | 数值 |
|:--|--:|
| target reusable shared bytes | 24576 B |
| C18 source shared allocation | 32768 B |
| C18 LLVM/HSACO LDS | 32768 B |
| P2 LLVM/HSACO LDS | 32768 B |
| target 是否实现 | 否 |
| physical reuse 是否完整 | 否 |

C18 添加 score-V plan region 后，实际 shared allocation 没有降到 target
24576 B。说明当前 plan/lifetime metadata 还没有控制底层 shared allocation
的 coalescing；旧的 phase allocations 仍然活着，或者 lowerer 为保持安全
而保留了整个 32 KiB backing region。

### 10.2 不能错误解释的地方

- 32768 B 不是 C18 计划成功复用后的证明；
- 不能把 reuse_shared_after_source=true 属性当成真实 lifetime shortening；
- 不能把 C18 的 LDS 变化全部归因于 V producer；
- 需要在后续实验中使用实际 allocation/lifetime IR 和 final code object
  验证，而不是只看属性。

## 11. exact-LTO MIR 与 ISA

### 11.1 MIR 位置

C18 exact replay 目录：

codex_qwen_gfx942_c18_full_physical_region_t2048/exact_lto/

用于报告的代表 section：

- kernel_section_00.mir：pre-greedy；
- kernel_section_01.mir：post-greedy；
- kernel_section_08.mir：post-virtregrewriter；
- kernel_section_09.mir：post-prologepilog。

replay 由于 LTO 内部重复编译产生多个重复 section。本报告只使用目标
kernel header 完全一致的代表 section，不把重复 section 数量当成
instruction 数量。

C18 post-greedy 与 post-RA 代表 section 中没有：

- SI_SPILL_AV32_SAVE/RELOAD；
- SI_SPILL_AV64_SAVE/RELOAD；
- private segment spill sequence。

### 11.2 静态 ISA lexical count

以下是 kernel function body 的静态 lexical count：

| 指令族 | C18 | P2 |
|:--|--:|--:|
| v_mfma_f32_32x32x8_bf16 | 56 | 56 |
| ds_read_b64 | 112 | 0 |
| ds_read_b128 | 0 | 56 |
| ds_write_b16 | 64 | 48 |
| ds_write_b16_d16_hi | 64 | 32 |
| ds_write_b128 | 12 | 12 |
| global_load_dword | 80 | 80 |
| global_load_ushort | 32 | 48 |
| global_load_dwordx4 | 20 | 12 |
| global_store_short_d16_hi | 16 | 16 |
| s_waitcnt | 172 | 152 |
| s_barrier | 47 | 44 |
| ds_bpermute | 0 | 0 |

### 11.3 解释

C18 保持 56 条静态 MFMA32，说明 MFMA 数学工作和 K32 structure 没有被
删减或偷换。

C18 的 ds_read_b64 增加、ds_read_b128 消失，表示 C18 的 V/operand
feeding 采用了不同的 LDS packet width/fragment path。它证明了机器图变化，
但也解释了 LDS PMC 上升。

C18 的 global dwordx4 与 global_load_ushort 组合不同，说明 V source
producer 的 packet lowering 与 P2 不同。必须把这些 lexical 数和 PMC 分开
看：不能拿 56 条静态 MFMA 直接当动态 MFMA，也不能拿 112 条 ds_read_b64
直接当动态 LDS。

C18 新增少量 waitcnt/barrier lexical count，说明 compiler-owned V phase
仍然没有完成完整 shared lifetime coalescing。当前不能直接批量删 barrier。

## 12. code object resource

llvm-readobj 读取的 C18 exact code object：

| resource | C18 |
|:--|--:|
| VGPR | 116 |
| AGPR | 32 |
| SGPR | 30 |
| LDS/group segment | 32768 B |
| private segment | 0 |
| spill | 0 |

P2 exact code object：

| resource | P2 |
|:--|--:|
| VGPR | 132 |
| AGPR | 48 |
| SGPR | 30 |
| LDS/group segment | 32768 B |
| private segment | 0 |
| spill | 0 |

C18 的 code-object resource 结果比 P2 更轻，但不能把 code-object AGPR/VGPR
字段和 profiler 的 Accum_VGPR_Count 混为一个指标。两者来源不同，报告中
必须分别记录。

## 13. fresh T=2048 PMC

### 13.1 采集口径

C18 与 P2 都在 fresh Docker process、current HIP stream、no Graph 下，通过
rocprofv3 kernel trace + PMC 采集。每个 kernel 运行 warmup=2、repeat=5，
并取 7 个匹配 dispatch 的中位数。

C18 的 grid 是 131072 work-items、WG256，对应 512 CTA。
所有动态计数先除以 512，再形成 per-CTA 结果。

### 13.2 PMC 表

| metric | C18 total median | P2 total median | C18 / CTA | P2 / CTA |
|:--|--:|--:|--:|--:|
| MFMA | 81920 | 81920 | 160 | 160 |
| VMEM | 212992 | 229376 | 416 | 448 |
| LDS | 409600 | 303104 | 800 | 592 |
| VALU | 4101120 | 4338688 | 8010 | 8474 |
| SALU | 397312 | 399360 | 776 | 780 |
| OccupancyPercent | 14.918858 | 15.147486 | 14.918858 | 15.147486 |

C18 相对 P2：

- VMEM 减少 32 / CTA，约 7.14%；
- VALU 减少 464 / CTA，约 5.48%；
- SALU 减少 4 / CTA，约 0.51%；
- LDS 增加 208 / CTA，约 35.14%；
- MFMA 不变；
- occupancy 中位数略低，不能据此做单独因果结论。

### 13.3 trace 诊断值

| kernel | trace median |
|:--|--:|
| C18 | 43.745 us |
| P2 | 44.947 us |

C18 比 P2 约低 1.202 us，约 2.67%。这只是 diagnostic trace，不能替代
正式 latency body benchmark，也不能作为 Eager public API 排名。

### 13.4 与冻结参考的关系

| kernel | MFMA/CTA | VMEM/CTA | LDS/CTA | VALU/CTA | SALU/CTA |
|:--|--:|--:|--:|--:|--:|
| Z5B | 160 | 672 | 672 | 7072 | 768 |
| P2 | 160 | 448 | 592 | 8474 | 780 |
| C18 | 160 | 416 | 800 | 8010 | 776 |
| native WG256 | 160 | 140 | 480 | 3376 | 660 |

C18 机器图在 VMEM 和 VALU 方向相比 P2 有小幅改善，但仍远离 native。
同时 C18 LDS 高于 P2 和 native。因为 C18 没有完成统一 physical region
ownership，所以本轮不把这张表转换成“C18 已经超过 P2”的性能结论。

## 14. correctness gate

### 14.1 full sequence

C18 与 Z5B 做 separate-env fresh process correctness，对以下长度通过：

| T | BF16 output/H | V-new | final state | finite |
|--:|:--:|:--:|:--:|:--:|
| 64 | PASS | PASS | PASS | PASS |
| 128 | PASS | PASS | PASS | PASS |
| 512 | PASS | PASS | PASS | PASS |
| 1024 | PASS | PASS | PASS | PASS |
| 2048 | PASS | PASS | PASS | PASS |
| 4096 | PASS | PASS | PASS | PASS |
| 8192 | PASS | PASS | PASS | PASS |
| 16384 | PASS | PASS | PASS | PASS |

C18 与 Z5B 的 max abs difference = 0，非零计数 = 0。

### 14.2 edge cases

以下 edge case 全部通过：

- T=64、8192、16384；
- caller-owned output；
- zero V-new；
- NaN-prefilled output；
- output prefill 最终无 NaN；
- finite 检查；
- C18 与 Z5B BF16 byte-exact。

结果：

| T | C18 vs Z5B | finite | NaN count |
|--:|:--:|:--:|--:|
| 64 | byte-exact | PASS | 0 |
| 8192 | byte-exact | PASS | 0 |
| 16384 | byte-exact | PASS | 0 |

### 14.3 为什么 correctness 通过不能覆盖 ownership 不完整

correctness 只证明当前 C18 source 与 lowering 仍然实现了正确数学。
由于 H/K 仍然保留 P2 consumer path，C18 的 correctness 不能证明
FullPhysicalRegionPlan 已经拥有全部 producer、consumer 和 shared lifetime。

因此 correctness gate = PASS，但 full physical ownership gate 仍然 FAIL。

## 15. regression 与 build

### 15.1 compiler build

以下 build targets 在 Docker 中通过：

~~~text
ninja -j2 \
  lib/Dialect/AveLang/IR/libave-lang-dialect.a \
  lib/IR/libave-lang-mlir.a \
  lib/Dialect/AveLang/Transforms/libave-lang-dialect-transforms.a \
  _avelang_bindings
~~~

### 15.2 static/regression tests

最新 Docker 结果：

~~~text
python3 -m py_compile \
  qwen_gdn_bt64_native_chunko_stage6z_c18_full_physical_region.py \
  bench_qwen_gdn_bt64_stage6z_c18_machine_pmc.py \
  test_qwen_gdn_bt64_stage6z_c18_full_physical_region.py

pytest -q \
  test_qwen_gdn_bt64_stage6z_c17_full_physical_plan.py \
  test_qwen_gdn_bt64_stage6z_c18_full_physical_region.py

7 passed in 0.18s
~~~

回归覆盖：

- C17 existing physical-plan contract；
- C18 env/plan/compiler attrs；
- C18 V source equality and BF16/rank contract；
- no source handwritten V phase_vec；
- no source handwritten V MFMA loop；
- no allocator/RA change；
- no public Qwen-specific C18 op；
- old C17 compatibility。

### 15.3 exact-LTO replay

C18 exact replay return code = 0。

代表文件：

- exact_lto/kernel_section_00.mir；
- exact_lto/kernel_section_01.mir；
- exact_lto/kernel_section_08.mir；
- exact_lto/kernel_section_09.mir；
- exact_lto/linked.hsaco.lto.s。

未发现 SI_SPILL_AV32/AV64_SAVE 或 RELOAD。

## 16. 产物索引

### 16.1 compiler source

- lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc
- lib/IR/Intrinsics/amdgpu_module.cc
- lib/Dialect/AveLang/IR/AveLangOps.cc
- lib/Target/GPU/gpu_outlining.cc

### 16.2 experimental source/tests

- test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_c18_full_physical_region.py
- test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_stage6z_c18_machine_pmc.py
- test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_stage6z_c18_full_physical_region.py

### 16.3 T2048 machine artifacts

目录：

codex_qwen_gfx942_c18_full_physical_region_t2048/

主要内容：

- ir/final_mlir.mlir
- ir/post_block_dot_lowering.mlir
- ir/post_bounded_packet_schedule.mlir
- ir/post_kfrag_load_lowering.mlir
- ir/post_kfrag_rewrite.mlir
- ir/post_recurrence_step_lowering.mlir
- ir/preopt_llvm.ll
- ir/postopt_llvm.ll
- link_debug/amdgpu-link-0.argv.txt
- exact_lto/kernel_section_00.mir
- exact_lto/kernel_section_01.mir
- exact_lto/kernel_section_08.mir
- exact_lto/kernel_section_09.mir
- exact_lto/linked.hsaco.lto.s
- pmc/
- pmc_p2/

### 16.4 machine-readable JSON

以下 9 个 JSON 已生成并通过 JSON parse：

- stage6z_c18_current_vs_target_ownership.json
- stage6z_c18_p2_convergence_point.json
- stage6z_c18_full_physical_region.json
- stage6z_c18_v_full_region_ownership.json
- stage6z_c18_shared_lifetime_machine.json
- stage6z_c18_full_correctness.json
- stage6z_c18_machine_evidence.json
- stage6z_c18_pmc_t2048.json
- stage6z_c18_regression_results.json

它们位于：

test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/

## 17. 最终门禁判断

| gate | result |
|:--|:--:|
| C18 compiler plan created | PASS |
| C18 V source ownership | PASS |
| C18 first divergence before LLVM | PASS |
| C18 LLVM/MIR/ISA differs from P2 | PASS |
| full correctness | PASS |
| edge correctness | PASS |
| scratch/spill | PASS |
| Q/H/K all plan-owned | FAIL |
| full shared lifetime realized | FAIL |
| full physical-region completeness | FAIL |
| formal latency benchmark | NOT RUN |
| public Eager API | NOT RUN |
| production integration | NOT RUN |

最终状态：

C18_MACHINE_DISTINCT=true

C18_V_OWNERSHIP=true

C18_FULL_PHYSICAL_REGION_COMPLETE=false

C18_FORMAL_PERFORMANCE_COMPLETE=false

C18_DECISION=NO_GO

## 18. 复盘与下一步边界

### 18.1 本轮回答了什么

本轮首次证明了一个重要的 compiler capability：

AveLang 可以在不创建新的 Qwen public op、且不修改 allocator/RA 的前提下，
让一个 logical V block-dot source 由 compiler-internal plan 生成：

- BF16x8 global producer；
- shared physical slot；
- consumer-owned score@V feeding；
- 与旧手写 Phase-C 不同的 LLVM/MIR/ISA；
- 正确的 BF16 输出。

这比单纯添加 metadata 更严格，因为最终 HSACO 和 ISA 已经不同。

### 18.2 本轮没有回答什么

本轮不能证明：

- Q/H/K/V 全部已经使用 unified FullPhysicalRegionPlan；
- shared lifetime 已经 coalesce；
- C18 正式 latency 已经优于 Z5B；
- C18 已经接近 native；
- C18 可以进入 X2 或 production；
- AveLang 的全部差距只来自 block-dot lowering。

### 18.3 停止条件

按照 C18 任务预注册的严格门禁，本轮必须在这里停止：

- 不做 formal latency benchmark；
- 不做 Eager public API；
- 不建立 selector；
- 不接 X2；
- 不实现下一条 ownership candidate；
- 不修改 allocator/RA；
- 不把 C18 的局部 V 成功宣传为 full-region 成功。

如果后续继续，下一条任务必须先解决完整 plan 的 Q/H/K physical ownership
和真实 shared lifetime coalescing，并重新通过 machine-distinct 和 correctness
门禁；不能在 C18 名义下继续叠加第二个未登记结构变化。

## 19. 一句话结论

**C18 已经证明 FullPhysicalRegionPlan 能真实改变 V producer-consumer 的
lowering，并保持完整 correctness；但 Q/H/K 仍未完全由该 plan 拥有，shared
lifetime 也未兑现，因此 C18 是 machine-distinct 的 partial proof，不是新的
full-recurrence 性能 baseline。**
