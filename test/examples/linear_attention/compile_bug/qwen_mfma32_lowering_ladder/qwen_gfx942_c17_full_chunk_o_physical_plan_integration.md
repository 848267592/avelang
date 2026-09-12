# Qwen gfx942 Stage 6Z C17：Full Chunk-O Static Physical-Plan Integration

## 结论先行

本轮 C17 完成了一个 **full-region、experimental-only** 的编译器 gate，且完整
correctness 通过；但是没有达到 C17 预注册的 machine-closure 条件。因此最终决策
是：

```text
STOP_C17_FULL_MACHINE_NO_GO
```

这不是数值正确性失败，也不是编译失败。准确地说：

1. C17 在 `T=64/512/1024/2048/4096/8192/16384` 上相对冻结的 Z5B
   输出 BF16 byte-exact，finite 和 caller-owned/NaN-prefill 边界也通过。
2. C17 确实在 full-scope block-dot 的 post-block MLIR 中消费了一个统一的
   `ChunkOPhysicalPlan`，并把 Q/H/K 的 distributed/shared/dot/transform 计划附着到
   对应的 first-class MFMA operand。
3. Q 的 single-producer/dual-consumer 关系在 full source/MLIR 中得到证明：一个 Q
   cache producer 同时供 `Q@H`、`Q@K` 两个逻辑消费者使用。
4. C17 的 static affine 计划把 post-block `arith.divui/remui` 从 BDV2-P2 的
   `10/8` 降到 `1/0`，并且最终 ISA 没有 `ds_bpermute` 泛化回退。
5. 但是该变化在 LLVM/LTO 中收敛为已有 BDV2-P2 的最终机器图：C17 和已有 P2 的
   HSACO hash 相同，ISA 静态指令分类也相同。也就是说，C17 的计划属性到达了
   post-block IR，但没有形成一个新的 final LLVM/MIR/ISA machine graph。
6. 相对 Z5B，C17 的 T=2048 动态 VMEM 从 `672` 降到 `448`、LDS 从 `672` 降到
   `592`，但 VALU 从 `7072` 升到 `8474`/CTA，code-object VGPR/AGPR 从
   `104/32` 升到 `132/48`。这个 VALU/寄存器反向增长不满足 full machine
   closure，也没有理由提前宣称性能收益。
7. C17 的 plan 表中虽然有 V entry，但当前 full-scope source 的 Phase-C V
   producer/consumer 仍是手写路径，`V_source_level_consumer_owned_by_c17=false`。
   因此本轮不能把“V entry 被登记”误写成“V 已经被统一 physical plan 真正接管”。

本轮没有运行正式 body latency benchmark、T8192/T16384 性能或 Eager public API
排名。这是遵守 C17 任务中“先完成 full correctness + machine/resource closure，
只有 machine GO 才进入 C18 性能”的规则，而不是漏测。

---

## 1. C17 要回答的问题

此前的 C13、C14、C15、C16 分别完成了 physical encoding、codegen gate、V
real-tile numerical closure 和 Q/H/K real-tile closure，但这些证据仍然主要是
单 tile 或局部 operand 级别。

C17 第一次把问题提升到完整 chunk-o：

```text
Q/H/K producer
    -> Q@H
    -> Q@K
    -> score / causal / g
    -> V-new producer
    -> score@V-new
    -> final output
```

目标不是再发明一个 Qwen 专用 kernel，也不是做一个新的 Q、H、K、V 局部变体，而
是让一个统一的 `ChunkOPhysicalPlan` 在 full-region lowering 入口被创建一次，并
同时描述：

- Q/H/K/V 的 distributed ownership；
- shared region 和 physical encoding；
- dot operand 和 MFMA consumer；
- Q 的两个消费者关系；
- source phase 到 last-use 的生命周期；
- full-scope block-dot 的静态物理计划。

冻结的 machine contract 是：

| 项目 | C17 contract |
|:--|:--|
| target | `gfx942` |
| wavefront | 64 |
| workgroup | 256 |
| waves/CTA | 4 |
| CTA ownership | 2 CTA/chunk-head |
| sequence tile | `BT64` |
| output tile | `BV64` |
| K stage | `BK32` |
| MFMA | `v_mfma_f32_32x32x8_bf16` |
| reduction order | K32 order unchanged |
| public output | BF16, caller-owned output |
| production path | unchanged |

这里的 C17 C++ gate 只在显式设置
`AVELANG_STAGE6Z_FULL_PHYSICAL_PLAN=c17` 时生效。默认路径不会自动进入 C17。

---

## 2. 代码改动与实验边界

### 2.1 Experimental source wrapper

新增入口：

```text
test/examples/linear_attention/vllm_compare/
  qwen_gdn_bt64_native_chunko_stage6z_c17_full_physical_plan.py
```

该 wrapper 的重要性质是：

- 复用已有 `qwen_gdn_bt64_native_chunko_stage6z_block_dot_v2_full_scope` source
  contract；
- 不创建新的 Qwen kernel schedule；
- 不改变 BT/BV/BK、CTA、WG、MFMA、数学或 output ABI；
- 只设置 `AVELANG_STAGE6Z_FULL_PHYSICAL_PLAN=c17`；
- 固定 `lowering="specialized"`、`planner="bdv2_p1_affine"`、
  `preservation="p2_first_class"`；
- public production selector 没有修改。

这个事实需要特别记住：C17 是“在已有 full-scope logical source contract 上验证
统一 compiler physical plan”的实验，不是一个完全重新编写 source schedule 的新
kernel。报告因此不会把它包装成 source-level Qwen 算法改进。

### 2.2 Compiler gate

主要实现位于：

```text
lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc
```

关键入口：

| 位置 | 作用 |
|:--|:--|
| 约 784 行 | 读取 C17 gate |
| 约 871 行 | 构造 `makeC17StaticAffineLayoutPlan` |
| 约 927 行 | 从统一 `ChunkOPhysicalPlan` 取角色计划 |
| 约 940 行 | 给 block-dot/operand 附加 C17 plan attrs |
| 约 2454 行 | full-scope operand mode 消费 C17 plan |
| 约 3361 行 | pattern 将统一 plan 传入 lowering |
| 约 3644 行 | 每个 FuncOp 创建并 verify 一个 plan |

C17 的主要设计是：

```text
FuncOp
  -> one ChunkOPhysicalPlan
  -> full-scope H/K block-dot lowering
  -> first-class MFMA operand
  -> existing MLIR/LLVM/LTO pipeline
```

而不是：

```text
每个 block_dot 独立猜一套地址
```

### 2.3 Static affine map

C17 的 affine ownership map 只使用固定 contract 下的 shift/mask 和 compile-time
plan 参数，不建立 Qwen-specific 的 4096 元素地址表。其核心抽象包括：

```text
wave      = tid >> 6
lane      = tid & 63
lane_col  = tid & 31
lane_group= (tid >> 5) & 1
row_half  = tid >> 7
value_half= (tid >> 6) & 1
```

H packet ownership 和 K producer packet ownership 使用同一个 generic planner，
由 logical role、transpose 和 encoding 参数区分。C17 还把以下 attrs 附加到
post-block IR：

```text
c17.full_physical_plan
c17.role
c17.consumer
c17.distributed
c17.shared
c17.dot
c17.transform
c17.mfma
c17.operand_lifetime
c17.phase_boundary
c17.q_dual_consumer
```

### 2.4 没有做的事情

本轮没有：

- 创建 C17-Q/H/K/V 局部候选；
- 修改 V transpose/swizzle；
- 做 packet width sweep；
- 做 Q duplicate、g residency、barrier/wait、scheduler 或 pipeline sweep；
- 使用 generic `ds_bpermute` 解决 full integration；
- 修改 allocator、RA、recurrence 或 production selector；
- 运行 C18 正式性能 benchmark；
- 把 diagnostic C15 V tensor 地址公式带进 full C17。

---

## 3. Unified physical plan 内容

机器可读计划位于：

```text
stage6z_c17_full_physical_plan.json
```

### 3.1 Distributed encodings

| logical operand | shape | size/thread | threads/wave | waves/CTA | order |
|:--|:--|:--|:--|:--|:--|
| Q | `[64,32]` | `[1,8]` | `[16,4]` | `[4,1]` | `[1,0]` |
| H | `[64,32]` | `[1,8]` | `[16,4]` | `[4,1]` | `[1,0]` |
| K | `[32,64]` | `[8,1]` | `[4,16]` | `[1,4]` | `[0,1]` |
| V | `[64,64]` | `[4,8]` | `[8,8]` | `[2,1]` | `[1,0]` |

Q/H/K 的真实 full-scope block-dot attrs 都能在 C17 post-block MLIR 中看到。V
的 plan entry 同样存在，但它是“计划记录”而不是“当前 source V op 已由该计划
接管”的证明，见第 5 节。

### 3.2 Shared encoding 和生命周期

| region | kind | vector | per phase | max phase | 计划 bytes |
|:--|:--|--:|--:|--:|--:|
| Q | swizzled shared | 4 | 2 | 8 | 8192 |
| H | swizzled shared | 1 | 1 | 1 | 8192 |
| K | swizzled shared | 4 | 2 | 8 | 8192 |
| score | planned phase | - | - | - | 8192 |
| V | amd rotating shared | 4 | 1 | 16 | 8192 |

计划声明的两个生命周期带是：

```text
source_Q_H_K: source_prologue -> source_release
score_V:      score_and_causal_g -> output
```

C17 的 code object 实际 LDS 是 `32768 B`，与 Z5B 相同；这个事实说明当前
计划 attrs 并没有把所有记录的 region 同时真实分配成五份独立物理 buffer。也
不能反过来把 JSON 中的 region bytes 直接当成硬件实际峰值。最终以 code object
metadata 和运行时资源为准。

### 3.3 Q single producer / dual consumer

Q 证据位于：

```text
stage6z_c17_q_dual_consumer_full.json
```

逻辑关系是：

```text
one Q global/cache producer
       |
       +--> Q@H / inter_acc
       |
       +--> Q@K / score_acc_source_half_0
       |
       +--> Q@K / score_acc_source_half_1
```

机器可检查的结果：

- source 中 kernel Q pointer producer region = 1；
- Q cache logical shape `[256,32] BF16`，16 KiB；
- H/K consumers 收到相同 Q cache SSA/memref physical source；
- H consumer 的 C17 Q dual attrs = 2；
- K consumer 的 C17 Q dual attrs = 2；
- generic `ds_bpermute` fallback = 0。

这是 C17 中最清晰的 full integration 正结果。

---

## 4. Full correctness

### 4.1 正确性方法

正确性脚本：

```text
test/examples/linear_attention/vllm_compare/
  check_qwen_gdn_bt64_stage6z_c17_full_physical_plan.py
```

每个 case 在 `ljd_qwen_vllm_avelang_rocm722` 内独立 Python process 执行，使用：

- 相同 seeded `q/k/v_new/h/g`；
- BF16 contiguous input/output；
- Z5B 作为冻结参考；
- caller-owned output；
- `torch.equal` 做 BF16 byte-exact；
- `torch.isfinite` 做 finite 检查；
- zero-V-new 和 NaN-prefilled output 边界。

没有通过放宽误差阈值、跳过 output 检查或忽略 GPU fault 来形成“通过”。

### 4.2 全部结果

| T | 场景 | finite | 对 Z5B BF16 byte-exact | max abs |
|--:|:--|:--:|:--:|--:|
| 64 | random | PASS | PASS | 0 |
| 64 | zero-V + NaN output | PASS | PASS | 0 |
| 512 | random | PASS | PASS | 0 |
| 1024 | random | PASS | PASS | 0 |
| 2048 | random | PASS | PASS | 0 |
| 4096 | random | PASS | PASS | 0 |
| 8192 | random | PASS | PASS | 0 |
| 8192 | zero-V + NaN output | PASS | PASS | 0 |
| 16384 | random | PASS | PASS | 0 |
| 16384 | zero-V + NaN output | PASS | PASS | 0 |

该结果证明的是：在现有 full-scope source contract 下，C17 gate 没有破坏现有
数学、BF16 rounding、causal/g 语义、output ABI 或长序列寻址。它并不证明 C17
已经产生了优于 Z5B 的机器图。

机器可读结果：

```text
stage6z_c17_full_correctness.json
```

### 4.3 Correctness 与 machine closure 必须分开

本轮最容易混淆的地方是：

```text
正确性 PASS != machine closure PASS
```

C17 的结果属于“数学和 ABI 安全、但物理计划还没有完整转化为新机器工作”的
状态。因此不能因为所有 T 都 byte-exact 就直接晋级性能 baseline。

---

## 5. V physical plan 的关键限制

C17 计划 JSON 明确记录：

```json
"V_plan_entry_present": true,
"V_source_level_consumer_owned_by_c17": false
```

含义如下：

1. `ChunkOPhysicalPlan` 的结构中有 V 的 distributed/shared/dot encoding；
2. C15 已经证明过 V real-tile 的独立 physical semantic/codegen reference；
3. 但是当前复用的 BDV2 full-scope source 在 Phase-C 仍然拥有手写 V
   producer/consumer；
4. C17 本轮没有新增一个允许的 C17-V 变体，也没有把手写 Phase-C 改成统一
   first-class `block_dot` physical-plan consumer；
5. 因而不能声称 full graph 的 Q/H/K/V 四条链都已经由同一个 plan 真正控制。

这不是为了否定 C15。C15 的结论仍然是 V real-tile 局部语义闭环 PASS。这里的
问题是“局部 V oracle 已有”与“full C17 source 中 V 已经由统一 plan 接管”是两
个不同命题。

它也是 C17 不应晋级的结构性证据之一：当前 C17 不是一个完整意义上的

```text
Q/H/K/V producer -> shared -> dot consumer
```

统一计划机器图，而是：

```text
统一 plan 管 Q/H/K
+
V plan entry/attrs
+
existing handwritten Phase-C V path
```

报告将这一点保留为 limitation，而不是用 JSON 中的 V entry 掩盖它。

---

## 6. MLIR 到 ISA 的 machine evidence

### 6.1 工件目录

T=2048 C17 machine capture：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
  codex_qwen_gfx942_c17_full_physical_plan_t2048_skip/
```

目录中保留：

- post-plan/post-block MLIR；
- post operand materialization MLIR；
- staged MLIR；
- pre-opt/post-opt LLVM；
- pre-LTO AMDGCN；
- exact LTO replay argv；
- exact LTO pre/post-greedy MIR；
- virtregrewriter MIR；
- final physical MIR；
- final ISA；
- HSACO；
- `machine_summary.json`。

目录名的 `_skip` 不是跳过 machine capture，而是跳过 initial `get_mlir()`：
driver 的 initial `get_mlir()` 会 SIGSEGV，导致一次不完整的初始尝试；使用
`--skip-initial-mlir` 后 post-plan 及其后所有阶段成功保存。这个工具入口问题
不能被写成 C17 kernel correctness failure。

### 6.2 Post-block MLIR：C17 确实有差异

C17 post-block MLIR 的可观测证据：

| 项目 | C17 | 已有 BDV2-P2 |
|:--|--:|--:|
| `c17` plan attrs | 44 | 0 |
| `arith.divui` | 1 | 10 |
| `arith.remui` | 0 | 8 |
| Q dual relation | present | absent as C17 attrs |
| H/K full-scope plan attrs | present | absent as C17 attrs |

这说明 C17 不是一个完全没有执行的环境变量。它确实在 block-dot lowering
边界产生了不同的 MLIR representation。

### 6.3 LLVM/LTO/ISA：C17 与 P2 收敛

但是后续层次的证据是：

```text
C17 HSACO SHA256:
f612f23968d1e9b9e205b1daf75a70cef2d9016d56f44df24fef6d3ae20c7947

existing BDV2-P2 HSACO:
same hash
```

C17 与 P2 的 final ISA 统计相同：

| static family | C17/P2 |
|:--|--:|
| MFMA32 | 56 |
| global load | 140 |
| global store | 16 |
| `ds_read` | 104 |
| `ds_write` | 92 |
| `s_barrier` | 44 |
| `ds_bpermute` | 0 |
| private segment | 0 |
| VGPR/SGPR spill words | 0/0 |

主要地址类静态计数也相同：

| ISA pattern | count |
|:--|--:|
| `v_add_u32` | 55 |
| `v_lshl_add_u32` | 7 |
| `v_and_b32` | 46 |
| `v_lshrrev_b32` | 66 |

因此必须区分两件事：

- C17 改善了 post-block IR 表达，使静态 affine 计划可见；
- 现有 LLVM/LTO pipeline 又把这个表示 canonicalize 成已经存在的 P2 机器图。

这就是 C17 没有完成“与 B0/P2 不同的 full machine graph”的直接证据。C17 与
Z5B 的 HSACO hash 确实不同，但 C17 相对 Z5B 的新图其实就是 BDV2-P2 已有图，
并不是 C17 专属的 full physical-plan machine result。

### 6.4 Exact-LTO MIR 和 spill

exact-LTO sections 的审计结果：

| section | 阶段 | 行数约数 | spill save/reload | virtual regs |
|:--|:--|--:|:--:|:--:|
| `kernel_section_00.mir` | pre-greedy | 4136 | 0 | present |
| `kernel_section_01.mir` | post-greedy | 4136 | 0 | present/RA stage |
| `kernel_section_02.mir` | post-virtregrewriter | 4130 | 0 | rewritten |
| `kernel_section_08.mir` | final physical | 4097 | 0 | none |
| `kernel_section_09.mir` | prolog/epilog | 3994 | 0 | none |

对应的 code object 没有 private segment，也没有 VGPR/SGPR spill。这里的结论是
“C17 没有触发 v29 式 spill cliff”，不是“C17 已经达到了 native resource
shape”。

### 6.5 Hash 和 artifact manifest

| artifact | SHA256 |
|:--|:--|
| source contract | `aee02e0309152ab34893836720c38266c6519c57499ef640db3e4ff48152047e` |
| post-block MLIR | `dbec4d837c6c30e05904c27a5896ab69ed8f8e9e9e082bd619c069687e1cb74f` |
| post operand MLIR | `7a2218b4cb9aaaab774e7c3e06a129c0cf5183d37e8d3d69d800914d55ca27bb` |
| final MLIR | `2338b4b885b02156e93b5a902bfb951f20c4a4b7bcfc2219c2501a62cdeabdd5` |
| lowered LLVM | `70283b959b73da3e1a1bd8cea63710df586e52de6ec228b1634af418b569db9a` |
| pre-LTO AMDGCN | `4f614815af6fb5cce83527c298559d2e4f46624f3477d5df8d3fe4160a77040d` |
| final ISA | `51839fc535f4438b989866816cd3060be74085e0233537a777a90805c34f52c2` |
| C17 HSACO | `f612f23968d1e9b9e205b1daf75a70cef2d9016d56f44df24fef6d3ae20c7947` |

---

## 7. T=2048 dynamic PMC

### 7.1 归一化方法

C17 fresh run 的 grid work-items 是 `131072`，WG 是 `256`，所以：

```text
CTA count = 131072 / 256 = 512
per-CTA metric = raw counter / 512
```

以下是 dynamic PMC，不是从 ISA static lexical count 推导的数量。

### 7.2 Dynamic instruction table

| candidate | MFMA/CTA | VMEM/CTA | LDS/CTA | VALU/CTA | SALU/CTA |
|:--|--:|--:|--:|--:|--:|
| Z5B frozen | 160 | 672 | 672 | 7072 | 768 |
| C17 fresh | 160 | 448 | 592 | 8474 | 780 |
| native WG256 diagnostic | 160 | 140 | 480 | 3376 | 660 |

相对 Z5B：

- MFMA：`0%`，数学工作保持不变；
- VMEM：`-224/CTA`，约 `-33.3%`；
- LDS：`-80/CTA`，约 `-11.9%`；
- VALU：`+1402/CTA`，约 `+19.8%`；
- SALU：`+12/CTA`，约 `+1.6%`。

相对 native WG256：

- C17 VMEM 仍是 native 的约 `3.20x`；
- C17 LDS 约 `1.23x`；
- C17 VALU 约 `2.51x`；
- C17 SALU 约 `1.18x`。

这组数据足以说明：C17 的 static affine 计划产生了某些 operand materialization
收益，但没有把 full graph 的地址/layout/fragment 机器工作收敛到 native 级别。

### 7.3 Code object 与 profiler resource 必须分开

C17 code object metadata：

| field | C17 |
|:--|--:|
| `.vgpr_count` | 132 |
| `.agpr_count` | 48 |
| `.sgpr_count` | 30 |
| LDS block size | 32768 B |
| private segment | 0 B |
| scratch | 0 |
| VGPR spill | 0 |
| SGPR spill | 0 |

C17 本次 rocprof resource fields：

| field | value |
|:--|--:|
| `VGPR_Count` | 88 |
| `Accum_VGPR_Count` | 88 |
| `SGPR_Count` | 112 |
| `LDS_Block_Size` | 32768 B |
| `Scratch_Size` | 0 |
| `OccupancyPercent` | 0.433535871 |

`.agpr_count=48`、`Accum_VGPR_Count=88`、`.vgpr_count=132` 是三个不同层次的
指标，不能互相替代。occupancy 在不同 rocprof session 中有明显波动，本轮只
记录，不把它作为唯一因果结论。

Z5B 的同口径 code object 记录为 `VGPR/AGPR=104/32`、LDS `32768 B`、无 spill；
这进一步说明 C17 的 VALU 降低尚未免费获得，反而带来了更大的 register/resource
表示。

---

## 8. 为什么没有继续跑性能

C17 任务预先规定了顺序：

```text
full correctness
    -> machine/resource closure
    -> C18 performance
```

C17 在 correctness 层面 PASS，但 machine closure 的核心条件失败：

| 预注册条件 | 结果 | 说明 |
|:--|:--:|:--|
| full correctness | PASS | 所有规定 T 通过 |
| MFMA 160/CTA | PASS | 与 Z5B/native reference 相同 |
| no scratch/spill | PASS | exact MIR 和 HSACO 均为 0 |
| Q producer once/dual consumer | PASS | Q JSON + post-block attrs |
| no generic `ds_bpermute` | PASS | final ISA 为 0 |
| real H/K physical plan | PASS at block-dot boundary | attrs 可见 |
| V source consumer owned by C17 | FAIL | 仍为手写 Phase-C V path |
| full final graph distinct from existing P2 | FAIL | LLVM/LTO/ISA/HSACO 收敛 |
| VMEM lower than Z5B | PASS | 672 -> 448 |
| VALU not reverse-growth | FAIL | 7072 -> 8474 |
| full graph close to native | FAIL | VALU 2.51x、VMEM 3.20x |

所以如果此时继续跑 T2048/T8192 latency，很难回答收益来自什么：

- 是 VMEM/LDS 下降；
- 是 VALU 增长抵消；
- 是 C17 与 P2 本来相同的机器图；
- 还是 GPU session 的短时波动。

在没有一个真正新的、完整 V-inclusive machine graph 的情况下，继续测出来的
数字不会回答 C17 的主要问题，反而会把“correctness pass”误读成“性能候选”。

---

## 9. Regression 与构建结果

### 9.1 Compiler build

Docker 内构建命令：

```bash
cd /workspace/project/avelang
cmake --build build-vllm-rocm722 -j2
```

结果：PASS，9 个相关 target relink/build 成功。

### 9.2 CTest

```bash
ctest --test-dir build-vllm-rocm722 \
  -R "static_physical_layout|c14_static_physical_codegen|amdgpu_codegen|lower_to_llvm|gpu_outlining" \
  --output-on-failure
```

结果：`5 passed, 0 failed`。

覆盖：

- `static_physical_layout_test`；
- `lower_to_llvm_test`；
- `gpu_outlining_test`；
- `amdgpu_codegen_test`；
- `c14_static_physical_codegen_test`。

### 9.3 Python static tests

```bash
python3 -m pytest -q \
  test_qwen_gdn_bt64_stage6z_c17_full_physical_plan.py \
  test_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py
```

结果：`12 passed`。

这些测试检查了：

- C17 gate 是 experimental-only；
- source 复用已有 full-scope contract；
- 统一 plan 创建和 static affine helper 存在；
- plan attrs 能到达 first-class operand boundary；
- 没有新增 Qwen-specific physical op；
- BDV2 full-scope 旧 contract 没有被静态回归破坏。

机器可读回归结果：

```text
stage6z_c17_regression_results.json
```

---

## 10. C17 与历史实验的关系

### 10.1 C13/C14/C15/C16 仍然有效

C17 的 No-Go 不会撤销此前局部闭环：

- C13 typed physical attrs 和 encoding 仍然有效；
- C14 specialized block-dot codegen gate 仍然有效；
- C15 V real-tile numerical/codegen oracle 仍然有效；
- C16 Q/H/K real-tile source/consumer mapping 仍然有效。

它们证明了：AveLang 可以在受控 local tile boundary 表达和生成这些 physical
语义。

### 10.2 C17 新增的事实

C17 新增的是 full composition 的边界事实：

```text
local physical correctness
并不自动推出
full-region physical-plan consumption
```

更具体：

1. Q/H/K 的统一 plan attrs 可以进入 full-scope block-dot lowering；
2. Q dual-consumer 的 physical identity 可以保持；
3. static affine plan 可以降低 post-block 的动态索引表达；
4. 但 LLVM/LTO 仍会把这部分表示收敛到已有 P2 machine graph；
5. V 仍然缺少一个当前 source 合法、且被同一 plan 真正消费的 full-scope
   first-class boundary；
6. 统一物理计划还没有成为完整 chunk-o 的 final machine scheduling/operand
   representation。

### 10.3 这不是“C17 算法错了”

C17 的 source 数学没有被判错，full output 也完全正确。问题是 compiler
representation 到 machine representation 的闭环不完整：

```text
计划被创建
  -> 部分 operand attrs 可见
  -> post-block 有差异
  -> LLVM/LTO canonicalize
  -> final machine graph 回到已有 P2
```

因此结论应写成“full physical plan 没有形成新的 machine closure”，而不能笼统
写成“AveLang 不支持 Q/H/K/V”或“算法不正确”。

---

## 11. 最终决策与停止范围

### 11.1 决策

```text
C17_FULL_CORRECTNESS_PASS
  + C17_POST_BLOCK_PLAN_DIFFERENT
  + C17_FINAL_GRAPH_CONVERGES_TO_BDV2_P2
  + V_NOT_FULLY_OWNED_BY_PLAN
  + VALU_RESOURCE_REVERSE_GROWTH
  = STOP_C17_FULL_MACHINE_NO_GO
```

### 11.2 本轮不晋级的原因排序

按证据强度排序：

1. **最终机器图没有独立性**：C17 与已有 BDV2-P2 HSACO hash 相同，ISA static
   family 相同；因此 C17 的新 affine attrs 没有改变最终机器实现。
2. **full plan 没覆盖实际 V consumer**：V entry 存在，但 source-level V path
   仍由手写 Phase-C 控制。
3. **VALU 反向增长**：即使 VMEM/LDS 下降，C17 dynamic VALU 为 `8474/CTA`，
   高于 Z5B 的 `7072/CTA`，并显著高于 native 的 `3376/CTA`。
4. **寄存器表示变重**：code object `VGPR/AGPR=132/48` 高于 Z5B
   `104/32`；虽然没有 spill，这仍然不是 native-like closure。

### 11.3 明确关闭的后续行为

本报告不登记也不实现：

- C17-Q/K/V 后继局部变体；
- C17.1/C17.2 placement sweep；
- 继续调 static affine 常量；
- 继续调 packet width、barrier、waitcnt 或 RA；
- 直接把 C17 接入 X2/full production；
- 用 private body 或 profiler trace 宣布正式性能胜出。

在 C17 任务边界内，下一步不是再堆一个局部优化，而是先决定是否值得补齐
一个真正可被同一 plan 消费的 V first-class full-scope representation。若不能
在不引入新的 C17 局部路线的条件下完成这一表示闭环，应保持 Z5B 为 isolated
performance baseline，并关闭 C17 full integration 线。

---

## 12. 复现实验命令

以下命令以 Docker 容器 `ljd_qwen_vllm_avelang_rocm722` 为准。宿主机和容器的
workspace 挂载必须确认同步；C17 compiler build 在容器内完成。

### 12.1 Build

```bash
docker exec ljd_qwen_vllm_avelang_rocm722 bash -lc '
  cd /workspace/project/avelang &&
  cmake --build build-vllm-rocm722 -j2
'
```

### 12.2 Correctness

```bash
export PYTHONPATH=/workspace/project/avelang/build-vllm-rocm722/python:\
/workspace/project/avelang/python:\
/workspace/project/avelang/test/examples/linear_attention/vllm_compare

export AVELANG_STAGE6Z_FULL_PHYSICAL_PLAN=c17

python3 check_qwen_gdn_bt64_stage6z_c17_full_physical_plan.py --T 64
python3 check_qwen_gdn_bt64_stage6z_c17_full_physical_plan.py --T 512
python3 check_qwen_gdn_bt64_stage6z_c17_full_physical_plan.py --T 1024
python3 check_qwen_gdn_bt64_stage6z_c17_full_physical_plan.py --T 2048
python3 check_qwen_gdn_bt64_stage6z_c17_full_physical_plan.py --T 4096
python3 check_qwen_gdn_bt64_stage6z_c17_full_physical_plan.py --T 8192
python3 check_qwen_gdn_bt64_stage6z_c17_full_physical_plan.py --T 16384
```

边界：

```bash
python3 check_qwen_gdn_bt64_stage6z_c17_full_physical_plan.py \
  --T 64 --zero-v --nan-prefill
python3 check_qwen_gdn_bt64_stage6z_c17_full_physical_plan.py \
  --T 8192 --zero-v --nan-prefill
python3 check_qwen_gdn_bt64_stage6z_c17_full_physical_plan.py \
  --T 16384 --zero-v --nan-prefill
```

### 12.3 Machine capture

初始 `get_mlir()` 会触发工具入口 SIGSEGV，因此正式 capture 使用：

```bash
python3 dump_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope_machine.py \
  --T 2048 \
  --variant specialized \
  --planner bdv2_p1_affine \
  --preservation p2_first_class \
  --out-dir codex_qwen_gfx942_c17_full_physical_plan_t2048_skip \
  --skip-initial-mlir
```

正式工件应检查：

```bash
grep -R "c17.full_physical_plan\|c17.q_dual_consumer" \
  codex_qwen_gfx942_c17_full_physical_plan_t2048_skip/ir/post_block_dot_lowering.mlir

grep -R "ds_bpermute" \
  codex_qwen_gfx942_c17_full_physical_plan_t2048_skip/final_isa.s

grep -R "SI_SPILL_AV\|SPILL" \
  codex_qwen_gfx942_c17_full_physical_plan_t2048_skip/exact_lto/*.mir
```

### 12.4 Dynamic PMC

T=2048 的 C17 fresh PMC 工件位于：

```text
codex_qwen_gfx942_c17_full_physical_plan_t2048_pmc/
  rocprof_bdv2_p2_specialized_T2048/
```

机器可读汇总：

```text
stage6z_c17_pmc_t2048.json
```

---

## 13. 机器可读产物索引

| 文件 | 作用 |
|:--|:--|
| `stage6z_c17_full_correctness.json` | 全长度 correctness 和边界结果 |
| `stage6z_c17_full_physical_plan.json` | unified plan、encoding、lifetime、V limitation |
| `stage6z_c17_q_dual_consumer_full.json` | Q single producer/dual consumer 证据 |
| `stage6z_c17_machine_evidence.json` | MLIR/LLVM/MIR/ISA/HSACO hash 和收敛结论 |
| `stage6z_c17_pmc_t2048.json` | fresh T2048 dynamic PMC 与 per-CTA 归一化 |
| `stage6z_c17_regression_results.json` | build、CTest、Python tests、benchmark rule |

这些文件与本报告位于同一目录：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
```

---

## 14. 给后续复盘的学习要点

### 14.1 为什么 post-block IR 变好仍然不够

`divui/remui` 的减少只说明某个 lowering 阶段看见了更好的 affine
representation。它不保证：

- LLVM 不会重新生成等价地址计算；
- LTO 不会把不同 SSA 表示 canonicalize；
- MIR register pressure 会下降；
- final ISA 会改变；
- dynamic PMC 会改善；
- latency 会改善。

因此必须沿着：

```text
MLIR -> LLVM -> pre-greedy MIR -> post-greedy MIR -> ISA -> PMC
```

逐级确认“差异是否保留”。C17 正好展示了一个完整的 representation-loss
案例：post-block 有差异，final machine 没有相对于 P2 的差异。

### 14.2 为什么没有把 C17 VMEM 下降直接当胜利

C17 的 VMEM 从 672 降到 448 是真实 dynamic PMC，值得记录。但同时：

- LDS 只降到 592；
- VALU 上升到 8474；
- code-object VGPR/AGPR 上升；
- V physical plan 尚未真正消费；
- final graph 与 P2 相同。

所以只能说 C17/P2 这一类 full-scope physical lowering 方向改变了机器工作
分布，不能说“统一 plan 已经解决 chunk-o 瓶颈”。

### 14.3 为什么没有把 V entry 当作 V closure

plan schema 先拥有一个 V 字段，不等于当前 source operation 真的带着 V plan
进入 first-class lowering。学习 compiler 集成时，必须区分：

```text
schema knows V
vs
source owns V
vs
MLIR carries V attrs
vs
LLVM/MIR/ISA uses V encoding
```

C17 的 JSON 把这几层明确拆开，这是本轮最重要的负面证据之一。

### 14.4 C17 的正面价值

即使最终 No-Go，这轮仍然留下了四个可复用成果：

1. 一个真正可开关的 full-region compiler gate；
2. 一个统一 plan 创建点，而不是 per-op 猜测；
3. Q single-producer/dual-consumer 的 full-scope 验证方法；
4. 一份从 post-block 到 final machine convergence 的可重复证据链。

这比再做一个局部 C17-Q/C17-K 性能 patch 更有学习和工程价值。

---

## Final decision

```json
{
  "candidate": "C17_FULL",
  "correctness": "PASS",
  "machine_closure": "FAIL",
  "decision": "STOP_C17_FULL_MACHINE_NO_GO",
  "formal_performance_benchmark": "NOT_RUN",
  "production_rollout": "NO",
  "primary_blockers": [
    "C17 final graph converges to existing BDV2-P2",
    "V source-level Phase-C consumer is not owned by the unified plan",
    "VALU and register representation grow relative to Z5B"
  ]
}
```
