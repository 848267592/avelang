# Qwen gfx942 Stage 6Z：BDV2-P2 First-Class MFMA Operand Preservation

## 1. 本轮结论

本轮完成了一个**正确、可编译、确实改变最终机器图、但没有性能晋级**的
通用 `block_dot_bf16_f32` compiler experiment。

P2 的核心结果不是“packed LDS load 已经更快”，而是回答了一个更基础的问题：

> AveLang 是否可以让同一个通用 block-dot source，在不创建 Qwen/chunk-o
> 专用 public op 的前提下，把 MFMA operand 的 logical role、shared physical
> encoding 和 consumer identity 保存到 GPU-module late lowering，并最终影响
> LLVM、MIR、ISA 和 HSACO？

答案是：**可以。**

但是当前第一版 P2 的具体物化方式是显式 `i64`/B64 LDS read 加 bitcast。它没有
降低最终机器工作的关键部分，反而增加了 LDS read：

```text
P1/BDV2-S dynamic LDS = 464/CTA
P2 dynamic LDS         = 592/CTA
```

P2 的 VMEM 和 MFMA 保持与 BDV2/P1 相同，但 T=2048/T=8192 body latency 都稳定
慢于 Z5B、P1-S 和 native diagnostic。因此：

```text
Z5B       = 当前 Stage 6Z isolated performance baseline
P2        = correctness PASS + machine-representation PASS + performance No-Go
P2 infra  = 保留为通用 compiler infrastructure / regression evidence
production/X2 = 不修改
```

本轮没有接入 X2、production selector、external recurrence HSACO，也没有修改
allocator/RA、MFMA geometry、WG、ownership 或数学。

---

## 2. 背景：为什么需要 P2

### 2.1 Z5B、BDV2 和 P1 的位置

Stage 6Z 的当前 isolated baseline 是 Z5B dedicated-Q-cache/direct-Q-consumer。
它已经把 Q 的 global producer pass 从 3 次降到 1 次，并保持了正确性。

之后的 BDV2 full-scope generalization 让同一个高层
`al.amdgpu.block_dot_bf16_f32` 负责更完整的数据链：

```text
logical K/H global block
  -> producer ownership
  -> typed/global packet
  -> shared placement
  -> MFMA-B consumer
```

BDV2 的优势是机器工作的一部分下降：

| arm | MFMA/CTA | VMEM/CTA | LDS/CTA | VALU/CTA | SALU/CTA |
|:--|--:|--:|--:|--:|--:|
| Z5B | 160 | 672 | 672 | 7072 | 768 |
| BDV2-S/P1-S | 160 | 448 | 464 | 8410 | 780 |

但 BDV2 的 VALU 增加了 `1338/CTA`，latency 没有收益，长文本反而退化。

P1 试图做两件事：

1. 由同一个 `LogicalBlockLayoutPlan` 复用 K/H 的 affine ownership/index；
2. 让 consumer 直接请求 `<4xbf16>`，减少 `<8xbf16>` 后再
   `extractelement/insertelement` 重建 fragment。

P1 在 lowered LLVM 文本中确实减少了：

```text
extractelement: 130 -> 66
insertelement:  160 -> 96
```

但 P1 与 BDV2-S 最终收敛到相同 HSACO：

```text
d17483b933a7abf68b34b0e193aa4e3e5d9f93f48a12cfa8d84b285c9091af5a
```

因此 P1 证明的是“LLVM 表示更紧凑”，没有证明“最终 AMDGPU machine graph
保留了这个意图”。P2 的任务就是补上这条证据链。

### 2.2 本轮冻结边界

P2 与 BDV2/P1 使用相同的：

- gfx942、wave64；
- BT64、BV64、BK32；
- WG256、每 chunk-head 两个 CTA；
- BF16 Q/K/H/V-new/output ABI；
- FP32 g 和 accumulator；
- dedicated full-Q LDS cache；
- phase-separated accumulator 顺序：`inter_acc -> score_acc0 -> score_acc1 -> intra_acc`；
- causal mask、K32 accumulation order、MFMA32 geometry；
- global layout、caller-owned output；
- source-level `block_dot_bf16_f32` logical contract；
- allocator/RA、production selector、X2 recurrence HSACO。

P2 唯一新增的是 compiler-only preservation selector：

```text
AVELANG_BLOCK_DOT_OPERAND_PRESERVATION=none
AVELANG_BLOCK_DOT_OPERAND_PRESERVATION=p2_first_class
```

没有新增 Qwen/chunk-o public op，也没有新增 Python kernel source variant。

---

## 3. P2 的设计

### 3.1 内部 first-class operand op

新增内部 AveLang IR op：

```text
ave.gpu.amdgpu_block_dot_mfma_operand
```

这个 op 不是最终硬件 intrinsic。它是一个短生命周期、target-aware 但通用的
内部表示，用来保存 block-dot 的 operand contract：

```text
operand_role:       A/B
source_role:        K/H
logical_shape:      32x32x32
transpose:          none/rhs_transposed
physical_encoding:  gfx942_shared_b32_mfma32
fragment_mapping:   lane_group_word_pair_k32_order
target_mfma:        f32_32x32x8_bf16
```

其操作数还显式携带：

- A/B shared stage；
- persistent FP32 accumulator；
- A row、B row；
- operand word；
- K-stage。

因此 lowering 不需要根据 kernel 名字猜测 K/H，也不需要在 Qwen pass 中写死
4096 个地址。K 和 H 的差别仍由 `source_role`、transpose 和 logical layout
plan 表达。

### 3.2 统一 planner

P2 复用 P1 的 `LogicalBlockLayoutPlan`。planner 计算并传递：

```text
tid
wave
lane
laneCol
laneGroup
rowHalf
valueHalf
kStage
producerLinear
packetRow
packet
packetCol
feature
```

K/H 进入同一条 `lowerFullScopeOperandMode` 路径，没有独立的 K planner 或 H
planner。未来 Q 的 A operand、V-new 的 B operand 也可以复用这一层表示。

### 3.3 late materialization

P2 在 GPU outlining 后运行：

```text
createLowerQwenBlockDotMfmaOperandPass()
```

这个 pass 将内部 first-class op 变成：

```text
static LDS address calculation
  -> addrspace(3) pointer
  -> packed LDS load
  -> BF16 fragment
  -> existing MFMA32 intrinsic
```

当前实验版本的 packed load 是：

```llvm
%word = load volatile i64, ptr addrspace(3) ..., align 8
%frag = bitcast i64 %word to <4 x bfloat>
```

`volatile` 只用于把本轮实验的 load representation 可靠地保留到后续观察点，
不是生产语义要求。它不能被解读为最终优化方案。

---

## 4. 编译 pipeline 和证据点

### 4.1 pipeline 顺序

本轮相关顺序为：

```text
block_dot source
  -> LowerQwenBlockDotPass
  -> GPU outlining
  -> post_gpu_outlining snapshot
  -> pre_block_dot_operand_materialization snapshot
  -> LowerQwenBlockDotMfmaOperandPass
  -> post_block_dot_operand_materialization snapshot
  -> GPU-module canonicalizer/CSE
  -> LLVM lowering
  -> ROCm LTO
  -> MIR/RA
  -> ISA/HSACO
```

代码位置：

```text
lib/Dialect/AveLang/IR/AveLangOps.td
lib/Dialect/AveLang/IR/AveLangOps.h
lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.h
lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc
lib/Target/GPU/lower_to_llvm.cc
```

### 4.2 initial MLIR 的工具限制

当前 Docker runtime binding 的 `get_mlir()` 在 initial MLIR printer 阶段会
SIGSEGV。因此本轮 dump 使用：

```text
--skip-initial-mlir
```

这意味着不能声称拿到了 initial/pre-branch MLIR 的字节级 hash，也不能把
source SHA 冒充 initial MLIR hash。

本轮可以严格证明的是：

1. generic/P1/P2 使用同一个 Python source；
2. P2 的 `post_gpu_outlining.mlir` 仍包含内部 operand op；
3. P2 的 post-materialization MLIR 已经变成显式 packed LDS load；
4. P2 的 lowered LLVM、pre-LTO AMDGCN、final ISA 和 HSACO 与 P1 不同。

### 4.3 artifact 清单

P2 machine artifact：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
  codex_qwen_bt64_stage6z_bdv2_p2_machine_specialized/
```

主要文件：

```text
source.py.txt
lowered_llvm.ll
pre_lto_amdgcn.s
final_isa.s
machine_summary.json
ir/post_gpu_outlining.mlir
ir/pre_block_dot_operand_materialization.mlir
ir/post_block_dot_operand_materialization.mlir
exact_lto/kernel_section_00.mir ... kernel_section_19.mir
exact_lto/summary.json
bdv2_p2_specialized.hsaco
```

机器 dump driver：

```text
test/examples/linear_attention/vllm_compare/
  dump_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope_machine.py
```

动态 PMC driver：

```text
test/examples/linear_attention/vllm_compare/
  profile_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py
```

fresh-process body driver：

```text
test/examples/linear_attention/vllm_compare/
  bench_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py
```

---

## 5. same-source identity

### 5.1 source SHA256

P2 使用的 source：

```text
test/examples/linear_attention/vllm_compare/
  qwen_gdn_bt64_native_chunko_stage6z_block_dot_v2_full_scope.py
```

source SHA256：

```text
aee02e0309152ab34893836720c38266c6519c57499ef640db3e4ff48152047e
```

`none`、P1、P2 都通过 environment selector 选择 lowering；不是 Python source
fork。source-level 的 launch、operand ownership、math、output contract 不变。

### 5.2 机器层 SHA256

| artifact | P1-S | P2-S |
|:--|:--|:--|
| lowered LLVM | `8e2122d3ac09a80f12996558f48862c33bc7266d548a2fa70cb299b30ccf1152` | `09d1092954f25206250df6e9eefb8db31da948ae95deadc449316061dc3a41f4` |
| pre-LTO AMDGCN | `96c3130391f63a777f2a0cac32a2db8b5528f770076fd3feefca8dc2904286d70` | `bfb43182ef7266b781c75b49c9864d0a845c3dbcae719fb8f2766498c6c1f28b` |
| final ISA | `a0d4f0bf616fec55f1757111a141d965ebd5f911ff5277ba694a18a86140d454` | `3f33aafb94f6cf2aac88ee463636833fd34f06a56503af23132060ecfa98c939` |
| HSACO | `d17483b933a7abf68b34b0e193aa4e3e5d9f93f48a12cfa8d84b285c9091af5a` | `f612f23968d1e9b9e205b1daf75a70cef2d9016d56f44df24fef6d3ae20c7947` |

P2 与 P1 的 HSACO 不同，满足“如果第一轮收敛，继续修复一次”的要求。

### 5.3 MLIR 证据

P2 的 `post_gpu_outlining.mlir` 中出现例如：

```mlir
ave.gpu.amdgpu_block_dot_mfma_operand ...
  {avelang.block_dot.first_class_operand,
   avelang.block_dot.logical_shape = "32x32x32",
   avelang.block_dot.physical_encoding = "gfx942_shared_b32_mfma32",
   avelang.block_dot.fragment_mapping = "lane_group_word_pair_k32_order",
   avelang.block_dot.source_role = "K/H"}
```

P2 的 `post_block_dot_operand_materialization.mlir` 中出现：

```mlir
llvm.load volatile ... : !llvm.ptr<3> -> i64
llvm.bitcast ... : i64 to vector<4xbf16>
```

内部 op 在该 late pass 之后不再残留。这个顺序证明 operand identity 并非只
存在于 metadata，而是被一个真实的 consumer lowering 消费。

### 5.4 LLVM 证据

P1 的 consumer 形式：

```llvm
load <4 x bfloat>, ptr addrspace(3) ..., align 2
```

P2 的 consumer 形式：

```llvm
load volatile i64, ptr addrspace(3) ..., align 8
bitcast i64 ... to <4 x bfloat>
```

这不是把 `i64` 当作数学数据类型改变 MFMA 语义；它只是 packed LDS word 的
表示。P2 仍把相同 BF16 bits 作为 MFMA operand 提供给既有
`f32_32x32x8_bf16` intrinsic。

### 5.5 MIR 和 ISA 证据

P2 exact-LTO replay：

- return code `0`；
- pre-greedy section 没有 `SI_SPILL_AV32_SAVE`；
- 没有 `SI_SPILL_AV64_SAVE`；
- `spill_virtual_registers=[]`；
- private segment `0`。

P2 static ISA：

| static family | P1-S | P2-S |
|:--|--:|--:|
| MFMA32 | 56 | 56 |
| global load | 140 | 140 |
| global store | 16 | 16 |
| ds_write | 92 | 92 |
| ds_read | 56 | 104 |
| s_barrier | 44 | 44 |

进一步拆分 P2 `ds_read`：

```text
ds_read_b64  = 96
ds_read_b128 = 8
```

P1 主要是：

```text
ds_read_b128 = 56
```

因此 P2 的 final ISA 差异是真实的，但方向是“更窄的多次 LDS read”，不是
“更少的 machine work”。

---

## 6. correctness gate

### 6.1 正常随机/多 chunk 矩阵

P2 与 Z5B 做 BF16 byte-exact 比较：

| T | P2 vs Z5B | finite | max abs vs Z5B |
|--:|:--:|:--:|--:|
| 64 | PASS | PASS | 0.0 |
| 512 | PASS | PASS | 0.0 |
| 1024 | PASS | PASS | 0.0 |
| 2048 | PASS | PASS | 0.0 |
| 4096 | PASS | PASS | 0.0 |
| 8192 | PASS | PASS | 0.0 |
| 16384 | PASS | PASS | 0.0 |

### 6.2 caller-owned output / NaN-prefill

以下 edge cases 均通过：

- T=64 zero-V-new + NaN-prefilled caller-owned output；
- T=8192 zero-V-new + NaN-prefilled caller-owned output；
- T=16384 zero-V-new + NaN-prefilled caller-owned output；
- finite output；
- output reuse contract；
- P2 与 Z5B 的 BF16 bytes 完全一致。

### 6.3 旧 regression

本轮 Docker regression command：

```bash
python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/
    test_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py \
    test_qwen_gdn_direct_k64_block_dot_ab.py \
    test_qwen_gdn_direct_k64_block_dot_bv32_typed_operand_ab.py \
    test_qwen_gdn_bt64_bf16_recurrence_full_stage6s.py -s
```

结果：

```text
21 passed in 87.21s
```

这说明 P2 没有破坏 full-scope block-dot contract、direct-K64 block-dot tests
或 Stage6S BF16 recurrence bridge tests。

### 6.4 正确性解释

P2 只改变 operand materialization 的表示；它没有改变：

- MFMA 调用次数；
- K32 accumulation order；
- K/H logical row identity；
- phase-separated accumulator 顺序；
- BF16 input/output bits；
- causal mask 或 Q cache ownership。

所以 correctness PASS 只能说明新的 packed LDS consumer 保持语义，不等于它已经
有更好的调度、访存重叠或 latency。

---

## 7. T=2048 dynamic PMC

### 7.1 采集口径

P2 使用：

- gfx942 Docker；
- `rocprofv3`；
- T=2048；
- WG256；
- Grid_Size `131072`；
- 512 CTA；
- counter collection 采集真实 kernel；
- 下表为原始 counter 除以 512 的 per-CTA dynamic result。

明确区分：这些不是 static ISA count，也不是 memory bytes。

### 7.2 per-CTA 动态结果

| arm | MFMA | VMEM | LDS | VALU | SALU | profiler VGPR | profiler AccVGPR | occupancy |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Z5B | 160 | 672 | 672 | 7072 | 768 | 76 | 100 | 14.498718% |
| BDV2/P1-S | 160 | 448 | 464 | 8410 | 780 | 88 | 88 | 14.549514% |
| P2-S | 160 | 448 | 592 | 8474 | 780 | 88 | 88 | 14.880388% |
| native diagnostic | 160 | 140 | 480 | 3376 | 660 | artifact-specific | artifact-specific | artifact-specific |

P2 相对 P1：

```text
MFMA: unchanged 160
VMEM: unchanged 448
LDS: +128/CTA (+27.6%)
VALU: +64/CTA (+0.76%)
SALU: unchanged 780
```

P2 相对 Z5B 的 VMEM 优势仍在，但 P2 没有继续减少 global work；P2 的代价
主要落在 LDS consumer materialization。

### 7.3 machine metadata

P2 code-object metadata：

```text
VGPR=132
AGPR=48
SGPR=30
LDS=32768 B
private_segment=0
vgpr_spill=0
sgpr_spill=0
```

P2 profiler metadata：

```text
VGPR_Count=88
Accum_VGPR_Count=88
SGPR_Count=112
Scratch_Size=0
```

两组数字来自不同层次，不能混写成同一个 register metric。P2 没有出现
scratch、private memory 或 MIR spill。

---

## 8. fresh-process Eager body benchmark

### 8.1 口径

本轮使用 caller-owned isolated body diagnostic：

- current HIP stream；
- no CUDA Graph；
- 预分配 input/output；
- `warmup=10`；
- `repeat=50`；
- 7 个 fresh-process sessions；
- rotating arm order；
- 比较 Z5B、P1-S、P2-S、native same-shape diagnostic。

这不是 public Eager full graph 排名，because 当前实验只观察 chunk-o body。它的
价值是 same-shape block-dot body A/B。

### 8.2 T=2048

| arm | mean of 7 session medians (ms) | P2 relative |
|:--|--:|--:|
| Z5B | 0.066862214 | 1.0000x |
| P1-S | 0.070825143 | 1.0592x |
| P2-S | 0.073120214 | 1.0936x |
| native | 0.042594643 | 0.6371x of Z5B |

P2 的 7 个 paired `P2-Z5B` 差值为：

```text
min +5.0875 us
max +7.4315 us
mean +6.2580 us
```

所有 session 都是正差值。P2 相对 native 的 session ratio 平均为 `1.7167x`。

### 8.3 T=8192

| arm | mean of 7 session medians (ms) | P2 relative |
|:--|--:|--:|
| Z5B | 0.157708641 | 1.0000x |
| P1-S | 0.168587500 | 1.0690x |
| P2-S | 0.176261858 | 1.1177x |
| native | 0.090960786 | 0.5769x of Z5B |

P2 的 7 个 paired `P2-Z5B` 差值为：

```text
min +17.6860 us
max +19.1485 us
mean +18.5532 us
```

所有 session 仍为正差值。P2 相对 native 的 session ratio 平均为 `1.9378x`。

### 8.4 endpoint per-chunk slope

这里使用 T=2048 到 T=8192 的 endpoint slope：

```text
slope = (latency_8192 - latency_2048) / 96 chunks
```

| arm | endpoint slope (us/chunk) |
|:--|--:|
| Z5B | 0.946317 |
| P1-S | 1.018358 |
| P2-S | 1.074392 |
| native | 0.503814 |

P2 不仅 T=2048 较慢，长文本 slope 也比 Z5B 高约 `13.5%`。因此不能把 P2
描述成“可能只是固定 launch/intercept 变差”；其 per-chunk 机器工作也没有
改善。

### 8.5 T=16384

P2 的 T=16384 correctness 已通过，但性能测试没有运行。预注册条件是 T=2048
或 T=8192 先观察到稳定正收益后，才扩展到 T=16384。两处均为稳定负收益，
所以本轮不运行条件性长文本测试，避免把已经明确 No-Go 的 arm 扩大成无信息
的 benchmark。

---

## 9. P2 为什么“机器图不同但性能更差”

### 9.1 P2 确实没有白做

P2 回答了表示层问题：

```text
same source
  -> first-class block-dot operand IR
  -> late packed LDS consumer
  -> different LLVM
  -> different MIR/ISA
  -> different HSACO
```

这与 P1 的“LLVM 不同、final HSACO 相同”是实质区别。对于编译器团队，P2
证明了 AveLang 可以建立一个让 operand identity 穿过 outlining 到 late GPU
lowering 的控制点。

### 9.2 当前具体物化方式的问题

P2 的内部 `vector<4xbf16>` 语义被实现为一个个显式 `i64` 读：

```text
每个 packed word
  -> ds_read_b64
  -> bitcast
  -> MFMA consumer
```

而 P1 主要使用：

```text
vector<4xbf16> local load
  -> ds_read_b128
  -> MFMA consumer
```

结果是 P2 的 final ISA 中 `ds_read_b64` 数量增加，dynamic LDS 由 464/CTA
增加到 592/CTA。P2 没有减少 VMEM，因为 Q/K/H/V-new global producer graph
没有改；它只改了 LDS-to-fragment 的表示。

### 9.3 latency 与 ISA 的关系

P2 的 latency 回退与以下证据一致：

- MFMA 不变：不是计算工作增加；
- VMEM 不变：不是 global load 下降失败；
- LDS 增加：新的 packed consumer materialization 有额外 LDS issue；
- VALU 小幅增加：address/bitcast/fragment feeding 没有变成 native-style
  低地址计算；
- slope 变差：额外 consumer work 按 chunk 重复。

因此本轮不能声称“first-class operand representation 已经实现 native
performance”。只能声称“representation-preserving lowering 的控制点已经
实现并可观测”。

---

## 10. 与 recurrence 语义问题的关系

本轮只修改 chunk-o full-scope block-dot lowering，不改变 recurrence 语义。
它没有重新实现 pred、state feedback、V-new、BF16 boundary 或 full-v29
nonzero-W recurrence。

此前 recurrence 审计中出现的区别仍然成立：

```text
recurrence semantic preservation
  !=
block-dot operand physical preservation
```

P2 只能说明 block-dot 的 operand identity 在当前 source/IR pipeline 中可以
保存到 late lowering。它不能据此证明：

- full v29 recurrence 的 nonzero-W correctness 已经解决；
- current Triton 的全部 pred/update scheduling 已被复现；
- Z5B/native 的 latency gap 都来自一个 operand lowering pass。

这也是为什么 P2 完成后仍保持 Stage6Z isolated scope，不接 X2。

---

## 11. 回归与修改文件

### 11.1 compiler

```text
lib/Dialect/AveLang/IR/AveLangOps.td
  AMDGPUBlockDotMfmaOperandOp

lib/Dialect/AveLang/IR/AveLangOps.h
  forward declaration

lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.h
  pass factory declaration

lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc
  P2 selector
  operand row/word plan
  internal op creation
  late packed LDS/MFMA materialization

lib/Target/GPU/lower_to_llvm.cc
  GPU outlining 后的 P2 pass 和 snapshot
```

### 11.2 source/harness

```text
test/examples/linear_attention/vllm_compare/
  qwen_gdn_bt64_native_chunko_stage6z_block_dot_v2_full_scope.py
  bench_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py
  check_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py
  dump_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope_machine.py
  profile_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py
  test_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py
```

source 文件没有新建 P2 kernel；P2 只新增 selector 参数。

### 11.3 regression assertions

无设备 contract test 新增检查：

- source 暴露 `set_block_dot_operand_preservation`；
- source 允许 `p2_first_class`；
- compiler 中存在 `AMDGPUBlockDotMfmaOperandOp`；
- late pass factory 被加入 GPU pipeline；
- pre/post operand materialization snapshot 存在。

---

## 12. 报告与复现命令

### 12.1 correctness

```bash
python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/
    test_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py \
    test_qwen_gdn_direct_k64_block_dot_ab.py \
    test_qwen_gdn_direct_k64_block_dot_bv32_typed_operand_ab.py \
    test_qwen_gdn_bt64_bf16_recurrence_full_stage6s.py -s
```

结果：`21 passed in 87.21s`。

### 12.2 P2 machine dump

```bash
python3 test/examples/linear_attention/vllm_compare/
  dump_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope_machine.py \
  --variant specialized \
  --planner bdv2_p1_affine \
  --preservation p2_first_class \
  --T 2048 \
  --skip-initial-mlir \
  --out-dir test/examples/linear_attention/compile_bug/
    qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_bdv2_p2_machine_specialized
```

### 12.3 P2 dynamic profile

```bash
python3 test/examples/linear_attention/vllm_compare/
  profile_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py \
  --arm bdv2_p2_specialized --T 2048 --warmup 2 --repeat 5 \
  --out-dir test/examples/linear_attention/compile_bug/
    qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_bdv2_p2
```

### 12.4 body benchmark

```bash
python3 test/examples/linear_attention/vllm_compare/
  bench_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py \
  --T 2048 --sessions 7 --warmup 10 --repeat 50 \
  --arms z5b bdv2_p1_specialized bdv2_p2_specialized native \
  --out codex_qwen_bt64_stage6z_bdv2_p2_bench_T2048_sessions7.json

python3 test/examples/linear_attention/vllm_compare/
  bench_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py \
  --T 8192 --sessions 7 --warmup 10 --repeat 50 \
  --arms z5b bdv2_p1_specialized bdv2_p2_specialized native \
  --out codex_qwen_bt64_stage6z_bdv2_p2_bench_T8192_sessions7.json
```

原始 JSON 位于：

```text
codex_qwen_bt64_stage6z_bdv2_p2_bench_T2048_sessions7.json
codex_qwen_bt64_stage6z_bdv2_p2_bench_T8192_sessions7.json
```

---

## 13. 最终回答本轮问题

### 13.1 specialized lowering 是否真正保留到 LLVM/MIR/ISA？

是。P2 在 post-GPU-outlining 中存在 first-class internal op，late pass 生成
显式 packed `i64` LDS load，P2 的 LLVM/pre-LTO/final ISA/HSACO 均与 P1 不同。

### 13.2 相比 P1 减少了哪些机器工作？

没有减少最终动态机器工作。P2 保持 MFMA/VMEM，但增加了 LDS read：

```text
static ds_read: 56 -> 104
dynamic LDS:    464 -> 592/CTA
```

因此本轮不能把 P2 写成性能优化成功。

### 13.3 距离 current Triton W=0 control 还剩多少差距？

在同 shape isolated body：

| T | P2/native |
|--:|--:|
| 2048 | 1.7167x |
| 8192 | 1.9378x |

P2 仍明显慢于 native diagnostic。native 的更低 VMEM/VALU 与更紧凑的 operand
feeding 尚未由当前 generic block-dot planner 复现。

### 13.4 剩余差距是否已经证明来自 CTA ownership/BV32？

不能。P2 冻结的是当前 BV64/WG256 full-scope path，且只改变 operand
representation。它证明了一个更窄的事实：当前 first-class operand 的**具体
packed LDS materialization**没有带来收益。不能由此推导 CTA ownership 是唯一
根因，也不能把 P2 的回退归咎于 recurrence。

### 13.5 P2 是否成为新的 native full-recurrence baseline？

不是。P2 是 chunk-o block-dot compiler infrastructure。Z5B 继续作为 Stage6Z
isolated performance baseline；P2 不接 X2、不接 production。

---

## 14. 下一步边界

本轮只登记以下结论，不实施新的性能变体：

1. `BlockDotMfmaOperandPlan`/`MfmaOperandLayout` 这类通用内部表示值得保留；
2. P2 具体的 `volatile i64 -> vector<4xbf16>` materialization 不应直接升级为
   default lowering；
3. 下一次若继续，应在同一个 generic operand representation 上审计并减少
   packed LDS read，而不是创建 Qwen-specific op；
4. 不修改 allocator/RA，不恢复 full-v29 nonzero-W correctness 支线，不接入
   X2/production selector；
5. 任何新 arm 必须继续保留 Z5B、P1、P2 的 source/MLIR/LLVM/MIR/ISA/HSACO
   分层证据，并重新通过全 correctness gate。

本报告与设计文档、Stage6Z 总报告共同记录：P2 的价值是让 compiler 控制点和
机器差异变得可见，不是把一条不成熟的 packed LDS lowering 宣称为性能胜者。

---

## 15. 历史架构对照：为什么 R0 能保留语义，而 P1 会再次丢失

这一节是本轮最重要的编译器解释。P2 不是凭空增加一个内部 op，而是把已经
在 persistent recurrence 上验证过的 semantic-boundary 方法，应用到
`block_dot_bf16_f32` 的更低层。

### 15.1 R0/R1 的成功边界

R0 引入的是 first-class recurrence region：

```text
ave.gpu.amdgpu_qwen_persistent_recurrence
  -> QwenRecurrenceSchedulePlan
  -> recurrence planner
  -> late recurrence lowering
```

当前 Avelang GPU pipeline 中，相关顺序可以概括为：

```text
createLowerAveLangGPUToIntrinsicsPass
  -> createLowerAveLangToMemRefPass
  -> createFormQwenPersistentRecurrencePass
  -> snapshot persistent_recurrence_formed
  -> createPlanQwenPersistentRecurrencePass
  -> snapshot post_recurrence_joint_planner
  -> createQwenModuloSoftwarePipelinePass
  -> snapshot post_software_pipeline_scheduler
  -> createLowerQwenGdnRecurrenceStepPass
  -> createLowerQwenPersistentRecurrencePass
  -> ordinary GPU/LLVM/AMDGPU lowering
```

在 `createFormQwenPersistentRecurrencePass` 之后，compiler 仍能同时看到：

```text
loop-carried FP32 state
  -> W/pred
  -> corrected
  -> BF16 V-new/V-decay boundary
  -> K update
  -> FP32 feedback
```

所以 `QwenRecurrenceSchedulePlan` 可以联合规划 producer、consumer、LDS lifetime、
next chunk 和 state feedback，而不是让 pred/update 先各自变成普通 vector SSA。

R0 本身保持旧 B0 code object byte-identical；它的价值是把语义边界建立起来。
随后 R1 joint planner 利用这个边界，在 T=2048 获得约 `21.49%` 的正收益；R4
继续成为 native recurrence baseline。这说明 first-class semantic region 不是
抽象装饰，而是能改变后续完整 machine schedule 的控制点。

### 15.2 当前 block-dot 的旧边界

`block_dot_bf16_f32` 的 source helper 先创建：

```text
AMDGPUBlockDotBF16F32Op
```

但在当前 pipeline 中，`createLowerQwenBlockDotPass` 位于 GPU outlining 之前。
旧路径大致为：

```text
AMDGPUBlockDotBF16F32Op
  -> LowerQwenBlockDotPass
  -> emitFullScopeProducer
  -> emitGenericOperandBPair
  -> memref/vector/arithmetic SSA
  -> GPU outlining
  -> LLVM
  -> AMDGPU/LTO
```

P1 的 `LogicalBlockLayoutPlan` 只是
`lower_qwen_block_dot_pass.cc` 内部的 C++ 临时对象。它保存过
`wave/lane/packet/row/feature`，但没有以 first-class IR value/type/interface
留在下一个 pass。

因此第一次丢失“这是 MFMA A/B dot operand + physical layout”的边界，不是在
最终 LTO，也不是在 RA，而是在 `LowerQwenBlockDotPass` 的 operand expansion
中：

```text
emitGenericOperandBPair
  -> ordinary memref/vector load
  -> <4xbf16> / <8xbf16> vector
  -> extract/insert 或普通 LLVM vector load
```

从这一刻开始，后续 AMDGPU/LTO 只看到“若干 BF16 vector 和 LDS address”，看不到
它们必须作为某个 MFMA32 operand、使用哪种 lane/register fragment mapping、以及
哪一种 physical shared encoding。P1 的 planner 只改善了这一步之前生成的 SSA，
没有改变这个 lifetime boundary。

### 15.3 旧 block-dot specialized 为什么能成功

历史 block-dot 同源 specialized lowering 的实验结果是：

```text
generic     = 0.724316 ms
specialized = 0.519632 ms
speedup     = 1.394x
```

那次 specialized lowering 并不只是把一条 vector load 改宽。它同时控制了：

```text
producer staging once
  -> shared operand
  -> multiple consumer uses
```

也就是说 producer、shared staging、consumer reuse 仍处于同一个 lowering scope。
差异能够继续穿过 LLVM、MIR、ISA 和 HSACO，说明那个 lowering 保留了足够真实的
producer-consumer graph，而不是只添加了不会被后端读取的 metadata。

这正是本轮不能简单下结论“block-dot first-class op 没用”的原因：**旧路径已经
证明，合适的 semantic boundary 可以带来 1.394x；P1 失败只说明这次 boundary
放得太早或表达得不够强。**

### 15.4 P1 为什么被 AMDGPU/LTO 收敛

P1 的 source/LLVM 差异是：

```text
<8xbf16> load
  -> extract/insert
```

尝试变为：

```text
<4xbf16> load
```

这减少了 lowered LLVM 的 `extractelement/insertelement` 文本数量，但对后端来说
仍然是普通 vector value。P1 没有一个真实的 consumer op 继续携带：

```text
operand role
physical shared encoding
lane/register mapping
MFMA fragment identity
```

因此 AMDGPU/LTO 可以把 P1 和 BDV2-S 识别为等价的 vector/LDS/MFMA 程序，并在
最终 machine pipeline 中重新生成相同 graph：

```text
P1 LLVM != BDV2 LLVM
P1 pre-LTO != BDV2 pre-LTO
P1 final HSACO == BDV2 final HSACO
P1 PMC == BDV2 PMC
```

这不是“LLVM pass 把一条指令优化坏了”的直接证据，而是更精确的表示结论：
`<4xbf16>` 本身不足以表达一个稳定的 MFMA operand/layout contract。

### 15.5 P2 把哪个 boundary 向后延长

P2 的 boundary 变为：

```text
block_dot
  -> BlockDot/MfmaOperandPlan
  -> GPU outlining
  -> first-class internal operand op
  -> target-specific late materialization
  -> packed LDS/register operand
  -> MFMA
```

因此 P2 没有把 operand identity停留在 C++ planner 或 metadata，而是通过
`AMDGPUBlockDotMfmaOperandOp` 让后续 pass 必须真正消费它。P2 的当前实现虽然
选择了不够高效的 B64 materialization，但它成功改变了最终 machine graph；这
正是 P1 缺少的控制点。

### 15.6 这次能证明什么，不能证明什么

可以证明：

1. 原始 `block_dot_bf16_f32` API 足够承载通用 full-scope logical block；
2. K/H 可以共享一个通用 B-operand planner；
3. first-class operand representation 可以存活到 GPU-module late lowering；
4. 真实 consumer 能把这个 representation 变成不同的 LLVM/MIR/ISA/HSACO；
5. P2 当前的 packed LDS 物化方式是性能负担，不是语义失败。

不能证明：

1. full v29 recurrence nonzero-W correctness 已解决；
2. P2 已经复现 Triton 的 register/shared ownership；
3. 当前全部 Z5B/native 差距都来自 block-dot lowering；
4. 仅仅保留 first-class op 就自动得到 1.394x 历史收益。

---

## 16. 对未来 Q 和 V-new 的复用路径

P2 选择的内部 contract 是 operand-role/layout contract，不是 K/H 名字：

```text
BlockDotOperandPlan
  operand role A/B
  logical shape
  transpose
  resident/global-backed source
  MFMA shape
  physical shared encoding
  lane/register fragment mapping
```

因此未来可以按同一机制扩展：

### Q 的 MFMA-A

Q 不需要新建 `qwen_q_fragment_load`。它可以把 dedicated Q cache 作为 resident
source，然后构造：

```text
role=A
resident=Q-cache
transpose=logical-Q
physical_encoding=<target encoding>
```

由同一个 operand planner 生成 MFMA-A fragment。

### V-new 的 MFMA-B

V-new 也不需要新建 chunk-o 专用 op。它可以把 BF16 V-new producer 或 CTA-local
resident tile 描述为：

```text
role=B
source_role=V-new
residency=global/shared/register
physical_encoding=<target encoding>
```

这样 producer ownership、shared placement、dot consumer 和 reuse identity 仍
在一个通用 representation 中，不需要为 Q 或 V-new 复制一套 lowering pass。

### 当前边界

本轮只实现并验证 K/H B operand。Q A 和 V-new B 只登记为可复用设计，不在本轮
接入，也不因为 P2 结果不佳而追加新的 source variant。
