# Stage 6Z Z4C Accumulator Live-Range and Provenance Audit

## 结论

本轮是 **read-only exact-LTO MIR / LLVM / ISA 审计**。没有修改 kernel
source、compiler、allocator/RA、selector、recurrence HSACO 或 X2，也没有运行
新的性能实验。

最终结论选择 **Case A**：

> Z4C 的资源增长主要由 `inter_acc`、`score_acc0`、`score_acc1` 在同一个
> K32 融合循环中的同时存活解释；额外的 fragment/copy 临时值是伴随成本，
> 但不是独立的主要根因。

这不是“64 个 AGPR 都是 accumulator”的结论。精确机器证据显示：

- Z4C 的三个早期逻辑 accumulator 各需要一个 16-AGPR MFMA 目的区，合计
  48 个 AGPR；
- Z4C 另外出现 `$agpr48..$agpr63` 的 copy/fragment bridge 区域，使 code
  object 的 AGPR 上界达到 64；
- Z2 采用 phase-separated 顺序，MFMA accumulator 物理区在 `$agpr0..31`
  内复用，code object AGPR 为 32；
- Z4C 的 code object VGPR/AGPR 从 `168/32` 上升到 `220/64`，但没有
  spill/private memory；
- profiler 的 `Accum_VGPR_Count=32 -> 52` 是硬件资源计数，不是 MIR
  virtual-register 数量，也不是“52 个对应的 source accumulator”。它与
  AGPR 物理分配扩大相符，但不能按一比一关系映射到 SSA 对象。

因此下一步只登记一个候选，不在本轮实现：

**Z5A：dedicated full-Q LDS cache**

- 以 fixed Z2 为起点；
- 额外约 16 KiB Q LDS，使 Q producer pass 从 3 次降到 1 次；
- 保持 Z2 原始 `Phase A -> Phase B0 -> Phase B1` accumulator 顺序；
- 不把三个 accumulator 融合到同一个 K32 loop；
- 不改 RA、MFMA geometry、WG256、BV64、BT64、BK32、数学或 X2；
- 仅登记，不实现，不接入 selector 或 production。

## 1. 审计对象和证据边界

### 1.1 两个 arm

| arm | kernel | 角色 |
|:--|:--|:--|
| fixed Z2 | `_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z2` | Stage 6Z 唯一 AveLang baseline |
| Z4C | `_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4c_full_k32_q_fusion` | Q pass 3 -> 1 的 diagnostic arm |

两者都固定为 `gfx942`、`BT64/BV64/BK32`、`WG256`、MFMA32、BF16 ABI、
16 KiB LDS、相同输出数学和相同 20 条静态 MFMA32 指令。Z4C 只改变 Q
producer 的 lifetime/consumer 组织，因此本轮可以把资源变化与
`inter_acc/score_acc0/score_acc1` 的 live graph 对齐。

### 1.2 工件

本轮使用的主要工件如下：

```text
test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z4_q_ladder.py

test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
  codex_qwen_bt64_stage6z_z4_q_machine/z2/
  codex_qwen_bt64_stage6z_z4_q_machine/z4c/
```

每个 arm 内部使用：

```text
lowered_llvm.ll
pre_lto_amdgcn.s
exact_lto/kernel_section_03.mir  # exact LTO, before greedy RA
exact_lto/kernel_section_08.mir  # exact LTO, after virtregrewriter
final_isa.s
machine_evidence.json
exact_lto/summary.json
```

`kernel_section_03.mir` 的正式标记是
`IR Dump Before Greedy Register Allocator (greedy)`。它适合观察 virtual
register 的 def/use 和 accumulator class。`kernel_section_08.mir` 已经没有
virtual register，适合观察 physical AGPR/VGPR 分配。

初始 high-level MLIR capture 在两个 arm 都被现有 Docker MLIR printer 的
segmentation fault 中止。因此本报告不把不存在的 initial MLIR 当成证据，
也不声称已经完成 high-level MLIR 的逐行比较。lowered LLVM、exact-LTO MIR、
ISA 和 HSACO 是本轮实际可复核证据。

### 1.3 证据等级

| 等级 | 含义 | 本报告用途 |
|:--|:--|:--|
| A | source/LLVM 定义和 exact MIR def/use 直接可见 | 判断 accumulator 来源、MFMA def/use、对象顺序 |
| B | post-RA physical assignment、live-in、COPY/REG_SEQUENCE 可见 | 判断物理 AGPR 区域和伴随临时值，不宣称精确 cycle lifetime |
| C | profiler/code-object 汇总资源字段 | 判断资源结果和相关性，不做 vreg 一比一归因 |

当前 exact 工件没有经过专门 `LiveIntervals` 文本导出。因此下面的
“live interval”分为两类：

1. **逻辑/SSA lifetime proxy**：来自 source 对象、LLVM alloca、MIR
   accumulator def/use；
2. **物理 allocation evidence**：来自 post-RA MIR 的 AGPR live-in、COPY
   和 MFMA destination。

不能把 MIR 字节偏移直接称为真实硬件 cycle lifetime。

## 2. 先分清三个不同的资源指标

### 2.1 Code object metadata

这是 HSACO metadata 中的静态资源上界，不是 profiler 运行时的
`Accum_VGPR_Count`。

| 指标 | fixed Z2 | Z4C | 增量 |
|:--|--:|--:|--:|
| code-object VGPR | 168 | 220 | +52 |
| code-object AGPR | 32 | 64 | +32 |
| SGPR | 44 | 36 | -8 |
| LDS block | 16384 B | 16384 B | 0 |
| private segment | 0 B | 0 B | 0 |
| VGPR spill count | 0 | 0 | 0 |
| SGPR spill count | 0 | 0 | 0 |
| HSACO SHA256 | `2afaa8867da656421a9c87988ed4dcad757796c1b1d3b93e27307fb401a1c09a` | `f9d12fbce1d82be7a6ba10abc48ca4de13ae98093dcb9f1d978a297b6a68f307` | different |

直接来源：两个 arm 的 `machine_evidence.json` 和 `llvm-readobj` metadata。

### 2.2 T=2048 profiler PMC

这是 rocprof 运行时收集的动态工作量和硬件资源字段。动态指令数按
`Grid_Size=131072 / 256 = 512` 个 CTA 归一化。

| 指标 | fixed Z2 | Z4C | 解释 |
|:--|--:|--:|:--|
| `Accum_VGPR_Count` | 32 | 52 | profiler 资源字段，不是 vreg ID |
| `VGPR_Count` | 88 | 100 | profiler 运行资源字段 |
| `SQ_INSTS_MFMA / CTA` | 160 | 160 | 数学工作没有减少 |
| `SQ_INSTS_VMEM / CTA` | 928 | 672 | Z4C Q 重复加载减少 |
| `SQ_INSTS_LDS / CTA` | 928 | 672 | Z4C 相关 staging 减少 |
| `SQ_INSTS_VALU / CTA` | 11400 | 7514 | 动态非 MFMA 工作减少 |
| `SQ_INSTS_SALU / CTA` | 1072 | 778 | 地址/控制工作减少 |
| `LDS_Block_Size` | 16384 B | 16384 B | LDS 容量没有变化 |
| `Scratch_Size` | 0 | 0 | 没有 spill 路径 |

原始数据：

```text
codex_qwen_bt64_stage6z_z4_q_pmc_T2048/stage6z_z2_T2048_rocprof.json
codex_qwen_bt64_stage6z_z4_q_pmc_T2048/stage6z_z4c_T2048_rocprof.json
```

`Accum_VGPR_Count` 的变化不能写成“Z4C 有 52 个 accumulator”。正确说法
是：Z4C 使代码对象的 AGPR 分配上界从 32 增长到 64，profiler 观察到的
accumulator-related resource count 从 32 增长到 52；二者方向一致，但统计
口径不同。

### 2.3 MIR virtual register 数量

对 exact pre-greedy MIR 使用 `%id:register_class` 去重得到：

| class | fixed Z2 | Z4C | 结论 |
|:--|--:|--:|:--|
| `areg_512_align2` token 数 | 6 | 4 | 不是逻辑 accumulator 数的可靠计数，Z2 有 CFG/phi clone |
| `av_512_align2` token 数 | 4 | 5 | Z4C 多一个宽 fragment/copy 形态 |
| `vreg_64_align2` token 数 | 358 | 292 | Z4C 并非简单地产生更多 64-bit address vreg |
| `vgpr_32` token 数 | 1462 | 1286 | raw vreg 数下降，不能解释物理 VGPR 上升 |

这个结果很重要：Z4C 的资源 cliff 不是由“virtual register 总数更多”直接
造成的。真正的差异是少数宽 accumulator/fragment 值的**同时可分配区间**扩大，
使 greedy RA 必须使用更高的物理 AGPR/VGPR 编号。

## 3. 高级源码层的 lifetime 差异

### 3.1 fixed Z2：Phase-separated

源码文件：

```text
test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py
```

关键位置：

```text
line 81   inter_acc = al.full((16,), 0.0, al.f32)
line 85   Phase A k_stage loop
line 103  inter_acc MFMA32 update
line 111  for source_half in al.range(2)
line 113  score_acc = al.full((16,), 0.0, al.f32)
line 142  score_acc MFMA32 update
line 154  score_acc element consumer / score materialization
line 174  intra_acc = al.full((16,), 0.0, al.f32)
line 182  intra_acc MFMA32 update
```

Z2 的关键结构是：

```text
inter_acc
  -> Phase A MFMA
  -> Phase A 结束

score_acc(source_half=0)
  -> score-half-0 MFMA
  -> score-half-0 consumer

score_acc(source_half=1)
  -> score-half-1 MFMA
  -> score-half-1 consumer

intra_acc
  -> intra MFMA
```

源码虽有 loop-carried MFMA accumulator，但 score accumulator 在
`source_half` 内部创建并消费。编译器可以把同一个 private alloca 和同一组
物理 AGPR 在两个 source-half 之间复用。

### 3.2 Z4C：full-K32-Q-fusion

源码文件：

```text
test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z4_q_ladder.py
```

Z4C 关键位置：

```text
line 504  inter_acc  = al.full((16,), 0.0, al.f32)
line 505  score_acc0 = al.full((16,), 0.0, al.f32)
line 506  score_acc1 = al.full((16,), 0.0, al.f32)
line 510  一个共同的 for k_stage in al.range(4)
line 528  inter_acc MFMA32
line 546  score_acc0 MFMA32
line 564  score_acc1 MFMA32
line 568  两个 score half 在全部 K32 reduction 后才 serialize
line 601  intra_acc 才创建
```

Z4C 的逻辑图是：

```text
preheader:
  define inter_acc
  define score_acc0
  define score_acc1

for k_stage:
  use/update inter_acc
  use/update score_acc0
  use/update score_acc1

after all K stages:
  consume score_acc0
  consume score_acc1

later:
  define/use intra_acc
```

这不是“同一条 MFMA 同时执行三个 accumulator”。它的含义是：三个
16-element FP32 accumulator object 的生命周期在同一个 loop region 中重叠，
而不是像 Z2 那样把 score accumulator 的生命周期限制在各自的 source-half
phase 内。

## 4. LLVM 层 provenance

### 4.1 Z2 private accumulator allocas

`z2/lowered_llvm.ll` 第 53-55 行：

```llvm
%44 = alloca float, i64 16, align 4, addrspace(5) ; inter_acc
%45 = alloca float, i64 16, align 4, addrspace(5) ; score_acc, reused by source_half
%46 = alloca float, i64 16, align 4, addrspace(5) ; intra_acc
```

对应的 machine-visible LLVM operation：

```text
line 118  store zero -> %44
line 299  load <16 x float> %44
line 300  MFMA -> %255
line 301  store %255 -> %44
line 318  load %44
line 319  MFMA -> %273
line 320  store %273 -> %44

line 340  store zero -> %45
line 560  load %45
line 561  MFMA -> %473
line 562  store %473 -> %45
line 579  load %45
line 580  MFMA -> %491
line 581  store %491 -> %45

line 751  store zero -> %46
line 846  load %46
line 847  MFMA -> %708
line 848  store %708 -> %46
line 865  load %46
line 866  MFMA -> %726
line 867  store %726 -> %46
```

Z2 的 score family 在 LLVM 层是一个 `%45` private object，通过控制流和
loop 复用，而不是两个长期同时存在的 score allocas。

### 4.2 Z4C private accumulator allocas

`z4c/lowered_llvm.ll` 第 55-58 行：

```llvm
%46 = alloca float, i64 16, align 4, addrspace(5) ; inter_acc
%47 = alloca float, i64 16, align 4, addrspace(5) ; score_acc0
%48 = alloca float, i64 16, align 4, addrspace(5) ; score_acc1
%49 = alloca float, i64 16, align 4, addrspace(5) ; intra_acc
```

三个早期 accumulator 在第 121-123 行连续初始化：

```text
line 121  store zero -> %46
line 122  store zero -> %47
line 123  store zero -> %48
```

实际 MFMA call sites：

```text
%46 / inter_acc:
  lines 304-306, 323-325

%47 / score_acc0:
  lines 480-482, 499-501

%48 / score_acc1:
  lines 660-662, 679-681

%49 / intra_acc:
  lines 993-1012 附近
```

LLVM 的 private alloca 仍然是后续 AMDGPU lowering 的输入，但它已经直接
暴露了 A/B 的核心差异：Z2 有一个 score alloca，Z4C 有两个 score allocas，
并且三者都在共同 loop 之前完成初始化。这个差异发生在 RA 之前，所以不能
归因成 RA 自己“凭空创造了三个 accumulator”。

## 5. Exact pre-greedy MIR accumulator provenance ledger

### 5.1 逻辑对象到 virtual register

| arm | 逻辑对象 | exact MIR vreg/class | 初始定义 proxy | MFMA tied-def/use | 最后可见 consumer/proxy | K-stage loop carried | 同时 live 结论 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| Z2 | `inter_acc` | `%771:areg_512_align2` | 1952-2192B；MFMA 输入准备后 | 11904, 12000, 12096, 12192B | 12192B tied-def；之后进入 phase-end/copy path | 是，跨 4 个 K stage | 与 score 初始化不重叠 |
| Z2 | `score_acc` half-0/CFG value | `%3962:areg_512_align2` | 13984B，另有 `%3963` phi/branch clone at 13968B | `%3962` 20256/20352B；`%3963` 20912/21008B | 21024B copy/branch merge | 是，限于 score phase | inter 已结束 |
| Z2 | `intra_acc` | `%3071:areg_512_align2` | loop path before 64032B | 64032, 64128, 64224, 64320, 64608, 64624, 64720, 64912B | element copies 65312-73776B | 是，跨 intra consumers | 不与早期三者重叠 |
| Z2 | intra CFG clone | `%3989/%4014:areg_512_align2` | 40144/40160B | 79936B onward and 80592/80688B | later output path | 是，CFG clone | 物理 AGPR 仍复用 |
| Z4C | `inter_acc` | `%825:areg_512_align2` | 4768-5008B | 13088, 13184, 13280, 13664B | 13664B tied-def，13792B copy-out path | 是，跨共同 K loop | score0/score1 已先初始化 |
| Z4C | `score_acc0` | `%3562:areg_512_align2` | 5296B | 14032, 14128, 15264, 15360B | 15376B copy to score materialization | 是，跨 K stages | 与 inter 的逻辑 lifetime 重叠 |
| Z4C | `score_acc1` | `%3520:areg_512_align2` | 5280B | 16096, 16192, 16544, 16640B | 16656B copy to score materialization | 是，跨 K stages | 与 inter/score0 的逻辑 lifetime 重叠 |
| Z4C | `intra_acc` | `%2671:areg_512_align2` | later phase before 58768B | 58768, 58864, 58960, 59056, 59344, 59360, 59456, 59728B | element copies 60128-68512B | 是，跨 intra consumers | 在三者之后才出现 |

说明：`areg_512_align2` 是一个 512-bit accumulator fragment class；在这个
MFMA32 路径中，其物理目的区表现为 16 个连续 AGPR 32-bit lanes。表中的
`areg` token 数不能直接相加为 code-object AGPR 数，因为 CFG/phi、subreg
和 copy 可能产生多个 virtual token，而 RA 可以复用物理区。

### 5.2 Z4C overlap 的最直接证据

在 `z4c/exact_lto/kernel_section_03.mir` 中：

```text
4768-5008B   %825  inter_acc 初始化
5280B        %3520 score_acc1 初始化
5296B        %3562 score_acc0 初始化

13088-13664B %825  inter MFMA tied-def chain
14032-15360B %3562 score_acc0 MFMA chain
16096-16640B %3520 score_acc1 MFMA chain
```

特别是 score0 和 score1 的初始化早于 inter MFMA，而不是在 inter phase
之后创建。source、LLVM alloca 和 pre-greedy MIR 三层都给出同一顺序，因此
这是同一个高层改动在后端的保留，而不是 post-RA 反推出来的猜想。

### 5.3 Z2 phase separation 的对照证据

`z2/exact_lto/kernel_section_03.mir` 中：

```text
1952-2192B   %771 inter_acc 初始化
11904-12192B %771 inter MFMA chain

13968-13984B %3963/%3962 score phase 初始化
20256-20352B %3962 score MFMA
20912-21008B %3963 score MFMA

40144-40160B %4014/%3989 intra CFG 初始化
64032B       %3071 intra MFMA region
```

虽然 Z2 因为 CFG/phi 也出现多个 `areg_512_align2` token，但 inter 的 MFMA
region 在 score accumulator 初始化之前结束；score family 完成后才进入
intra。因此 Z2 的逻辑 lifetime 是阶段化的，物理 AGPR 可以循环利用。

## 6. Post-RA physical AGPR evidence

### 6.1 Z2：两组 AGPR 物理区循环复用

`z2/exact_lto/kernel_section_08.mir` 的 MFMA destination 主要是：

```text
inter region       $agpr0..$agpr15
score branch A     $agpr16..$agpr31
score branch B     $agpr0..$agpr15
intra region       $agpr0..$agpr15 和 $agpr16..$agpr31 复用
```

最大物理 AGPR index 为 31，和 code-object `.agpr_count: 32` 一致。

### 6.2 Z4C：三组 MFMA accumulator destination 同时出现

`z4c/exact_lto/kernel_section_08.mir` 的关键位置：

```text
13088-13664B  inter_acc   -> $agpr32..$agpr47
14032-15360B  score_acc0  -> $agpr16..$agpr31
16096-16640B  score_acc1  -> $agpr0..$agpr15
58768-59728B  intra_acc   -> $agpr0..$agpr15
```

这组分配和三个逻辑 accumulator 的 provenance 一一对应：

| 逻辑对象 | post-RA MFMA destination | 区域 |
|:--|:--|:--|
| `inter_acc` | 4 条 MFMA32 | `$agpr32..47` |
| `score_acc0` | 4 条 MFMA32 | `$agpr16..31` |
| `score_acc1` | 4 条 MFMA32 | `$agpr0..15` |
| `intra_acc` | 8 条 MFMA32 | 后续复用 `$agpr0..15` |

因此 Z4C 不是增加了 MFMA 数量。它是把前三组 accumulator 的物理容纳
需求从“两组可复用”推到“三组早期可用”。这直接解释了 AGPR metadata
从 32 到至少 48 的增长。

### 6.3 为什么 code object 是 64 而不是 48

在 Z4C inter setup 附近，`kernel_section_08.mir` 还出现：

```text
5776-6000B  $agpr32..$agpr46 的 COPY 链
6096B       $agpr63 = COPY $agpr47
6112-6352B  $agpr63..$agpr48 -> VGPR copy 链
```

这些不是第四个 MFMA accumulator。它们是 inter accumulator 的 element
materialization、fragment bridge 和后续 scalar/DS consumer 所需的 COPY
临时值。由于三组早期 accumulator 已经占据 `$agpr0..47`，这组 bridge
临时值不能再完全复用低编号物理区，最高编号扩展到 `$agpr63`。

所以：

```text
Z4C code-object AGPR 64
  = 三组 accumulator destination 约 48
  + 伴随 fragment/copy bridge 约 16
```

这里的“约”表示资源分区的解释，不是对每个 code-object AGPR 做硬性
语义标注。post-RA MIR 的 COPY 和 live-in 是证据，硬件并没有提供一个
标注为“this register belongs to score_acc0”的字段。

### 6.4 VGPR 168 -> 220 的解释

Z4C 的 raw pre-greedy `vgpr_32` token 数反而低于 Z2，说明 `220` 不是
“源码新写了更多 VGPR scalar”。更符合证据的解释是：

1. 三个 `areg_512_align2` 早期同时可用，使 RA 不能像 Z2 那样尽早回收
   accumulator 的 fragment bridge；
2. Z4C 的 Q 共享 consumer 让 Q/score operand、MFMA input packet 和
   accumulator copy 更容易跨过同一个 phase 边界；
3. 物理 VGPR 的高编号区域用于这些 input/copy/address bridge，最终 code
   object 上界扩展到 220；
4. Z4C 的动态 VMEM/VALU/SALU 反而低于 Z2，所以这不是简单的“地址指令更多”
   解释，而是 live-range/packing/physical allocation 的解释。

这部分的精确归因等级为 B：MIR 可以证明高编号 VGPR/AGPR 和 COPY/fragment
区域存在，但当前没有独立 LiveIntervals dump 能给出每个 physical VGPR
的完整 cycle lifetime。

## 7. Final ISA 对照

### 7.1 Z4C

`z4c/final_isa.s` 中可直接看到：

```text
inter_acc:
  a[32:47]  at ISA offsets 0x2690, 0x26C8, 0x26D8, 0x26E0

score_acc0:
  a[16:31]  at ISA offsets 0x2724, 0x272C, 0x27D8, 0x27E0

score_acc1:
  a[0:15]   at ISA offsets 0x2930, 0x2938, 0x29E4, 0x29EC

intra_acc:
  later reuses a[0:15]
```

Z4C final ISA 静态统计：

| 指令类别 | 数量 |
|:--|--:|
| MFMA32 | 20 |
| global load | 120 |
| global store | 16 |
| DS read | 20 |
| DS write | 72 |
| `s_waitcnt` | 108 |
| `s_barrier` | 8 |
| `v_lshl_add` | 138 |
| `v_add` | 152 |

### 7.2 Z2

`z2/final_isa.s` 中 inter/score/intra MFMA destination 在 `a[0:15]` 和
`a[16:31]` 之间循环复用，没有 `a[32:47]` 或更高的 accumulator
destination。Z2 静态统计为：

| 指令类别 | 数量 |
|:--|--:|
| MFMA32 | 20 |
| global load | 136 |
| global store | 16 |
| DS read | 20 |
| DS write | 88 |
| `s_waitcnt` | 127 |
| `s_barrier` | 9 |
| `v_lshl_add` | 159 |
| `v_add` | 187 |

ISA 证明了两点：

1. Z4C 没有通过减少 MFMA 数量获得动态工作量下降；
2. Z4C 的低 VMEM/LDS/VALU/SALU 与更高 AGPR/VGPR 上界可以同时出现，
   这是一个典型的“减少重复 producer work，但扩大 live graph”的 trade-off。

## 8. 逐项回答用户要求

### 8.1 各 accumulator family 占用的 class 和数量

每个 logical MFMA32 accumulator 在 MIR 中是 `areg_512_align2`，对应 16
个 32-bit accumulator lanes。Z4C 的三个早期 family 映射为：

```text
inter_acc   -> %825:areg_512_align2 -> $agpr32..47
score_acc0  -> %3562:areg_512_align2 -> $agpr16..31
score_acc1  -> %3520:areg_512_align2 -> $agpr0..15
```

`intra_acc` 是后续的 `%2671:areg_512_align2`，它不和这三个早期 family
同时处于同一 source phase，且 post-RA 复用 `$agpr0..15`。

### 8.2 Z4C 新增的同时-live accumulator fragment 数

相对 Z2 的 phase-separated 物理复用：

- Z2：早期最多保留两组可复用的 16-AGPR 区域；
- Z4C：inter、score0、score1 对应三组早期区域；
- 净新增：一组约 16-AGPR 的早期同时-live accumulator capacity；
- 因为 bridge/copy 临时值，code-object AGPR 的最大编号再扩展到 64。

### 8.3 是否有额外 fragment/copy/address temporary

有，证据等级 B：

- Z4C post-RA 在 `$agpr32..63` 有 inter accumulator 的 copy/fragment
  bridge；
- Z4C final ISA 的地址/布局指令数量没有上升，反而从 Z2 的
  `v_lshl_add=159, v_add=187` 降到 `138, 152`；
- 因此这些临时值是 accumulator overlap 带来的 materialization/packing
  伴随物，不是一个独立的“地址算术爆炸”主因。

### 8.4 是否出现更高物理寄存器区间

是：

- Z2 最高 MFMA accumulator destination 为 `$agpr31`；
- Z4C MFMA inter destination 到 `$agpr47`；
- Z4C bridge COPY 到 `$agpr63`；
- code-object metadata 对应 `AGPR=32 -> 64`，VGPR 对应 `168 -> 220`。

### 8.5 是否有 spill/scratch

没有：

- 两边 `.private_segment_fixed_size = 0`；
- 两边 `.vgpr_spill_count = 0`；
- 两边 `.sgpr_spill_count = 0`；
- exact-LTO `kernel_section_07.mir` 没有 `SI_SPILL_AV32_SAVE` 或
  `SI_SPILL_AV64_SAVE`；
- 这是 register pressure/resource cliff，但尚未跨过 spill 阈值。

### 8.6 Final ISA 的 MFMA accumulator 区域

已在第 7 节列出。最关键的机器差异是：

```text
Z2:  a[0:15] / a[16:31] 复用
Z4C: inter=a[32:47], score0=a[16:31], score1=a[0:15]
```

## 9. Case A/B/C 判定

### Case A：通过

选择 Case A 的证据链为：

1. source：Z4C 在共同 K32 loop 前定义三组 accumulator；
2. LLVM：Z4C 有 `%46/%47/%48` 三个独立 FP32 alloca，Z2 只有一个
   score `%45` alloca；
3. pre-greedy MIR：`%825/%3562/%3520` 三个 `areg_512_align2` 在同一
   loop region 内分别承担 inter/score0/score1 的 tied MFMA chain；
4. post-RA MIR：三组 MFMA destination 分别占据 `$agpr32..47`、
   `$agpr16..31`、`$agpr0..15`；
5. final ISA：Z4C 的 inter/score0/score1 目的区没有被压缩成 Z2 的两组
   复用形式；
6. 结果：MFMA 数不变、LDS 容量不变、scratch 为 0，但 AGPR/VGPR 上界
   上升。

### 为什么不是 Case B

Case B 要求 accumulator overlap 存在，但主要增长来自另一个独立的
fragment/copy/address temporary。当前证据不支持“主要来自其他对象”：

- 额外 AGPR32-47 直接就是 inter accumulator 的 MFMA destination；
- score0/score1 各自占用另一组 16-AGPR；
- bridge/copy 约 16 AGPR 是后续影响，不是最初从 32 到 48 的主要来源；
- 地址/布局动态工作还下降了，不能把资源增长首要归因为地址计算。

### 为什么不是 Case C

Case C 要求无法从现有 MIR 判断 accumulator 的具体来源。这里 source、LLVM、
pre-greedy MIR、post-RA MIR 和 ISA 对同一映射形成闭环，虽然没有精确
LiveIntervals cycle dump，但已足以判断逻辑 lifetime 和物理 AGPR 分区。

## 10. 唯一后续候选：登记 Z5A，不实现

由于 Case A 成立，只登记以下一个候选：

```text
Z5A: dedicated full-Q LDS cache
```

设计约束：

1. 从 fixed Z2 分叉；
2. 额外约 16 KiB Q LDS；
3. Q producer pass 3 -> 1；
4. 恢复/保持 Z2 的 Phase A -> Phase B0 -> Phase B1 accumulator 顺序；
5. 不把 `inter_acc`、`score_acc0`、`score_acc1` 融合成共同 K32 loop；
6. 不修改 K/H/V-new/g/output、MFMA、WG、BV、BT、BK、数学和 ABI；
7. 不修改 allocator/RA、selector、production、X2 或 recurrence HSACO；
8. 本轮没有实现 Z5A，也没有生成 Z5A kernel 或 benchmark 结果。

Z5A 的理由不是“Q LDS 一定更快”，而是：它尝试保留 Z4C 的 Q producer
工作量收益，同时避免 Z4C 已被证明的三 accumulator overlap。它需要单独的
correctness、LDS/occupancy、exact-LTO 和 fresh-process 性能 gate，不能把
本报告的结论当成 Z5A 已验证。

## 11. 审计闭环

本轮完成了：

- Z2/Z4C code-object metadata 对照；
- profiler `Accum_VGPR` 与 code-object AGPR 的口径分离；
- source accumulator 定义和 phase 顺序；
- lowered LLVM private object provenance；
- exact pre-greedy MIR accumulator class/def/use；
- post-RA physical AGPR destination；
- final ISA MFMA destination区域；
- spill/scratch/private memory 检查；
- Case A/B/C 单一结论；
- 唯一后续候选登记。

未完成且没有伪造的内容：

- 初始 high-level MLIR 逐行对比，因为现有 printer 崩溃；
- LLVM LiveIntervals 的精确 cycle-level 输出；
- Z5A 的实现、编译、性能和 public API 接入。

这份报告的正确表述是：**Z4C 的三 accumulator overlap 已有跨 source、
LLVM、MIR、ISA 的强证据，足以解释主要资源增长；但它不是对每个物理
register 的 cycle-level 生命周期证明。**
