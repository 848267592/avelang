# Stage 6Z Z5B 剩余 VMEM Operand Provenance Ledger

## 0. 本轮范围与冻结结论

本轮不是重新审计 Z5A/Z5B 的资源差异，也不是重新验证 Z5B 是否应该晋级。
本轮只补完此前 fixed-Z2/Z5A provenance audit 中仍未能唯一归属的
`672 VMEM/CTA`：把 Z5B 的 Q-fill、K、H、V-new、g、output 六类 global
operand 与同形状 native WG256 chunk-o 对齐，区分逻辑字节、静态 ISA
指令、可建模的 issuing-load 数和 rocprof 动态 PMC。

冻结对象是：

```text
Z5B = qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer.py
BT64 / BV64 / BK32 / gfx942 / wave64 / WG256
2 CTA per chunk-head
BF16 ABI, MFMA32, 160 dynamic MFMA per CTA
```

本轮没有修改 kernel source、compiler、lowering、allocator/RA、selector、X2
或 production dispatch，也没有重新运行性能 benchmark。Z5B 的既有性能、
correctness 和资源结论全部冻结：T=2048 的动态 `VMEM=672/CTA`、
`LDS=672/CTA`、`VALU=7072/CTA`、`SALU=768/CTA`、`MFMA=160/CTA`，
code object `LDS=32768 B`、`VGPR=104`、`AGPR metadata=32`、scratch/spill=0。

## 1. 证据与证据等级

### 1.1 Z5B 工件

使用的 Z5B 文件为：

- source：`vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer.py`
- lowered LLVM：`codex_qwen_bt64_stage6z_z5b_machine_stage1/lowered_llvm.ll`
- exact-LTO MIR：`codex_qwen_bt64_stage6z_z5b_machine_stage1/exact_lto/`
- final ISA：`codex_qwen_bt64_stage6z_z5b_machine_stage1/final_isa.s`
- machine summary：`codex_qwen_bt64_stage6z_z5b_machine_stage1/machine_evidence.json`
- T=2048 PMC：`codex_qwen_bt64_stage6z_z5b_rocprof/stage6z_z5b_T2048_rocprof.json`

Z5B exact machine identity：

```text
HSACO SHA256: 979889c1a10ac53064cd1d2da91298b67e4ef7f2db3baadc171872cb56b0ed67
workgroup: 256
LDS: 32768 B
VGPR/AGPR metadata: 104/32
private segment: 0
AV32/AV64 spill save/reload: 0
```

### 1.2 同形状 native 工件

native 对照是 current-vLLM selected `chunk_fwd_kernel_o` 的 WG256、
BT64/BV64/BK32 capture，不是根据旧 T=2048/T=8192 数据推断的 selector。
使用的工件位于：

`codex_qwen_bt64_stage6z_native_chunko/native/T2048/trace_capture/selected/`

其中包括 captured source/TTIR-like text、TTGIR、LLVM、final ISA、HSACO
和 PMC。native 的 T=2048 归一化结果为：

| 指标 | Z5B | native WG256 | Z5B/native |
|:--|--:|--:|--:|
| dynamic MFMA/CTA | 160 | 160 | 1.00x |
| dynamic VMEM/CTA | 672 | 140 | 4.80x |
| dynamic LDS/CTA | 672 | 480 | 1.40x |
| dynamic VALU/CTA | 7,072 | 3,376 | 2.09x |
| dynamic SALU/CTA | 768 | 660 | 1.16x |
| profiler VGPR | 76 | 100 | 0.76x |
| profiler Accum_VGPR | 100 | 36 | 2.78x |
| SGPR | 112 | 96 | 1.17x |
| LDS bytes | 32,768 | 24,576 | 1.33x |
| scratch | 0 | 0 | - |

`LDS_Block_Size=0` 出现在 native 的一份 collector metadata 中，但 native
capture 的 shared allocation 为 `24576 B`，且动态 LDS PMC 为 `480/CTA`。
这里把 `24576 B` 作为 native shared footprint；`0` 视为 collector metadata
缺失，不能当作 native 没有 LDS。

### 1.3 证据等级定义

| 等级 | 含义 |
|:--|:--|
| A | source phase/loop、LLVM pointer/type、ISA family 和控制流可以闭合，或由 PMC 直接观测 |
| B | source/LLVM/ISA 能确定 family 和方向，但没有 debug provenance 把同一 mnemonic 精确分到 operand |
| C | 只能由静态相似性、地址形态或总数差值推断，不能用来声称精确动态归属 |

本报告不把 static lexical count 当作 dynamic transaction count，也不把
VMEM instruction count 当作访问字节数。

## 2. Machine-level 总体对账

### 2.1 Z5B exact static ISA count

对 `final_isa.s` 的 lexical mnemonic 统计为：

| family | count | 说明 |
|:--|--:|:--|
| `global_load_ushort` | 112 | BF16 scalar/窄 load family；Q/H/K/V 不能仅凭 mnemonic 完全拆分 |
| `global_load_dword` | 80 | FP32/scalar load family；主要与 g/索引路径相关，但不能仅凭 mnemonic 完全拆分 |
| `global_store_short_d16_hi` | 16 | BF16 output/store family |
| `ds_write_b16` | 112 | shared producer 的窄 BF16 store |
| `ds_write_b16_d16_hi` | 32 | packed BF16 word 的另一半 store |
| `ds_read_b128` | 56 | fragment/operand local read |
| `s_waitcnt` | 214 | 静态 wait 点，不能直接等于动态等待次数 |
| `s_barrier` | 32 | 静态 barrier，不能直接等于动态同步次数 |
| `v_mfma_f32_32x32x8_bf16` | 56 | lexical MFMA；PMC 确认为 160 dynamic MFMA/CTA |

最重要的限制是：Z5B 的 source debug provenance 没有把 112 条
`global_load_ushort` 逐条标记成 Q/H/K/V，也没有把 80 条
`global_load_dword` 逐条标记成 g/地址辅助。因此下面的 operand ledger 会
把 source/LLVM 可证明的 issuing model 与 ISA family 分开写，不能把
`112 + 80` 直接分配给六类 operand。

### 2.2 Z5B PMC

T=2048 capture 有 512 个 WG256 CTA：

```text
SQ_INSTS_MFMA = 81920       -> 160/CTA
VMEM           = 344064      -> 672/CTA
LDS            = 344064      -> 672/CTA
VALU           = 3620864     -> 7072/CTA
SALU           = 393216      -> 768/CTA
VGPR           = 76
Accum_VGPR     = 100
SGPR           = 112
LDS block      = 32768 B
Scratch        = 0 B
Occupancy      = 14.5401
```

这组 PMC 是动态 kernel instruction 类计数，不是字节计数。特别是
`VMEM=672` 不能解释成 `672 bytes`，也不能仅凭它反推出 672 条每类
operand 的 exact 分配。

## 3. Z5B source ownership 与 logical tile

Z5B 的 source 结构决定了每个 CTA 的逻辑 ownership。关键 source 区域如下：

| source 区域 | source 行 | 逻辑工作 |
|:--|:--:|:--|
| dedicated Q cache allocation | 93-98 | 32 KiB shared；Q cache 位于低地址，旧 phase 区域在其上方 |
| Q fill | 100-111 | `k_stage=0..3`、`rep=0..7`，一次把当前 chunk 的 scaled BF16 Q 写入 dedicated cache |
| Phase A H/inter | 119-138 | H producer 后直接从 Q cache 形成 Q*H 的 inter accumulator |
| Phase B score half 0/1 | 140-165 | 两个 source half；每个 half 4 个 K32 stage、4 个 rep，Q 直接从 Q cache 取 |
| score/g | 174-178 | target/source g 参与 score/gating 与 score store |
| V-new | 181-190 | 从 BF16 V-new producer 写入当前 phase band |
| intra update | 193-201 | 使用 V-new 与 score 进行 MFMA32 update |
| final g/output | 203-209 | final scaling 和 BF16 output store |

在 source/LLVM 上可以确认：Q cache 使 kernel Q pointer `%0` 的 global Q
producer 只有一个 source region。Phase A/B 的后续 Q 消费不是再次从 `%0`
读取，而是从 addrspace(3) 的 dedicated cache 取。此前 Z5A→Z5B 的唯一结构变化
是删掉 Q cache 到旧 phase Q rows 的 republish；因此 Z5B 的 global producer
graph 与 Z5A 相同，dynamic VMEM 也相同，Q duplicate/global reload 不是当前
剩余 VMEM 的新问题。

## 4. Z5B per-operand provenance ledger

### 4.1 总表

下面的 `modeled issuing` 是由 source loop、wave participation 和 LLVM load
type 建立的 issuing opportunity 模型，不是硬件 transaction byte 数。`PMC
share` 只有在能够唯一闭合时才给 exact；否则写成 unknown 或区间。

| operand | logical bytes/CTA | source producer pass | LLVM load type | ISA family | modeled issuing contribution | PMC 可否唯一分摊 | native 对应路径 | VALU 伴随成本 | 证据 |
|:--|--:|:--|:--|:--|--:|:--|:--|:--|:--:|
| Q-fill | 16,384 B | 1 个 Q-cache fill；4 stage × 8 rep | `load bfloat` from `%0` | `global_load_ushort` family | 约 128 scalar issuing opportunities 上界/模型 | unknown；与 H/K/V 共用 family | Q packet 4×`buffer_load_dwordx4`，cache/local-load 后复用 | 地址、BF16 scale/round、cache indexing | A/B |
| K | 16,384 B | 2 source-half × 4 K32 stage × 4 rep | `load bfloat` from `%1` | `global_load_ushort` family | 约 128；按 source-half 分工，不等同 duplicate logical tile | unknown；同 family | K packet 4×`buffer_load_dwordx4`，typed block/layout 供多个 dot consumer | K index、phase address、fragment reconstruction | A/B |
| H | 16,384 B | Phase A；4 stage × 8 rep | `load bfloat` from `%3` | `global_load_ushort` family | 约 128 | unknown；不能从 112 条静态 ushort 唯一拆出 | H packet 4×`buffer_load_dwordx4`，local load 后供 dot | H address、load/fragment layout | A/B |
| V-new | 8,192 B | `rep=0..15` 的 BF16 producer | `load bfloat` from `%2` | `global_load_ushort` family | 约 64 | unknown；最终 intra ownership/mask 可能少于上界 | native 2×`buffer_load_dwordx4`，typed local block | V address、BF16/fragment movement | B |
| g | 256 B（64×FP32 的逻辑 chunk/head tile） | score target/source 两角色 + final scaling | `load float` from `%4` | `global_load_dword` family | 约 192：score target/source 约128 + final约64 | unknown；80 条 static dword 与 loop mask 不一一对应 | native block `tt.load`/global dword path，g tile 多 consumer 复用 | g address、target/source index、exp/sub/scale | A/B |
| output | 8,192 B（BF16） | final `rep=0..15` store | `store bfloat` to `%5` | `global_store_short_d16_hi` family | 约 32-64 issuing opportunities；mask/ownership使 exact share unresolved | unknown；不能把总 PMC 剩余项硬扣给 output | native 4×`buffer_store_dwordx2` packet | output address、BF16 conversion/store | B |

模型中 Q/H/K/V-new/g 的合计约为 `128+128+128+64+192=640` 个 issuing
opportunity；output 的 source/mask 上界约 `32..64`。这与总 PMC `672` 相容，
但不产生唯一的 operand-to-PMC partition：某些 scalar load 的 inactive lanes、
循环展开、以及同一 instruction family 的不同 source region 都会改变硬件计数。
因此本报告把 `672` 作为总量实测，把各 operand 数字作为模型/上下界，而不把
`672` 伪造为逐 operand exact 数。

### 4.2 Q-fill：重复已消除，但一次 fill 仍偏窄

Z5B 的 Q pointer `%0` 只在 lowered LLVM 的 Q producer 区域出现：

```text
lowered_llvm.ll:168-169
%136 = getelementptr ... bfloat, ptr %0
%137 = load bfloat
```

Phase A/B 的 Q fragment 在 LLVM 约 293-304 及其后续 extraction 区域从 shared
address space 的 `<4 x i32>` load 形成。`%0` 后面没有新的 Q global load region。
这给出等级 A 的结论：Z5B 已消除 Z2/Z4 中的三次 Q producer/重复 global Q
load问题。

不过 Q-fill 自身仍由 source 的 `k_stage×rep` ownership 逐个产生 BF16 元素，
因此它仍贡献约 128 个 scalar issuing opportunities 的模型上界，并伴随
64-bit 地址形成、BF16 scale/round 和 shared cache indexing。它不是 Q duplicate，
而是“每个逻辑 Q tile 只生产一次，但 producer packet 仍窄”的问题。native
则以 `tensor<64x32xbf16>` block load 和 4 条 `buffer_load_dwordx4` lexical
packet 形成 Q local block，再让同一 Q block 同时服务 `Q*H` 与 `Q*K` dot。

### 4.3 K：source-half 分工不等于重复 producer

Z5B K producer 在 source 行 140-165，按两个 source half、四个 K32 stage 和
四个 rep 组织。LLVM 的 K global load 位于：

```text
lowered_llvm.ll:448-449
%362 = getelementptr ... bfloat, ptr %1
%363 = load bfloat
```

仅凭 final ISA 中 112 条 `global_load_ushort` 不能把其中多少条精确标成 K；
但 source ownership 能证明每个 source half 是对应 MFMA update 的必要输入分区，
不是同一个 K logical tile 被两个 consumer 无条件重读。两个 source half 加上
四个 K32 stage 的循环拆分说明 K 仍可能有较窄的 packet/materialization 成本，
却不足以在本轮把 K 宣称为最大的重复 producer offender。

native 的 TTGIR 以 `tensor<32x64xbf16>` K block，并在 dot consumer 前使用
typed local load；LLVM 有 4 条 K 侧 `raw.ptr.buffer.load.v4i32` lexical
packet。native 的关键差异是 block ownership 和 consumer layout 同时被表达，
而不是单独把一条 raw load 改成 x4。

### 4.4 H：一个逻辑 tile，两个 value-half 不足以证明重复 global H

Z5B H load 在 source 行 119-138，LLVM 为：

```text
lowered_llvm.ll:251-252
%198 = getelementptr ... bfloat, ptr %3
%199 = load bfloat
```

source 中 H producer 是 Phase A 的 inter path。它大约有 `4×8×4=128`
个 scalar issuing opportunities 的模型上界，但这并不等于两个 value-half
把同一 H tile 重复生产了两次。当前 source/LLVM 没有 debug-level per-lane
mapping 可以把这一上界进一步折成 exact PMC share。native TTGIR 的 H 是
`tensor<64x32xbf16>` packet/local allocation，随后直接参与 typed dot operand
路径。

结论：H 有明显 packet/layout 差距，但没有证据把它排在 g 的多角色重复读取之上。

### 4.5 V-new：source producer 与 intra consumer 的边界

V-new load 位于 LLVM：

```text
lowered_llvm.ll:721-722
%585 = getelementptr ... bfloat, ptr %2
%586 = load bfloat
```

其 source producer 为行 181-190 的 `rep=0..15` 路径，模型约 64 个 scalar
issuing opportunities。V-new 的 global tensor 逻辑大小是 8,192 B/CTA，
但当前 Z5B source 仍把 producer/phase materialization 与 intra consumer
绑定；从现有 LLVM/ISA 不能证明所有 V-new load 都是重复 global producer，
也不能精确从总 PMC 中扣除 V-new share。

因此 V-new 是需要关注的 operand-layout 成本，但本轮没有足够证据把它选为
唯一最大 offender。native 使用 2 条 `buffer_load_dwordx4` lexical V block
packet，并用 local/typed block 消费。

### 4.6 g：唯一具有多角色重复读取证据的剩余候选

Z5B score 阶段在 source 行 174-178 同时读取 target g 和 source g；final
阶段在行 203-209 再读取 target/final g。LLVM 对应位置是：

```text
lowered_llvm.ll:639-648  score target/source g
lowered_llvm.ll:905-906  final g
```

因此 g 不是一个只被单个 dot consumer 使用的输入：同一个 current chunk 的
g logical tile 被 score/gating 和 final scaling 两个 phase 角色使用。按照 source
loop/ownership，score target/source 约 128 个 FP32 scalar issuing opportunities，
final scaling 约 64 个，合计约 192 的模型上界。这是六类 operand 中最高的可建模
issuing 数量，而且每次都是 FP32 `global_load_dword` + 地址/索引路径。

native TTGIR 在约 320、323 行把 g 作为 block `tt.load`，随后让 g tile 参与
多个 scaling/gating consumer；native final ISA 的 `global_load_dword` lexical
count 为 17，不能直接等同 dynamic count，但与 block residency/typed packet
结构一致。native 的优势不是 g 数学被删除，而是 g 的逻辑 packet 与 consumer
生命周期被统一规划。

等级 A 的结论是：Z5B 的 g 存在两个以上 logical consumer 角色，且 source/LLVM
能证明重复 load opportunity；等级 B 的限制是不能把 80 条 static dword load
精确等同于 192 个动态 g load。这个证据组合足以把 g 选为下一轮唯一候选，但不
足以声称 g 占 PMC 672 的 exact 百分比。

### 4.7 output：窄 BF16 store 可见，但不是最大 load offender

Z5B output store 在 LLVM 933-934：

```text
%772 = getelementptr ... bfloat, ptr %5
store bfloat
```

final ISA 有 16 条 `global_store_short_d16_hi` lexical store family；source
ownership 给出约 `32..64` 个 issuing opportunities 的范围。output 是必需的
BF16 public ABI 写回，且不能简单与 native 的 packet store 做相同数学语义替换。
它可能贡献窄 store/address VALU，但从已知 672 总 PMC 和 source role 看，不是
当前最大剩余 VMEM offender。native 使用 4 条 `buffer_store_dwordx2` lexical
packet，把 public BF16 output 作为 block store。

## 5. Native 对应 ledger：为什么是 140/CTA

native TTGIR/LLVM/ISA 的可验证结构如下：

| native logical operand | TTGIR/LLVM 表达 | static ISA family | ownership/reuse 证据 |
|:--|:--|:--|:--|
| Q | `amdg.buffer_load tensor<64x32xbf16>`；4 条 `raw.ptr.buffer.load.v4i32` lexical | `buffer_load_dwordx4` | Q local block 同时给 Q*H 与 Q*K dot |
| K | `amdg.buffer_load tensor<32x64xbf16>`；4 条 `v4i32` lexical | `buffer_load_dwordx4` | K block 按 dot operand layout 给 update consumer |
| H | `tensor<64x32xbf16>` block load；4 条 `v4i32` lexical | `buffer_load_dwordx4` | H local block 进入 Q*H consumer |
| V-new | `amdg.buffer_load tensor<64x64xbf16>`；2 条 `v4i32` lexical | `buffer_load_dwordx4` | V block 作为 update operand，不回到窄 scalar path |
| g | block `tt.load`（TTGIR 约320、323） | `global_load_dword` family | g tile 被多个 scaling/gating consumer 复用 |
| output | 4 条 `raw.ptr.buffer.store.v2i32` | `buffer_store_dwordx2` | public BF16 output block store |

native 仍有 local allocation 和 LDS 指令：静态可见
`ds_read2_b64=8`、`ds_read2st64_b64=8`、`ds_read_b64=24`、`ds_read_u16=32`、
`ds_write2st64_b64=4`、`ds_write_b128=4`、`ds_write_b16=16`、
`ds_write_b32=8`、`ds_write_b64=8`，动态 LDS 是 `480/CTA`。所以 native
并非“完全不经过 LDS”，而是 packet/block ownership、shared layout 和 dot
operand encoding 一起设计，避免 Z5B 把每个 BF16 logical element 在多个 phase
中重新 materialize。

native 的 `buffer_load_dwordx4=14`、`buffer_store_dwordx2=4` 是 static lexical
模板计数，不是 140/CTA 或 output bytes 的直接代数等式。140/CTA 来自 PMC；
TTGIR/LLVM 只能证明 packet 宽度、reuse 和 loop ownership，不能替代硬件计数。

## 6. VALU 7072 对 3376 的 provenance 分类

Z5B 的 dynamic VALU 是 `7072/CTA`，native 是 `3376/CTA`。以下分类只用于
归因方向，不把 static opcode count 冒充 dynamic category count。

| 类别 | Z5B machine evidence | 绑定的 operand/phase | native 对照 | 判断 |
|:--|:--|:--|:--|:--|
| address/index arithmetic | final ISA static proxy `v_add=202`、`v_lshl_add=130` | Q fill、K stage/source-half、H/V phase、g target/source/final、output | packet/block pointer arithmetic | Z5B 维护更多 phase/rep/address tuple |
| BF16 fragment reconstruction | LLVM shared `<4xi32>` load、bitcast、extract/insert chains，约 304-372、507-577、795-844 | Phase A、score half0/1、intra MFMA | TTGIR typed local-load/dot operand | 这是与 operand feeding 直接绑定的主要 VALU 类别 |
| Q scale/round | Q `%0` load 后的 FP32 scale、BF16 round/conversion | Q-fill | native block load 后按其 dot mapping处理 | 必需数学存在，但 Z5B scalar ownership 放大辅助工作 |
| g score/gating | target/source load、subtract/exp/scale相关指令 | score 与 final scaling | native g block 多 consumer | load/address可减少，exp/sub数学不能删除 |
| LDS/materialization | `ds_write_b16`/`d16_hi` 与 `ds_read_b128` | Q/H/K/V phase materialization | native packed LDS families | 动态 LDS 仅 1.4x，不是当前最大总差距解释 |
| output conversion/store | BF16 scalar store 与地址形成 | final output | native block store | 有可见窄 store成本，但规模不如 g 多角色/全体 operand path |

`v_add` 和 `v_lshl_add` 的 static count 说明存在大量地址/索引辅助，但不能
说它们合计等于 `7072-3376=3696`。更可信的结论是：Z5B 的 scalar producer、
phase-specific address tuple 和 fragment reconstruction 同时推高 VMEM 与 VALU；
native 的 typed block path 把同一逻辑 operand 的 producer/consumer layout 一起
编码，因而这些辅助工作更少。

## 7. A-F 问题的明确回答

### A. Q duplicate 是否已经消除？

是。Z5B 的 Q pointer global producer 只有一个 source/LLVM region，Phase A/B
从 dedicated Q cache 读。剩余问题是一次 Q-fill 的 scalar/narrow ownership，
不是三次 Q global reload。证据等级 A（一次 producer）、B（ISA family 的精确
operand拆分）。

### B. K 是否在 source-half/K32 stage/wave 间重复 materialize？

K 仍有窄 load、phase 和 fragment feeding 成本，但当前工件不能证明存在同一
logical K tile 被跨 consumer 无条件重复 producer。两个 source half 是 update
数学与 ownership 的必要分区。证据不足以把 K 排为唯一最大 offender。

### C. H 是否被两个 value-half/wave 重复加载？

source/LLVM 只能确认 Phase A 的 H producer 和约128个 issuing opportunity
模型上界，不能确认同一 H tile 被两个 value-half 完整重复 global load。native
有一个 H block packet path。结论是 layout/packet差距明确，duplicate证据不足。

### D. V-new 是否重复读取？

Z5B 的 V-new producer 与 intra consumer 位于不同 source phase band，存在窄
BF16 load/fragment materialization成本；现有 LLVM/ISA 没有足够 provenance 证明
全部 V-new PMC 都是重复 global read。不能把它硬列为最大 offender。

### E. g 是否有多角色重复读取？

是。score target/source 与 final scaling 都读取 g；source/LLVM 可见两个阶段，
模型约192个 FP32 scalar issuing opportunities，是当前六类中最大的可建模
角色总量。由于 static dword family 与 loop mask不能一一映射，exact PMC share
仍 unknown，但这是最强的 remaining offender 证据。

### F. output store 是否值得首先优化？

output 是窄 BF16 store，存在 packet-width 差距，但它是必须的 public ABI store，
模型约32-64 issuing opportunities，低于 g 的约192，也低于多类 BF16 producer
合计成本。它不是下一轮唯一首选。

## 8. 排序与唯一下一候选

### 8.1 剩余 offender 排序

| 排名 | operand/path | 证据支持的原因 | 是否已证明是最大 |
|--:|:--|:--|:--|
| 1 | g target/source/final residency | 约192 FP32 scalar issuing model；至少两个 consumer role；与 global dword/address VALU绑定；native 为 block g tile | 是，按当前可建模证据 |
| 2 | K typed producer/consumer | 约128 BF16 issuing model；K layout/fragment reconstruction明显比 native重；重复producer尚未证明 | 否 |
| 3 | Q-fill packet width | 约128，但Q duplicate已由Z5B消除；主要是一次fill偏窄 | 否 |
| 4 | H producer/packet | 约128上界；是否跨 value-half重复未闭合 | 否 |
| 5 | V-new | 约64上界；consumer ownership/mask未闭合 | 否 |
| 6 | output | 约32-64上界；public BF16 store必需 | 否 |

### 8.2 唯一登记的下一候选：Z6G typed FP32 g-tile residency

只登记设计，不在本轮实现：

```text
Z5B current chunk/head g logical tile
    -> one typed FP32 CTA-local packet/cache
    -> score target/source consumers
    -> final scaling consumer
```

候选名称：`Z6G typed FP32 g-tile residency`。

它的约束必须是：

1. 从 Z5B 分叉，只改 g 的 producer/consumer lifetime；
2. 对当前 chunk/head 先建立 typed FP32 g tile（逻辑上为 64-token g
   ownership；精确 packet shape 由下一轮 source/plan 验证）；
3. score target/source 和 final scaling 都只从该 cache/packet 读取；
4. Q-fill、K、H、V-new、output、BT64/BV64/BK32、WG256、MFMA32、数学、BF16
   ABI、caller-owned output 完全不变；
5. 不修改 allocator/RA、X2、selector、production，也不把一条 raw
   `buffer_load` 单独改成 x4 便宣称完成；
6. 必须做 same-source/producer-consumer evidence，证明减少的是 g 的重复
   materialization，而不是把 g 变成大型 private array；
7. 先做 T=64/512/2048 correctness/resource gate，再决定是否跑性能。

预期验证信号不是“VMEM 一定下降到某个数字”，而是：g 的 source producer
pass/LLVM global dword region减少，Q/K/H/V/output不变，g 相关 address/VALU
同步减少，且没有新的 LDS/occupancy/register cliff。

## 9. 最终结论

1. Z5B 和 Z5A 的 dynamic VMEM 都是 `672/CTA`；Z5B 的收益来自删除 Q-cache
   到旧 phase 区域的 Q republish，使 dynamic LDS 从 `1440` 降到 `672`，不是
   VMEM 下降。
2. Z5B 已经消除 Q duplicate/global reload，因此不能再把 Q duplicate 作为
   当前最大问题。
3. K/H/Q-fill/V-new 各自有约128/128/128/64的 modeled issuing upper bound，
   但现有工件不能证明它们存在比 g 更大的重复 producer。
4. g 在 score target/source 与 final scaling 中具有多角色重复读取，约192个
   FP32 issuing opportunity，是当前唯一同时满足“可建模 VMEM大、比 native更窄、
   伴随地址/VALU成本”的 remaining offender。
5. 因此唯一下一候选是 `Z6G typed FP32 g-tile residency`；本轮不实现，不接
   X2，不改变 Z5B baseline。
