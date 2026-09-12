# `block_dot_bf16_f32` V2 设计说明

## 目的

Stage 6Z 的 Z5B 已经把 Q 的 64x128 BF16 tile 放进 dedicated LDS cache，
但 K/H 的 producer、shared placement 和 MFMA-B consumer 仍然由 chunk-o
源码中的 phase staging 组织。Z7B 把两个调用换成同一个 block-dot 入口，
证明了通用 API 可以建立 source-level same-source A/B；不过两条 lowering
最后收敛到同一个 HSACO。

本轮 BDV2 的目标不是为 chunk-o 创建新的 intrinsic，而是把既有
`al.amdgpu.block_dot_bf16_f32` 演进成一个可以描述完整 operand scope 的
通用 block-dot contract：逻辑 global block、producer ownership、typed BF16
packet、shared placement 和 dot consumer 都由同一个 late lowering 边界管理。

## 对外语义

BDV2 复用现有 `AMDGPUBlockDotBF16F32Op`，因此旧的 direct-K64 caller 不变。
新增的两个 source helper 只是给同一个 op 附加通用 metadata：

```text
block_dot_bf16_f32_logical
block_dot_bf16_f32_logical_transposed
```

metadata 描述的是可复用的逻辑事实，而不是 Qwen 专用地址：

| metadata | 含义 |
|:--|:--|
| `operand_mode=full_scope` | producer 和 consumer 属于同一 lowering scope |
| `scope=logical_block` | 第三个 operand 是逻辑 global B block |
| `source_role=K/H` | B operand 的通用源角色 |
| `transpose=none/rhs_transposed` | B 的逻辑转置关系 |
| `logical_shape=32x32x32` | block-dot 的逻辑 M/N/K |
| `logical_source_operand=source_block` | source 不是已经手工展开的 fragment |
| `lhs_residency=existing_shared` | A operand 是调用者已拥有的 resident block |
| `rhs_producer=global_packet` | lowering 负责 packet producer |
| `reuse_key=lhs_ssa_block` | 相同 resident A 的复用身份 |
| `layout_intent=typed_shared_dot` | 允许目标 lowering 选择物理布局 |

这些字段只表达 block identity、transpose、residency 和 reuse intent；没有
把 4096 个地址、wave ID 或某个 Qwen kernel 名称编码进 API。

## 两条 lowering

### BDV2-G generic

generic arm 根据同一 packet ownership 展开为 BF16 标量 global load/store，再
调用现有 MFMA-B consumer。它是 target-independent fallback 和行为基线。

### BDV2-S gfx942 specialized

specialized arm 对同一 ownership 生成 `<8 x bfloat>` global load 和 packed
shared store，然后复用同一 MFMA-B consumer。它没有改变数学、MFMA geometry、
CTA mapping 或 accumulator phase 顺序。

两臂的 K/H 分支都经过 `emitFullScopeProducer` 和
`emitGenericOperandBPair`；K/H 没有独立的 Qwen intrinsic 或独立 planner。
H 的 rank-5 layout 使用 `rhs_transposed` metadata，K 的 rank-4 layout 使用
`none` metadata。

## Residency 与 ownership

高层 source 保持 Z5B 的完整 Q cache：

```text
global Q -> one dedicated [256, 32] BF16 shared cache
```

H/K logical block-dot 调用把 `q_cache` 作为 resident A。它们不再在 source
层手写 phase Q producer；BDV2 lowering 只生产 H/K B operand。K 的 logical op
由整个 CTA 调用，使 lowering 可以插入 CTA-uniform barrier；真正发出 K
producer 和 update MFMA 的 wave 是 `value_half == 0`，即 wave 0 和 wave 2，
与原 Z5B ownership 一致。这个细节很重要：早期实现误用了 `tid < 128`，只让
wave 0 发出 K work，T=64 correctness 立刻失败；修复后使用
`wave=tid//64`、`value_half=wave%2`，所有长度恢复 BF16 byte-exact。

## 编译器边界

`lower_qwen_block_dot_pass.cc` 在 full-scope metadata 上分叉。block-dot op
在该 pass 被展开，后续 LLVM/MIR 中不再保留高层 op 名称；因此“保留到 late
lowering”并不意味着 ISA 中会出现一个新的 block-dot instruction，而是指：

1. producer ownership 和 typed packet 的选择晚于 source/memref contract；
2. specialized 的 vector producer 能进入 lowered LLVM；
3. 两臂继续生成不同的 LTO/MIR/ISA/HSACO，而不是只在 Python 源码中不同。

如果未来要实现真正 native-style operand reuse，下一步应在这个通用 scope
表示上继续做 convergence bisect，而不是再添加一个 Qwen-specific intrinsic。

## 兼容性与测试策略

- 旧 `block_dot_bf16_f32` 注册和 direct-K64 tests 保留。
- logical helper 仍构造同一个 `AMDGPUBlockDotBF16F32Op`。
- generic arm 保留 fallback，specialized arm 只在
  `AVELANG_BLOCK_DOT_LOWERING=specialized` 时启用。
- BDV2 correctness driver 使用 caller-owned BF16 output、zero-V 和 NaN
  prefill，不能通过未写 output 或放宽阈值伪装成功。
- 机器证据必须区分 initial MLIR、lowered LLVM、LTO MIR、ISA、HSACO 和
  profiler counters。当前 binding 的 initial `get_mlir()` 会 SIGSEGV，因而
  任何报告都必须写 N/A，不能伪造 pre-branch MLIR hash。

## BDV2-P1：通用 affine layout planner 与表示收敛审计

### 目标

BDV2 full-scope 在 T=2048 保持 `MFMA=160/CTA` 的同时，把 VMEM/LDS 从 Z5B 的
`672/672` 降到 `448/464`，但 VALU 从 `7072` 增加到 `8410/CTA`。P1 只选择一个
通用控制杆：让 producer 和 consumer 共享同一份 affine ownership/layout plan，
并让 MFMA consumer 直接请求 `<4 x bf16>` typed fragment。

P1 不增加 Qwen-specific op，不修改 block-dot logical contract、K/H ownership、
BT64/BV64/BK32、WG256、MFMA32 或 accumulator phase。

### Planner 表示

`lower_qwen_block_dot_pass.cc` 新增内部 `LogicalBlockLayoutPlan`，保存：

```text
tid, kStage, wave, lane, laneCol, laneGroup,
rowHalf, valueHalf, producerLinear,
packetRow, packet, packetCol, feature
```

`lowerFullScopeOperandMode` 只创建一次该 plan，然后将同一个 plan 传给
`emitFullScopeProducer` 和 `emitGenericOperandBPair`。K/H 使用完全相同的 planner；
`source_role` 和 transpose metadata 只改变 logical block 的解释，不改变 planner
的 ownership 实现。

环境选择为：

```text
AVELANG_BLOCK_DOT_LAYOUT_PLANNER=legacy
AVELANG_BLOCK_DOT_LAYOUT_PLANNER=bdv2_p1_affine
```

### Typed fragment 分支

legacy consumer 是：

```text
LDS <8xbf16> -> extract 4 values -> insert <4xbf16> -> MFMA
```

P1 尝试直接使用：

```text
LDS <4xbf16> -> MFMA
```

generic lowering 保留 scalar fallback；gfx942 specialized lowering 使用 typed
vector local load。该分支只改变 fragment representation，不改变 fragment 数学值
或 MFMA 次序。

### 分层证据

P1 specialized 相对 BDV2 specialized 的 lowered LLVM 文本统计为：

```text
extractelement: 130 -> 66
insertelement:  160 -> 96
```

但 P1 pre-LTO 仍出现额外 affine arithmetic，且最终 AMDGPU/LTO 将两条路径收敛到
相同 HSACO：

```text
HSACO SHA256: d17483b933a7abf68b34b0e193aa4e3e5d9f93f48a12cfa8d84b285c9091af5a
```

normalized final ISA body 也相同。两臂 code object 均为 `VGPR=132, AGPR=48,
SGPR=30, LDS=32768 B, private=0, spill=0`，T=2048 dynamic PMC 均为
`MFMA/VMEM/LDS/VALU/SALU=160/448/464/8410/780`。

因此 P1 当前是一个**表示层和回归基础设施结果**，不是性能优化结果。它证明
`<4xbf16>` intent 可以进入 lowered LLVM，但还没有证明该 intent 被保留到
AMDGPU MIR/ISA。后续若继续，应在 LLVM 到 AMDGPU/MIR 的 representation-preserving
边界做通用 convergence experiment，不能添加 K/H/Qwen 特例。

### 回归要求

P1 由现有的无设备 contract test 和 direct-K64 block-dot test 覆盖。full-scope
correctness 覆盖 `T=64/512/1024/2048/4096/8192/16384`、finite、caller-owned
output、zero-V-new 和 NaN-prefill。initial MLIR 仍因当前 runtime binding
`get_mlir()` SIGSEGV 而显式标记 unavailable，不能把 source hash 当成 MLIR hash。

## BDV2-P2：First-Class MFMA Operand Preservation

P2 是在同一 `block_dot_bf16_f32` source 上增加的 compiler-only 表示层，不是
新的 Qwen/chunk-o public op。它针对 P1 的证据缺口：P1 可以在 lowered LLVM
中写出较紧凑的 `<4xbf16>` fragment，但该意图在 AMDGPU/LTO 前没有稳定的
consumer identity。

### 内部表示

P2 增加内部 `amdgpu_block_dot_mfma_operand` op。它保存：

```text
operand role A/B
source role K/H
logical shape 32x32x32
transpose
gfx942 shared b32 MFMA32 physical encoding
lane_group_word_pair_k32_order mapping
target MFMA f32_32x32x8_bf16
```

这个 op 只在同一个 full-scope lowering 中创建；K/H 仍共享同一 planner，未来
Q/V-new 可以复用同一 contract。它在 GPU outlining 后、普通 GPU module
canonicalization 前由 `createLowerQwenBlockDotMfmaOperandPass()` 消费。这样
`post_gpu_outlining.mlir` 仍能审计 operand identity，消费后则必须不再残留内部
op。

### 当前 gfx942 materialization

P2 late lowering 将静态 LDS BF16 stage 的 consumer load 保持为显式 packed
64-bit 读：

```text
addrspace(3) LDS pointer
    -> volatile llvm.load i64, align 8
    -> llvm.bitcast i64 to vector<4xbf16>
    -> existing MFMA32 intrinsic
```

这是一次有意的 machine-graph repair：第一版使用普通 `load <4xbf16>`，最终
与 P1 收敛到相同 HSACO；当前版本保留 `i64` packed load 和
`avelang.block_dot.first_class_lds_b64_i64` 标记，因此最终 HSACO/ISA 已经不同。
它不是性能承诺，`volatile` 只用于保证这条实验性 load 表示不会在实验过程中
被合并掉。

选择开关：

```text
AVELANG_BLOCK_DOT_OPERAND_PRESERVATION=none
AVELANG_BLOCK_DOT_OPERAND_PRESERVATION=p2_first_class
```

`none` 保持 BDV2/P1 路径；`p2_first_class` 只要求 specialized lowering，generic
path 不接受该内部 operand plan。

### P2 结果

P2 保持正确性和 MFMA 数学工作，但当前 packed LDS materialization 增加了 LDS
读取：P1 的静态 `ds_read_b128=56` 变为 P2 的 `ds_read_b64=96` 加
`ds_read_b128=8`，T=2048 动态 LDS 从 `464` 变为 `592/CTA`。VMEM 和 MFMA
仍为 `448/CTA`、`160/CTA`；因此 P2 是一个可复用的 operand-preserving
compiler infrastructure，但不是当前性能 winner。详细证据与 fresh-process
结果见 `qwen_gfx942_stage6z_block_dot_v2_p2_first_class_mfma_operand.md`。

## BDV2-P3：Packed Operand Reuse / Wide LDS Consumer

### 目标与边界

P3 是 P2 的单一 compiler-only 后续，不是新的 Qwen/chunk-o public op。它保持
同一份 `block_dot_bf16_f32` source、同一份 K/H full-scope planner、同一套
BT64/BV64/BK32/WG256/MFMA32/accumulator phase 和 BF16 ABI，只改变 operand
consumer 的 late materialization。

P2 的第一版按 fragment 生成 B64/i64 LDS read，导致两个相邻的 `<4xbf16>`
operand 各自读一次。P3 把这两个相邻 fragment 分组为一条 aligned 128-bit
LDS read：

```text
vector<8xbf16> LDS load
    -> low/high vector<4xbf16> slices
    -> two existing MFMA32 consumers
```

这不是 MFMA64，也不是两个 MFMA 的数学融合；每个 MFMA 仍保持原来的 operand
和 K32 accumulation order。

### 通用实现

选择器为：

```text
AVELANG_BLOCK_DOT_OPERAND_PRESERVATION=p3_packed_reuse
```

`lower_qwen_block_dot_pass.cc` 中的 P3 helper 是通用的
`emitPackedLdsLoad128`、`splitPackedFragmentPair` 和
`emitPackedConsumerGroup`。它们只依赖 operand role、source role、logical
shape、physical encoding 和 consumer pair metadata；K/H 仍走同一 planner，
没有 `kSpecialCase`、`hSpecialCase` 或 Qwen-specific address table。

P3 的 MLIR 证据为：

```mlir
%packed = llvm.load %lds_ptr {alignment = 16,
    avelang.block_dot.first_class_lds_b128_group}
    : !llvm.ptr<3> -> vector<8xbf16>
%lo = vector.extract_strided_slice %packed {offsets = [0], sizes = [4]}
%hi = vector.extract_strided_slice %packed {offsets = [4], sizes = [4]}
%acc0 = mfma32(..., %lo, ...)
%acc1 = mfma32(..., %hi, %acc0)
```

### 结果与设计含义

P3 的 final HSACO 为：

```text
d17483b933a7abf68b34b0e193aa4e3e5d9f93f48a12cfa8d84b285c9091af5a
```

它不同于 P2 的 B64 版本，但与 P1/BDV2-S 收敛。T=2048 dynamic LDS 从 P2
的 `592/CTA` 回到 `464/CTA`，static `ds_read` 从 `104` 回到 `56`；MFMA
仍为 `160/CTA`，VMEM/VALU/SALU 回到 P1 的 `448/8410/780`。因此 P3 证明
了“两个相邻 typed fragments 可以共享一个宽 LDS consumer load”，但没有
证明它能消除 BDV2/P1 共有的 `8410 VALU/CTA` layout/address 工作。

P3 在 T=2048 和 T=8192 都慢于 Z5B，故不晋级性能 baseline；它保留为通用
block-dot compiler infrastructure 和 regression evidence。后续性能研究应
转向 BDV2/P1 共有的 address/layout/fragment feeding provenance，不应继续
枚举 P2/P3 的 B64/B128 load-width 变体。

当前 runtime binding 的 initial `get_mlir()` 仍会 SIGSEGV，P3 dump 使用
`--skip-initial-mlir`，因此设计文档不声称 pre-branch MLIR hash 已验证。

## BDV2-P4：Accumulator Forwarding Representation

P4 是在 P3 之后登记的唯一通用 compiler-internal accumulator 控制杆。它不
新增 Qwen/chunk-o public op，也不改变 `block_dot_bf16_f32` 的 source contract。
P4 由 `AVELANG_BLOCK_DOT_OPERAND_PRESERVATION=p4_accumulator_reuse` 选择，
使用 `AccumulatorForwardingMap`：同一 accumulator 在 enclosing reset boundary
之外可以复用已经产生的 SSA value；遇到新的 vector accumulator reset 时停止
forward，避免错误跨越 score-half 边界。full-scope lowering 同时保留真实
MFMA 低 tile，不再生成 source 后续不会消费的 duplicated high tile。

K/H 仍共享 `LogicalBlockLayoutPlan` 和同一个 operand planner，P4 没有
`kSpecialCase`、`hSpecialCase`、kernel-name判断或 Qwen 地址表。相关内部标记
是：

```text
avelang.block_dot.p4_low_tile_ssa
avelang.block_dot.p4_ssa_forwarding
avelang.block_dot.p4_ssa_forwarding_reset_boundary
```

P4 的 source SHA 与 P3 相同，但 post-materialization MLIR、lowered LLVM、
pre-LTO AMDGCN、exact-LTO MIR、final ISA 和 HSACO 均不同。因此这是一次真实
的 representation-level compiler experiment，而不是只改报告或重新命名。

机器结果需要分开解释：P4 code-object `VGPR/AGPR=104/32`，比 P3 的 `132/48`
更小，scratch/spill 仍为零；但 final `v_accvgpr_read/write`、MFMA、global/LDS
主要族没有下降，`v_mov_b32` 的静态 lexical 减少被 e64 `v_cndmask` 增加抵消，
dynamic `VALU=8410/CTA` 不变。因而 P4 证明了 SSA/fragment 表示可以改变最终
资源分配，却没有证明这一个 forwarding 控制杆能减少 full-scope 的实际机器
工作。P4 的完整归因和性能 No-Go 记录在：

```text
qwen_gfx942_stage6z_block_dot_v2_p4_final_machine_valu_provenance.md
```

该文档以及同目录的 machine-readable JSON 必须与 P3/P2 一起作为 compiler
regression evidence 保存；不得把 P4 提升为 production selector。
