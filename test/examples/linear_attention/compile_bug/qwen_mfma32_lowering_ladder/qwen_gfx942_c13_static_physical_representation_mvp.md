# Qwen gfx942 C13-SPR：Static Physical Representation & Plan MVP

## 1. 结论先行

本轮任务是 **C13-SPR Static Physical Representation & Plan MVP**。它只补齐
表示层、静态 mapping algebra、MLIR pass-through 和单元测试，不实现任何新的
chunk-o kernel，也不运行 benchmark、PMC、LLVM/AMDGPU codegen 或 public Eager。

最终状态：

```text
C13_SPR_MVP_GO_FOR_CODEGEN
```

这个 GO 的含义是：C12 需要的最小 physical representation 已经可以被
AveLang 的 typed MLIR attr + compiler-owned C++ plan 表示，能够经过当前
canonicalizer 和 `lower_qwen_block_dot` pass 边界，并且 mapping verifier、负例、
MLIR 文本 parse/print 全部通过。

它**不**表示 C13 codegen 已经完成，更不表示已经得到新的 chunk-o HSACO 或性能
收益。下一阶段如果开始 codegen，仍必须单独实现 shared packet、dot operand、
in-thread transform 和 MFMA lowering，并重新做 correctness/resource gate。

本轮最重要的认识修正是：

> “AveLang 当前无法表达 static full-region encoding”过于绝对。
> 准确说法是：AveLang 早已能表达普通静态 shape/stride，也能让 block-dot 携带
> 字符串 metadata；缺失的是一套 typed、可跨 pass 保存的 physical encoding，以及
> 能同时描述 Q/H/K/V、dual consumer 和 phase lifetime 的 compiler-owned full plan。

## 2. 严格边界

本轮明确没有做以下事情：

- 没有新建 Qwen/chunk-o 性能 candidate。
- 没有修改 Qwen Python schedule 或任何 production selector。
- 没有运行 GPU kernel、benchmark、PMC、rocprof 或 Eager public API。
- 没有实现 C13 到 LLVM、ROCDL、AMDGPU packet、MFMA 或 ISA 的 lowering。
- 没有修改 recurrence、RA、allocator、shared swizzle、packet width 或 scheduler。
- 没有创建 Qwen-specific public layout op。
- 没有复制整个 TritonGPU dialect，也没有 generic runtime convert-layout fallback。

## 3. 复用审计

### 3.1 `al.make_layout` 已经能做什么

`lib/IR/layout_operation.cc` 中的 `al.make_layout` 接收 shape 和 stride，形成
AveLang 普通 memory layout。`lower_ave_lang_to_memref_pass.cc` 会把它转换为
普通 `StridedLayoutAttr`。因此它适合回答：

```text
logical tensor shape + ordinary memory stride
```

它不应该同时承担以下语义：

```text
register/lane/wave ownership
shared swizzle/rotation
MFMA dot operand slot
fixed in-thread register transform
full-region buffer lifetime
```

本轮没有扩展 `make_layout`，避免把 memory layout 和 GPU physical encoding 混成
一个接口。

### 3.2 现有 block-dot IR

`AMDGPUBlockDotBF16F32Op` 已经是合适的 source-level block-dot 边界，并且能携带
`operand_role`、`source_role`、`transpose`、`physical_encoding` 等 metadata。

`AMDGPUBlockDotMfmaOperandOp` 已经是内部 operand boundary，适合在后续 lowering
阶段承载 planned MFMA operand。它目前还不是 Triton `#ttg.dot_op` 的 typed
等价物，本轮没有把它直接改成 codegen op。

### 3.3 现有 Qwen planner 模式

`QwenRecurrenceSchedulePlan` 已证明仓库里存在成熟模式：由 compiler-owned
静态对象保存 target、shape、ownership、phase 和 schedule，再由 pass 消费，而
不是把所有规划信息降成 runtime SSA。

C13 复用了这个架构思想，但没有把 Qwen recurrence plan 直接硬编码成 chunk-o
layout。新增的是通用的 `ChunkOPhysicalPlan`，其内容是 Q/H/K/V physical block
关系、consumer graph、shared region 和 lifetime。

### 3.4 旧 full-scope lowering 的缺口

`lower_qwen_block_dot_pass.cc` 中的 `LogicalBlockLayoutPlan` 当前保存：

```text
tid, kStage, wave, lane, laneCol, laneGroup,
rowHalf, valueHalf, producerLinear,
packetRow, packet, packetCol, feature
```

但是它通过 `DivUI`、`RemUI`、加法和乘法在 lowering 时创建 runtime index SSA。
这对一般动态操作可以工作，但对固定 gfx942/BT64/BV64/BK32/WG256 的 physical
facts 来说，表示层已经知道的东西被过早物化成了 runtime arithmetic。

同时，当前 full-scope block-dot path 主要以 K/H source role 为中心，不能在一个
统一对象中保存：

- 同一个 Q physical source 同时服务 Q@H 和 Q@K；
- Q/H/K source phase 结束后释放并复用 shared capacity；
- V 的 blocked 到 linear1/in-thread transform；
- score/V phase 的 shared relationship。

## 4. C13 hybrid architecture

C13 采用两层表示，职责明确分开。

### 4.1 Typed MLIR attributes

新增文件：

- `lib/Dialect/AveLang/IR/AveLangAttrs.td`
- `lib/Dialect/AveLang/IR/AveLangAttrs.h`
- `lib/Dialect/AveLang/IR/AveLangAttrs.cc`

新增五个最小 typed attrs：

| attr | 表达内容 |
|:--|:--|
| `DistributedEncodingAttr` | logical shape、sizePerThread、threadsPerWave、wavesPerCTA、order |
| `SharedEncodingAttr` | kind、vec、perPhase、maxPhase、order、rotating |
| `MfmaEncodingAttr` | target、version、instruction shape、warpsPerCTA、transposed |
| `DotOperandEncodingAttr` | opIdx、kWidth、parent MFMA encoding |
| `StaticTransformAttr` | identity/fixed transpose/fixed permutation、source/target encoding |

数组字段使用 `DenseI64ArrayAttr`，所以 attr 能正常生成稳定的 MLIR assembly
格式并进行文本 parse/print。它们是 typed semantics，不再依赖长期维护的
`StringAttr physical_encoding` 字符串作为唯一事实来源。

### 4.2 Compiler-owned `ChunkOPhysicalPlan`

新增：

```text
lib/Dialect/AveLang/IR/static_physical_layout.h
lib/Dialect/AveLang/IR/static_physical_layout.cc
```

`ChunkOPhysicalPlan` 保存：

```text
q, h, k, v
scoreShared
mfma
consumer relationships
phase lifetimes
shared regions
```

它不是 source-facing Qwen op，也不是 MLIR SSA value。它的职责是保存 region-level
关系，尤其是 tensor attr 不适合独自表达的：

```text
Q -> Q@H
Q -> Q@K
samePhysicalSource = true

source_Q_H_K lifetime
    -> source_release

score/V lifetime
    -> output
```

typed attrs 用于随 IR value/op 通过 pass；C++ plan 用于 planner 内部在一个完整
region 中联合看见 producer、consumer、shared region 和 lifetime。二者不是两个
独立的 semantic source of truth，而是两个层级的载体。

## 5. C12 plan 的完整承载

C12 输入工件为：

```text
stage6z_c12_native_chunk_o_physical_plan.json
codex_qwen_bt64_stage6z_native_chunko/native/T2048/trace_capture/selected/chunk_fwd_kernel_o.ttgir
```

`ChunkOPhysicalPlan::makeC12T2048WG256()` 对应以下链路。

### 5.1 Q

```text
#blocked2
    -> #shared
    -> dot_op0
```

表示字段：

```text
logicalShape    = [64, 32]
sizePerThread   = [1, 8]
threadsPerWave  = [16, 4]
wavesPerCTA     = [4, 1]
order           = [1, 0]
shared          = kind=swizzled_shared, vec=4, perPhase=2, maxPhase=8
dot             = opIdx=0, kWidth=4
```

计划额外保存：同一 Q physical source 同时服务 `Q@H` 与 `Q@K`，没有把 dual
consumer 拆成两个逻辑 producer。

### 5.2 H

```text
#blocked2
    -> #shared1
    -> fixed transpose
    -> dot_op1
```

H 与 Q 使用同一 distributed family，但 shared encoding 为 `#shared1`，并且
通过 `StaticTransformAttr(kind=fixed_transpose, permutation=[1,0])` 表示 H 到
MFMA B operand 的固定转置。

### 5.3 K

```text
#blocked1
    -> #shared2
    -> dot_op1
```

表示字段为：

```text
logicalShape    = [32, 64]
sizePerThread   = [8, 1]
threadsPerWave  = [4, 16]
wavesPerCTA     = [1, 4]
order           = [0, 1]
shared          = kind=swizzled_shared, vec=4, perPhase=2, maxPhase=8
dot             = opIdx=1, kWidth=4
```

### 5.4 V

```text
#blocked
    -> fixed in-thread transpose / #linear1
    -> #shared4 rotating
    -> dot_op1
```

表示字段为：

```text
logicalShape    = [64, 64]
sizePerThread   = [2, 8]
threadsPerWave  = [8, 8]
wavesPerCTA     = [4, 1]
order           = [1, 0]
shared          = kind=amd_rotating_shared, vec=4, perPhase=1, maxPhase=16
dot             = opIdx=1, kWidth=4
transform       = fixed_transpose, #blocked -> #linear1
```

这里必须区分“表示了什么”和“已经能生成什么”：C13 已经能表示 V 的静态元素
变换关系和 native `amdg.in_thread_transpose` 的语义位置，但没有声称已经恢复
每个 lane 的最终物理寄存器表。该未知事实在 JSON 和报告中显式标记，没有根据
Triton 变量名猜测。

### 5.5 MFMA

统一保存：

```text
target       = gfx942
version      = 3
instrShape   = [32, 32, 8]
warpsPerCTA  = [2, 2]
transposed   = true
```

## 6. 纯 compile-time layout algebra

`static_physical_layout.h/.cc` 提供以下查询：

```text
hardware(register, lane, wave) -> logical coordinate
logical coordinate -> owners(register, lane, wave)
logical coordinate -> shared byte offset
logical coordinate -> MFMA operand slot
logical coordinate -> fixed transform result
```

这些 API 只接收 C++ 整数、数组和静态 plan，不接收 `mlir::Value`。因此它们不会
生成：

```text
arith.divui
arith.remui
arith.addi
arith.muli
select/cndmask ownership IR
```

Distributed mapping 使用静态 `order` 解码 lane/wave 坐标。Shared mapping 使用
固定的有限 phase-xor recipe 做表示层 bijection 检查。这个 recipe 的作用是验证
“给定 encoding 可以形成无 alias、覆盖完整 tile 的静态地址函数”，不是声称它
就是 native Triton bank formula。真正的 AMDGPU LDS packet lowering 留到后续阶段。

## 7. mapping verifier

新增 `static_physical_layout_test.cc`，测试覆盖：

1. C12 `ChunkOPhysicalPlan` 整体 verify。
2. Q/H/K/V 全 tile ownership enumeration。
3. 每个 logical element 恰好一个 owner。
4. owner 反查 logical coordinate 可逆。
5. Q/H/K/V/score shared byte offset 无重复、无越界、无 alias。
6. dot operand slot 的 opIdx、rowTile、row、kGroup、word 有效且无重复。
7. H/V fixed transpose 的 cardinality 和元素集合保持不变。
8. duplicate order、shape mismatch、unknown shared kind、invalid permutation、
   invalid dot width 都被拒绝。

实测结果：

```text
7 tests from 1 test suite
7 passed
```

Q/H/K 每个测试 tile 是 2048 个元素，V 是 4096 个元素。完整数值和失败计数见：

[`stage6z_c13_c12_roundtrip.json`](stage6z_c13_c12_roundtrip.json)

## 8. old runtime recipe 与 static representation 对照

旧 `makeLogicalBlockLayoutPlan` 的动态 recipe 包括：

```text
wave       = tid / 64
lane       = tid % 64
laneCol    = lane % 32
laneGroup  = lane / 32
rowHalf    = wave / 2
valueHalf  = wave % 2
H packetRow/packet = tid / 4, tid % 4
K producerLinear  = (wave / 2) * 64 + lane
packetCol         = packet * 8
feature           = kStage * 32 + packetCol
```

C13 不删除旧 planner，也不改变旧 codegen。它增加一个可验证的静态等价视图：

```text
wave/lane/laneCol/laneGroup
    -> DistributedEncoding

packet/feature
    -> static distributed order + DotOperandEncoding(kWidth=4)

shared row/word
    -> SharedEncoding

H/V transform
    -> StaticTransformAttr

Q dual consumer / phase lifetime
    -> ChunkOPhysicalPlan
```

逐项结果见：

[`stage6z_c13_old_vs_static_layout.json`](stage6z_c13_old_vs_static_layout.json)

结论是：对本 C13 子集，固定 physical facts 不再需要 runtime SSA 才能表达。

## 9. pass-through survival proof

测试构造最小 MLIR module：

```text
module
  func.func @c13_survival()
    attributes {
      c13.distributed = typed ave.distributed_encoding
      c13.shared      = typed ave.shared_encoding
      c13.dot         = typed ave.dot_operand_encoding
      c13.transform   = typed ave.static_transform
    }
```

然后依次运行：

```text
initial annotated IR
    -> canonicalizer
    -> nested lower_qwen_block_dot pass
    -> MLIR print
    -> MLIR parse
```

`lower_qwen_block_dot` 在该最小 fixture 上没有 block-dot operation，因此不会做
operation rewrite；这正是一个安全的 pass-through survival fixture，而不是把
“无 block-dot”冒充成 lowering 已完成。它证明相关 pass boundary 不会丢掉 typed
attrs。随后 module 的 textual MLIR 能打印出五类 attr 的 mnemonic，并成功重新
parse。

plan 本体是 compiler-owned C++ 对象，不能直接作为 MLIR attr 序列化。本轮采用
的生命周期约束是：

```text
create C12 plan
    -> verify static algebra
    -> attach/forward typed attrs and plan identity at planner boundary
    -> consume before physical lowering
```

不会把 plan 强行伪装成 runtime SSA。具体 pass-survival 证据见：

[`stage6z_c13_pass_survival.json`](stage6z_c13_pass_survival.json)

## 10. `physical_encoding` legacy metadata 迁移

当前 block-dot 的字符串 metadata 不能在本轮直接删除，因为已有 IR 工件和旧
regression 仍依赖它。因此迁移策略是：

```text
legacy physical_encoding string
    -> adapter/decoder at planning boundary
    -> typed Distributed/Shared/Mfma/Dot/Transform attrs
    -> compiler-owned ChunkOPhysicalPlan
```

未来 typed attrs 和 plan 才是 experimental internal path 的 semantic source of
truth。legacy 字符串继续作为兼容输入/调试 metadata 保留，但不应该再独立定义
lane ownership、shared swizzle 或 dot operand 语义。

本轮没有修改已有 block-dot source ABI，也没有新增 Qwen public abstraction。

## 11. 与“当前 Avelang 做不到”的准确关系

本轮可以明确区分三个层次：

### 已经能表达

- 普通 shape/stride memory layout。
- block-dot op 和已有 internal MFMA operand op。
- 编译器自己的静态 recurrence plan 模式。
- C13 typed distributed/shared/MFMA/dot/static-transform attr。
- Q/H/K/V full-region producer-consumer/lifetime 的 C++ plan。
- 静态 ownership/shared/dot/transform 的整数验证。

### 本轮证明可以跨过

- MLIR attr 生成、注册、打印、解析。
- canonicalizer。
- 当前 `lower_qwen_block_dot` pass boundary。
- 不退化成 string-only metadata。

### 仍然没有实现

- attr/plan 到 `memref`/`amdg`/ROCDL 的 codegen lowering。
- exact Triton lane/register-level V in-thread transpose table。
- native exact bank-conflict/swizzle formula 的机器级证明。
- typed LDS store/load、dot operand fragment 的 ISA 生成。
- 新 kernel correctness、resource 或性能。

因此，证明的是“static full-region encoding 在 AveLang compiler representation 层
可以实现”，不是“现有 compiler 已经能自动生成 native Triton ISA”。

## 12. 构建与测试

由于宿主 build cache 使用 `/workspace/project/avelang`，本轮在
`ljd_qwen_vllm_avelang_rocm722` 容器中执行：

```bash
cmake --build build --target static_physical_layout_test -j2
build/lib/Dialect/AveLang/IR/static_physical_layout_test --gtest_color=no
```

最终结果：

```text
build/link: PASS
tests: 7
passed: 7
failed: 0
```

本轮没有运行 full Avelang build、Qwen recurrence suite 或 GPU benchmark，因为
任务硬边界要求在 representation/pass-through/unit test 完成后停止。已有工程中
与 Qwen 相关的 dirty worktree 变更没有被回滚。

## 13. 输出文件

- [`stage6z_c13_physical_representation.json`](stage6z_c13_physical_representation.json)
- [`stage6z_c13_c12_roundtrip.json`](stage6z_c13_c12_roundtrip.json)
- [`stage6z_c13_old_vs_static_layout.json`](stage6z_c13_old_vs_static_layout.json)
- [`stage6z_c13_pass_survival.json`](stage6z_c13_pass_survival.json)
- `lib/Dialect/AveLang/IR/AveLangAttrs.td`
- `lib/Dialect/AveLang/IR/static_physical_layout.h/.cc`
- `lib/Dialect/AveLang/IR/static_physical_layout_test.cc`

## 14. 最终判断

```text
C13_SPR_MVP_GO_FOR_CODEGEN
```

允许下一阶段开始讨论 codegen，但下一阶段必须重新注册自己的 correctness、
resource、MIR/ISA 和性能 gates；本报告不为任何未执行的 codegen 或性能结果背书。
