# Qwen gfx942 C12：Full Chunk-O Native-Shaped Physical Lowering

## 结论先行

本轮 C12 完成了两件事：

1. 建立了 external native chunk-o HSACO upper-bound control，并证明
   Avelang 的 caller、当前 HIP stream、caller-owned output、launch ABI 可以
   直接调用 selected native `chunk_fwd_kernel_o`。
2. 从 selected native 的 TTGIR、LLVM、MIR/ISA 工件恢复了完整的 Q/H/K/V
   physical chain，并执行了 feasibility gate。

最终决策是：

```text
STOP_C12_PHYSICAL_PLAN_INCOMPLETE
```

这不是 native 算法不可恢复，而是当前 Avelang compiler representation 还不能
在一份 static full-region plan 中同时保存并传递 native 所需的 Q/H/K/V
distributed encoding、shared encoding、dot operand encoding、固定 V transpose
和 phase lifetime。因而本轮**没有实现 C12 kernel、没有生成 C12 HSACO、没有运行
C12 correctness/PMC/performance**，也没有创建 C12-Q/H/K/V 局部变体。

机器可读 physical plan：

[stage6z_c12_native_chunk_o_physical_plan.json](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/stage6z_c12_native_chunk_o_physical_plan.json)

external control 记录：

[stage6z_c12_external_native_control.json](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/stage6z_c12_external_native_control.json)

## 1. 实验边界

本轮只研究 chunk-o。persistent recurrence 不在本轮重做，因为当前 public
experimental path 已经使用 current-vLLM/Triton equivalent recurrence HSACO；
本轮的剩余主要差距是 chunk-o 的 machine graph。

冻结内容：

- gfx942、wave64、BT64、BV64、BK32；
- T=2048 的 selected native target 为 WG256、2 CTA/chunk-head；
- MFMA 为 `v_mfma_f32_32x32x8_bf16`；
- BF16 Q/K/V-new/H/output，FP32 g；
- Q/H/K/V 的数学、causal mask、K32 reduction、输出 ABI；
- Z5B、Z8W 及历史 Stage6Z 局部路线；
- allocator/RA、recurrence HSACO、production selector；
- 不进行 packet width、LDS swizzle、barrier、waitcnt、WG 或 tile sweep。

特别注意：native 在长文本 T=8192 会选择另一个 WG128/2-warp specialization。
本报告将 T=2048 WG256 作为 same-shape physical-plan 目标；T8192 只用于证明
external control 能复现另一个 selected native code object，不能把两个
specialization 的资源数混成一张 same-shape 表。

## 2. Step 0：external native HSACO control

### 2.1 调用链

新增的实验 launcher 只做 external module loading 和真实 native launch：

- C++ bridge：
  `codex_qwen_bt64_stage6z_native_chunko/c12_native_chunk_o_bridge.cpp`
- Python ABI wrapper：
  `codex_qwen_bt64_stage6z_native_chunko/qwen_gdn_bt64_c12_external_native_chunko.py`
- 构建脚本：
  `codex_qwen_bt64_stage6z_native_chunko/build_c12_external_native_chunk_o_bridge.sh`

真实 kernel symbol：

```text
chunk_fwd_kernel_o
```

参数顺序为：

```text
q, k, v_new_bf16, h_bf16, g, output_bf16, scale, T, null, null
```

wrapper 检查 dtype、shape、contiguous layout、current stream、T-specific
HSACO hash、grid、workgroup 和 dynamic LDS。output 是 caller-owned，计时前
完成预分配；没有 CUDA Graph、没有 module load、没有 allocation 计时。

### 2.2 native code-object identity

| T | grid | WG | stages | dynamic LDS | HSACO SHA256 |
|:--:|:--|--:|--:|--:|:--|
| 2048 | `(2,32,8)` | 256 | 3 | 24576 B | `9cc107ecf4f9b8529694fd07f4532174ba98593aab206fbddf628dbf53bfb95c` |
| 8192 | `(2,128,8)` | 128 | 2 | 12288 B | `e201dd58c83e64565f10754789ac294df53343b9c9bbbe764e8f667817066ee5` |

T2048 selected artifact：

[native T2048 selected artifacts](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_native_chunko/native/T2048/trace_capture/selected)

T8192 selected artifact：

[native T8192 selected artifacts](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_native_chunko/native/T8192/trace_capture/selected)

### 2.3 correctness

| T | external vs selected native BF16 | external max abs | Z5B max abs vs native |
|--:|:--:|--:|--:|
| 2048 | exact | `0` | `3.0517578125e-05` |
| 8192 | exact | `0` | `3.0517578125e-05` |

external control 与 selected native 使用同一个 code object，输出逐元素
`torch.equal`。Z5B 与 native 的极小差异来自不同的 BF16/累加顺序路径，不是
本轮 external bridge 的错误；Z5B 的完整 correctness 仍以 Z5B 自己的 reference
门槛为准。

### 2.4 七 session body timing

口径：fresh process、current HIP stream、caller-owned output、no Graph，
warmup=10、repeat=50；每个 session 轮换顺序。所有原始 session 保存在：

- [T2048 sessions](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_native_chunko/c12_external_native_T2048_sessions7.json)
- [T8192 sessions](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_native_chunko/c12_external_native_T8192_sessions7.json)

下表是七个 session median 的中位数，不是单次最好值。

| T | Z5B ms | external native HSACO ms | selected native direct body ms | Z5B / external | Z5B / native |
|--:|--:|--:|--:|--:|--:|
| 2048 | `0.0671000` | `0.0300245` | `0.0426835` | `2.235x` | `1.572x` |
| 8192 | `0.1570135` | `0.0771345` | `0.0895130` | `2.037x` | `1.755x` |

paired vectors：

```text
T2048 external - native, us:
[-12.5790, -13.6805, -12.2780, -13.8610, -12.1785, -11.5375, -12.2780]

T2048 Z5B - external, us:
[37.9770, 36.8150, 36.7145, 36.9955, 36.3745, 36.0535, 36.5945]

T8192 external - native, us:
[-12.5985, -15.2225, -3.3245, -14.5620, -12.7590, -7.5520, -13.2800]

T8192 Z5B - external, us:
[81.3605, 79.9590, 71.1460, 79.4785, 80.3390, 75.6130, 79.9990]
```

external launcher 比当前 direct-native body 更快并不改变结论：它使用的是
selected native 的同一 HSACO/hash，证明 Avelang caller/ABI/stream/launch 没有
把 native code object 的收益挡住。`selected_native_direct_body` 是 public
selector warm-up 后的 direct body control，不是 public-Eager 端到端排名。

因此 Step 0 通过：

```text
external native control reaches the native code-object upper bound: PASS
```

这给出了 C12 的理论 upper bound：在相同 caller/body 口径下，T2048 至少存在
约 `0.037 ms` 的 Z5B-to-external gap，T8192 至少存在约 `0.080 ms` 的 gap。

## 3. selected native 的完整 physical plan

### 3.1 CTA ownership

T2048 selected native 的程序 ID 是：

```text
i_v = program_id(0)
i_t = program_id(1)
i_bh = program_id(2)
grid = (2, 32, 8)
```

一个 CTA 负责一个 `[BT=64, BV=64]` output tile、一个 chunk、一个 value
head。一个 chunk-head 的 `V=128` 由两个 CTA 覆盖。Q/K 的 score 和 inter
state 都按四个 BK32 stage 完成；score@V-new 只需要 K=64。

这不是“每个 V16 CTA 重复做一遍完整 Q/K”的 Z5B ownership。native 同一个
Q physical source 在一个 CTA 内服务 Q@H 与 Q@K，两个 V64 CTA 各自读取 disjoint
的 H/V-new value half。

### 3.2 TTGIR encoding 总表

selected T2048 TTGIR 头部直接给出了以下 encoding：

```text
#blocked  = sizePerThread [2,8], threadsPerWarp [8,8],
            warpsPerCTA [4,1], order [1,0]
#blocked1 = sizePerThread [8,1], threadsPerWarp [4,16],
            warpsPerCTA [1,4], order [0,1]
#blocked2 = sizePerThread [1,8], threadsPerWarp [16,4],
            warpsPerCTA [4,1], order [1,0]
#shared   = vec 4, perPhase 2, maxPhase 8, order [1,0]
#shared1  = vec 1, perPhase 1, maxPhase 1, order [1,0]
#shared2  = vec 4, perPhase 2, maxPhase 8, order [0,1]
#shared3  = vec 1, perPhase 1, maxPhase 1, order [0,1]
#shared4  = amd_rotating_shared, vec 4, perPhase 1,
            maxPhase 16, order [0,1]
#mma      = amd_mfma version 3, warpsPerCTA [2,2],
            instrShape [32,32,8], isTransposed true
```

这些不是从 Triton 变量名推断，而是来自：

[selected native TTGIR](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_native_chunko/native/T2048/trace_capture/selected/chunk_fwd_kernel_o.ttgir:1)

### 3.3 Q physical chain

```text
global Q BF16 #blocked2
  -> buffer_load tensor<64x32xbf16,#blocked2>
  -> #shared two-slot Q memdesc [2,64,32]
  -> local_load #ttg.dot_op opIdx=0,kWidth=4
  -> Q@H MFMA
  -> same Q physical source
  -> Q@K MFMA
```

Q shared allocation 出现在 TTGIR line 124。初始 producer 在 lines 144 和
174；K loop 内的下一 slot producer 在 lines 232、258-259。Q 的 shared source
在 lines 233、266、272 被直接 local-load 为 MFMA A operand。

最关键的事实是：Q 没有一个“Q@H 专用副本”和一个“Q@K 专用副本”。同一个
rotating Q memdesc 在同一 K-stage 中被两个 dot consumer 读取。这正是 Z5B
source-level Q cache 只完成了一半的问题：Z5B 有 Q residency，但没有 native
的完整 physical Q/dot encoding。

### 3.4 H physical chain

```text
global H BF16 #blocked2
  -> #shared1 two-slot H memdesc [2,64,32]
  -> local_load #linear
  -> fixed tt.trans order=[1,0]
  -> #ttg.dot_op opIdx=1,kWidth=4
  -> Q@H MFMA
```

H 的 `#shared1` 是 token-major/simple shared layout，TTGIR line 251 的
`local_load` 返回 `#linear`，line 252 通过固定 `tt.trans` 形成 dot B operand。
这是一个 compile-time layout transform，不是运行时 generic transpose loop。

### 3.5 K physical chain

```text
global K BF16 #blocked1
  -> buffer_load tensor<32x64xbf16,#blocked1>
  -> #shared2 two-slot K memdesc [2,32,64]
  -> local_load #ttg.dot_op opIdx=1,kWidth=4
  -> Q@K MFMA
```

K 的 producer 和 consumer 编码都是 K-major：`#blocked1`、`#shared2` 和
`#dot_op opIdx=1` 一致。TTGIR lines 161、175-176、243-244、260-277 给出
了这条链。native 没有把 K 先降成一个普通 BF16 scalar vector，再由 generic
fragment loop 重建。

### 3.6 V physical chain

```text
global V-new BF16 #blocked
  -> amdg.in_thread_transpose #blocked -> #linear1
  -> #shared4 amd_rotating_shared
  -> local_load #ttg.dot_op opIdx=1,kWidth=4
  -> score@V-new MFMA
```

TTGIR lines 397-406 是直接证据：

```text
%b_v_279 = amdg.buffer_load %v[...] : tensor<64x64xbf16,#blocked>
%b_v_285 = amdg.in_thread_transpose %b_v_279
              : tensor<64x64xbf16,#blocked>
             -> tensor<64x64xbf16,#linear1>
%b_v_286 = ttg.local_alloc %b_v_285 -> #shared4
%b_v_287 = ttg.local_load %b_v_286 -> #ttg.dot_op opIdx=1
%b_o_288 = tt.dot score, %b_v_287, ...
```

`#linear1` 的 register/lane/warp encoding也在 TTGIR line 5 明确给出。它不是
`ds_bpermute` 大循环，也不是 Z5B 的 phase-vector generic gather。LLVM 中对应
的输入是 `raw.ptr.buffer.load.v4i32`，随后产生 addrspace(3) packed store；
final ISA 中能看到 `buffer_load_dwordx4`、`ds_write_b128/b64`、`ds_read` 和
MFMA 的固定序列。

### 3.7 lifetime

native 的 source phase 先分配：

```text
Q #shared  : 2 * 64 * 32 * 2 = 8192 B
H #shared1 : 2 * 64 * 32 * 2 = 8192 B
K #shared2 : 2 * 32 * 64 * 2 = 8192 B
峰值        = 24576 B
```

Q/H/K 最后一次 dot 完成后，TTGIR lines 278-280 执行三个 `local_dealloc`。
随后才进入 score/V-new phase：

```text
score #shared3 : 64 * 64 * 2 = 8192 B
V #shared4     : 64 * 64 * 2 = 8192 B
```

这样 source Q/K/H buffers 不会和 score/V buffers 同时存活。selected code
object 的 T2048 dynamic LDS 为 24576 B，与这条 lifetime 解释一致。

## 4. Avelang current representation audit

### 4.1 `block_dot` 的 source contract

当前 op 定义在：

[AveLangOps.td](/home/jiandongliu/project/avelang/lib/Dialect/AveLang/IR/AveLangOps.td:467)

它已经能保留一个 experimental `block_dot_bf16_f32`，并有内部的
`amdgpu_block_dot_mfma_operand`。这是此前 BDV2/P1 的有效基础，但它不是
native C12 所需的 complete physical plan。

当前 full-scope lowering 的明确限制在：

[lower_qwen_block_dot_pass.cc](/home/jiandongliu/project/avelang/lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc:1340)

```text
full-scope block-dot source role must be K or H
```

也就是说，虽然 op ABI 中存在 `sourceVNew` 等兼容字段，真正的 full-scope
producer lowering 只为 K/H 建立 producer-to-shared 逻辑。它没有为 Q 和 V
建立同级别的 physical plan。

### 4.2 当前 planner 是运行时 layout arithmetic

`makeLogicalBlockLayoutPlan` 位于：

[lower_qwen_block_dot_pass.cc](/home/jiandongliu/project/avelang/lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc:167)

它用运行时 SSA 值计算：

```text
wave    = tid / 64
lane    = tid % 64
laneCol = lane % 32
laneGroup = lane / 32
rowHalf = wave / 2
valueHalf = wave % 2
packetRow = producerLinear / 4
packet = producerLinear % 4
packetCol = packet * 8
feature = kStage * 32 + packetCol
```

这些计算对固定 gfx942/BT64/BV64/BK32/WG256 的大部分情况本来可以是
compile-time physical encoding 的一部分，但当前 pass 把它们 materialize
成 runtime `DivUI/RemUI` 等 IR。这个形态与 C12 gate 的“不能使用 runtime
generic layout planner”直接冲突。

### 4.3 当前 producer lowering 的粒度

`emitFullScopeProducer` 仍然按 K/H 两种 source role 分叉：

- specialized arm 为 K/H 生成 `vector.load` + shared `vector.store`；
- generic arm 生成逐元素 `memref.load`/`memref.store`；
- shared layout 是 Avelang 自己的 consumer-oriented memref shape，不是
  native `#shared/#shared1/#shared2` encoding；
- 后续 `AMDGPUBlockDotMfmaOperandOp` 再按显式 row/word 值形成 MFMA operand。

这可以改变一部分 LLVM/MIR/ISA，已经被 BDV2 证明；但它仍不是 Q/H/K/V
统一的 native physical chain。

### 4.4 V transpose 表示缺口

本轮在 `lib/Dialect` 和 `lib/IR` 审计中没有找到与 native
`amdg.in_thread_transpose`、`#linear1`、`#shared4 amd_rotating_shared` 三者
相连的 AveLang source/IR representation。已有的 `tt.trans`/历史 transpose
相关代码不能自动证明可以生成 native 的固定 register-slot permutation。

因此若现在直接写 C12 source，最可能的结果只能是：

```text
V global load
-> generic view/memref
-> scalar/register reconstruction
-> ordinary shared store
-> generic local load
```

这正是 C12 明确禁止的回退路径，不能把它命名为 native-shaped。

## 5. C12 feasibility gate

| gate | 结果 | 证据 |
|:--|:--:|:--|
| selected native encoding 足够恢复 | PASS | TTGIR 明确给出 blocked/shared/linear/dot/mfma encoding |
| lane/shared/fragment map 可由当前 AveLang static representation 表达 | FAIL | 当前 op 没有这些 encoding 类型；planner 使用 runtime DivUI/RemUI |
| Q/H/K/V 统一 full-region plan | FAIL | current full-scope source role 只有 K/H |
| 不依赖 runtime generic layout planner | FAIL | `makeLogicalBlockLayoutPlan` 生成运行时 ownership/packet arithmetic |
| 不需要 full-tile VGPR/private ring | 未到达 | representation gate 已失败，不能生成 candidate 后再猜资源 |
| 不改 MFMA/math/K32/ABI | 未到达 | 没有 C12 candidate |

因此不是“native 工件不完整”，而是“native 工件完整，但 Avelang 当前表示
能力不完整”。按任务预注册规则，必须停止：

```text
STOP_C12_PHYSICAL_PLAN_INCOMPLETE
```

## 6. 与 Z5B 的 machine/resource 边界

Z5B 当前 machine artifact：

[Z5B machine evidence](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_z5b_machine_stage1/machine_evidence.json)

| kernel | dynamic MFMA/CTA | dynamic VMEM/CTA | dynamic LDS/CTA | dynamic VALU/CTA | dynamic SALU/CTA | code VGPR | code AGPR | LDS | scratch |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Z5B | 160 | 672 | 672 | 7072 | 768 | 104 | 32 | 32768 B | 0 |
| native T2048 | 160 | 140 | 480 | 3376 | 660 | selected native metadata | selected native metadata | 24576 B | 0 |

native static T2048 counts from the selected artifact are：MFMA32 40、buffer
load 14、buffer store 4、ds_read 72、ds_write 40、barrier 10。它们是 lexical
static counts，不被当作 dynamic PMC。Z5B static counts similarly come from its
ISA audit and are not substituted for runtime counters.

C12 没有 machine delta，因为 feasibility gate 在 candidate 生成前失败。不能
填写一个虚构的“C12 VMEM/VALU/LDS”，也不能把 external native 的数写成
Avelang C12 的性能。

## 7. 对任务十个问题的逐项回答

### 1. external native chunk-o HSACO 能否达到 native latency？

能。它使用同一个 selected native symbol/HSACO/hash，T2048 和 T8192 correctness
exact，并在 body 计时中分别为 `0.0300 ms` 和 `0.0771 ms`；没有 caller/ABI/stream
障碍把 native code object 的收益挡住。

### 2. 因而 chunk-o 理论 compiler-codegen 上限是多少？

在当前 isolated body 口径下，T2048 从 Z5B `0.0671 ms` 到 external native
`0.0300 ms`，可见上限约 `36.7 us`；T8192 从 `0.1570 ms` 到 `0.0771 ms`，
可见上限约 `80.0 us`。这是 code-object upper bound，不是 Avelang C12 已实现
收益。

### 3. selected native 的完整 Q/H/K/V physical plan 是什么？

Q 使用 `#blocked2 -> #shared -> dot_op0` 并同时服务 Q@H/Q@K；H 使用
`#blocked2 -> #shared1 -> #linear -> fixed tt.trans -> dot_op1`；K 使用
`#blocked1 -> #shared2 -> dot_op1`；V 使用
`#blocked -> amdg.in_thread_transpose -> #linear1 -> #shared4 -> dot_op1`。
Q/H/K source buffers 在 score/V phase 前 dealloc。完整 JSON 已记录所有字段和证据路径。

### 4. 哪些 physical facts 以前 AveLang IR 没有保存？

缺少：lane/warp blocked encoding、swizzled shared 参数、rotating shared 参数、
dot-op `opIdx/kWidth`、Q 同一 physical source 的多 consumer 关系、V 的固定
in-thread transpose slot map、以及 Q/H/K 到 score/V 的统一 lifetime plan。

### 5. C12 是否是 full-region plan 而不是局部 operand patch？

设计要求是 full-region；本轮 gate 没通过，所以没有实施局部 patch，也没有
把 K/H 的已有 first-class operand 误称为 C12。

### 6. static transforms 是否避免 runtime planner？

selected native 是的：TTGIR encoding 和 `amdg.in_thread_transpose` 是固定
physical recipe。当前 AveLang 不是：现有 full-scope planner 使用 runtime
DivUI/RemUI 和 generic memref/vector lowering。

### 7. VMEM/VALU/LDS/SALU 能否向 native 缩小？

本轮没有 C12 machine graph，所以不能报告 C12 缩小值。已知 Z5B 与 native
的 body/PMC 差距是真实的：Z5B dynamic `672/7072/672/768` 对 native
`140/3376/480/660`（VMEM/VALU/LDS/SALU）。external control 证明这个差距
不是 launch ABI 问题，但不能证明只改某一项就能复现 native。

### 8. T2048/T8192/T16384 提升多少？

Z5B 到 external native 的 T2048/T8192 body upper-bound 分别约 `55.2%`、
`50.9%`。C12 没有实现，因此 T16384 没有 C12 数据，不能填补或外推。

### 9. public Eager 最终提升多少？

本轮没有 C12 public path，故为 `N/A`。external body control 不是 public
Eager replacement，也没有修改 selector/production dispatch。只有未来一个
真正 correct 的 full-region AveLang C12 candidate 成为 isolated winner 后，
才有资格做 public Eager 验证。

### 10. C12 是否值得继续？

作为 Qwen 局部 patch，当前 C12 应停止；作为独立 compiler research project，
值得继续建立 **native-style static distributed-layout infrastructure**。这个
基础设施应先成为通用 compiler representation，再重新承载一个 full-region
chunk-o plan；不应继续枚举 C12-Q/H/K/V、transpose 或 packet-width 变体。

## 8. 为什么本轮没有实现 candidate

如果强行继续，至少要新增以下 compiler-owned 能力：

1. 静态 `DistributedEncoding`：表达 size-per-thread、threads-per-warp、
   warps-per-CTA、order 和 dot operand slot。
2. 静态 `SharedEncoding`：表达 swizzle/rotating 的 vec、perPhase、maxPhase、
   order 以及两个-slot/phase 生命周期。
3. `MFMAOperandEncoding`：表达 `opIdx=0/1`、kWidth=4、A/B operand 的
   register/ LDS map。
4. 固定 `in_thread_transpose` recipe：从 V 的 global blocked encoding 生成
   `#linear1`，不能降成 generic scalar transpose。
5. `ChunkOPhysicalPlan`：一次同时管理 Q/H/K/V producer、shared destination、
   dot consumer、phase lifetime、score/causal/g 和 output。

这不是给现有 `lower_qwen_block_dot_pass.cc` 再加一个 `if (sourceRole == V)`，
也不是把 native 的 lane table 硬编码进 Qwen pass。若直接这样做，会违反 C12
“一个完整 region plan、compile-time map、无 runtime generic planner”的成功条件。

## 9. 产物与复现

外部 control benchmark：

```bash
cd /workspace/project/avelang
export PYTHONPATH=/opt/avelang/python:$PYTHONPATH

python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_native_chunko/bench_qwen_gdn_bt64_c12_external_native_chunko.py \
  --T 2048 8192 \
  --warmup 10 --repeat 50 --sessions 7
```

主要产物：

- [C12 physical plan JSON](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/stage6z_c12_native_chunk_o_physical_plan.json)
- [C12 external control JSON](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/stage6z_c12_external_native_control.json)
- [T2048 raw sessions](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_native_chunko/c12_external_native_T2048_sessions7.json)
- [T8192 raw sessions](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_native_chunko/c12_external_native_T8192_sessions7.json)
- [native specializations](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_native_chunko/z0_native_specializations.json)

本轮没有 `stage6z_c12_machine_delta.json`，因为没有 C12 candidate。这个缺失
是有意的 stop-rule 结果，不是测试遗漏。

## 10. 最终决策

```json
{
  "external_native_upper_bound": "PASS",
  "native_physical_plan_recovered_from_artifacts": true,
  "avelang_static_full_region_representation": false,
  "c12_candidate_implemented": false,
  "c12_status": "STOP_C12_PHYSICAL_PLAN_INCOMPLETE",
  "public_eager_validation": "N/A",
  "next_allowed_direction": "independent native-style static distributed-layout compiler infrastructure"
}
```

C12 到此闭账。不要自动开始 C12-Q/H/K/V、transpose sweep、packet-width sweep、
新的 scheduler、g/P-op、recurrence 或 production rollout。
