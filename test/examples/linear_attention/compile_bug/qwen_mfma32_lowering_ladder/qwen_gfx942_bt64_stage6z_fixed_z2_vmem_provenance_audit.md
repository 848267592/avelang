# Qwen gfx942 BT64 Stage 6Z：Fixed Z2 VMEM Provenance Audit

## 结论先行

本报告是对 fixed Z2 与 exact native selected WG256 chunk-o 的只读
VMEM provenance 审计。它不修改 kernel source，不实现 Z4，不接入 X2，
也不修改 allocator、RA、recurrence HSACO、selector 或 production dispatch。

审计对象固定为：

```text
gfx942 / wave64 / BT64 / BV64 / BK32 / MFMA32 / WG256
2 CTA per chunk-head / BF16 q,k,h,v_new / FP32 g / BF16 output
```

T=2048 的 fresh same-shape PMC 是：

| metric | fixed Z2 | native selected WG256 | Z2/native |
|:--|--:|--:|--:|
| CTA 数 | 512 | 512 | 1.00x |
| dynamic MFMA per CTA | 160 | 160 | 1.00x |
| dynamic VMEM per CTA | 928 | 140 | 6.63x |
| dynamic LDS per CTA | 928 | 480 | 1.93x |
| dynamic VALU per CTA | 11,400 | 3,376 | 3.38x |
| dynamic SALU per CTA | 1,072 | 660 | 1.62x |

本轮新增的最强 provenance 结论是：

1. **Q 的重复 materialization 已经被 source 与 LLVM 直接证明。**
   Z2 在 Phase A 为 inter-state MFMA 读取一遍完整 Q tile，随后在
   Phase B 为 score 又重新读取同一逻辑 Q tile，并且 Phase B 位于
   `source_half=0,1` 两次循环中。native 的 TTGIR 则在同一个 K-step
   中把一个 typed Q operand 同时送给 `Q*H` 和 `Q*K` 两个 dot consumer。
2. **Z2 的 BF16 输入 global load 是窄化的 scalar load。** Q/K/H/V_new
   在 lowered LLVM 中都是 `load bfloat`，final ISA 是
   `global_load_ushort`。native 的 Q/K/H/V typed block 在 LLVM 中是
   `raw.ptr.buffer.load.v4i32`，对应 `buffer_load_dwordx4`，一个静态
   load 携带 8 个 BF16，即 16 bytes。这个结论证明了 load-width 和
   operand materialization 差异，但不能单独证明某一 operand 占据了
   928 个动态 VMEM 中的最大份额。
3. **K 的标量化已证明存在，但 K 是最大 offender 目前仍不能从现有
   证据推出。** 没有 per-operand VMEM transaction counter，静态 ISA
   lexical count 也不能换算成动态字节数。
4. **V_new 只有一个 source-level tile pass，窄 load 是确定的，但它不具备
   Q 那样的跨 phase 重复证据。**
5. 因而本轮只登记一个下一步 source 候选：
   **让 Phase-A 产生的 Q tile 在 Phase B 的两个 score half 中保持可复用，
   消除 Phase-A 到 Phase-B 的 Q global reload，并保持 WG256/BV64/BK32/
   MFMA32/数学和 ABI 不变。** 本候选没有在本轮实现。

这是一条由 source-visible dataflow 支持的下一步，不是对 compiler lowering
单独根因的最终证明。Z2 与 native 是不同 high-level program 的异源比较；
它能证明当前 Z2 的 materialization 结构和 native 的 typed operand 结构不同，
不能仅凭这一轮证明“同一 AveLang IR 的 LLVM lowering 必然错误”。

## 1. 审计边界与证据等级

### 1.1 本轮冻结项

- fixed Z2 是唯一 AveLang baseline。
- Z2 workgroup contract 固定为 256，host launch 为 `(256, 1, 1)`。
- native 使用 T=2048 fresh public selection 后锁定的 WG256 配置：
  `BK=32, BV=64, num_warps=4, num_stages=2, num_ctas=1`。
- 不使用修复 Phase-B dead K overread 之前的旧 Z2 timing、PMC 或 ISA。
- 不使用旧 WG128 native direct-body 数据作为本轮对照。
- 不运行 kernel source 修改，不运行 Z4，不接 X2。

### 1.2 使用的工件

固定工件根目录：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
  codex_qwen_bt64_stage6z_fixed_z2_vs_native_audit/
```

关键文件：

| 内容 | 文件 |
|:--|:--|
| fixed Z2 source | [`qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py`](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py) |
| Z2 lowered LLVM | `machine/z2_T2048/lowered_llvm.ll` |
| Z2 final ISA | `machine/z2_T2048/final_isa.s` |
| Z2 exact-LTO MIR | `machine/z2_T2048/exact_lto/kernel_section_*.mir` |
| native TTIR/TTGIR/LLVM | `machine/native_T2048/chunk_fwd_kernel_o.ttir`, `.ttgir`, `.llir` |
| native final ISA | `machine/native_T2048/final_isa.s` |
| dynamic PMC | `pmc3/stage6z_fixed_z2_vs_native_T2048_pmc.json` |
| raw counter CSV | `pmc3/z2/z2_T2048_counter_collection.csv`, `pmc3/native/native_T2048_counter_collection.csv` |
| pinned body timing | `body_T2048_pinned5.json` |

### 1.3 证据等级定义

本报告把结论分成四级：

| 等级 | 含义 |
|:--|:--|
| A | source、LLVM pointer/type、final ISA family 三者一致，能够证明逻辑归属或 load width |
| B | source/IR 能证明结构，但无法从静态模板恢复真实动态执行次数 |
| C | 只能说明存在可能的 materialization 或 address 成本，不能确认重复字节 |
| N/A | 当前工件没有足够证据，不进行推断 |

本轮特别遵守：

```text
static lexical instruction count != dynamic instruction count
dynamic VMEM instruction count != accessed byte count
```

例如 Z2 的 136 条静态 global-load 文本行不能写成每 CTA 136 次访问，
更不能写成固定字节数。动态 928/CTA 来自 PMC，load 的逻辑字节需要由
source shape、element type、lane execution 和有效 mask 另行推导。

## 2. Per-CTA 逻辑字节下界

一个 Z2/native 对照 CTA 负责：

```text
一个 64-token chunk
一个 value head
一个 BV64 value block
一个 key head
```

在 full-valid T=2048 情况下，单 CTA 的 unique logical tile 下界如下：

| operand | logical shape | dtype | unique logical bytes/CTA | 说明 |
|:--|:--|:--|--:|:--|
| Q | `[64,128]` | BF16 | 16,384 | 当前 chunk 的全部 K feature |
| K | `[64,128]` | BF16 | 16,384 | 两个 source half，各 32 token，合计 K128 |
| H | `[64,128]` | BF16 | 16,384 | 当前 chunk 的一个 value-head state tile |
| V_new | `[64,64]` | BF16 | 8,192 | 当前 chunk、当前 BV64 block |
| g | `[64]` | FP32 | 256 | score 的 target/source 都来自同一个 64-token g tile；这是 unique 下界 |
| output | `[64,64]` | BF16 | 8,192 | public output tile |
| **合计** | | | **65,792** | 仅是 unique logical I/O 下界 |

这里的 `65,792 B` 不是实测硬件流量，也不是 VMEM counter。它只表示：
如果每个 CTA 对每个逻辑 tile 只读一次、g 只保留一个 64-element tile，
并写一次 output，至少需要覆盖这些逻辑数据。它不包括 shared staging、
LDS round-trip、地址指令或重复读取。

### 2.1 Z2 已能从 source 直接推出的重复量

Z2 source 的 Phase A 在 `k_stage=0..3` 中产生完整 Q `[64,128]`，因此
产生一次 `16,384 B` 的 Q tile pass。Phase B 的 `source_half=0,1`
各自再次执行四个 K32 stage，每个 source half 又产生一次完整 Q tile pass。

因此，忽略边界 mask 和编译器执行细节，仅按 source logical dataflow：

```text
Q Phase A: 1 x 16,384 B
Q Phase B: 2 x 16,384 B
--------------------------------
Q source-level passes: 3 x 16,384 B = 49,152 B
unique Q lower bound:                  16,384 B
```

这不是把 49,152 B 当成 GPU transaction，而是一个 source-level duplicate
证据：Z2 对同一逻辑 Q tile 至少组织了三次独立 producer pass。最终 dynamic
VMEM 的精确 Q 字节仍需要 per-operand transaction counter，当前没有这种
证据，因此不把 `49,152 B` 写成硬件实际流量。

K、H、V_new 在 Z2 source 中分别是：

- K：Phase B 两个 source half，各自生成一个 `[32,128]` K half，合计覆盖
  一次 `[64,128]` K tile。当前 source 没有证明同一个 K32 stage 在同一 phase
  内重复加载。
- H：Phase A 只对当前 H tile 做一次 `[64,128]` staging pass。
- V_new：Phase C 只对当前 `[64,64]` V_new tile 做一次 staging pass。

这些对象的 scalar/narrow load 是确定的，但“是否最大”不是由 source pass
数量 alone 决定的。

## 3. Fixed Z2 source -> LLVM -> ISA provenance

### 3.1 Program mapping 与 shared layout

Z2 source 的固定映射在：

[`qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:60`](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:60)

核心内容：

```text
program_id    = block_id(0)
v_block_idx   = program_id % 2
value_head    = (program_id // 2) % 8
chunk_idx     = program_id // (2 * 8)
value_base    = v_block_idx * 64
chunk_start   = chunk_idx * 64
```

shared phase 是：

```python
phase = al.make_shared((256, 32), al.bf16)   # 16 KiB
phase_vec = al.view(phase, al.Tensor((256, 4, 4), al.i32))
```

这意味着 Q、K、H、score、V_new 都反复使用同一个 phase buffer 的不同
行区间。它能节约 LDS capacity，但也把每一阶段的 global load、shared store、
shared read 和 fragment view 串在一起。

### 3.2 Q：Phase A 到 Phase B 的重复 producer

Phase A source：

[`qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:85`](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:85)
到
[`qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:105`](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:105)：

```python
for k_stage in al.range(4):
    ...
    phase[row, col] = BF16(q[chunk_start + row, key_head, k_stage * 32 + col] * scale)
    phase[64 + row, col] = h[... k_stage * 32 + col]
    barrier
    ... inter_acc = mfma(h_frag, q_frag, inter_acc)
```

lowered LLVM 的 Q pointer/type 证据是：

```text
machine/z2_T2048/lowered_llvm.ll:169-170
  %136 = getelementptr ... bfloat, ptr %0, ...
  %137 = load bfloat, ptr %136, align 2
```

Phase B source：

[`qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:111`](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:111)
到
[`qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:145`](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:145)：

```python
for source_half in al.range(2):
    for k_stage in al.range(4):
        ...
        phase[score_stage_base + row, col] = BF16(q[... k_stage * 32 + col] * scale)
        phase[score_stage_base + 64 + row, col] = k[...]
        barrier
        ... score_acc = mfma(k_frag, q_frag, score_acc)
```

LLVM 中对应的第二个 Q load template 是：

```text
machine/z2_T2048/lowered_llvm.ll:391-392
  %324 = getelementptr ... bfloat, ptr %0, ...
  %325 = load bfloat, ptr %324, align 2
```

LLVM 的 loop body 还明确包含：

- `source_half` 的 2 次循环，约在 `lowered_llvm.ll:217-220` 开始；
- 每个 source half 的 `k_stage` 4 次循环，约在 `lowered_llvm.ll:408-411`；
- Q pointer 仍然是 kernel 参数 `%0`，没有从 Phase A 的 shared tile 直接
  传入 Phase B。

所以 Q duplicate 的证据等级为 **A**：

| 证据层 | 观察 |
|:--|:--|
| source | Phase A 和 Phase B 各有独立的 q expression |
| control flow | Phase B 的 q expression 被 `source_half=0,1` 包围 |
| LLVM | 两个独立的 `%0` GEP/load 模板，均为 `load bfloat` |
| ISA | 对应 `global_load_ushort` family，不是 shared-only reuse |
| 结论 | Q logical tile 在 phase boundary 被重新 materialize |

### 3.3 K：存在窄化，但暂不能证明重复

K source 位于：

[`qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:126`](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:126)
到
[`qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:132`](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:132)：

```python
for rep in al.range(4):
    ...
    phase[score_stage_base + 64 + row, col] = k[
        chunk_start + source_half * 32 + row,
        key_head,
        k_stage * 32 + col,
    ]
```

LLVM 证据：

```text
machine/z2_T2048/lowered_llvm.ll:453-454
  %378 = getelementptr ... bfloat, ptr %1, ...
  %379 = load bfloat, ptr %378, align 2
```

ISA 证据：K 与其它 BF16 input 一样落入：

```text
global_load_ushort v..., v[...], off
```

K 的结论分开写：

- **确定**：K 的 global element load 是 scalar 16-bit narrow load，且经过
  `phase` scalar store 后再被 `phase_vec`/fragment consumer 读取。
- **未证明**：当前静态工件没有证明同一个 K element 在同一 source half、
  同一 K32 stage 中被重复 global load。
- **未证明**：K 的窄 load 是否贡献了 928 dynamic VMEM 的最大份额。

因此 K 是一个真实的 vectorization/materialization opportunity，但不是本轮
唯一最大 offender。

### 3.4 H：一次性 source tile，scalar BF16 load

H source 位于 Phase A：

[`qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:94`](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:94)

LLVM 中 H 指针是 `%3`，对应：

```text
machine/z2_T2048/lowered_llvm.ll:199-200
  %165 = getelementptr ... bfloat, ptr %3, ...
  %166 = load bfloat, ptr %165, align 2
```

H 没有像 Q 那样在 Phase B 中再次从 `%3` 读取的 source evidence。它的
主要成本是 scalar BF16 global load、scalar shared store 和后续 packed LDS
read/view，而不是已证明的 cross-phase global duplicate。

### 3.5 g：FP32 scalar load，角色拆分不能从静态数唯一恢复

Phase B score 的 g/exp 在：

[`qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:147`](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:147)
到
[`qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:158`](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:147)：

```python
g[target] - g[source]
```

LLVM 对应两个 `%4` FP32 loads：

```text
machine/z2_T2048/lowered_llvm.ll:642-651
  %533 = getelementptr ... float, ptr %4, ...
  %534 = load float, ptr %533, align 4
  %541 = getelementptr ... float, ptr %4, ...
  %542 = load float, ptr %541, align 4
```

最终输出缩放再次读 g：

```text
machine/z2_T2048/lowered_llvm.ll:906-907
  %757 = getelementptr ... float, ptr %4, ...
  %758 = load float, ptr %757, align 4
```

因此可以确定：

- Z2 的 `global_load_dword` family 由 FP32 g load 构成，类型映射是 A 级证据。
- g 在 score 和最终 output scaling 中有不同 source use。
- 但 80 条静态 `global_load_dword` 文本行不能直接拆成 target/source/final
  三个动态 byte ledger。循环展开、predicate、lane execution 和 LTO 共同
  决定实际动态次数。
- 只能把 g 标为“有重复 use，确切 VMEM 归属 unresolved”，不能把它指定为
  最大 offender。

### 3.6 V_new：Phase C 单次 tile pass，窄 load 确定

V_new source 位于：

[`qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:165`](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:165)
到
[`qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:172`](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:172)。

LLVM：

```text
machine/z2_T2048/lowered_llvm.ll:723-724
  %598 = getelementptr ... bfloat, ptr %2, ...
  %599 = load bfloat, ptr %2-address, align 2
```

V_new 直接对应 `global_load_ushort`，并写入 phase 的 V transpose 区域。
source 只显示一个 `[64,64]` pass，没有 Q 那样的 Phase-A/Phase-B duplicate
证据。它的明确问题是 narrow load + LDS transpose/read，而不是已证明的
重复 global tile。

### 3.7 output：一次 public BF16 store，静态模板被循环复用

output source 位于：

[`qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:185`](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:185)
到
[`qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:191`](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py:191)。

LLVM 中 output pointer 是 `%5`，最终为 scalar BF16 store：

```text
machine/z2_T2048/lowered_llvm.ll:934-935
  %784 = getelementptr ... bfloat, ptr %5, ...
  store bfloat %765, ptr %784, align 2
```

ISA 中对应 `global_store_short_d16_hi`。这属于 unavoidable public output
逻辑写入，但 Z2 是窄 scalar store，而 native 使用 packed buffer store。

## 4. Native vLLM source/TTGIR/LLVM provenance

native 工件来自 exact selected WG256 cache，而不是凭变量名猜测。

### 4.1 Q/K/H typed block load

native TTGIR 中的 typed block producer：

```text
machine/native_T2048/chunk_fwd_kernel_o.ttgir:142
  %b_q_89 = amdg.buffer_load %q[...] : tensor<64x32xbf16, #blocked2>

machine/native_T2048/chunk_fwd_kernel_o.ttgir:159
  %b_k_106 = amdg.buffer_load %k[...] : tensor<32x64xbf16, #blocked1>

machine/native_T2048/chunk_fwd_kernel_o.ttgir:170
  %b_h_117 = amdg.buffer_load %h[...] : tensor<64x32xbf16, #blocked2>
```

在 loop body 内还有对应的 next-tile typed loads：

```text
machine/native_T2048/chunk_fwd_kernel_o.ttgir:195
  %b_q_274 = amdg.buffer_load %q[...] : tensor<64x32xbf16, #blocked2>
machine/native_T2048/chunk_fwd_kernel_o.ttgir:206
  %b_k_285 = amdg.buffer_load %k[...] : tensor<32x64xbf16, #blocked1>
machine/native_T2048/chunk_fwd_kernel_o.ttgir:213
  %b_h_292 = amdg.buffer_load %h[...] : tensor<64x32xbf16, #blocked2>
```

关键不是“native 没有任何 loop load”，而是一个 typed Q tile 在同一个
loop iteration 中被两个 consumer 复用：

```text
machine/native_T2048/chunk_fwd_kernel_o.ttgir:216-217
  %b_o_295 = tt.dot %b_q_275, %b_o_294, ...
  %b_A_296 = tt.dot %b_q_275, %b_k_286, ...
```

`%b_q_275` 来自：

```text
ttg.local_load %b_q_254 -> tensor<64x32xbf16, #ttg.dot_op>
```

也就是说，native 的 source/TTGIR graph 没有为 `Q*H` 和 `Q*K` 各自再建
一个独立 scalar Q producer。它通过 shared memdesc 和 typed dot-op 保持
一个 Q tile 的双 consumer 复用。

### 4.2 Native H transpose 与 K dot operand

native TTGIR 先把 H 放进 typed shared memdesc，再由：

```text
ttg.local_load %b_h_256 -> tensor<64x32xbf16, #linear>
tt.trans order=[1,0]
-> tensor<32x64xbf16, #ttg.dot_op>
```

然后与同一个 Q typed operand 做 `tt.dot`。K 直接从 `shared2` memdesc
load 为 dot operand：

```text
ttg.local_load %b_k_255
-> tensor<32x64xbf16, #ttg.dot_op>
```

这说明 native 的 K path 仍然有 LDS materialization，但 layout/operand type
在 high-level graph 中已经是 typed block，而不是 Z2 的 scalar BF16 load
加 `i32` view 加 fragment extraction 链。

### 4.3 Native V_new 与 output

TTGIR：

```text
machine/native_T2048/chunk_fwd_kernel_o.ttgir:354
  %b_v_240 = amdg.buffer_load %v[...] : tensor<64x64xbf16, #blocked>
machine/native_T2048/chunk_fwd_kernel_o.ttgir:358-359
  %b_o_244 = ttg.local_alloc / local_load score operand
machine/native_T2048/chunk_fwd_kernel_o.ttgir:360-362
  %b_v_246 = amdg.in_thread_transpose %b_v_240
  %b_v_247 = ttg.local_alloc %b_v_246
  %b_v_248 = ttg.local_load %b_v_247 -> dot operand
machine/native_T2048/chunk_fwd_kernel_o.ttgir:363
  %b_o_249 = tt.dot %b_o_245, %b_v_248, ...
machine/native_T2048/chunk_fwd_kernel_o.ttgir:379
  amdg.buffer_store ... : tensor<64x64xbf16>
```

native 也没有魔法般消除 V 的 shared path。它做的是 typed 64x64 load、
明确的 in-thread transpose、typed local allocation/local load、dot 和
packed output store。因而 native 与 Z2 的差别不是“有没有 LDS”，而是
producer/consumer layout 是否在 IR 中保持为可被后端直接消费的 typed block。

### 4.4 Native g

TTGIR 的 g 是两组 64-element FP32 `tt.load`：

```text
machine/native_T2048/chunk_fwd_kernel_o.ttgir:277
  %b_g_166 = tt.load ... tensor<64x!tt.ptr<f32>>
machine/native_T2048/chunk_fwd_kernel_o.ttgir:280
  %b_g_169 = tt.load ... tensor<64x!tt.ptr<f32>>
```

这与 native 的 score target/source 结构一致。静态 ISA 中相应主体是
17 条 `global_load_dword`，但同样不能把 17 条 lexical instructions 当作
动态 17 次或固定字节数。

## 5. Static ISA ledger：指令族和宽度

### 5.1 Fixed Z2

从 `machine/z2_T2048/final_isa.s` 统计的主 kernel lexical family：

| ISA family | static lexical count | type/width | provenance |
|:--|--:|:--|:--|
| `global_load_ushort` | 56 | 16-bit scalar | Q/K/H/V_new BF16 element load |
| `global_load_dword` | 80 | 32-bit scalar | g FP32 load，LLVM pointer `%4` |
| `global_load*` total | 136 | mixed | 不等于 dynamic VMEM |
| `global_store_short_d16_hi` | 16 | BF16 scalar store family | public output |
| `buffer_load*` | 2 | helper/resource load | 不按 Q/K/H/V ledger 强行归属 |
| `ds_write_b16` | 88 | 16-bit LDS store | scalar BF16 phase staging |
| `ds_read` | 20 | LDS operand read family | fragment consumer |
| `s_waitcnt` | 127 | synchronization | 非 VMEM bytes |
| `s_barrier` | 9 | CTA barrier | 非 VMEM bytes |
| `v_mfma_f32_32x32x8_bf16` | 20 | MFMA32 | static lexical，动态按 PMC 计 |

Z2 的 56 条 `global_load_ushort` 是静态文本中若干循环 body 的模板，
不是 per-CTA 的动态 56 条。所有 BF16 输入在 LLVM 层都是 scalar `load bfloat`，
因此可确定“窄化 load”，但不能仅由静态条数拆出 Q/K/H/V_new 各占多少
动态 VMEM。

### 5.2 Native selected WG256

从 `machine/native_T2048/final_isa.s` 统计：

| ISA family | static lexical count | type/width | provenance |
|:--|--:|:--|:--|
| `buffer_load_dwordx4` | 14 | 4 dwords = 16 bytes per issuing instruction | typed BF16 block packet，Q/K/H/V path |
| `global_load_dword` | 17 | 32-bit scalar | g FP32 path |
| `buffer_store_dwordx2` | 4 | 2 dwords per issuing instruction | packed public BF16 output backend path |
| `ds_read_u16` | 32 | LDS read family | typed operand lowering |
| `ds_read_b64` | 40 | 64-bit LDS read family | typed operand lowering |
| `ds_write_b16` | 16 | 16-bit LDS write family | selected shared path |
| `ds_write_b32` | 8 | 32-bit LDS write family | selected shared path |
| `ds_write_b64` | 8 | 64-bit LDS write family | selected shared path |
| `ds_write_b128` | 4 | 128-bit LDS write family | selected shared path |
| `s_waitcnt` | 48 | synchronization | 非 VMEM bytes |
| `s_barrier` | 11 | CTA barrier | 非 VMEM bytes |
| `v_mfma_f32_32x32x8_bf16` | 40 | MFMA32 | static lexical，动态按 PMC 计 |

这里的 `buffer_load_dwordx4=14` 是 ISA opcode width 证据，不等于每 CTA
只读 14 次，也不等于 native 总 global byte count。它说明 native 的 typed
BF16 operand 在 ISA 入口保留了更宽的 packet load；dynamic execution 仍由
wave/lane、loop 和 mask 决定。

### 5.3 Static 与 dynamic 不可直接相乘

本次必须明确禁止以下错误推导：

```text
136 static Z2 global-load lines -> 136 dynamic loads/CTA       # 错
56 global_load_ushort * 2 bytes -> 112 bytes/CTA               # 错
14 native dwordx4 * 16 bytes -> 224 bytes/CTA                  # 错
928 VMEM / 6.63 -> exact excess bytes                                # 错
```

正确表达是：

- static ISA 告诉我们编译后出现了哪些 opcode family 和 operand width；
- LLVM/source 告诉我们这些 family 的逻辑 pointer/type/phase 来源；
- PMC 告诉我们运行时每 CTA 执行了多少 VMEM instructions；
- 当前没有 per-operand transaction-byte counter，所以 exact byte split 保持
  unresolved。

## 6. Fixed Z2 vs native per-CTA logical operand ledger

下表把逻辑下界、已观察的 static family 和 provenance 置信度放在一起。
“excess type”描述的是可证明的结构类型，不是已测得的额外字节数。

| operand | unique logical bytes/CTA | fixed Z2 source phase | Z2 static family | native corresponding graph | duplicate/reuse evidence | excess type | certainty |
|:--|--:|:--|:--|:--|:--|:--|:--|
| Q | 16,384 | A inter + B score, B repeated for 2 source halves | scalar `global_load_ushort` | typed Q block, one Q dot operand reused by QH/QK in loop | Z2 duplicate across A/B and B half; native same Q value feeds two `tt.dot` | repeated global load + narrow load + scalar address/materialization | A for duplicate, C for exact dynamic share |
| K | 16,384 | B, 2 source halves x 4 K32 stages | scalar `global_load_ushort` | typed `32x64xbf16` block and dot operand | no same-stage duplicate proven in Z2; native typed block | narrow load + scalar address + scalar-to-LDS staging | A for narrow, N/A for dominance |
| H | 16,384 | A inter | scalar `global_load_ushort` | typed `64x32xbf16` block and shared memdesc | Z2 source only one full tile pass | narrow load + scalar LDS staging | A for narrow, B for total impact |
| V_new | 8,192 | C score-times-V | scalar `global_load_ushort` | typed `64x64xbf16` block, in-thread transpose | one Z2 source tile pass | narrow load + transpose/materialization | A for narrow, B for impact |
| g | 256 unique lower bound | B score target/source + final scale | scalar `global_load_dword` | two 64-element `tt.load` groups | multiple role uses; exact dynamic reuse unresolved | repeated role use + address arithmetic | A for type/use, N/A for share |
| output | 8,192 | final result loop | scalar `global_store_short_d16_hi` | typed BF16 `buffer_store` / `v2i32` packet | one logical output tile | narrow/scalar output store vs packed store | A for width, B for impact |

### 6.1 每类 operand 的“必要”和“可疑”部分

#### Q

必要的是每 CTA 至少读取一次 Q `[64,128]`。已经被证明的可疑部分是：

```text
Phase A Q producer -> inter MFMA
Phase B Q producer -> score MFMA, source_half 0
Phase B Q producer -> score MFMA, source_half 1
```

这三次 source-level pass 中，后两次与第一份 Q 的 logical tile 相同。
这是目前唯一同时拥有 source loop、LLVM pointer 和 native reuse 对照的
cross-phase duplicate 证据。

#### K

必要的是 `[64,128]` K tile。Z2 的 K path 以 `[32,32]` stage 分片，
每个 BF16 element 用 scalar global load，再写进 phase。native 同样需要
K 的 shared/dot operand，但 TTGIR 类型是 `tensor<32x64xbf16>`，最终
buffer load 是 dwordx4。这里首先应区分“scalarized bytes”和“duplicate bytes”。
本轮只能证明前者，不能证明后者或其动态占比。

#### H

必要的是一个 H tile。Z2 scalar load + LDS staging 是固定成本；没有 source
证据说明 H 在两个 phase 间重复 global read。不能把 H 的 phase reuse buffer
视为 global duplicate。

#### V_new

必要的是一次 `[64,64]` V_new tile。Z2 以 scalar BF16 load 写入 transpose
区，native 以 typed block load 后做 in-thread transpose。V_new 的窄化确定，
但由于没有第二个 source-level V pass，不能指定为最大 excess。

#### g

必要的是当前 chunk 的 64 个 g FP32 value。score 中的 target/source role
和最终输出 scaling 都使用 g。Z2 的 scalar `global_load_dword` 与复杂
address arithmetic 明确存在，但静态/动态工件没有把 928 VMEM 分解成 g
target、g source、g final 三类。因此保持 unresolved。

#### output

写一次 public BF16 output tile 是必要工作。Z2 的 scalar BF16 store 与
native packed store 说明有 store-width 差异，但 output 是 producer 端最后一
次写，不能把它误标成 input duplicate。

## 7. Address arithmetic 与 load width

### 7.1 Z2 address recipe

Z2 lowered LLVM 对每个 BF16 element 先计算 element index，再做：

```text
GEP bfloat pointer
load bfloat align 2
```

Phase A Q、Phase B Q、K、H、V_new 都遵循这一模式。final ISA 中形成大量
`v_lshl_add_u64`/地址加法后接 `global_load_ushort`。已有 static audit 的
地址 proxy 也显示：

| static address proxy | fixed Z2 | native |
|:--|--:|--:|
| `v_add` | 187 | 54 |
| shift family proxy | 约 166 | 42 |
| permute | 0 | 0 |

这些是 static lexical proxy，不能直接当动态地址数量；但它们与 Z2 的
scalar element GEP 和 928/CTA VMEM、11,400/CTA VALU 的方向一致。

### 7.2 Native address recipe

native TTGIR 先构造 tensor-wide blocked pointer tensors，再由：

```text
amdg.buffer_load tensor<64x32xbf16> / tensor<32x64xbf16>
amdg.buffer_load tensor<64x64xbf16>
-> typed shared memdesc
-> typed dot operand
```

LLVM 中对应：

```text
@llvm.amdgcn.raw.ptr.buffer.load.v4i32
-> bitcast <4 x i32> to <8 x bfloat>
```

这是“load packet 和 source layout 在高层保持 typed”的证据。它不代表
native 没有地址计算，也不代表其所有 LDS 工作都消失；它只说明 native
减少了 Z2 那种逐 BF16 element global load/GEP 的表达和执行压力。

## 8. 动态 PMC 与静态 ledger 的对应关系

PMC 原始值来自：

```text
codex_qwen_bt64_stage6z_fixed_z2_vs_native_audit/pmc3/
```

按 512 CTA 归一化：

| counter | Z2 total | native total | Z2/CTA | native/CTA | 解释 |
|:--|--:|--:|--:|--:|:--|
| MFMA | 81,920 | 81,920 | 160 | 160 | 数学工作一致 |
| VMEM | 475,136 | 71,680 | 928 | 140 | Z2 机器 global/buffer memory 指令更多 |
| LDS | 475,136 | 245,760 | 928 | 480 | Z2 phase round-trip 更多 |
| VALU | 5,836,800 | 1,728,512 | 11,400 | 3,376 | address/layout/packing 等复合成本 |
| SALU | 548,864 | 337,920 | 1,072 | 660 | loop/index/control 成本 |

这张表可以支持“Z2 做了更多机器工作”，但不能进一步支持：

```text
928 - 140 = 788 VMEM
=> 某个 operand 恰好多了 788 个 global load
```

因为 PMC counter 没有按 Q/K/H/V_new/g/output 分桶，ISA 的静态文本行又
被 loop/lane 执行复用。对 exact byte provenance，当前结论必须保留
`per-operand dynamic byte split = unresolved`。

## 9. 为什么本轮选择 Q，而不是 K 或 V_new

### Q duplicate：已经满足“最大候选”的证据门槛

- source 中有两个不同 phase 的 Q producer；
- Phase B producer 被两个 source half 重新执行；
- LLVM 有两个独立 Q GEP/load 模板，均从 kernel Q pointer `%0` 读取；
- native TTGIR 中一个 Q dot operand 同时进入 `Q*H` 和 `Q*K`；
- native loop 的 typed shared memdesc 允许同一 Q value 被两个 consumer 使用。

这能直接形成一个单变量 source experiment：保持所有 shape、ownership、
MFMA、barrier contract 不变，只让 Q 的 phase-A producer 结果跨到 Phase B
并在两个 score half 复用。它是一个明确的 producer-consumer residency
候选，而不是“把 load 往前挪一点”的泛化建议。

### K scalarization：真实但不能证明最大

- K 的 scalar `load bfloat` 和 `global_load_ushort` 已经由 LLVM/ISA 证明；
- native K 是 typed `32x64xbf16` block/dot operand；
- 但 Z2 source 没有证明 K element 重复 global load；
- 没有 per-operand PMC 证明 K 的 narrow load 贡献大于 Q duplicate、g
  role loads 或 V/H materialization。

所以本轮不凭经验选择 K。

### V_new narrow load：确定但规模较小

V_new 逻辑 tile 是 8,192 B，Z2 只有一个 source pass。即使 narrow load
会增加 VMEM instruction 数，它缺少已证明的重复 phase，因此不是当前最大
证据候选。

## 10. 唯一下一 source 修改点，只登记不实施

### Matched Q tile residency across Phase A -> Phase B

唯一登记项：

```text
保持 WG256 / BT64 / BV64 / BK32 / MFMA32 / output ABI / 数学不变，
让 Phase A 产生的 scaled-Q tile 以明确的 shared/register lifetime
跨过 inter-state MFMA，供 Phase B 的两个 source_half score consumer 复用。
```

预期结构变化：

```text
当前 Z2:
Q -> Phase-A shared -> inter MFMA
Q -> Phase-B shared -> score MFMA (half 0)
Q -> Phase-B shared -> score MFMA (half 1)

候选 source graph:
Q -> one typed/packed phase producer
  -> inter MFMA consumer
  -> score MFMA consumer half 0/1
```

本轮没有实现该 graph，也没有改变 phase buffer 或 barrier。下一轮若实施，
必须首先检查：

1. Q tile lifetime 是否引起 LDS/VGPR cliff；
2. Q 的 BF16 scaling/rounding 与原 Z2 bit-exact contract 是否一致；
3. Phase-A inter consumer 与 Phase-B score consumer 的跨-wave RAW/WAR；
4. VMEM/CTA 是否下降，且下降来自 Q reload 而不是减少 MFMA；
5. K/V_new/g 不得在同一轮同时修改，避免失去归因。

这一候选不是 Z4，不是 compiler RA patch，也不是 native HSACO bridge。

## 11. 本轮未能证明的事项

以下问题在当前工件中必须明确标记为 unresolved：

- Z2 928 dynamic VMEM 中 Q/K/H/V_new/g/output 各自的精确 instruction share；
- 每类 operand 的实际 memory transaction bytes；
- native 140 dynamic VMEM 中每类 operand 的 exact byte split；
- 由 static `global_load_ushort` 数量直接反推出的字节差；
- native 11 个 barrier 的逐一 source provenance；
- Z2 与 native 在完全相同 pre-lowering IR 下的 compiler-only A/B 结论。

这不是数据缺失被隐藏，而是当前 rocprof/ISA 工件没有 per-operand
transaction-byte counter。继续写出一个精确的 byte ledger 会变成猜测，
本报告不这样做。

## 12. 复现与审计命令

### 静态 ISA family

```bash
cd /workspace/project/avelang
rg -c 'global_load_ushort|global_load_dword|global_store_short_d16_hi|buffer_load_dwordx4|buffer_store_dwordx2|ds_read|ds_write|s_waitcnt|s_barrier|v_mfma' \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_fixed_z2_vs_native_audit/machine/z2_T2048/final_isa.s \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_fixed_z2_vs_native_audit/machine/native_T2048/final_isa.s
```

### LLVM/source provenance

```bash
nl -ba test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_fixed_z2_vs_native_audit/machine/z2_T2048/lowered_llvm.ll \
  | sed -n '155,220p;380,470p;630,680p;710,755p;895,945p'

nl -ba test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py \
  | sed -n '83,191p'

rg -n 'amdg\.buffer_load|ttg\.local_load|tt\.dot|amdg\.buffer_store|tt\.load' \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_fixed_z2_vs_native_audit/machine/native_T2048/chunk_fwd_kernel_o.ttgir
```

### Dynamic PMC

```bash
sed -n '1,240p' \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_fixed_z2_vs_native_audit/pmc3/stage6z_fixed_z2_vs_native_T2048_pmc.json
```

## 13. 审计结论状态

| item | status |
|:--|:--|
| same-shape WG256 comparison | PASS |
| dynamic MFMA alignment | PASS，160/CTA each |
| static Z2 BF16 scalar load provenance | PASS |
| native typed block load provenance | PASS |
| Z2 Phase-A/Phase-B Q duplicate | PASS，source + LLVM |
| K scalarization provenance | PASS |
| exact per-operand dynamic VMEM split | UNRESOLVED，当前没有 counter |
| exact per-operand byte split | UNRESOLVED |
| source-only next candidate selection | PASS，唯一选择 Q residency |
| kernel modification | NONE |
| Z4/X2 integration | NONE |

最终状态：**fixed Z2 仍是当前唯一 AveLang isolated baseline；本轮没有
产生新的 kernel 或性能排名。下一步只登记 Q Phase-A/Phase-B residency
实验，等待单变量 correctness/resource/PMC gate。**
