# Qwen GDN chunk-o T=8192：Z5B 与 selected Triton 的 exact ISA gap ledger

## 结论摘要

本轮是 **read-only machine-dataflow audit**。复用了已有 Q@H、Q@K、score@V
consumer-closure 结果，没有重复 correctness，没有 benchmark、rocprof、调度实验，
也没有修改 kernel、compiler、ABI、WG、layout 或 production selector。

结论按执行顺序是：

1. 两边的 MFMA opcode 都已经是
   `v_mfma_f32_32x32x8_bf16`，不存在新的 MFMA intrinsic 缺口。
2. **FIRST structural divergence** 出现在 MFMA 之前的 CTA/wave ownership 和
   producer/shared/dot-operand contract，而不是 MFMA 本身：Z5B 用 4-wave/2-CTA
   加 ordinary shared view；Triton 用 2-wave selected mapping、blocked/swizzled
   shared encoding 和 `ttg.local_load -> tt.dot`。
3. 最大的已证实机器工作差异是完整的 producer-to-consumer materialization：Z5B
   大量 scalar BF16 global/LDS staging、普通 view 和 phase barrier；native 使用
   packet load、typed shared layout 和 dot operand。它不是某一条 MFMA 指令的问题。
4. 若必须只登记一个下一步 source change，选择已有 ledger 支持的 **Z6G-S g-tile
   residency**：只把 64-token FP32 `g` tile 从 global producer 提升为一次 CTA-local
   cache，并让 score target/source/final scaling 复用它。它是当前唯一有单一 operand
   provenance、且 modeled issuing opportunity 约 192 的候选；本报告不实现它，也不
   声称它能一次消除全部 layout 差距。

## 审计边界与冻结身份

### Z5B pure AveLang

产物目录：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
  codex_qwen_bt64_stage6z_z5b_machine_t8192/
```

| 项目 | 值 |
|---|---|
| kernel | `_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z5b_direct_q_cache` |
| target | gfx942 |
| contract | BT64/BV64/BK32，2 CTA/chunk-head |
| workgroup | 256 threads，4 waves |
| HSACO SHA256 | `979889c1a10ac53064cd1d2da91298b67e4ef7f2db3baadc171872cb56b0ed67` |
| final ISA SHA256 | `2daa24b11de95dc744c7b9bc0f596b6541e52b4d36446cf4a507ce3bf941915` |
| lowered LLVM SHA256 | `1a88164cbf264d9c8c7fa876c22c2f8909f2148dcec25721489ea65d99fbf29c` |
| code-object VGPR/AGPR/SGPR | 104 / 32 / 28 |
| LDS | 32768 B |
| private/spill | 0 B / 0 |

### selected Triton/native

产物目录：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
  codex_qwen_bt64_stage6z_native_chunko/native/T8192/selected/
```

它由 T=8192 public API capture 实际选出，不是用 T=2048 或 T=8192 的旧表格推断。

| 项目 | 值 |
|---|---|
| kernel | `chunk_fwd_kernel_o` |
| target | gfx942，wave64 |
| contract | BT64/BV64/BK32 |
| workgroup | 128 threads，2 waves |
| stages | 2 |
| MFMA | `v_mfma_f32_32x32x8_bf16` |
| HSACO SHA256 | `e201dd58c83e64565f10754789ac294df53343b9bbbe764e8f667817066ee5` |
| final ISA SHA256 | `00f0647036be4632891e48ecea3b937357e4814aace57c84f4f127ef94614df5` |
| LLVM SHA256 | `9c2d3a9c142c6c8dcd4eb965abf12dfcb0d2d84aab39c1736ff1ea1aac9001c9` |
| JSON shared field | 12288 B |
| ELF note fixed LDS | 0 B；与 JSON 字段不一致，不能合并解释 |
| ELF VGPR/AGPR/SGPR | 220 / 64 / 76 |
| private/spill | 0 B / 0 |

native 的 `num_ctas=1` 是 Triton module 的 cluster/launch metadata，不能当作逻辑
CTA 数。逻辑 ownership 取自实际 `program_id` 公式和 host grid 映射。

## Ownership：第一处结构差异

### Z5B

source：

```text
test/examples/linear_attention/vllm_compare/
  qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer.py:78-111
```

核心映射是：

```text
v_block_idx = program_id % 2
value_head = (program_id // 2) % H_V
chunk = program_id // (2 * H_V)
workgroup = 256
wave_id = tid // 64
row_half = wave_id // 2
value_half = wave_id % 2
```

因此一个 logical chunk/value-head 被拆成两个 CTA；每个 CTA 处理一个 BV64 block，
内部四个 wave 再用 `row_half/value_half` 组织 output ownership。Q cache 和 phase
区是同一个 32768 B shared allocation 的不同 row band。

### native

selected `.source`/TTGIR 的 program mapping 使用：

```text
z_block = program_id_z
head_or_block = z_block % 8
group = z_block // 8
chunk = group * ceil(T / 64) + program_id_y
```

Q/K/H/V/output 的 pointer base 由这些 program id 和 `T` 计算得到。selected TTGIR
还明确给出：

```text
#mma: warpsPerCTA = [1, 2], instrShape = [32, 32, 8], isTransposed = true
Q/H: #blocked2, sizePerThread=[1,8], threadsPerWarp=[16,4]
K:   #blocked1, sizePerThread=[8,1], threadsPerWarp=[4,16]
```

也就是说 native 的 first-class dot ownership 不是由 Z5B 的 `wave_id/value_half`
分支手工重建的。两边逻辑上都覆盖对应的 output block，但不是同一个 per-wave、
per-lane physical contract。

## Whole-kernel static ISA ledger

以下全部是 final ISA 的 lexical count，不是动态硬件 transaction，也不是 latency。
`global_load` 与 `buffer_load` 是不同 opcode family，不能相加后当作字节数。

| static family | Z5B | native selected | 证据 |
|---|---:|---:|---|
| `v_mfma_f32_32x32x8_bf16` | 56 | 80 | final ISA |
| `global_load*` | 192 | 18 | final ISA |
| `buffer_load*` | 6 | 28 | final ISA |
| `global_store*` | 16 | 0 | final ISA |
| `buffer_store*` | 6 | 8 | final ISA |
| `ds_write*` | 144 | 36 | final ISA |
| `ds_read*` | 56 | 56 | final ISA |
| `v_perm_b32` | 0 | 48 | final ISA |
| `s_barrier` | 32 | 11 | final ISA |
| `s_waitcnt` | 214 | 63 | final ISA |

不能从 56 vs 80 推断 Z5B 动态 MFMA 少做了工作；两边的 wave/CTA ownership 和
unrolling 不同。本轮没有采集动态 PMC，因此不把任何 static count 写成动态 count。

## 三个 consumer 的 machine-dataflow ledger

### Q@H

逻辑形状都是 `Q[64,32] x H[32,64] -> FP32[64,64]`。

| 链路 | Z5B | native selected |
|---|---|---|
| global producer | Q 在前置 Q-cache fill 中用 `global_load_ushort`；H 在 Phase A 用 `global_load_ushort` | Q/H producer 区使用 `buffer_load_dwordx4` packet |
| shared write | Q/H 进入普通 phase/Q-cache row，用 `ds_write_b16`；Q cache 后续不再从 Q pointer reload | Q/H 使用 swizzled shared descriptor 与 `ds_write2st64_b64`/`ds_write_b128` 等宽路径 |
| shared read | `ds_read_b128` 读取 ordinary view，再由 `al.view` 解释为 BF16 fragment | `ttg.local_load` 直接产出 `#ttg.dot_op` operand |
| permutation | 没有 `v_perm_b32`；重排由索引、view 和 LDS 地址承担 | Q/H 相关的 layout/transpose 由 shared/dot encoding 表达；native 全 kernel 有 `v_perm_b32` |
| static MFMA | 32 条全 kernel lexical MFMA 位于 Q@H window | 32 条 Q@H MFMA window |
| barrier/wait | 受 Z5B phase loop 影响；全 kernel 32/214 | 全 kernel 11/63，phase 精确归属需结合 scheduled ISA |

Z5B 代表性链，见 `z5b_machine_t8192/final_isa.s:882-925`：

```text
H global_load_ushort
  -> ds_write_b16 offset:20480/22528
  -> s_barrier
  -> ds_read_b128 offset:20480/0/20512/32
  -> v_mfma_f32_32x32x8_bf16
```

Q 来自前置 Q cache，其 producer 在 `final_isa.s:36-97` 同样是窄 global load +
`ds_write_b16`。这说明 Q@H 的主要结构差异在 MFMA 前的 materialization，不在
MFMA opcode。

### Q@K

逻辑形状都是 `Q[64,32] x K[32,64] -> FP32[64,64]`。

| 链路 | Z5B | native selected |
|---|---|---|
| global producer | 每个 `source_half`/K32 stage 从 K pointer 读 `global_load_ushort` | K producer 使用 `buffer_load_dwordx4`，TTGIR 类型为 `tensor<32x64xbf16>` |
| shared write | K 写入 phase row，`ds_write_b16`；代表性 offset `20480/22528` | K 写入 `#shared2`，其 physical encoding 为 `vec=4, perPhase=2, maxPhase=8, order=[0,1]` |
| shared read | Q cache/K phase 都经 `ds_read_b128`，再做动态 word/view 组装 | `ttg.local_load` 直接形成 `#ttg.dot_op<opIdx=1,kWidth=4>` |
| permutation | `v_perm_b32=0`，使用 word/row address 公式 | native 的全 kernel permutation 由 shared/dot path 分担，不能把全局 48 条全部归给 Q@K |
| static MFMA | 16 条 Q@K window；只在对应 `value_half` 参与 | 32 条 Q@K window |
| barrier/wait | K source-half/stage 的 phase sync；全 kernel 32/214 | 全 kernel 11/63 |

Z5B 代表性链见 `final_isa.s:926-1004`：

```text
K global_load_ushort
  -> ds_write_b16 phase row
  -> ds_read_b128 offset:20480/4096/20512/4128
  -> v_mfma_f32_32x32x8_bf16
```

这里可以确认 K 的 producer-to-consumer 路径不是 native 的 typed local-load dot
operand。不能仅凭 `global_load_ushort` 的条数把所有差异归咎为“重复 K”；同一
source-half/stage 的逻辑 K 工作与 ordinary phase materialization 必须分开审计。

### score@V

逻辑形状都是 `score[64,64] x V[64,64] -> FP32[64,64]`。

| 链路 | Z5B | native selected |
|---|---|---|
| global producer | V 使用窄 `global_load_ushort`；g 还以 `global_load_dword` 参与 score/final scaling | V 使用 typed `buffer_load_dwordx4`，score/V 分别进入 shared3/shared4 |
| shared write | V 写 phase rows，典型 offset `24576/26624/28672/30720`，均为 `ds_write_b16` | V 经过 `tt.in_thread_transpose` 后写 rotating shared，ISA 可见 `ds_write_b64`/`ds_write2st64_b64` |
| shared read | barrier 后 `ds_read_b128`，典型 offset `24576/16384/24608/16416` | `ds_read_b64`/`ds_read2st64_b64` 形成 dot operands |
| permutation | Z5B 没有 `v_perm_b32`，依赖 ordinary view/index | native score/V window 有对应 `v_perm_b32`，whole-kernel 共 48 条 |
| static MFMA | 8 条 score@V window（4-wave/2-CTA 分摊） | 16 条 score@V window |
| output | Z5B 有 16 条 `global_store_short_d16_hi` | native whole kernel 是 `buffer_store*` 8，未见 `global_store*` |
| barrier/wait | score/V phase 代表性链含 `s_barrier`，全 kernel 32/214 | score/V window 使用 native 的 11/63 全 kernel同步预算 |

Z5B 代表性链见 `final_isa.s:2813-3041`：

```text
V global_load_ushort
  -> ds_write_b16 offset:24576/26624/28672/30720
  -> s_barrier
  -> ds_read_b128 offset:24576/16384/24608/16416
  -> v_mfma_f32_32x32x8_bf16
  -> BF16 global_store_short_d16_hi
```

native TTGIR 的直接证据是：

```text
chunk_fwd_kernel_o.ttgir:354  amdg.buffer_load V
chunk_fwd_kernel_o.ttgir:358-359  score shared3 -> local_load dot_op
chunk_fwd_kernel_o.ttgir:360-362  V in_thread_transpose -> shared4 -> local_load dot_op
chunk_fwd_kernel_o.ttgir:363  score@V tt.dot
```

native ISA 的对应窗口在 `.amdgcn:1892-2077`。它的 V transpose 是显式且 typed 的；
这不是“native 多了无意义 permutation”，而是 native physical operand contract 的
一部分。是否值得由 AveLang 用不同方式表达，属于后续实验，当前不改。

## Backtrace 与 offset 对照

### Z5B effective offset

Z5B source 的 shared logical row 是：

```text
Q cache: k_stage * 64 + row
H:       256 + 64 + row
K:       256 + source_half * 128 + 64 + row
score:   256 + token_offset * 2 + source_half
V:       256 + 128 + value_offset * 2 + token_offset // 32
```

BF16 byte offset 是 logical element offset 乘 2。消费者另有：

```text
word = kt * 2 + lane_group
q row = k_stage * 64 + row_half * 32 + lane_col
```

这套 path 先把数据写入普通 shared element layout，再用 `i32 view`/word arithmetic
重建 MFMA fragment。

### native effective layout

native TTGIR 的 shared encoding 是：

```text
Q: #swizzled_shared<vec=4, perPhase=2, maxPhase=8, order=[1,0]>
H: #swizzled_shared<vec=1, perPhase=1, maxPhase=1, order=[1,0]>
K: #swizzled_shared<vec=4, perPhase=2, maxPhase=8, order=[0,1]>
score: #swizzled_shared<vec=4, perPhase=1, maxPhase=16, order=[1,0]>
V: #amd_rotating_shared<vec=4, perPhase=1, maxPhase=16, order=[0,1]>
```

因此 native 的 offset 不是一条可直接替换 Z5B `row*32+col` 的 scalar 公式，而是
由 distributed ownership、swizzle/rotating encoding 和 `local_load` 的 dot operand
encoding 联合决定。代表性 native ISA：

```text
Q/K/H producer: chunk_fwd_kernel_o.amdgcn:162-235
score/V LDS offsets: chunk_fwd_kernel_o.amdgcn:1892-2077
```

## 资源与同步证据边界

### Code-object resources

| 资源 | Z5B | native selected | 说明 |
|---|---:|---:|---|
| VGPR | 104 | 220 | ELF metadata |
| AGPR | 32 | 64 | ELF metadata |
| SGPR | 28 | 76 | ELF metadata |
| LDS fixed field | 32768 B | 0 B | Z5B code note / native ELF note |
| native JSON `shared` | N/A | 12288 B | Triton selected JSON |
| private segment | 0 B | 0 B | ELF/metadata |
| VGPR spill | 0 | 0 | ELF/metadata |
| SGPR spill | 0 | 0 | ELF/metadata |

native 的 `shared=12288` 与 ELF `.group_segment_fixed_size=0` 是两个来源的不同
字段；本报告保留原始值，不把其中一个强行改写成另一个。两边寄存器数量也不能
脱离 workgroup/ownership 直接比较快慢；这里仅作为机器身份和资源证据。

### Barrier/waitcnt

精确可复核的 whole-kernel static count 是：

```text
Z5B:    s_barrier=32, s_waitcnt=214
native: s_barrier=11, s_waitcnt=63
```

source debug location 在 LLVM/ISA scheduling 后不能把每一个 waitcnt 唯一归属于
某一个 logical consumer。因此本轮只报告 whole-kernel exact count 和上述 phase
代表性链，不把同步差异伪造成 Q@H/Q@K/score@V 的精确动态分摊。

## Parity matrix

| contract | 结果 | 证据 |
|---|---|---|
| A. MFMA opcode/dtype | **PASS** | 两边均为 exact `v_mfma_f32_32x32x8_bf16` |
| B. MFMA count | **NOT PARITY / dynamic unmeasured** | static Z5B=56、native=80；ownership/unrolling不同 |
| C. wave/output ownership | **FAIL** | Z5B WG256/4 waves/2 CTA；native WG128/2 waves，grid由 program mapping决定 |
| D. lane-level fragment mapping | **FAIL** | Z5B ordinary view/word arithmetic；native blocked/shared/dot encoding |
| E. operand orientation | **PARTIAL / NOT PARITY** | native H/V 有 transpose contract；Z5B 用手工 view/index path |

已有 Q@H、Q@K、score@V isolated oracle 只证明 AveLang 能正确发出 MFMA32 并通过
对应的 fragment contract；它们没有证明完整 Z5B 与 native 的 lane-level machine
identity，也没有替代本轮的 full-kernel ISA ledger。

## 唯一下一步登记项

只登记一个 source-level candidate，不在本轮实现：

```text
Z6G-S stable g-tile residency
```

范围必须严格限定为：

1. 从当前 Z5B 分叉；
2. 当前 CTA/chunk/head 只把 64-token FP32 `g` tile 从 global producer 读一次；
3. score target、score source 和 final scaling 均从同一 CTA-local cache 取值；
4. Q/K/H/V-new/output、WG、MFMA geometry、causal mask、数学顺序和 ABI 全部不变；
5. 不同时改 typed global load、LDS layout、barrier、schedule 或 selector。

选择依据不是静态总数，而是上一份 remaining-VMEM ledger：g 在 source/LLVM 中有
多个可确认的 consumer role，modeled issuing opportunity 约 192，是当前唯一能
单独归属、且同时可能减少 global address/index VALU 的剩余 operand。这个 192 不是
PMC 672 的 exact share，下一轮仍须以 correctness 后的 fresh-process latency 和
dynamic PMC 验证。

本轮到此停止，不实施 Z6G-S，不运行性能，不接入 X2 或 production。
