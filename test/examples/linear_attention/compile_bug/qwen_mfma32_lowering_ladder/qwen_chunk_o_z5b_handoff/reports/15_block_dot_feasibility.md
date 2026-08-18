# Qwen gfx942 BT64 Stage 6Z：BF16 Operand Feeding 与 Layout Feasibility Audit

## 0. 结论先行

本轮是只读审计：没有修改 Z5B kernel、compiler、lowering、allocator/RA、X2、
selector 或 production，也没有重新运行旧 benchmark。

审计对象：

```text
AveLang Z5B:
  vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer.py

current-vLLM native:
  codex_qwen_bt64_stage6z_native_chunko/native/T2048/pmc_capture/selected/
```

核心结论：

1. Z5B 与 native 的 dynamic MFMA 都是 `160/CTA`，所以剩余差距不是 MFMA 数量
   或 MFMA32 几何缺失。
2. Z5B 的 dedicated Q full-cache 已把 Q global producer 收敛为一次；当前问题
   不是 Z2 式三次 Q global reload。
3. Q、K、H、V-new 的数学 tile 都能在高层描述，但 Z5B 仍通过 scalar BF16
   producer、通用 shared materialization、`i32` view 和 fragment extract/insert
   进入 MFMA。native 在 TTGIR 中保留了 `blocked`、`swizzled_shared`、
   `amd_rotating_shared` 与 `dot_op` typed operand。
4. 因此缺口不是“连续 scalar store 完全不能向量化”。C0.5S 已证明连续
   producer store 会被后端合并成 `ds_write_b128`。真正的缺口是
   **producer ownership / physical shared layout 到 MFMA consumer fragment 的
   完整桥梁**。
5. 本轮还不能把它写成已经完成的“纯 compiler bug”铁证，因为 Z5B 高层表示
   本身没有 native 等价的 first-class dot-operand encoding；目前最准确的分类
   是 API/representation 与 lowering 的组合边界，同时包含真实 ownership mismatch。
6. 只登记一个下一步候选，不在本轮实现：

   **Stage 6Z-BF16-DOT：same-source typed BF16 operand late-lowering A/B。**

   该候选固定高层 ownership、shared shape、barrier、MFMA 和数学，只让
   `block_dot_bf16_f32` 的 generic lowering 与 gfx942 specialized lowering 分叉。

## 1. 冻结对象、数据和证据等级

### 1.1 Z5B contract

```text
gfx942 / wave64
BT64 / BV64 / BK32 / WG256 / 2 CTA per chunk-head
Q/K/H/V-new/output: BF16
g: FP32
accumulator: FP32
MFMA: v_mfma_f32_32x32x8_bf16
```

T=2048 已有 same-shape PMC（每 CTA）：

| 指标 | Z5B | native WG256 diagnostic |
|:--|--:|--:|
| dynamic MFMA | 160 | 160 |
| dynamic VMEM | 672 | 140 |
| dynamic LDS | 672 | 480 |
| dynamic VALU | 7,072 | 3,376 |
| dynamic SALU | 768 | 660 |
| shared/LDS allocation | 32,768 B | 24,576 B |
| private segment/spill | 0 / 0 | 0 / 0 |

`VMEM` 和 `LDS` 是动态指令类 PMC，不是字节数；static ISA lexical count、
code-object resource metadata 与 profiler PMC 也不互相替代。

### 1.2 工件

Z5B 工件：

```text
source:
  test/examples/linear_attention/vllm_compare/
  qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer.py

machine:
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
  codex_qwen_bt64_stage6z_z5b_machine_stage1/
```

Z5B HSACO SHA256 为：

```text
979889c1a10ac53064cd1d2da91298b67e4ef7f2db3baadc171872cb56b0ed67
```

Z5B static 参考：global load `192`、global store `16`、`ds_write=144`、
`ds_read=56`、lexical MFMA32 `56`、`s_barrier=32`；这些数字不被用来替代
dynamic PMC。

native selected 文件：

```text
codex_qwen_bt64_stage6z_native_chunko/native/T2048/pmc_capture/selected/
  chunk_fwd_kernel_o.ttir
  chunk_fwd_kernel_o.ttgir
  chunk_fwd_kernel_o.llir
  chunk_fwd_kernel_o.amdgcn
  chunk_fwd_kernel_o.hsaco
  chunk_fwd_kernel_o.json
```

native metadata 原样记录 `gfx942 / warp_size=64 / num_warps=2 / num_stages=3 /
shared=24576 B / profile_scratch_size=0`。已有 Stage 6Z same-shape capture
把 selected native 按 WG256 diagnostic 对齐；metadata 的 `num_warps=2` 不在本轮
被擅自改写，后续严格复现时必须同时检查 launch 与 code-object metadata。

### 1.3 证据等级

| 等级 | 含义 |
|:--|:--|
| A | source/IR/ISA 或 PMC 能闭合证明 |
| B | 能确定 operand family、方向和 source region，但不能逐条分配 dynamic PMC |
| C | 只有 lexical、地址形态或总数差值，不能做精确归因 |
| N/A | 工件抽象掉了该信息，不能诚实恢复 |

## 2. 数学 block 与 MFMA microtile

一个 CTA 固定一个 64-token chunk、一个 value head、一个 value block。数学路径：

```text
inter = Q @ transpose(H)
score = Q @ transpose(K)
intra = score @ V_new
```

逻辑 tensor：

```text
Q      [query_token=64, reduction_k=128] BF16
K      [source_token=64, reduction_k=128] BF16
H      [value=64, reduction_k=128] BF16
V_new  [source_token=64, value=64] BF16
```

每个 `k_stage=0..3` 对应 `k0=32*k_stage` 的 K32 slice：

| dot | 数学 A | 数学 B | slice |
|:--|:--|:--|:--|
| inter | `Q[64,k0:k0+32]` | `H[value_base:value_base+64,k0:k0+32]^T` | 64 行、64 列、32 reduction |
| score half s | `Q[64,k0:k0+32]` | `K[32s:32s+32,k0:k0+32]^T` | 64 行、32 source 列、32 reduction |
| intra half s | `score[64,32s:32s+32]` | `V_new[32s:32s+32,value_base:value_base+64]` | 64 行、64 value 列、32 reduction |

每次 `v_mfma_f32_32x32x8_bf16` 使用 8 个 BF16 reduction 元素；源码的 `kt`、
`word` 和 fragment 切分覆盖一个 K32 slice。数学 operand 顺序与 Z5B intrinsic
参数顺序不同：数学上 Q 是 inter/score 的左矩阵 A，但 source 调用是
`mfma(h_frag,q_frag,...)`、`mfma(k_frag,q_frag,...)`；intra 是
`mfma(v_frag,score_frag,...)`。报告中按数学 A/B 与 intrinsic ABI 分开记。

## 3. Z5B source ownership 与地址

### 3.1 CTA/wave/lane

Z5B 第 78-91 行定义：

```text
lane       = tid & 63
lane_col   = lane & 31
lane_group = lane >> 5
wave_id    = tid >> 6
row_half   = wave_id >> 1
value_half = wave_id & 1

v_block_idx = program_id % 2
value_head  = (program_id // 2) % 8
chunk_idx   = program_id // (2*8)
value_base  = v_block_idx * 64
chunk_start = chunk_idx * 64
key_head    = value_head >> 1
```

### 3.2 Global logical byte formula

Z5B source 第 44-75 行的 strides 给出：

```text
Q[t,k] = 2 * ((chunk_start+t)*512 + key_head*128 + k)
K[t,k] = 2 * ((chunk_start+t)*512 + key_head*128 + k)
H[v,k] = 2 * (chunk_idx*8*16384 + value_head*16384
             + (value_base+v)*128 + k)
V[t,v] = 2 * ((chunk_start+t)*1024 + value_head*128
             + value_base+v)
```

这是 logical element 到 global byte 的映射，不是 hardware transaction byte。

### 3.3 Q full-cache producer

第 93-98 行分配 `[512,32] BF16` shared；低 256 行是 dedicated Q cache，高 256
行是 phase 区。第 100-111 行做：

```text
for k_stage in 0..3:
  for rep in 0..7:
    idx = tid + rep*256
    row = idx // 32
    col = idx % 32
    q_cache[k_stage*64+row,col] = BF16(FP32(Q[chunk_start+row,
                                                k_stage*32+col]) * scale)
```

一个 stage 的 unique tile 是 `64*32 BF16=4096 B`，四 stage 的完整 Q cache
是 `64*128 BF16=16384 B/CTA`。lane 例子：

```text
tid=0,  rep=0 -> (row,col)=(0,0)
tid=1,  rep=0 -> (0,1)
tid=31, rep=0 -> (0,31)
tid=32, rep=0 -> (1,0)
tid=0,  rep=1 -> (8,0)
```

因此相邻 lane 能形成连续 producer 邻域，但每 lane 的 global source op 仍是
scalar BF16 load。Q 的 global duplicate 已被 Z5B 消除；Q 的 packet-to-dot
feeding 尚未等同 native。

### 3.4 H producer/inter

第 119-138 行将 `H[value_base+row,k_stage*32+col]` 写入
`phase[320+row,col]`，也是 `idx=tid+rep*256`、4 stage、8 rep。unique
logical bytes 为 `64*128*2=16384 B/CTA`。消费者以
`phase_vec[320+value_half*32+lane_col,word]` 取 word，再用
`al.view(...,(2,4,1),bf16)` 形成 `h_frag`。

### 3.5 K producer/score

第 140-165 行对两个 source-half、四个 K32 stage、四个 rep 执行：

```text
idx = tid + rep*256
row = idx // 32
col = idx % 32
phase[320+source_half*128+row,col] =
  K[chunk_start+source_half*32+row,k_stage*32+col]
```

每 source-half 每 stage 是 `32*32 BF16=2048 B`，两个 half 合计
`16384 B/CTA`。这两个 source-half 是 score 数学的必要分区，当前工件没有
证明同一 K logical tile 被无条件重复 global producer。

score consumer 的 q/k word 关系是：

```text
q_words = q_cache_vec[k_stage*64+row_half*32+lane_col,word]
k_words = phase_vec[320+source_half*128+lane_col,word]
q_frag/k_frag = view(word,(2,4,1),bf16)
```

### 3.6 V-new/intra

第 181-190 行用 `rep=0..15`：

```text
value_offset = idx // 64
token_offset = idx % 64
phase[...] = V_new[chunk_start+token_offset,value_base+value_offset]
```

unique logical tile 是 `64*64 BF16=8192 B/CTA`。例如 `tid=0` 是
`(value_offset=0,token_offset=0)`，`tid=63` 是 `(0,63)`，`tid=64` 是
`(1,0)`。intra 第 192-201 行再从 phase word/view 形成 score/V fragment。

## 4. Native TTIR/TTGIR/LLVM/ISA

### 4.1 TTIR 的直接 block 证据

native `chunk_fwd_kernel_o.ttir`：

```text
115  tt.load tensor<64x32xbf16>       Q
125  tt.load tensor<32x64xbf16>       K
129  tt.load tensor<64x32xbf16>       H
130  tt.trans H -> tensor<32x64xbf16>
131  tt.dot Q, transpose(H), inter
132  tt.dot Q, K, score
150  tt.load tensor<64xf32>           g
200  tt.load tensor<64x64xbf16>       V-new
203  trunc score FP32 -> BF16
204  tt.dot score, V-new, intra
210  tt.store BF16 output
```

这些是实际 captured tensor type，不是根据变量名猜测。

### 4.2 TTGIR encoding 与 memdesc

native `chunk_fwd_kernel_o.ttgir` 第 1-12 行定义：

```text
#blocked  sizePerThread=[4,8], threadsPerWarp=[8,8], warpsPerCTA=[2,1]
#blocked1 sizePerThread=[8,1], threadsPerWarp=[4,16], warpsPerCTA=[1,2]
#blocked2 sizePerThread=[1,8], threadsPerWarp=[16,4], warpsPerCTA=[2,1]
#shared   swizzled_shared vec=4, perPhase=2, maxPhase=8, order=[1,0]
#shared1  swizzled_shared vec=1, perPhase=1, maxPhase=1, order=[1,0]
#shared2  swizzled_shared vec=4, perPhase=2, maxPhase=8, order=[0,1]
#shared3  swizzled_shared vec=4, perPhase=1, maxPhase=16, order=[1,0]
#shared4  amd_rotating_shared vec=4, perPhase=1, maxPhase=16, order=[0,1]
#mma      amd_mfma version=3, instrShape=[32,32,8], isTransposed=true
```

TTGIR 第 124-126 行分配 Q/H/K memdesc；第 173-178 行把 blocked tensor
local_store 到 shared；第 233-254 行 local_load 为 Q/K dot operand，并做
Q*H/Q*K；第 258-277 行轮换下一组 buffer；第 401-406 行把 score/V-new
local_load 成 dot operand 并做 intra。

| native operand | input encoding | shared memdesc | consumer encoding |
|:--|:--|:--|:--|
| Q | `tensor<64x32,#blocked2>` | `2x64x32,#shared` | `dot_op opIdx=0` |
| K | `tensor<32x64,#blocked1>` | `2x32x64,#shared2` | `dot_op opIdx=1` |
| H | `tensor<64x32,#blocked2>` | `2x64x32,#shared1` | linear load -> `tt.trans` -> `dot_op1` |
| score | `tensor<64x64,#mma>` | `#shared3` | `dot_op0` |
| V-new | `tensor<64x64,#linear1>` | `#shared4 amd_rotating_shared` | `dot_op1` |

### 4.3 LLVM/ISA 的 packet evidence

native `.llir` 第 70-102 行出现 Q/K/H 的
`llvm.amdgcn.raw.ptr.buffer.load.v4i32`，bitcast 为 `<8xbf16>`；后续可见
`<4xbf16>`/`<4xi32>` shared stores。native ISA 的对应 family 包括：

```text
buffer_load_dwordx4
ds_write2st64_b64 / ds_write_b128
ds_read2_b64 / ds_read2st64_b64
v_mfma_f32_32x32x8_bf16
```

native 的 ISA 还能直接看到 load->waitcnt/barrier->LDS read->MFMA 的 block
feeding 顺序。这里的 static family count 仅用于证明 packet/encoding 形状，
140/CTA 仍只来自 PMC。

### 4.4 native lane mapping 的边界

TTGIR encoding 可恢复 per-thread logical tile extent：例如 `#blocked2` 的
每线程 extent 是 `[1,8]`、`threadsPerWarp=[16,4]`；`#blocked1` 是 `[8,1]`、
`threadsPerWarp=[4,16]`。但当前 dump 已将 `ttg.local_load -> dot_op` 后的
element map 抽象掉，不能诚实给出某一 lane 的精确 LDS byte 或某一 MFMA word
对应的 token/feature 集合。最终 ISA 的寄存器/address 也丢失原始 tensor
provenance。此字段标为 N/A，而不是按指令顺序猜测。

## 5. Q/K/H/V-new operand ledger

| operand | unique logical bytes/CTA | Z5B producer | Z5B consumer path | native path | classification |
|:--|--:|:--|:--|:--|:--|
| Q | 16,384 B | 4 stage x 8 rep，scalar BF16 load | Q cache -> i32 view -> q_frag | 64x32 packet -> swizzled shared -> dot_op0，复用 QH/QK | B+C |
| K | 16,384 B | 2 half x 4 stage x 4 rep，scalar BF16 load | phase BF16 -> i32 view -> k_frag | 32x64 packet -> shared2 -> dot_op1 | B+C |
| H | 16,384 B | 4 stage x 8 rep，scalar BF16 load | phase BF16 -> i32 view -> h_frag | 64x32 packet -> shared1 -> transpose -> dot_op1 | B+C |
| V-new | 8,192 B | 16 rep，scalar BF16 load | phase token/value map -> i32 view -> v_frag | 64x64 packet -> rotating shared -> dot_op1 | B+C |

`B` 是 scalarization/fragment reconstruction 的 representation cost；`C` 是
producer natural contiguous direction 与 MFMA consumer lane direction 不同，
需要 swizzle/transpose/typed dot encoding。`C` 不表示硬件不可行，而表示当前
公开 source ownership 和表示没有闭合。

### 5.1 Q

Z5B Q pointer 只在 Q-fill source region 出现，Q cache 被 inter 和两个 score
half 使用；这是“global residency 已成功”。剩余问题是 Q cache 到
consumer-compatible MFMA fragment 的 generic word/view bridge。native 的 Q
`tensor<64x32>` 在同一 K32 loop 中成为 dot operand。

### 5.2 K

没有证据证明两个 source-half 是无条件 duplicate producer；有证据证明
K 的 scalar row/col producer、phase address 和 `k_frag` reconstruction 比
native `#blocked1/#shared2/dot_op1` 更早展开了 operand layout。

### 5.3 H

H 是一个 64x128 unique tile。native 显式 `tt.trans`，说明 H 的转置关系被
保留在 IR。Z5B 用 phase row 与 word view 完成等价数学，但没有 first-class
transpose-to-dot encoding。无法证明两个 value-half 完整重复 H global load。

### 5.4 V-new

V-new 是 8 KiB unique tile。native TTIR 直接有 `64x64xbf16` block load，
TTGIR 使用 rotating shared；Z5B 先按 `value_offset/token_offset` scalar
写 phase，再重建 V fragment。单独改变 `ds_write` width 不能同时解决 packet
ownership、shared physical layout 和 dot consumer。

## 6. word/half 到 MFMA slot：能证明什么，不能证明什么

Z5B 能证明的链：

```text
q_words/h_words/k_words/score_words/v_words
  -> view(word,(2,4,1),bf16)
  -> fragment[0], fragment[1]
  -> mfma_32x32x8_bf16_f32
```

不能从当前 source/LLVM/MIR/ISA 恢复：

```text
(logical token, feature)
  -> exact BF16 half in word
  -> exact physical MFMA A/B register lane
```

原因是 `al.view` 没有输出 distributed encoding，LLVM 只保留 vector
extract/insert 与 MFMA intrinsic，final ISA 只保留 physical register 和
dynamic LDS address。要闭合该映射，需要 Triton layout interpreter 或在
`ttg.local_load` lowering 前导出 element map，不能用本报告伪造。

## 7. Source expressibility 与 lowering boundary

| capability | 既有证据 | 结论 |
|:--|:--|:--|
| BF16x8 raw global load | C0.5S/D0-P compile-only | 可表达 |
| contiguous packed LDS store | C0.5S 的 `ds_write_b128` | 可表达 |
| packed i32/BF16 view | Z5B source/LLVM | 可表达，但不是 dot layout |
| fixed lane exchange | D0-P probe | 局部可表达 |
| native-like distributed shared + dot operand | TTGIR 有，AveLang public source 无等价 first-class type | 缺口 |
| token-major non-contiguous gather -> typed MFMA fragment | C0.5S 留下 `builtin.unrealized_conversion_cast` | 当前不稳定 |

C0.5S 的报告 `qwen_direct_k64_source_vectorized_lds_c05s_report.md` 明确指出：
普通 source scalar-looking contiguous store 也能自动形成 `ds_write_b128`，
显式 u32 producer 没有释放额外优势。D0-P 的
`qwen_direct_k64_bv32_layout_feasibility_d0p_report.md` 明确指出：AveLang 能
发出部分 raw load/shuffle/packed LDS primitive，但完整 register-transpose
arm 在 T=64 correctness 失败，不能据此继续枚举 swizzle。

因此：

```text
连续 store vectorization:      不是主要缺口
Q/K/H/V-new typed dot feeding:  当前缺少稳定的表示/lowering桥梁
纯 compiler bug 铁证:           本轮尚未达到 same-source A/B 标准
```

## 8. 必须保留与可避免工作

必须保留：unique Q/K/H/V-new global input、Q scale/BF16 round、K32 reduction
order、H transpose math、causal mask、BF16 boundary、FP32 accumulation、真正
的跨-wave synchronization 以及 `160 dynamic MFMA/CTA`。

原则上可避免、且已有 IR/ISA 方向证据的工作：

```text
连续 BF16 block 被拆成 scalar global_load_ushort issuing paths
BF16 word -> generic i32 view -> extract/insert fragment reconstruction
producer row/col 与 consumer lane_col/word 坐标之间的重复地址 arithmetic
不能被 dot operand type 复用的 phase materialization 临时值
```

不能把这些类别逐条等同于 `672-140` 或 `7072-3376`；它们是 provenance
方向证据，不是伪造的 per-op PMC 分摊。

## 9. 四个关键假设

### A. Q reuse

逻辑上成立：Z5B Q full-cache 供 Q*H、Q*K0、Q*K1 使用。物理 feeding 不等同
native：Z5B 每个 consumer 仍走 generic i32/fragment view，native Q block
在 dot encoding 中复用。

### B. K/H common MFMA-B

数学上成立，表示上不等价。K 和 H.T 都是 inter/score 的右矩阵 B；Z5B 通过
phase rows/view 生成，native 通过 `#shared2/#shared1 + dot_op1` 生成。不能由此
声称有 duplicate global load，但可以确认 feeding path 更重。

### C. V-new packet

native 已经是 64x64 block operand；Z5B 仍是 scalar producer + phase/view。
“改一个 store width”不能闭合完整 packet-to-consumer path。

### D. phase_vec/i32 reconstruction

这是已确认的表示差异之一。Z5B LLVM 的 shared vector load、bitcast、
extract/insert 与 `v_add/v_lshl_add` address proxy 绑定到 operand feeding；
它不是唯一数学根因，不能只删一个 view 就保证 native 映射。

## 10. 唯一登记的下一候选：Stage 6Z-BF16-DOT

本轮不实现，只登记一个 compiler/representation arm：

```text
same-source generic/specialized typed BF16 operand late-lowering A/B
```

高层 source 只做最小的 logical block-dot 表达：

```python
inter = al.amdgpu.block_dot_bf16_f32(Q_block, H_block, inter_acc,
                                     layout="gfx942_mfma32")
score = al.amdgpu.block_dot_bf16_f32(Q_block, K_block, score_acc,
                                     layout="gfx942_mfma32")
intra = al.amdgpu.block_dot_bf16_f32(score_block, Vnew_block, intra_acc,
                                     layout="gfx942_mfma32")
```

两个 arm：

```text
A generic:
  block-dot -> 当前合法 generic shared/view/fragment lowering

B gfx942 specialized:
  block-dot + explicit distributed/shared/dot operand encoding
  -> typed vector packet/local load
  -> gfx942 MFMA32 operand
```

门槛不是预设资源数字，而是先证明同源：

1. pre-branch high-level MLIR hash 完全相同；
2. global IO、shared allocation、barrier、MFMA 数、K32 order 和数学相同；
3. A/B 差异只在 block-dot operand lowering；
4. B 的 typed packet/shared/dot feeding 仍出现在 LLVM/MIR/ISA，而不是退化为
   scalar BF16 load、generic i32 view 或大量 cross-lane shuffle；
5. T=64/512/2048 correctness 先通过，才测 dynamic PMC/body；
6. 继续分开记录 static ISA、dynamic PMC、logical bytes 和 transaction bytes。

如果 B 不能在同一高层 schedule 下成立，或只能靠改变 ownership/数学/同步取得
收益，则关闭“单独 block-dot lowering 足以追平 native”的假设，转向 matched
fused pipeline；本轮不预先承诺结果。

## 11. 最终判断

Z5B 已证明 AveLang 能做 BF16、MFMA32、Q residency 和部分 packed LDS；但当前
source/IR 没有把 Q/K/H/V-new 的 producer ownership、shared physical layout
和 MFMA dot operand 作为同一个 typed object 保留下来。native 的优势主要在
这条完整 representation/lowering 链，而不是某一条孤立 `ds_write` 或某个
MFMA intrinsic。

本报告支持继续做严格 compiler/representation A/B，但不支持现在就向 compiler
团队声称“纯 lowering same-source 铁证”已经完成。当前 Stage 6Z baseline 仍是
Z5B；Z6G-S/I 仍是 correctness-passing but performance-No-Go；BF16-DOT 候选
尚未实现。
