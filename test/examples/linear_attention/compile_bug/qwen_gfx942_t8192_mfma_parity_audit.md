# Qwen GDN chunk-o T=8192：Z5B 与实际选中 Triton kernel 的 MFMA parity audit

## 1. 最终审计结论（Q@H、Q@K、score@V）

本轮已经完成 reuse-first 的三个 consumer contract closure。Q@H 的既有强化
oracle保持通过；本轮新增 Q@K 和 score@V，均在指定 Docker 中重新编译并运行了
五组区分方向的 oracle。没有运行 benchmark、rocprof 性能采集、full chunk-o
集成、X2 接入或指令调度实验。

最终状态：**C. Q@K 和 score@V 均通过，三个 MFMA consumer contract 已闭合。**

已证明：

- Q@H、Q@K、score@V 都能使用 `v_mfma_f32_32x32x8_bf16`，没有新增硬件
  MFMA intrinsic 的缺口；
- Q@K 的 Q `[64,32]` × K `[32,64]`、score@V 的 score `[64,64]` × V `[64,64]`
  的 BF16 输入、FP32 accumulator、两 wave、双 32-column output half 均通过；
- 每个 probe 的 debug BF16 producer/readback、raw FP32 fragment、finite 检查均
  `max_abs=0`；
- Q@K 和 score@V 的独立 accumulator 链没有被错误合并；score@V 的 V 侧在
  AveLang 中实际出现了对应的 permutation/packed LDS consumer path；
- Q@H、Q@K、score@V 的实验 HSACO 均为 private segment=0、无 VGPR/SGPR spill。

尚未证明：

- 这些 isolated probe 的 lane/register 分配已经与完整 native Triton ISA 逐条
  byte-for-byte 相同；
- Z5B full chunk-o 与 native 的动态 MFMA/VMEM/LDS/VALU、waitcnt/barrier 或延迟
  已经 parity；本轮严格没有 benchmark；
- 因此不能把 contract closure 写成性能 parity，也不能据此修改 production/X2。

下一个允许的任务不是立刻调度，而是按固定顺序做 exact ISA gap ledger：

```text
global/buffer load
 -> ds_write
 -> ds_read
 -> permutation
 -> effective offset
 -> barrier/waitcnt
 -> register/spill
 -> scheduling
```

必须先把前三个 consumer 的机器工作拆开后，才允许选择一个唯一的调度或布局控制杆。

---

## 2. 冻结的两个 T=8192 产物

### 2.1 Z5B：纯 AveLang

Z5B 是在当前 gfx942 Docker 中重新执行 compile-only capture 得到的 T=8192 产物。
命令使用 `--skip-initial-mlir`，没有 launch kernel，没有运行 rocprof，也没有计时。

根目录：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
codex_qwen_bt64_stage6z_z5b_machine_t8192/
```

kernel symbol：

```text
_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z5b_direct_q_cache
```

冻结身份：

| 项目 | 值 |
|---|---|
| target | gfx942 |
| T | 8192 |
| workgroup | 256，4 waves |
| Z5B contract | BT64/BV64/BK32，2 CTA/chunk-head |
| MFMA | `v_mfma_f32_32x32x8_bf16` |
| HSACO SHA256 | `979889c1a10ac53064cd1d2da91298b67e4ef7f2db3baadc171872cb56b0ed67` |
| final ISA SHA256 | `2daa24b11bde95dc744c7b9bc0f596b6541e52b4d36446cf4a507ce3bf941915` |
| lowered LLVM SHA256 | `1a88164cbf264d9c8c7fa876c22c2f8909f2148dcec25721489ea65d99fbf29c` |
| code-object VGPR/AGPR/SGPR | 104 / 32 / 28 |
| LDS | 32768 B |
| private segment | 0 B |
| spill | 0 |
| compile-only | `launch_executed=false`, `rocprof_executed=false` |

机器证据文件：

```text
codex_qwen_bt64_stage6z_z5b_machine_t8192/
  lowered_llvm.ll
  pre_lto_amdgcn.s
  exact_lto/
  final_isa.s
  code_object_notes.txt
  machine_evidence.json
  z5b_fixed.hsaco
```

### 2.2 Triton/native：public API 实际选中版本

Triton 不是从 T=2048 或 T=8192 的旧表格推断，而是从 T=8192 public API capture
中实际选出的 `chunk_fwd_kernel_o`。选择证据在：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
codex_qwen_bt64_stage6z_native_chunko/native/T8192/native_capture.json
```

该文件记录：`BK=32`、`BV=64`、`num_warps=2`、`num_stages=2`、`num_ctas=1`，并且
`selection_match.match_score=12`、`loop_bk_match=true`、`v_load_bv_match=true`。
因此本报告比较的是实际 selected kernel，不是手工挑选的 Triton cache。

selected 目录：

```text
.../codex_qwen_bt64_stage6z_native_chunko/native/T8192/selected/
```

冻结身份：

| 项目 | 值 |
|---|---|
| kernel | `chunk_fwd_kernel_o` |
| target | gfx942，wave64 |
| workgroup | 128，2 waves |
| BT/BV/BK | 64 / 64 / 32 |
| stages | 2 |
| MFMA | `v_mfma_f32_32x32x8_bf16` |
| HSACO SHA256 | `e201dd58c83e64565f10754789ac294df53343b9bbbe764e8f667817066ee5` |
| ISA SHA256 | `00f0647036be4632891e48ecea3b937357e4814aace57c84f4f127ef94614df5` |
| LLVM SHA256 | `9c2d3a9c142c6c8dcd4eb965abf12dfcb0d2d84aab39c1736ff1ea1aac9001c9` |
| selected shared metadata | 12288 B |
| private segment | 0 B |
| Triton | 3.6.0 + ROCm 7.2.2，commit `4ed88892` |

---

## 3. 高层逻辑是否相同

两边都包含同样的三个逻辑矩阵乘阶段：

| 阶段 | 逻辑矩阵 |
|---|---|
| Q@H | Q `[64,32]` × H `[32,64]` -> FP32 `[64,64]` |
| Q@K | Q `[64,32]` × K `[32,64]` -> FP32 score `[64,64]` |
| score@V | score `[64,64]` × V `[64,64]` -> FP32 `[64,64]` |

Triton TTIR 中可以直接看到 3 个 `tt.dot`：

```text
chunk_fwd_kernel_o.ttir:131-132   Q@H, Q@K
chunk_fwd_kernel_o.ttir:204       score@V
```

Triton TTGIR 中的对应形状是：

```text
Q/H: tensor<64x32xbf16> * tensor<32x64xbf16>
V:   tensor<64x64xbf16>
```

Z5B 的 AveLang source 没有一个高层 `tt.dot` 等价的 first-class dot operand；它在
下面这些位置手工构造 MFMA 输入：

```text
qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer.py:116-138  Q@H
...:140-179  Q@K
...:181-201  score@V
```

因此，**数学阶段是对应的**，但“一个逻辑 block 如何分给 lane、如何写入 shared、
如何 local-load 成 MFMA fragment”不是同一份高层 contract。

---

## 4. MFMA opcode、dtype 与静态数量

### 4.1 Opcode 与 dtype

两边 final ISA 都是：

```text
v_mfma_f32_32x32x8_bf16
```

Z5B lowered LLVM 中使用：

```text
llvm.amdgcn.mfma.f32.32x32x8bf16.1k
```

Z5B 的 BF16 fragment 在 LLVM wrapper 中可能先表现为 `<4 x bfloat>`，再 bitcast 成
`<4 x i16>` 传给 AMDGPU intrinsic。这是表示转换，不是换成了另一种 MFMA。

Triton TTIR 有 `inputPrecision = tf32` 属性，但 selected ISA 仍然是 BF16 MFMA；这
个属性不能解释成“需要新增 TF32 MFMA intrinsic”。

**结论：没有 intrinsic 缺口。** AveLang 已经能够生成精确的 gfx942 MFMA32 opcode。

### 4.2 静态 MFMA 数量不是字面相等

| 计数口径 | Z5B | Triton/native |
|---|---:|---:|
| TTIR logical dot op | source-level 手工 MFMA blocks | 3 个 `tt.dot` |
| final ISA static MFMA32 | 56 | 80 |
| final ISA MFMA16 | 0 | 0 |
| 本轮 dynamic PMC | 未采集 | 未采集 |

56 与 80 不能被简单写成“Z5B 少做了 24 个 MFMA，所以更快”。两边的 WG、wave
ownership、loop unrolling 和 dot lowering 都不同，static lexical count 不是 dynamic
work，也不是结果 tile 数量。此前 Stage 6Z 的动态 PMC 只作为历史背景，**没有在本轮
重新采集，不能用来替代本轮证据**。

本轮能确认的是：两边都使用 MFMA32，不能确认两边的 dynamic MFMA/CTA 在 T=8192
下通过本轮 audit 达到数值 parity。

---

## 5. CTA、wave 与输出 ownership

这是第一个可直接确认的物理差异。

### Z5B ownership

Z5B source `:78-91` 明确写出：

```text
lane       = tid & 63
lane_col   = lane & 31
lane_group = lane >> 5
wave_id    = tid >> 6
row_half   = wave_id >> 1
value_half = wave_id & 1
v_block_idx = program_id % 2
```

所以 Z5B 是：

- 一个 workgroup 4 waves，即 256 threads；
- `wave 0/1` 负责 `row_half=0`，`wave 2/3` 负责 `row_half=1`；
- 相邻 wave 通过 `value_half` 区分 value 的两个 32-wide half；
- `program_id % 2` 把一个 64-wide value block 拆成两个 CTA；
- Q cache 是 `[512,32] BF16`，其中完整 Q cache 占前 256 行，phase 区域占后 256 行，
  总 LDS 32768 B。

### Triton/native ownership

T=8192 selected TTGIR 的 encoding 是：

```text
#mma = #ttg.amd_mfma<
  {version = 3, warpsPerCTA = [1, 2],
   instrShape = [32, 32, 8], isTransposed = true}>
```

并且：

```text
#blocked2: sizePerThread=[1,8], threadsPerWarp=[16,4], warpsPerCTA=[2,1]
#blocked1: sizePerThread=[8,1], threadsPerWarp=[4,16], warpsPerCTA=[1,2]
```

对应的实际 launch 是 128 threads、2 waves、1 CTA。Q/H operand 沿一个 logical 维度
分布，K operand 沿另一个 logical 维度分布；dot operand 由 `#mma` encoding 定义，
而不是由 source 中的 `wave_id/value_half` 分支定义。

### 是否拥有同一个 32x32 tile

不能这样说。两边最终都覆盖相同的逻辑输出区域，但：

| 检查项 | 结果 |
|---|---|
| logical output shape | 相同，都是对应的 64x64 结果块 |
| waves/CTA | 不同：Z5B=4，native=2 |
| CTA/chunk-head ownership | 不同：Z5B=2 CTA 拆 value block，native selected=1 CTA |
| per-wave tile identity | 不同，Z5B 由 `row_half/value_half` 组织，native 由 blocked/MMA encoding 组织 |
| 逐 lane 的输出寄存器/fragment 对应 | 尚未 parity，不能声称相同 |

因此 parity 项 C（wave/output ownership）为 **FAIL**，不是 MFMA opcode 问题。

---

## 6. lane-level operand mapping

### 6.1 Native：typed blocked/shared/dot operand

Triton TTGIR 的关键 local allocation：

```text
line 122: Q  !ttg.memdesc<1x64x32xbf16, #shared,  #smem, mutable>
line 123: H  !ttg.memdesc<1x64x32xbf16, #shared1, #smem, mutable>
line 124: K  !ttg.memdesc<1x32x64xbf16, #shared2, #smem, mutable>
```

steady loop 中：

```text
line 195-196: Q buffer_load -> local_load -> dot operand opIdx=0
line 206-207: K buffer_load -> local_load -> dot operand opIdx=1
line 213-215: H buffer_load -> local_load -> tt.trans -> dot operand
line 216:      Q@H
line 217:      Q@K
```

score@V 的路径在：

```text
line 354: V buffer_load
line 358-362: score/V local allocation and local_load
line 363: score@V tt.dot
```

已有 native mapping oracle `stage6z_c16_qhk_native_mapping.json` 给出了对应的 blocked
映射。核心特征是：

```text
Q logical [64,32]:
  row = lane_high*4 + lane_low
  col = packet_slot*4 + element
  lane_high = row % 16
  lane_low  = (col // 8) % 4

Q shared physical element:
  row*32 + ((((col//4) xor
    (((row>>1)&7) xor ((row>>4)&7))) << 2) + (col&3))

K logical [32,64]:
  distributed #blocked1, shared2 vec4
  shared logical coordinate = [col,row]
  shared offset = col*32 + row
```

T=8192 selected artifact 的具体 `warpsPerCTA` 是 `[1,2]`，所以本报告只把该 JSON
作为 mapping/formula oracle，不把 oracle 中其他 T 的 hash 或 wave 数冒充成 T=8192
selected artifact 的 identity。

### 6.2 Z5B：普通 shared view 与手工 word fragment

Z5B Q cache producer：

```text
source :100-111
idx    = tid + rep*WORKGROUP
row    = idx // BK
col    = idx - row*BK
q_cache[k_stage*BT + row, col] = BF16(F32(q)*scale)
```

Q@H consumer：

```text
source :130-137
word    = kt*2 + lane_group
q_words = q_cache_vec[k_stage*BT + row_half*32 + lane_col, word]
h_words = phase_vec[... + value_half*32 + lane_col, word]
q_frag  = view(q_words, Tensor((2,4,1), bf16))
h_frag  = view(h_words, Tensor((2,4,1), bf16))
MFMA(h_frag[0], q_frag[0]), MFMA(h_frag[1], q_frag[1])
```

Q@K 使用相同 Q cache，但 K 由 source-half/stage 重新写入 phase 区域：

```text
source :143-165
phase[Q_CACHE_ROWS + score_stage_base + 64 + row, col] = K[...]
q_words = q_cache_vec[k_stage*BT + row_half*32 + lane_col, word]
k_words = phase_vec[... + lane_col, word]
```

score@V 使用 phase 中的 score/V word view：

```text
source :193-201
score_words = phase_vec[... score ...]
v_words     = phase_vec[... V ...]
view -> MFMA
```

这不是 native 的 `local_load -> dot_op` contract，而是：

```text
global scalar BF16 load
  -> ordinary shared element store
  -> i32 view
  -> word index arithmetic
  -> BF16 fragment view
  -> MFMA
```

### 6.3 代表性 ISA 对照

Z5B T=8192 的 Q/H 代表性序列在 `final_isa.s:910-925`：

```text
ds_read_b128 v[56:59], ... offset:20480
ds_read_b128 v[60:63], ...
ds_read_b128 v[64:67], ... offset:20512
ds_read_b128 v[68:71], ... offset:32
v_mfma_f32_32x32x8_bf16 a[0:15], v[56:57], v[60:61], 0
v_mfma_f32_32x32x8_bf16 a[0:15], v[58:59], v[62:63], a[0:15]
```

但其 producer 路径仍大量出现 `global_load_ushort` 和 `ds_write_b16`；静态统计为：

```text
global_load 192
global_store 16
ds_write 144
ds_read 56
s_barrier 32
v_mfma32 56
v_perm_b32 0
```

native T=8192 代表性 Q/H 序列在 `chunk_fwd_kernel_o.amdgcn:162-235`：

```text
buffer_load_dwordx4 ...
ds_write2st64_b64 ...
ds_write_b128 ...
s_barrier
ds_read2_b64 ...
ds_read2st64_b64 ...
v_mfma_f32_32x32x8_bf16 a[0:15], v[12:13], v[16:17], 0
```

native 的代表性 static summary 是：

```text
buffer_load 28
global_load 18
buffer_store 8
global_store 0
ds_write 36
ds_read 56
s_barrier 11
mfma32 80
v_perm_b32 48
v_permlane 0
```

注意：`buffer_load` 和 `global_load` 是不同的文本统计 family，不能把它们直接相加
当作字节数；上述值全部是 static lexical instruction count，不是动态 transaction
数，也不是本轮 benchmark 结果。

---

## 7. 三个阶段的回溯链

### 7.1 Q@H

| 层次 | Z5B | Triton/native |
|---|---|---|
| global producer | `q[...]` scalar BF16 load，`source:100-109` | `amdg.buffer_load`，TTGIR `:142`，blocked Q `[64,32]` |
| shared write | Q cache/phase 的 element store，ISA `ds_write_b16` | `ds_write2st64_b64` / `ds_write_b128` |
| shared read | ordinary `phase_vec/q_cache_vec` view，ISA `ds_read_b128` 等 | `ttg.local_load` 到 dot operand，ISA `ds_read2*` |
| operand assembly | `al.view(... Tensor((2,4,1), bf16))` | `#ttg.dot_op`, `kWidth=4` |
| MFMA | `v_mfma...bf16` | 同 opcode |

Z5B 的有效 Q cache logical address 是：

```text
cache_row = k_stage*64 + row
row = (tid + rep*256) // 32
col = (tid + rep*256) % 32
```

consumer 再使用：

```text
cache_row = k_stage*64 + row_half*32 + lane_col
word = kt*2 + lane_group
```

native 对应的 shared 物理地址由 blocked/swizzled encoding 给出，已有 LLVM 片段显示
Q packet base 类似：

```text
base = ((tid << 4) & 4080) xor (tid & 56)
second_packet = base xor 8
```

consumer 侧再以 `base xor 16/32/48` 选择 MFMA packet。两者公式明显不同。

### 7.2 Q@K

Z5B：

```text
K producer: source:146-153
K phase row: Q_CACHE_ROWS + source_half*128 + 64 + row
K consumer word: source:156-164
```

这条路径每个 source-half/stage 重新把 K 写入普通 phase 区，然后通过 `i32` view
取出 fragment。native 则用独立的 K `[32,64]` blocked/shared2 descriptor，
`local_load -> dot_op<opIdx=1>`，并在 LLVM 中使用固定的 lane/packet 地址公式。

### 7.3 score@V

Z5B：

```text
V producer: source:182-190
score/V view: source:193-201
ISA representative: final_isa.s:2980-3021
```

native：

```text
V global buffer load: TTGIR:354
score local load: TTGIR:358-359
V local load: TTGIR:361-362
score@V dot: TTGIR:363
ISA representative region: approximately 1959-2077
```

native score@V 也不是通过 Z5B 的 `Tensor((2,4,1), bf16)` ordinary view 形成；它有
独立的 `#mma`/`#ttg.dot_op` layout。两者 opcode 相同，operand lane contract 不同。

---

## 8. 历史 full-kernel parity matrix（不覆盖本轮 consumer closure）

| parity 项 | 结论 | 证据 |
|---|---|---|
| A. opcode | **PASS** | 两边都是 `v_mfma_f32_32x32x8_bf16`；Z5B LLVM 有对应 AMDGPU intrinsic |
| B. count | **NOT PROVEN / static mismatch** | Z5B static 56，native static 80；dynamic count 本轮未采集 |
| C. wave/output ownership | **FAIL** | Z5B WG256/4 waves/2 CTA；native WG128/2 waves/1 CTA，encoding 也不同 |
| D. lane-level fragment mapping | **FAIL** | native 是 blocked/shared/dot encoding；Z5B 是 row_half/lane_group + i32 view |
| E. operand orientation | **PARTIAL / NOT PARITY** | native H 明确 `tt.trans` + `isTransposed=true`；Z5B 手工交换 view 参数，未证明 lane-level 等价 |

本节保留的是 Z5B full kernel 与 native full kernel 的历史对照；它不是本轮三个
isolated consumer oracle 的结果。 “PASS opcode”不能推出“PASS physical contract”。
MFMA 指令只是最后一步；在此之前
的 producer ownership、shared physical offset、local-load fragment encoding 才决定
每个 lane 实际把哪些 BF16 元素送进 src0/src1。

---

## 9. 寄存器范围的解释

寄存器范围不是跨编译器可直接比较的 ABI，但可以作为代表性机器证据：

| 阶段/产物 | src0 | src1 | accumulator | 说明 |
|---|---|---|---|---|
| Z5B ISA `:918` | `v[56:57]` | `v[60:61]` | `a[0:15]` | 第一组 Q/H 代表性 MFMA |
| Z5B ISA `:925` | `v[58:59]` | `v[62:63]` | `a[0:15]` | 同一组后续 K=8 packet |
| native ISA `:207` | `v[12:13]` | `v[16:17]` | `a[0:15]` | native Q/H 代表性 MFMA |
| native ISA `:220` | `v[12:13]` | `v[18:19]` | `a[16:31]` | 第二 output accumulator fragment |

两边都能出现相同的 `a[0:15]`，这不代表 fragment mapping 相同；物理寄存器编号是
RA 结果。真正有判别力的是输入 VGPR 的来源、shared 地址和 dot encoding，而不是
单独比较 `a[0:15]`。

---

## 10. 是否存在缺失的 AveLang MFMA intrinsic

**没有。**

现有 AveLang intrinsic：

```text
al.amdgpu.mfma_32x32x8_bf16_f32
```

已经能够通过 LLVM 生成：

```text
llvm.amdgcn.mfma.f32.32x32x8bf16.1k
```

最终 ISA 也已经是目标硬件指令。因此本轮不支持“需要新增一个更精确的 MFMA
intrinsic”这一判断。

真正缺的不是算术 opcode，而是一个能够把以下语义保留下来的表达：

```text
typed blocked shared operand
  -> dot-operand encoding
  -> lane-correct MFMA fragment
```

Z5B 目前使用 ordinary `al.view` 和 packed word extraction；它没有把 Triton 的
`#ttg.dot_op` / shared encoding 作为一等 operand contract 保留下来。

---

## 11. FIRST physical divergence

按执行流水顺序，最早能被证据确认的差异是：

1. **CTA/wave ownership 已经不同**：Z5B 在 source 中显式使用 256-thread、4-wave、
   `row_half/value_half` 和 2 CTA split；native selected 是 128-thread、2-wave、1 CTA，
   并使用 `#mma warpsPerCTA=[1,2]`。
2. 随后 **shared physical layout / dot operand construction 分叉**：native 使用
   blocked descriptors、swizzled shared encoding、`local_load -> dot_op`；Z5B 使用
   `[512,32]` ordinary shared buffer、`i32` view、动态 word/row 地址和 BF16 fragment
   view。
3. 最后两边才进入相同的 `v_mfma_f32_32x32x8_bf16` opcode。

所以“第一处差异”不是：

- MFMA32 vs MFMA16；两边都不是 MFMA16；
- BF16 vs FP16；这里的输入是 BF16；
- 某一条 `v_perm` 是否存在；Z5B 根本没有 `v_perm_b32`，不能把 native 的 48 条
  `v_perm_b32` 机械地当成缺失指令。

更准确的 root cause 表述是：

> Z5B 的高层 source 已经描述了正确的逻辑矩阵，但没有把 native 的 distributed
> ownership、physical LDS encoding 和 dot-operand fragment mapping 一起表达给
> lowering；因此 lowering 只能从 ordinary shared view 和手工 word index 重建 MFMA
> 输入。

这属于 **operand mapping / layout contract 差异**。它还不能单凭本报告断言某一个
具体 compiler pass 是唯一责任点；本轮没有做 pass-by-pass A/B，也没有做性能因果实验。

---

## 12. 历史下一步记录（已由第 13 节覆盖）

在 Q@K/score@V oracle 执行前，唯一建议曾是：

### 选择 (b)：fix wave/lane operand mapping

下一轮应建立一个 experimental-only、same-source 的 native-compatible operand mapping
实验，目标是让 AveLang 在不改变 MFMA32 数学的前提下表达：

```text
blocked global ownership
  -> native-compatible shared physical encoding
  -> typed local/dot operand
  -> same MFMA32 opcode
```

需要先固定并验证：

- Q/H/K/V 每个 lane 的逻辑元素；
- shared byte offset；
- source operand orientation；
- 两个 wave 的 output tile ownership；
- `kWidth=4` 的 dot operand packet；
- Q@H、Q@K、score@V 三个阶段分别的 mapping。

这个实验完成前，不要调 schedule、waitcnt、barrier 或 MFMA 指令顺序。若 mapping 已
完全 parity 后仍存在 latency 差距，再单独进入 instruction scheduling audit。

---

## 13. Final closure addendum：Q@K 与 score@V 实际执行结果

本节是本报告的最终结论，覆盖前文产生时尚未运行的 Q@K/score@V gate。前文关于
“Q@K/score@V 尚未继续”的文字是历史审计状态，不能覆盖本节结果。

### 13.1 执行环境与回归顺序

编译和运行均在 `ljd_qwen_vllm_avelang_rocm722` 容器中完成。镜像当前没有
`tmux` 可执行文件，因此采用同一容器的前台 PTY；GPU、源码挂载、ROCm 和 Python
binding 没有切换。编译器 binding SHA256 为：

```text
e163618f9241a1bd2d04d9d087c3f24f846a80256e109dcad9d61a44db9429aa
```

本次 compiler 只增加 score@V 的实验 gate：

```text
AVELANG_C15_SCOREV_FULL64=1
```

它复用现有 C15 V producer 和 C14 rotating-shared recipe，只把历史的“一条
accumulator chain 后 joinColumns(acc, acc)”改成低/高两个独立 32-column chain。
Q@H、Q@K 的 lowering 和 production selector 没有被打开或替换。

回归顺序如下：

1. Q@H strengthened oracle：`row_code`/`row_half` 的 B-high=1/2 全部 exact；
2. 默认 WG256 C16 Q/H/K regression：5 类 Q + 5 类 H + 5 类 K + 5 类 Q-dual，
   共 20/20 exact；
3. 新增 Q@K WG128 oracle：5/5 exact；
4. 新增 score@V WG128 oracle：5/5 exact。

### 13.2 Q@K closure

source：

```text
test/examples/linear_attention/vllm_compare/
repro_qwen_gdn_t8192_qk_wg128_fragment_oracle.py
```

实验 contract 是 Q `[64,32]` × K `[32,64]` -> FP32 `[64,64]`。两 physical waves
分别覆盖 32-row output block；每个 wave 使用两个独立的 output-column accumulator
half。Q stage 采用低/高 row-half 权重，K source 使用 zero、one-hot-low、
one-hot-high、row/column code、checker 五种模式，故 K row、K orientation、低/高
column half 和 accumulator ownership 不能靠全零数据偶然通过。

结果：

| gate | 结果 |
|---|---|
| debug source/readback | 5/5 BF16 exact |
| raw FP32 output | 5/5 exact，`max_abs=0` |
| finite | 5/5 |
| wave 0/1 | 均有非零 row/half case |
| MFMA | `v_mfma_f32_32x32x8_bf16` |
| independent accumulator | 2 chains |

逐 lane 公式和证明结果在：

```text
lane_fragment_orientation_audit_qk.json
qwen_t8192_qh_wg128_machine_evidence_qk.json
qwen_t8192_qh_wg128_closure/qk_wg128_fragment_observation.json
qwen_t8192_qh_wg128_closure/qk_capture/
```

关键机器数据：8 条静态 MFMA32、25 条 `ds_read`、123 条 `ds_write`、8 条
`v_perm_b32`、2 个 `s_barrier`、34 个 `s_waitcnt`；HSACO metadata 为
VGPR/AGPR/SGPR=`72/32/14`，LDS=16384 B，private=0，spill=0。

### 13.3 score@V closure

source：

```text
test/examples/linear_attention/vllm_compare/
repro_qwen_gdn_t8192_scorev_wg128_fragment_oracle.py
```

实验 contract 是 score `[64,64]` × V `[64,64]` -> FP32 `[64,64]`。score stage
是 shared identity tile，V 是真实 global BF16 `[64,64]` source，由 C15 producer
写入 rotating shared。每个 wave 覆盖 32 个 output row；两个独立 accumulator
分别覆盖 output columns `[0,32)` 和 `[32,64)`；两个 K32 stage 按原有 C15/C14
顺序累加。

结果：

| gate | 结果 |
|---|---|
| debug V producer/readback | 5/5 BF16 exact |
| raw FP32 output | 5/5 exact，`max_abs=0` |
| finite | 5/5 |
| output low/high half | wave 0/1 均有检查值 |
| MFMA | `v_mfma_f32_32x32x8_bf16`，16 条静态 MFMA |
| independent accumulator | 2 chains |
| private/spill | 0 / 0 |

逐 lane 公式和机器数据在：

```text
lane_fragment_orientation_audit_scorev.json
qwen_t8192_qh_wg128_machine_evidence_scorev.json
qwen_t8192_qh_wg128_closure/scorev_wg128_fragment_observation.json
qwen_t8192_qh_wg128_closure/scorev_capture/
```

`scorev_capture/ir/vfrag/` 保存了 pre/post block-dot MLIR、late-lowering MLIR、
pre-opt/post-opt LLVM 和 LLVM pass snapshots；`scorev_capture/scorev_wg128_full64.isa.s`
和 HSACO 是实际运行的最终 code object。这个最小 probe 的 compiler capture 没有单独
导出 exact-LTO MIR 文件，因此没有把 MIR 数字伪造进机器 evidence；资源字段来自
HSACO AMDGPU metadata，静态指令数来自 final ISA。

score@V probe 的 HSACO metadata 为 VGPR/AGPR/SGPR=`76/32/14`，LDS=16384 B，
private=0，spill=0；ISA 静态数据为 16 MFMA32、40 `ds_read`、104 `ds_write`、
19 `v_perm_b32`、2 barrier、39 waitcnt。

### 13.4 与 native selected score@V 的对照

冻结 native 文件：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
codex_qwen_bt64_stage6z_native_chunko/native/T8192/selected/
```

native TTGIR 直接给出：

```text
#mma      = warpsPerCTA=[1,2], instrShape=[32,32,8], isTransposed=true
score    = b_A_197 -> truncf -> shared3 -> local_load dot_op opIdx=0
V        = b_v_240 -> in_thread_transpose -> shared4(amd_rotating_shared)
           -> local_load dot_op opIdx=1
dot      = b_o_249: 64x64xbf16 x 64x64xbf16 -> 64x64xf32
```

native final ISA 的 score@V MFMA window（`chunk_o.py:137:50`，从第一次 score
operand `ds_read` 到最后一组 accumulator MFMA）包含 16 条同形
`v_mfma_f32_32x32x8_bf16`。该窗口之前能看到 V 的 `v_perm_b32`、`ds_write_b64`、
barrier；随后是 `ds_read_b64` 与 MFMA 交错。因此：

- opcode：Q@H/Q@K/score@V 与 native 均为 exact MFMA32；
- logical shapes：三阶段均一致；
- score@V 的 V transpose：native TTGIR 明确要求，ISA 也有对应 permutation；
- AveLang score@V probe：已实现对应的 C15 rotating-shared/packed consumer，并以
  5 组 pattern 证明 fragment 语义；
- full-kernel register allocation、waitcnt 距离、LDS bank schedule 和 dynamic
  transaction 数：本轮仍未宣称 parity。

### 13.5 最终 parity matrix

| 项目 | Q@H | Q@K | score@V |
|---|---|---|---|
| exact MFMA opcode/dtype | PROVEN | PROVEN | PROVEN |
| logical matrix shape | PROVEN | PROVEN | PROVEN |
| two-wave output-half oracle | PROVEN | PROVEN | PROVEN |
| BF16 producer/readback | PROVEN | PROVEN | PROVEN |
| FP32 raw fragment exact | PROVEN | PROVEN | PROVEN |
| private/spill in probe | 0/0 | 0/0 | 0/0 |
| full native ISA byte identity | NOT PROVEN | NOT PROVEN | NOT PROVEN |
| full-kernel dynamic work parity | NOT MEASURED | NOT MEASURED | NOT MEASURED |

### 13.6 FIRST DIVERGENCE 与唯一下一步

本轮没有在 Q@K 或 score@V oracle 中发现 correctness divergence；因此最终状态不是
“Q@K failed”或“score@V failed”。

对 **full Z5B 与 native** 而言，下一处待查差异仍然是机器数据流，不是 intrinsic：

```text
same MFMA opcode
 -> compare global/buffer load width and ownership
 -> compare ds_write layout
 -> compare ds_read fragment formation
 -> compare v_perm/transpose only where TTGIR proves it
 -> compare effective offsets
 -> compare barrier/waitcnt
 -> compare register/spill
 -> only then scheduling
```

本轮禁止的 benchmark 和 scheduling 不应被自动启动。下一次只能先建立上述
load/LDS/permutation ledger，并在证据中选一个最大差异点。

---

## 13. 本轮未做事项与证据边界

- 没有 benchmark；
- 没有运行 correctness matrix；
- 没有重新采集 dynamic MFMA/VMEM/LDS/VALU/SALU PMC；
- 没有修改任何 source、compiler、allocator、ABI、WG 或 production selector；
- static ISA count 没有被当作 dynamic hardware work；
- native `stage6z_c16_qhk_native_mapping.json` 只作为 layout/formula oracle，T8192
  的 selected identity 以 `native/T8192/selected/` 和 `native_capture.json` 为准；
- 本报告证明了“首个 physical divergence 在 ownership/operand mapping”，但没有在
  本轮证明修改 mapping 后一定能够达到 Triton 性能。

这一步的价值是把问题从“是否缺少 MFMA 指令”收窄为“是否能让 AveLang 表达和保留
native 的 lane/physical operand contract”。
