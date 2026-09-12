# Qwen GDN chunk-o T=8192：Reuse-first ISA parity closure

> **最终状态更新（2026-09-08）**：本文件前半部分记录的是 Q@H closure 之后、
> Q@K/score@V 尚未运行时的中间状态。该中间状态已被同目录的最终 parity audit
> supersede。现在 Q@K 与 score@V 的 WG128/full64 oracle 也已通过，最终状态为
> **C：三个 MFMA consumer contract 已闭合**。请以
> [qwen_gfx942_t8192_mfma_parity_audit.md](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_gfx942_t8192_mfma_parity_audit.md)
> 及其 `lane_fragment_orientation_audit_qk.json`、
> `lane_fragment_orientation_audit_scorev.json` 为准；本轮仍未进行 benchmark 或
> scheduling。

## 0. 审计状态

本报告执行的是 **reuse-first、受限修改的 ISA parity closure**。目标是把已经完成的
Z5B、native Triton、C13-C16、P2/P3、D0-P、C26 证据重新对齐，避免重复做已经关闭
的问题；然后只对 Q@H 建立 T=8192、native selected、WG128/两 wave 的最小 parity
入口。

除下方明确登记的 WG128 Q/H ownership closure 外，本轮没有：

- 修改 Qwen chunk-o kernel source；
- 修改 allocator、RA、production selector 或通用 layout planner；
- 增加 MFMA intrinsic；
- 运行 full benchmark、PMC 或 X2/full-graph 集成；
- 继续审计 Q@K、score@V、waitcnt、barrier 或 instruction scheduling。

本轮唯一的 compiler 变更是 `lower_qwen_block_dot_pass.cc` 中由
`AVELANG_C16_WG128_QH=1` 显式启用的 Q/H 两 wave physical ownership 分支；默认
环境和旧 WG256 C16 路径不变。

当前正式状态：

```text
T=8192 native selected artifact       已冻结
T=8192 pure-AveLang Z5B artifact     已冻结
MFMA opcode/type parity               已证明
T=8192 WG128 Q@H source probe         binding 已修复；WG128 producer/readback 已闭环
T=8192 WG128 Q@H physical coverage    已通过；lane-fragment/orientation parity 仍待审计
Q@K / score@V parity                  本轮禁止继续
benchmark                            本轮未运行
```

因此，本报告不能把旧的 WG256 C16 mapping closure 写成 T=8192 WG128 的完整
parity，也不能把当前 probe 的 physical readback closure 写成 Q@H lane-level parity。

---

## 1. 结论摘要

### 1.1 已经可以关闭的问题

在 Z5B 和 T=8192 native selected Triton 的最终机器工件中，以下事实已经有直接证据：

1. 两边都使用 gfx942 的
   `v_mfma_f32_32x32x8_bf16`。
2. MFMA 输入是 BF16，累加结果是 FP32；没有发现 AveLang 被迫使用错误的
   MFMA dtype，也没有发现需要另造一个 MFMA32 intrinsic。
3. AveLang 已经能够生成这个精确的 BF16-to-FP32 MFMA opcode。
4. AveLang 的 compiler/ISA 能够生成 `v_perm_b32`；硬件交换能力不是“完全不存在”。
5. C14/C15/C16 已证明 first-class physical operand 路径可以在此前 runtime/build 中
   通过 MLIR、LLVM、LTO MIR、ISA、HSACO 和数值测试。
6. C26 已纠正 MFMA 工作量归一化：不能再把旧的 160/CTA 与 80/CTA 误解成 native
   比 AveLang 少一半数学工作。
7. C16 已证明 `ds_bpermute` 和运行时 div/rem 不是构造正确 Q/H/K tile 的必然条件。
8. D0-P 已证明完整的 local LDS layout / register transpose 路线在当前 source/API
   表达下曾失败 correctness，但这不是“gfx942 硬件不能表达该布局”的证明。

### 1.2 当前仍未证明的问题

严格的 T=8192 Q@H parity 仍缺少一份可以执行的 WG128/两 wave AveLang side-by-side
机器工件。最初的 source binding 问题已经修复；本轮已经补齐 producer/readback 的
完整物理覆盖，但还没有把每个 MFMA operand 的 lane fragment 和 orientation 与
Triton 逐项对齐：

```text
binding: /workspace/project/avelang/build-software-pipeline/python/
         _avelang_bindings.cpython-312-x86_64-linux-gnu.so
Q@H WG128 probe: debug_written=2048/2048 raw_finite=True
```

当前 C16 lowerer 已增加一个显式、环境门控的 WG128 Q/H physical producer：每个
lane 负责两行、四个 BF16x4 packet，覆盖完整 64x32；对应 C13 plan 为
`sizePerThread=[2,8]`, `threadsPerWave=[16,4]`, `wavesPerCTA=[2,1]`，MFMA
parent 为 `[1,2]`。因此 **WG128 physical coverage gate 已通过**，但这只是
ownership/readback closure，不等同于 Triton 的 lane-level fragment/orientation
parity，也不是 full chunk-o correctness 或性能结果。

### 1.3 本轮唯一下一步

**在现有 first-class block-dot operand ABI 上继续完成 Q@H 的 lane-fragment/
orientation parity；只覆盖 Q@H，不扩展到 Q@K、score@V 或调度。**

source binding 已经通过容器环境修复，现有 C16/C13 physical plan 也已经能够表达
native selected 的两-wave ownership，且本轮已经生成 WG128 Q@H 的 MLIR/LLVM/ISA
审计工件。下一步只允许在同一 mapping 上核对 MFMA source fragment 与 orientation；
kernel schedule、MFMA opcode、Q@K、score@V 都冻结。

在 Q@H coverage、lane fragment 和 orientation 真正闭环之前，不允许进入 Q@K、
score@V、barrier、waitcnt、调度或 full benchmark。

---

## 2. 复用的证据清单

### 2.1 证据 inventory

| 属性 | 现有证明 | 主要 artifact | 本轮处理 |
|---|---|---|---|
| MFMA opcode | Z5B 和 native 都是 `v_mfma_f32_32x32x8_bf16` | Z5B `final_isa.s`；native `chunk_fwd_kernel_o.amdgcn` | 已关闭，直接复用 |
| MFMA dtype | BF16 输入，FP32 accumulator | ISA opcode、C14/C16 machine evidence | 已关闭，直接复用 |
| AveLang MFMA intrinsic | 能生成 exact gfx942 MFMA32 | C14、C16、Z5B | 已关闭，不加 intrinsic |
| `v_perm` 能力 | AveLang 后端可生成 `v_perm_b32` | C14/C16；native ISA | 已关闭，不把 v_perm 当缺失能力 |
| large `ds_bpermute` | R3/相关实验显示大量 cross-lane shuffle 会恶化资源 | R3/R4 相关报告、D0-P report | 已关闭为局部路线风险 |
| MFMA 工作量 | C26 完成 logical-unit/CTA 归一化 | `qwen_gfx942_c26_work_decomposition_ownership_audit.md`、oracle JSON | 已关闭旧误读 |
| C13 static physical plan | AveLang 可表达普通静态 blocked/shared shape | C13 report、mapping JSON | 作为 layout schema oracle |
| C14 synthetic codegen | exact MFMA、typed LDS load、无 spill | C14 report、machine evidence | 作为 compiler 能力证据 |
| C15 real tile | real V tile 数值与机器闭环 | C15 report、machine evidence | 作为 V-side 物理路径证据 |
| C16 Q/H/K | T2048 WG256 下 Q/H/K real tile closure | C16 report、source、JSON | 只能复用 mapping/formula oracle |
| P2 first-class MFMA operand | 正确且机器图不同，但不是性能胜者 | P2 report | 作为已有 first-class path 证据 |
| P3 packed operand | 机器图确实变化，但性能 No-Go | P3 report | 不重复实验 |
| D0-P | 完整 layout feasibility correctness 未过 | D0-P report | 不把失败扩大为硬件不可能 |
| T8192 native selected | 实际 selector 产物已冻结 | `native/T8192/native_capture.json`、`selected/` | 本轮直接复用 |
| T8192 Z5B | pure AveLang compile-only 产物已冻结 | `codex_qwen_bt64_stage6z_z5b_machine_t8192/` | 本轮直接复用 |
| T8192 WG128 Q@H | 最小 probe 已进入 codegen；coverage 已通过，lane parity 待审计 | `repro_qwen_gdn_t8192_qh_parity_wg128.py` | 本轮唯一未完成 gate |

### 2.2 为什么不重复 C13-C16

C13-C16 的价值不是“替代 T8192 parity”，而是提供可复用的物理表示和检查规则：

- C13 说明 static shape、shared encoding、dot operand schema 可以落到 AveLang IR。
- C14 说明 exact `llvm.amdgcn.mfma.f32.32x32x8bf16.1k` 和 typed LDS fragment
  load 可以生成。
- C15 说明 real tile 可以完成数值和机器闭环。
- C16 说明 Q/H/K real tile 在 T2048、WG256 下可以用统一 block-dot operand 路径
  完成，且 Q dual-consumer、H、K 的数值 gate 通过。

但 C16 的 contract 是 `WG256 / 4 waves`，不是本轮冻结的 native selected
`WG128 / 2 waves`。因此 C16 不能直接证明 T8192 native Q@H parity，只能用来：

1. 复用 lane/packet/offset 公式作为 oracle；
2. 复用已知能生成 exact MFMA 的 source-facing op；
3. 排除“需要新 MFMA intrinsic”的错误假设。

---

## 3. 冻结的 T=8192 native selected Triton

### 3.1 选择身份

native 工件来自 public API 的 T=8192 capture，不是根据 T=2048/T=8192 旧表格插值，
也不是手工挑选缓存。`native_capture.json` 记录：

| 项目 | 值 |
|---|---|
| kernel | `chunk_fwd_kernel_o` |
| target | gfx942 / HIP / wave64 |
| BT/BV/BK | 64 / 64 / 32 |
| workgroup | 128，2 waves |
| `num_warps` | 2 |
| `num_stages` | 2 |
| `num_ctas` | 1 |
| selector match score | 12 |
| loop BK match | true |
| V-load BV match | true |
| Triton | 3.6.0 + ROCm 7.2.2，commit `4ed88892` |

artifact 根目录：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
codex_qwen_bt64_stage6z_native_chunko/native/T8192/
```

selected 文件：

```text
selected/chunk_fwd_kernel_o.ttir
selected/chunk_fwd_kernel_o.ttgir
selected/chunk_fwd_kernel_o.llir
selected/chunk_fwd_kernel_o.amdgcn
selected/chunk_fwd_kernel_o.hsaco
selected/chunk_fwd_kernel_o.json
```

### 3.2 hash 和资源身份

| 文件/属性 | 值 |
|---|---|
| HSACO SHA256 | `e201dd58c83e64565f10754789ac294df53343b9c9bbbe764e8f667817066ee5` |
| final ISA SHA256 | `00f0647036be4632891e48ecea3b937357e4814aace57c84f4f127ef94614df5` |
| LLVM/LLIR SHA256 | `9c2d3a9c142c6c8dcd4eb965abf12dfcb0d2d84aab39c1736ff1ea1aac9001c9` |
| selected metadata shared | 12288 B |
| selected metadata scratch | 0 |
| code-object readobj VGPR/AGPR/SGPR | 220 / 64 / 76 |
| code-object readobj LDS fixed size | 0 B，见下方 mismatch 说明 |
| private segment | 0 |
| static MFMA32 | 80 |
| static ds_read/ds_write | 56 / 36 |
| static barrier/waitcnt | 11 / 63 |
| static buffer load/store | 28 / 8 |
| static `v_perm_b32` | 48 |

`native_capture.json` 的 selected metadata 报告 shared=12288 B，而
`code_object_readobj.txt` 报告 `.group_segment_fixed_size: 0`。这是现有 capture
metadata 与 code-object readobj 的不一致，不能把两个字段静默合并；本轮只把它记录
为 artifact mismatch，不据此推导新的 LDS 结论。

### 3.3 native Q/H 的可见机器链

selected TTGIR 给出三组关键 shared/blocked/dot encoding：

```text
#blocked  : sizePerThread=[4,8], threadsPerWarp=[8,8], warpsPerCTA=[2,1]
#blocked1 : sizePerThread=[8,1], threadsPerWarp=[4,16], warpsPerCTA=[1,2]
#blocked2 : sizePerThread=[1,8], threadsPerWarp=[16,4], warpsPerCTA=[2,1]
#mma      : MFMA version3, instrShape=[32,32,8], warpsPerCTA=[1,2], isTransposed=true
```

native Q/H/K 的 local allocation 在 TTGIR 中是：

```text
Q : 1x64x32xbf16, shared
H : 1x64x32xbf16, shared1
K : 1x32x64xbf16, shared2
```

Q@H/Q@K 周围的 TTGIR 机器语义是：

```text
buffer_load_dwordx4
  -> typed shared store
  -> local_load / dot operand
  -> MFMA32
```

ISA 中可以观察到 `buffer_load_dwordx4`、`ds_write2st64_b64`、`ds_write_b128`、
`ds_read2_b64`、`ds_read2st64_b64`、barrier 和 MFMA 的组合；exact selected ISA
没有 `ds_bpermute`。

---

## 4. 冻结的 T=8192 Z5B AveLang

### 4.1 产物身份

Z5B 是当前 pure AveLang 的 direct-Q-cache consumer，使用 BT64/BV64/BK32，
WG256、4 waves、2 CTA/chunk-head。它不是本轮新编译，只复用已经保存的 exact
T8192 compile-only capture：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
codex_qwen_bt64_stage6z_z5b_machine_t8192/
```

| 项目 | 值 |
|---|---|
| kernel | `_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z5b_direct_q_cache` |
| target | gfx942 |
| workgroup | 256，4 waves |
| HSACO SHA256 | `979889c1a10ac53064cd1d2da91298b67e4ef7f2db3baadc171872cb56b0ed67` |
| final ISA SHA256 | `2daa24b11bde95dc744c7b9bc0f596b6541e52b4d36446cf4a507ce3bf941915` |
| lowered LLVM SHA256 | `1a88164cbf264d9c8c7fa876c22c2f8909f2148dcec25721489ea65d99fbf29c` |
| code-object VGPR/AGPR/SGPR | 104 / 32 / 28 |
| LDS | 32768 B |
| private segment | 0 B |
| spills | 0 |
| launch/rocprof | 未执行 |

### 4.2 source ownership 和 phase

Z5B source 是：

```text
test/examples/linear_attention/vllm_compare/
qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer.py
```

关键 source contract：

```text
lane       = tid & 63
lane_col   = lane & 31
lane_group = lane >> 5
wave_id    = tid >> 6
row_half   = wave_id >> 1
value_half = wave_id & 1
program_id = block_id(0)
v_block    = program_id % 2
value_head = (program_id // 2) % H_V
chunk      = program_id // (2 * H_V)
```

Z5B 的 dedicated Q cache 是：

```text
shared = make_shared((512, 32), bf16)
```

即 512x32x2=32768 B；前 256 行是 Q cache，后续区域是旧 phase 区域。Q source
producer 位于 `k_stage`/`rep` 循环，source 仍以 scalar BF16 global read 的形式生成
并写入 shared view；Q@H/Q@K 再从这个 cache view 读取。

Q@H 的关键 source 形态是：

```text
H phase producer
    -> q_cache_vec / phase_vec
    -> BF16 fragment view
    -> mfma_32x32x8_bf16_f32
```

Z5B 不是 native selected 的 exact physical contract。特别是：

- Z5B 有 4 waves，native 有 2 waves；
- Z5B 每 chunk-head 使用 2 CTA，native selected 使用 1 CTA；
- Z5B source 有 `row_half/value_half` 的 4-wave ownership；
- Z5B source 使用普通 `al.view` 和 phase shared 区域；
- native selected 使用 TTGIR 的 blocked/shared/dot encodings 和 typed local-load。

### 4.3 Z5B static whole-kernel 计数

这些是 whole-kernel lexical ISA 计数，不是 Q@H isolated count，也不是 dynamic PMC：

| 指标 | Z5B whole kernel |
|---|---:|
| MFMA32 | 56 |
| global load | 192 |
| global store | 16 |
| ds_read | 56 |
| ds_write | 144 |
| s_barrier | 32 |
| s_waitcnt | 214 |
| `v_perm_b32` | 0 |
| `v_lshl_add` | 130 |
| `v_add` | 202 |

不能把这里的 56 与 native 的 80 直接解释为数学工作差异；C26 已证明正确的
logical MFMA work normalization 是 160/CTA，且需要区分 CTA 数和 logical output unit。

---

## 5. Q@H parity contract：应该比较什么

Q@H 的逻辑操作是一个 64-token、32-key-feature 的 Q block 与 H 的 64-token、
32-value-feature block 的矩阵乘，结果累加到 FP32 persistent/intermediate accumulator。
为了避免把语言层 tensor shape 与物理 MFMA operand 混为一谈，本报告把 parity 拆成
五个 gate：

| gate | 需要完全相同的内容 | 当前状态 |
|---|---|---|
| A opcode | exact `v_mfma_f32_32x32x8_bf16` | **通过** |
| B count | 对同一个 isolated Q@H block 的 MFMA 次数 | **未证明** |
| C ownership | wave 数、每 wave 负责的 32x32 output tile | **未通过宏观 contract**：WG 不同 |
| D lane fragment | 每 lane src0/src1 的逻辑 BF16 元素和寄存器 packet | **未证明** |
| E orientation | A/B 是否需要 transpose，MFMA operand orientation 是否一致 | **未证明** |

### 5.1 A：opcode parity

这一项已经闭环：

```text
native: v_mfma_f32_32x32x8_bf16
Z5B:    v_mfma_f32_32x32x8_bf16
```

因此没有理由新增类似 `mfma_32x32x8_bf16_f32_exact` 的硬件 intrinsic。
AveLang 的已有 `al.amdgpu.mfma_32x32x8_bf16_f32` 以及 first-class block-dot
lowering 已经可以到达同一个硬件 opcode。

### 5.2 B：count parity

全 kernel static summary 为：

```text
native selected: 80 MFMA32
Z5B:              56 MFMA32
```

这不能当成 Q@H isolated count，因为 native 和 Z5B 的 kernel 结构、CTA ownership、
phase 和 output work decomposition 不同。C26 的 corrected oracle 表明在相同 logical
work normalization 下应比较 160 MFMA/CTA，而不是仅比较 static ISA 行数。

当前 WG128 probe 已通过 binding gate 并完成 launch，但未通过 physical coverage
gate，尚未获得可以把 Q@H 独立出来的 AveLang machine artifact。因此 B 只能写成
`not proven`，不能写成 pass/fail。

### 5.3 C：wave/output ownership

这一项已经出现可见的结构性差异：

```text
native selected: WG128 = 2 waves, num_ctas=1
Z5B:             WG256 = 4 waves, 2 CTA/chunk-head
```

native TTGIR 的 MFMA encoding 明确使用 `warpsPerCTA=[1,2]`、
`isTransposed=true`；Z5B source 则以 `wave_id`、`row_half`、`value_half` 和
`v_block_idx` 在 WG256/2 CTA contract 下组织输出。

因此可以确认：**两边不是同一个 CTA/wave physical contract**。但因为当前 WG128
AveLang probe 没有进入 codegen，仍不能把“native 两个 wave 各自精确负责哪个
32x32 tile”写成已由 AveLang side 复现的事实。

### 5.4 D：lane-level src0/src1 fragment

native exact selected TTGIR 提供了 typed blocked/shared/dot encoding；C16 提供了可复用
的 native-style mapping oracle。对于 C16/WG256 real tile，已记录的 Q blocked2 公式为：

```text
wave        = row // 16
lane_high   = row % 16
lane_low    = (col // 8) % 4
lane        = lane_high * 4 + lane_low
packet_slot = (col % 8) // 4
element     = col % 4
reg_slot    = packet_slot * 4 + element
```

C16 Q shared physical mapping 为：

```text
phase        = (row >> 1) & 7
block_no     = (row >> 4) & 7
swizzle      = phase xor block_no
inner_group  = col >> 2
physical_col = ((inner_group xor swizzle) << 2) | (col & 3)
byte_offset  = 2 * (row * 32 + physical_col)
```

这些公式是 C16 `T=2048/WG256` 的 verified mapping oracle，不是本次 T=8192/WG128
的 exact output。当前报告不会把它们伪装成 native selected 的逐 lane 证明。

因此 D 的状态是：**mapping formula 可复用，但 T=8192/WG128 exact lane packet
parity 尚未证明**。

### 5.5 E：operand orientation

native TTGIR 的 `#mma` 写出 `isTransposed=true`，并且 Q/H 的 typed dot operand
分别来自不同 blocked/shared encoding。Z5B source 通过 `al.view` 得到 fragment，
但仅凭 Z5B source 和 whole-kernel final ISA，无法完整恢复每个 native lane 的
src0/src1 logical orientation。

因此 E 目前也是 `not proven`。不能因为两边 opcode 一样，就推断 A/B operand
orientation 已经一样。

---

## 6. 代表性 Q@H 机器链和 offset 证据

### 6.1 native selected 的链

从 T=8192 selected TTGIR/ISA 可观察到的代表性链是：

```text
buffer_load_dwordx4
    -> ds_write2st64_b64 / ds_write_b128
    -> s_barrier
    -> ds_read2_b64 / ds_read2st64_b64
    -> v_mfma_f32_32x32x8_bf16
```

native final ISA 的 Q/H 代表区域在约 `lines 162-235`，其中可以看到 typed/global
packet load、packed shared store、packed shared read 和 MFMA；exact selected ISA
没有 `ds_bpermute`，但存在 `v_perm_b32`。

从 C16/native-style LLVM mapping 中已经恢复出的主要地址 recipe 是：

```text
Q producer packet base:
  %77 = (tid << 4 & 4080) xor (tid & 56)
  next packet uses an additional xor 8

Q consumer:
  %120 = (%112 & 1984)
       | ((%114 & 56) xor ((tid >> 2) & 8))
       | ((tid << 4) & 2048)
  fragment steps use xor 16 / xor 32 / xor 48

H consumer:
  %127 = ((%114 & 56) xor ((tid >> 2) & 8))
       | (%112 & 1984)
       | ((tid << 5) & 2048)
  fragment steps use xor 16 / xor 32 / xor 48

K consumer oracle:
  shared_base + (%112 & 1984)
              + ((tid >> 2) & 8)
              + ((tid << 5) & 2048)
  fragment steps use xor 16 / xor 32 / xor 48
```

这些 recipe 说明 native 路径把 physical layout、lane ownership 和 packet offset 作为
一组稳定的 affine/bitwise mapping 来保留；它不是简单的 token-major scalar scatter。
由于这些公式来自 C16/native mapping 工件，且 C16 contract 是 WG256，报告把它们
标为 **可复用的 native-style offset oracle**，而不是 T8192/WG128 的完整 final proof。

### 6.2 Z5B 的链

Z5B source 的 Q@H 路径是：

```text
q global BF16 producer
    -> dedicated Q shared cache
    -> q_cache_vec / al.view
    -> q_frag
    -> MFMA32
```

source 中 Q cache producer 位于：

```text
qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer.py:100-111
```

Q@H H producer 和 consumer 位于约 `119-138`；Q@K 位于约 `140-179`。最终 ISA 的
Q/H 代表区域约在 `final_isa.s:910-925`，可见：

```text
ds_read_b128 v[56:59], v27 offset:20480
ds_read_b128 v[60:63], v26
ds_read_b128 v[64:67], v27 offset:20512
ds_read_b128 v[68:71], v26 offset:32
v_mfma_f32_32x32x8_bf16 a[0:15], v[56:57], v[60:61], 0
v_mfma_f32_32x32x8_bf16 a[0:15], v[58:59], v[62:63], a[0:15]
```

Z5B producer 部分存在 scalar `global_load_ushort` 和 `ds_write_b16` 家族；这与
native selected 的 `buffer_load_dwordx4`、packed shared store/read 不是同一个
memory packet contract。

### 6.3 首个可见 divergence

按从 launch/ownership 到 MFMA consumer 的顺序，当前已有 exact artifact 能支持的
首个结构性 divergence 是：

```text
native:  WG128/2 waves/1 CTA + blocked/shared/dot encoding
Z5B:     WG256/4 waves/2 CTA + row_half/value_half + ordinary view/phase layout
```

在 memory instruction 层面的下一处 divergence 是：

```text
native:  typed buffer packet -> packed LDS store/read -> MFMA operand
Z5B:     scalar BF16 global producer / narrow LDS publication -> view -> MFMA operand
```

这两处都早于 instruction scheduling。它们足以说明“当前 machine graph 不是 native
graph”，但还不能证明 T=8192/WG128 下某一个具体 lane 的第一个错误元素，因为
WG128 AveLang probe 尚未完成 lowering。

---

## 7. 最小 WG128 Q@H probe：首次失败运行记录

### 7.1 新增的实验文件

```text
test/examples/linear_attention/vllm_compare/
repro_qwen_gdn_t8192_qh_parity_wg128.py
```

这是一个 experimental-only 最小 probe：

- WG128、`num_warps=2`；
- 仅构造 Q@H 相关的 64x32/64x64 shared stage；
- 不接 full chunk-o；
- 不做 benchmark；
- 使用已有 `al.amdgpu.block_dot_bf16_f32_operand` first-class operand 入口；
- 要求 raw output finite，并检查 debug BF16 readback。

文件已经通过 host-side `py_compile`。第一次在 `ljd` 容器中运行时，默认旧扩展在
source visitor 阶段停止；这是本轮已修复的历史环境错误：

```text
error: Symbol not found: 'al.amdgpu.block_dot_bf16_f32_operand'
error: Unimplemented: 'Unsupported function call target'
RuntimeError: VisitFunctionDef failed due to compiler diagnostics
```

随后检查当前 Python module 也显示：

```text
AttributeError: module 'avelang.language' has no attribute 'amdgpu'
```

这是容器环境问题，不是 kernel 结果。已在 `ljd` 中完成可逆修复：

```text
旧扩展备份：
build-software-pipeline/python/_avelang_bindings.cpython-312-x86_64-linux-gnu.so.pre-binding-fix-20260907

默认入口切换为当前挂载源码对应的扩展：
build-software-pipeline/python/_avelang_bindings.cpython-312-x86_64-linux-gnu.so

当前默认入口 SHA256：
89ea0749f2d1f76be06e378b46f14f6bb810a4c32d15df399df203db23a3a63e
```

切换后，已有 C16 WG256 回归在 Docker 中通过：Q/H/K 五组 pattern、Q dual-consumer，
共 20 个角色/consumer 检查均 `debug_byte_exact=true`、raw exact、finite；这证明
first-class binding 和原有 WG256 path 已恢复。

首次运行本 WG128 probe 时，错误已经从 source binding 变成真实的 physical coverage：

```text
Q@H WG128 probe: debug_written=1024/2048 raw_finite=True
AssertionError: Q@H WG128 physical readback is incomplete
```

它说明 probe 已经完成编译和 GPU launch，但当时的 C16 lowerer 仍使用 WG256 physical
plan；不能把 raw finite 当成 lane mapping correctness。随后在本轮的 WG128 分支中
已修复该 coverage 问题，详见第 14 节。

因此首次 WG128 probe 没有形成可用于 parity 的完整 machine artifact；这个历史失败
状态已由第 14 节的 WG128 ownership closure 和 compile-only machine artifacts 修复。

在修复前，这一轮没有产生可用于 exact WG128 parity 的：

- Q@H 完整 physical readback；
- Q@H lane-level parity 结论；
- Q@H dynamic/static isolated counts。

### 7.2 为什么这不是 MFMA intrinsic failure

源码注册仍可在：

```text
lib/IR/Intrinsics/amdgpu_module.cc
```

附近看到已有注册名：

```text
block_dot_bf16_f32                  # around line 508
block_dot_bf16_f32_operand          # around line 525
block_dot_bf16_f32_operand_transposed # around line 541
```

并且 C16/P2 已经在另一套 runtime/build 中使用过这一类 first-class operand，生成过
正确的 MLIR/LLVM/MIR/ISA/HSACO。故本轮失败说明：

```text
当前 Docker 的 Python source binding / runtime build 不包含已有 operand symbol
```

而不是：

```text
gfx942 没有 MFMA32
AveLang compiler 没有 MFMA32 lowering
Q@H 数学错误
WG128 mapping 已经失败 correctness
```

这一区分很重要。若把 binding gate 失败误记为 kernel 失败，会错误地把下一步带到
MFMA intrinsic 或 layout redesign。

---

## 8. 机器工作差异：目前能说到什么程度

### 8.1 whole-kernel static summary

下面只用于说明机器图的结构差异，不能当作 Q@H isolated dynamic count：

| static lexical family | native selected | Z5B |
|---|---:|---:|
| MFMA32 | 80 | 56 |
| buffer load | 28 | 0（主要是 global load family） |
| buffer store | 8 | 0（主要是 global store family） |
| global load | 18 | 192 |
| global store | 0 | 16 |
| ds_read | 56 | 56 |
| ds_write | 36 | 144 |
| `s_barrier` | 11 | 32 |
| `s_waitcnt` | 63 | 214 |
| `v_perm_b32` | 48 | 0 |
| `v_lshl_add` | 76 | 130 |
| `v_add` | 119 | 202 |

解释边界：

- native 的 80 与 Z5B 的 56 是整 kernel static opcode 数；
- C26 已说明数学 work 需按 logical output/CTA 归一化；
- dynamic PMC 尚未在本轮采集；
- 不能用 static `ds_write` 或 `v_perm` 数推导动态 LDS bytes 或 latency；
- native 的 `v_perm_b32=48` 不是“AveLang 缺少 v_perm”的证据。

### 8.2 Q@H 代表性 memory path 对比

| 层次 | native selected | Z5B | parity 状态 |
|---|---|---|---|
| global input | typed `buffer_load_dwordx4` 可见 | scalar/narrow BF16 producer 可见 | 不同 |
| producer LDS store | packed `ds_write2st64_b64` / `ds_write_b128` | `ds_write_b16` 家族可见 | 不同 |
| shared physical layout | TTGIR blocked/shared encoding | dedicated cache + ordinary view/phase area | 不同 |
| local/dot load | typed blocked local-load | `q_cache_vec`/view path | 尚未 exact parity |
| MFMA | exact BF16/F32 MFMA32 | exact BF16/F32 MFMA32 | 相同 |
| lane mapping | selected TTGIR encoding，可部分恢复 | WG256 source ownership | T8192/WG128 尚未证明 |
| orientation | native `isTransposed=true` | source view 不能单独证明 exact orientation | 未证明 |

---

## 9. 逐项回答审计目标

### A. opcode

**Parity：通过。** 不需要新增 hardware intrinsic。

### B. count

**Parity：未证明。** Whole-kernel static numbers 不具备 Q@H isolated 可比性，且
WG/CTA contract 不同；必须等 WG128 probe 进入 codegen 后按同一 logical block 重新
计数。

### C. wave/output ownership

**Parity：当前宏观 contract 不同。** Native 是 WG128/2 waves/1 CTA；Z5B 是
WG256/4 waves/2 CTA。exact native per-wave 32x32 output assignment 尚未由 AveLang
WG128 side 重现。

### D. lane-level operand fragment mapping

**未证明。** C16 公式可作为 oracle，但其 verified contract 是 T2048/WG256，不能
冒充 T8192/WG128 exact proof。

### E. operand orientation

**未证明。** native TTGIR 明确 `isTransposed=true`，Z5B source 通过 view/fragment
形成 operand；没有可执行的 WG128 AveLang machine artifact 前，不能声称 orientation
相同或不同到某个具体 lane。

---

## 10. “AveLang 是否缺少精确 MFMA intrinsic”的判断

结论是：**不是。**

已有证据链：

```text
C14 synthetic exact MFMA
  -> C15 real tile
  -> C16 Q/H/K real tile
  -> Z5B final ISA exact MFMA32
```

当前最终失败发生在 WG128 physical ownership/readback，而不是 MFMA lowering。binding
修复后的 C16 WG256 回归和 exact MFMA 证据仍然有效。因此本轮不实现新的 intrinsic，
也不修改 `lower_qwen_block_dot_pass.cc`。

同理，不能因为 native 有 48 条 `v_perm_b32` 而立即实现 v_perm 专用 path：AveLang
已经具备生成 v_perm 的能力，且当前首个已知差异位于 ownership/typed shared/dot
operand contract。

---

## 11. 复用优先的停止决策（修复前状态）

严格 stop rule 现在在 Q@H WG128 physical ownership/mapping gate 处触发：

```text
不进入 Q@H exact machine parity
不进入 Q@K
不进入 score@V
不进入 scheduling
不进入 benchmark
```

在用户给定的四个候选动作中，本轮不能诚实地宣称已经通过 parity 后应执行某一个：

| 候选 | 本轮决策 | 原因 |
|---|---|---|
| (a) add exact MFMA intrinsic | 不做 | opcode 已相同，C14/C16/Z5B 已证明能力存在 |
| (b) fix wave/lane mapping | 本轮已完成最小 closure | 首次 partial coverage 已定位为 WG256 producer ownership 错用于 WG128；新分支及 machine artifact 见第 14 节 |
| (c) fix on-the-fly transpose/v_perm | 不做 | 不能绕过尚未建立的 exact mapping；v_perm 也不是缺失能力 |
| (d) instruction scheduling | 不做 | operand contract 尚未 parity，调度不是当前控制点 |

当时唯一允许的动作不是新的性能分支，而是：

> **在已有 first-class operand ABI 上完成一个最小 WG128 Q@H ownership/mapping
> closure，然后只重跑同一个 probe。**

coverage 已经完成；在 lane fragment 和 orientation 完成前，仍不得继续 Q@K、score@V
或调度。

---

## 12. Artifact index 与复现边界

### 12.1 本轮直接复用

```text
compile_bug/qwen_gfx942_t8192_mfma_parity_audit.md

compile_bug/qwen_mfma32_lowering_ladder/
  codex_qwen_bt64_stage6z_z5b_machine_t8192/
  codex_qwen_bt64_stage6z_native_chunko/native/T8192/

compile_bug/qwen_gfx942_c13_static_physical_representation_mvp.md
compile_bug/qwen_gfx942_c14_static_physical_codegen_mvp.md
compile_bug/qwen_gfx942_c15_real_tile_numerical_closure.md
compile_bug/qwen_gfx942_c16_qhk_native_mapping_real_tile_closure.md
compile_bug/qwen_gfx942_c26_work_decomposition_ownership_audit.md
compile_bug/qwen_direct_k64_bv32_layout_feasibility_d0p_report.md
compile_bug/qwen_gfx942_stage6z_block_dot_v2_p2_first_class_mfma_operand.md
compile_bug/qwen_gfx942_stage6z_block_dot_v2_p3_packed_operand_reuse.md
```

### 12.2 本轮新增/保留的最小 probe

```text
vllm_compare/repro_qwen_gdn_t8192_qh_parity_wg128.py
```

该文件是实验入口，不是 production kernel，也不代表 parity 已通过。它的失败状态
必须和“运行时 binding 尚未同步”一起保留，方便后续重建 Docker/runtime 后复查。

### 12.3 复现命令与预期状态

```bash
docker exec ljd_qwen_vllm_avelang_rocm722 \
  env AVELANG_C16_WG128_QH=1 \
  PYTHONPYCACHEPREFIX=/tmp/avelang_pycache \
  python3 \
  /workspace/project/avelang/test/examples/linear_attention/vllm_compare/repro_qwen_gdn_t8192_qh_parity_wg128.py
```

当前容器已完成 Python binding 路径修复；加上 `AVELANG_C16_WG128_QH=1` 后，此命令
能够完成编译并 launch，当前已观察到完整 readback coverage。以下 compile-only artifact
已经生成：

```text
WG128 Q@H MLIR
WG128 Q@H LLVM
WG128 Q@H exact-LTO MIR
WG128 Q@H final ISA
WG128 Q@H HSACO
```

这些 artifact 已经生成，但在 lane-fragment/orientation parity 通过以前，仍不应运行
性能或扩展到其他 chunk-o phase。

---

## 13. 最终结论

当前最可靠的结论不是“AveLang 缺 MFMA”，也不是“Z5B 已经和 Triton 完全 parity”。

准确结论是：

1. **MFMA opcode/type 已 parity。**
2. **Z5B 和 T=8192 native selected 的 CTA/wave ownership 与 shared/dot operand
   physical representation 明显不同。**
3. **Q@H 的 producer/readback ownership 已在 WG128/两 wave contract 下闭环，
   但 exact lane fragment、operand orientation 和 isolated MFMA count 尚未证明。**
4. **Docker Python source binding 已经修复；C13/C16 已增加显式 opt-in 的 WG128
   Q/H ownership，默认 WG256 contract 和回归均保持不变。**
5. **因此本轮不继续修改 layout/v_perm，也不修改 barrier、waitcnt、schedule，且不 benchmark。**
6. **唯一下一步是利用已生成的 WG128 Q@H MLIR/LLVM/ISA 审计 lane-level
   fragment/orientation parity，完成后才考虑 Q@K 或 score@V。**

## 14. WG128 Q@H ownership closure：本轮实际修复与证据

### 14.1 修改范围

本轮只修改了现有 C16/C13 lowering 的 WG128 Q/H 分支，没有新增 Qwen 专用 op，也
没有触碰 Q@K、score@V、barrier、waitcnt、调度、allocator/RA、production selector
或 full chunk-o。修改集中在：

```text
lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc
```

新增显式环境门控：

```text
AVELANG_C16_WG128_QH=1
```

默认不设置时，原来的 C16 WG256/4-wave 公式完全保留。开启时仅对 Q/H 使用：

```text
logical Q/H tile:       [64, 32]
registers per lane:     [2, 8]
threads per wave:       [16, 4]
waves per CTA:          [2, 1]
MFMA parent:            [1, 2]
packet width:           BF16x4
packets per lane:       4
```

旧 WG256 producer 的 row 公式是：

```text
row = wave * 16 + lane / 4
col = (lane % 4) * 8 + packet * 4 + element, packet in [0, 1]
```

它依赖四个 wave 才能覆盖 `row=0..63`。在 WG128 下只有两个 wave，因此只写出
前 32 行，正好解释了原来的 `1024/2048` 覆盖率。

新的 WG128 Q/H producer 使用：

```text
row = wave * 32 + (lane / 4) * 2 + (packet >> 1)
col = (lane % 4) * 8 + (packet & 1) * 4 + element
packet in [0, 3], element in [0, 3]
```

这是一个显式的两 wave ownership bijection：每个 lane 负责 2 行和 4 个连续
BF16x4 packet，覆盖完整的 `64*32=2048` 元素。它仍然使用已有 C16 shared-offset
公式和已有 first-class `block_dot_bf16_f32_operand` ABI；没有改变数学或 MFMA
opcode。

### 14.2 编译与运行证据

在持久化 Docker `ljd_qwen_vllm_avelang_rocm722` 中执行：

```bash
cmake --build /workspace/project/avelang/build \
  --target _avelang_bindings -j2

cp /workspace/project/avelang/python/_avelang_bindings.cpython-312-x86_64-linux-gnu.so \
   /workspace/project/avelang/build-software-pipeline/python/

AVELANG_C16_WG128_QH=1 \
PYTHONPYCACHEPREFIX=/tmp/avelang_pycache \
python3 /workspace/project/avelang/test/examples/linear_attention/vllm_compare/\
repro_qwen_gdn_t8192_qh_parity_wg128.py
```

本轮新 binding 的 SHA256：

```text
4042146bc3d5642d6c1b112491097af91306c08aa3493ad6f7e6b9c6ee61604e
```

实际结果：

```text
Q@H WG128 probe: debug_written=2048/2048 raw_finite=True
```

随后关闭 `AVELANG_C16_WG128_QH`，在同一容器重跑原有 WG256 C16 Q/H/K mapping
regression。Q/H/K 五组 pattern、Q dual-consumer 五组 pattern 全部通过：

```text
20 checks: debug_byte_exact=True, raw finite/exact=True
```

这证明新分支没有污染原来的 WG256 C16 contract。

### 14.3 这次修复证明了什么

已经证明：

1. Docker binding 路径修复后，AveLang 可以编译并启动 WG128 Q@H first-class
   operand probe。
2. 原先 `1024/2048` 的失败是具体的 WG256 producer ownership 被错误复用于
   WG128，而不是 MFMA intrinsic 缺失。
3. C13 plan 可以表达两 wave Q/H coverage，且 `[1,2]` MFMA parent 可以通过
   compiler validation。
4. 完整 Q/H shared readback 可以在 BF16 下通过 NaN coverage 与 finite gate。

尚未证明：

1. AveLang 与 Triton 每个 lane 的 MFMA source fragment 完全相同。
2. Q@H 的 operand orientation 和实际 `src0/src1` VGPR fragment 完全相同。
3. WG128 Q@K、score@V 或完整 chunk-o 正确性。
4. 任何 latency、PMC、occupancy 或与 native Triton 的性能关系。

因此本轮仍然不能进入 benchmark，也不能声称已经完成 T=8192 MFMA parity。

### 14.4 下一步唯一动作

在不改变当前 WG128 Q/H producer 的前提下，生成该 probe 的 MLIR、LLVM、exact-LTO
MIR、ISA，并把代表性 MFMA 的 `src0/src1 -> LDS -> producer` 反向追踪到 Triton
selected artifact。只有 Q@H 的 lane fragment 与 orientation 通过后，才允许审计
Q@K 或 score@V。调度、barrier、waitcnt 和性能优化仍然冻结。

这份 closure 现在记录的是一个已经通过 producer/readback gate、但仍未完成完整
lane-level parity 的中间状态；它不会把 coverage closure 夸大成最终性能结论。

### 14.5 WG128 compile-only machine artifacts

在同一 `ljd_qwen_vllm_avelang_rocm722` 容器内，使用 `num_warps=2` 显式调用同一
AveLang generator，已导出：

```text
test/examples/linear_attention/compile_bug/qwen_t8192_qh_wg128_closure/
```

其中包括：

```text
pre_block_dot_operand_materialization.mlir
post_block_dot_lowering.mlir
post_block_dot_operand_materialization.mlir
amdgpu_00_pre_common.mlir ... amdgpu_19_post_gpu_pipeline.mlir
preopt_llvm.ll
postopt_llvm.ll
llvm_pass_*.ll
wg128_qh_isa.s
wg128_qh.hsaco
machine_evidence.json
```

关键机器事实：

| 项目 | WG128 Q@H probe |
|---|---:|
| final HSACO SHA256 | `95be31f109e49e076528a7d1f0f90447c7ad8ffdf85b7c36a5119568f0683762` |
| final ISA SHA256 | `17e7536b3c108cc52db03c83da3f74c155b3942836c060055da4d88c5946eee7` |
| HSACO MFMA | `v_mfma_f32_32x32x8_bf16` |
| static MFMA lexical count | 4 |
| static `ds_read` / `ds_write` | 29 / 68 |
| static `s_barrier` / `s_waitcnt` | 2 / 10 |
| VGPR / AGPR / SGPR | 52 / 16 / 22 |
| LDS fixed | 16384 B |
| private / VGPR spill / SGPR spill | 0 / 0 / 0 |
| max flat workgroup | 128 |

`pre_block_dot_operand_materialization.mlir` 中可直接看到新的 Q operand 属性：

```text
c13.distributed = #ave.distributed_encoding<
  logicalShape = [64, 32],
  sizePerThread = [2, 8],
  threadsPerWave = [16, 4],
  wavesPerCTA = [2, 1],
  order = [1, 0]>
c13.mfma = #ave.mfma_encoding<warpsPerCTA = [1, 2] ...>
```

后端快照中还保留了 `c16.native_dot_lds_load` 和
`_avelang_amdgpu_rocdl_mfma_f32_32x32x8bf16_1k`，说明这次修复确实进入了
lowered MLIR/LLVM，而不是只改变了 Python 环境变量或 debug buffer。由于这个 probe
不是完整 chunk-o，以上 MFMA/VMEM/LDS 数不能与 native full chunk-o 的静态或动态
计数直接比较；它们只用于证明 WG128 Q@H physical path 已经生成可执行机器图。

## 15. WG128 lane-fragment/orientation audit：当前停止点

本节把 coverage closure 与 fragment parity 分开记录。机器可读版本位于：

```text
qwen_t8192_qh_wg128_closure/lane_fragment_orientation_audit.json
```

### 15.1 AveLang producer ownership 已可证明

在 `AVELANG_C16_WG128_QH=1` 下，Q/H `[64,32]` producer 使用：

```text
row = wave*32 + floor(lane/4)*2 + floor(packet/2)
col = (lane%4)*8 + (packet%2)*4 + element
```

其中 `wave=0..1`、`lane=0..63`、`packet=0..3`、`element=0..3`。这个有限域一共
产生 `2*64*4*4=2048` 个坐标，且覆盖 `[64,32]` 的每个 BF16 元素恰好一次。
现有 NaN readback 证据为 `2048/2048`，并且 debug BF16 与 source bit-exact。

这证明的是 producer ownership/readback，不是 MFMA lane fragment 正确性。

### 15.2 Native 侧能证明到什么程度

T=8192 实际 selected TTGIR 明确给出：

```text
#mma      warpsPerCTA=[1,2], instrShape=[32,32,8], isTransposed=true
#blocked2 sizePerThread=[1,8], threadsPerWarp=[16,4], warpsPerCTA=[2,1], order=[1,0]
#blocked1 sizePerThread=[8,1], threadsPerWarp=[4,16], warpsPerCTA=[1,2], order=[0,1]
#shared   vec=4, perPhase=2, maxPhase=8, order=[1,0]
#shared2  vec=4, perPhase=2, maxPhase=8, order=[0,1]
```

Native 的 source chain 是 `buffer_load -> local_store -> local_load(dot_op,kWidth=4)
-> tt.dot`。WG128 AveLang probe 的 chain 已进入 LLVM/ISA，但是 ordinary shared/view
和 packed LDS load 的表示，不能仅凭相同 MFMA opcode 推出与 Triton 的每 lane source
fragment 相同。

### 15.3 Parity matrix

| 项目 | 当前结论 | 证据边界 |
|:--|:--|:--|
| MFMA opcode/dtype | PASS | 两边均为 `v_mfma_f32_32x32x8_bf16`，输入 BF16、累加 F32 |
| AveLang Q/H producer coverage | PASS | 2048/2048 BF16-exact readback |
| coarse waves/CTA | PASS | 两边均为 WG128、2 waves；native 另有 num_ctas=1 |
| static/dynamic MFMA count | NOT PROVEN | 当前 probe 是最小物理 closure，不能与 full chunk-o 对数 |
| output tile ownership | NOT PROVEN | probe 尚未使用完整双 accumulator 的 Q@H 输出 contract |
| lane-level src0/src1 mapping | NOT PROVEN | native TTGIR 给出 encoding，不提供完整显式 lane table |
| operand orientation | NOT PROVEN | native 有 `isTransposed=true`，但未完成逐 lane 回溯 |

因此当前最早的可确认差异仍是 **consumer representation**：AveLang 侧的 ordinary
shared/view -> packed LDS load，而 native 侧是 typed `dot_op` encoding。不能把这一步
偷换成“MFMA intrinsic 缺失”，也不能在 fragment parity 前开始 instruction scheduling。

### 15.4 双 accumulator oracle 的实际结果

已在同一 `ljd_qwen_vllm_avelang_rocm722` Docker、WG128/2-wave、
`AVELANG_C16_WG128_QH=1` 下运行：

```text
Q@H WG128 dual-acc oracle: debug_exact=True finite=True
wave_rows_exact=False max_abs=1028
wave0_reused_exact=True wave0_reused_max_abs=0
first mismatch: tid=64, wave=1, lane=0, slot=0,
  actual=0, expected=1024
```

输入是 `[64,32]` BF16 row-code，B 是逻辑 identity（`B[k,k]=B[k,k+32]=1`）。
因此 `tid=64` 本来应该读出 row 32 的值 `1024` 开始；实际读出的是 row 0 的
`0,1,2,3,...`。把 wave 1 强制按 wave 0 的 row half 解释时，全部 raw 输出 exact。

为了区分 row-half 和 column-half，随后把 B 改成：`B[k,k]=1`、
`B[k,k+32]=2`。观察到：

```text
wave0 low : 0,1,2,3,8,9,10,11,...
wave1 low : 0,2,4,6,16,18,20,22,...
wave0 high: 0,1,2,3,8,9,10,11,...
wave1 high: 0,2,4,6,16,18,20,22,...
```

因此这个最小 closure 观察到的是 **wave 1 选择 B 的第二个输出列半区**，而不是
可以直接断言 wave 1 应该负责 Q 的第二个 row half。这个现象与 native 的
`#mma warpsPerCTA=[1,2]` / `isTransposed=true` 方向相容，但还不足以证明完整
`[64,64]` Q@H 的源 row half、输出 column half 和双 accumulator 三者完全一致。

这不是 MFMA opcode 缺失：MFMA 已正确生成且结果 finite；也不是 producer coverage
失败：debug readback 仍 BF16-exact。当前准确结论是：**已观测到 WG128 的 wave-column
行为，但完整 source-row/output-tile lane mapping 仍未闭环**。不能基于这个最小
closure 直接改 consumer mapping，也不能提前进入 instruction scheduling。

### 15.5 当前决策

本轮不接 Q@K、score@V，不跑 benchmark，不修改 waitcnt/barrier/schedule。下一步唯一
允许的动作是建立一个覆盖完整 `[64,64]` 的 Q@H lane oracle，分别让输入 Q 的 row
half 和 B 的 output column half 使用互相可区分的编码，再回放 native TTGIR 的
`#blocked2 -> #mma` ownership。只有当：

1. `debug_exact=True`；
2. 两个 source row half 与两个 output column half 的 raw mapping 都 exact；
3. 双 accumulator 的 `src0/src1` logical fragment 与 expected lane table 一致；

才允许继续 Q@K/score@V parity。机器可读结果已同步到
`qwen_t8192_qh_wg128_closure/lane_fragment_orientation_audit.json`。

### 15.6 强化后的行半/列半 ownership probe

为避免把一个 identity B 的偶然结果误判为 lane parity，本轮又运行了
`repro_qwen_gdn_t8192_qh_wg128_fragment_oracle.py` 的强化版。这个 probe 仍然只
覆盖一个最小 Q@H 物理 closure，不代表完整 chunk-o；它把两个来源维度分开编码：

- Q 的 source rows `0..31` 写成 `1`，`32..63` 写成 `7`；
- B 的低输出列半区权重固定为 `1`，高输出列半区分别使用 `1` 和 `2`；
- 两个 wave、两个结果半区都记录 `tid=0` 和 `tid=64` 的原始结果；
- 每个 case 都检查 BF16 debug readback exact 和 raw finite。

GPU 运行结果如下，原始 JSON 位于
`qwen_t8192_qh_wg128_closure/qh_wg128_full_tile_observation.json`：

| source pattern | B high-column weight | wave0 low/high | wave1 low/high |
|:--|--:|--:|--:|
| row half: low=1, high=7 | 1 | 1 / 1 | 1 / 1 |
| row half: low=1, high=7 | 2 | 1 / 1 | 2 / 2 |

这个结果有两个明确含义：

1. wave 1 能区分 B 的第二个输出列半区；当高列半区权重从 1 变成 2 时，wave 1
   的 raw 结果随之变成 2，而 wave 0 保持 1。
2. 在这个单 op closure 里，wave 0 和 wave 1 都观察到 Q 的低源行半区。因而不能
   把 `warpsPerCTA=[1,2]` 简化为“两个 wave 自动覆盖两个 Q source row half”。

这仍然不是“当前 kernel 一定错误”的充分证据，因为 native 的一个
`tt.dot` 是完整 `64x32 * 32x64 -> 64x64` 逻辑结果，而当前 probe 的 C16 source
op 只构造了一个局部 physical closure。它的价值是把当前 lowering 的缺口从模糊的
“可能是 lane mapping”进一步缩小到了可直接检查的 accumulator/result contract。

### 15.7 LLVM/MLIR 证据：当前 C16 path 没有真正保留两个输出 accumulator

强化 probe 对应的 lowered artifacts 给出了比数值样本更直接的证据：

- `lowerGenericOperandMode` 先 materialize `accLow`，然后只创建一条
  `amdgpu_block_dot_mfma_operand` 链；
- `post_block_dot_lowering.mlir` 中这条链包含 4 个
  `v_mfma_f32_32x32x8_bf16` consumer，累加器类型是 `vector<16xf32>`；
- 末尾的 `vector.from_elements` 把同一条 accumulator 的 16 个值重复放入
  结果的前后两个 16-element 半区，即等价于
  `joinColumns(accumulator, accumulator)`；
- `accHigh` 没有形成独立的第二条 consumer chain。

对应源码位置是：

```text
lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc
  lowerGenericOperandMode: approximately 2502
  C16 result construction: approximately 2623-2719
```

这解释了为什么双 accumulator probe 的两个结果半区相同：它实际上验证的是同一
个 16-lane accumulator 被复制后的结果，而不是两个独立的 32x32 output tile。
同时，T=8192 Triton TTGIR 明确表示：

```text
tt.dot tensor<64x32xbf16> * tensor<32x64xbf16>
  -> tensor<64x64xf32>
#mma warpsPerCTA=[1,2], instrShape=[32,32,8], isTransposed=true
```

所以目前可以确认的**第一处物理/表示分歧**不再只是“普通 LDS view 对 typed dot
operand”，而是：AveLang 的当前 WG128 C16 consumer 没有表达完整的
`64x64` 双输出 accumulator contract，随后通过复制单 accumulator 填满结果向量。
这不是缺少 `v_mfma_f32_32x32x8_bf16` 指令；该指令已经生成且执行 finite。也不能
靠 waitcnt、barrier 或 instruction scheduling 修复。

### 15.8 更新后的判断

| 候选方向 | 判断 |
|:--|:--|
| (a) 增加 MFMA intrinsic | `No-Go`：现有 opcode 已正确生成 |
| (b) 修复 wave/lane/accumulator mapping | **下一步**：需要在现有 `block_dot` 通用 lowering 中表达完整 64x64 consumer |
| (c) 先修 v_perm/LDS on-the-fly path | 暂缓；当前更早的缺口是 accumulator/result contract |
| (d) instruction scheduling | 暂缓；parity 尚未闭合 |

因此，下一轮应只做一个 experimental-only generic `block_dot` consumer contract：
让 `accLow` 与 `accHigh` 都有独立的 MFMA consumer ownership，并用同一强化 oracle
验证 source-row half 与 output-column half。仍不修改 production selector、不跑性能、不
调整 barrier/waitcnt，也不接 Q@K/score@V，直到这个 64x64 contract 先闭环。

补充的 native ISA 证据也已写入
`qwen_t8192_qh_wg128_closure/lane_fragment_orientation_audit.json`：selected Triton
kernel 的 final ISA 实际出现 `a[0:15]`、`a[16:31]`、`a[32:47]`、`a[48:63]` 四组
累加器目的区间；selected ISA 全函数有 80 条 MFMA lexical occurrences。这个数字
不能直接拿来和最小 probe 的 4 条 MFMA 做性能比较，但它明确证明 native 的
`64x64` dot 不是一个 16-lane accumulator 再复制成两半的 contract。

### 15.9 实验性 two-accumulator / 64x64 consumer contract 已闭环

按 15.8 的唯一允许动作，本轮新增了环境门控的
`AVELANG_C16_WG128_QH_FULL64=1` 分支。它没有增加 MFMA intrinsic，也没有改变
WG128、两 wave producer、LDS 容量、barrier 或任何默认 selector。改动只发生在
`lowerGenericOperandMode` 的 C16 Q/H consumer：

1. 独立 materialize `accLow` 和 `accHigh` 两个 `vector<16xf32>`；
2. 为两个输出列半区分别创建一条 `amdgpu_block_dot_mfma_operand` 链；
3. 用 `c16.wg128_qh_output_half = 0/1` 保留列半区语义；
4. 在晚期 operand lowering 中使用
   `virtualWave = sourceWave * 2 + outputHalf`，复用已验证的 WG256 physical
   address recipe；
5. 用 `joinColumns(low, high)` 生成真正的 32-lane logical result，而不是
   `joinColumns(accumulator, accumulator)`。

对应的 source 位置是：

```text
lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc:2657-2732
```

这次分支已经在 Docker `ljd_qwen_vllm_avelang_rocm722` 内重建并切换到运行时
binding。新 binding SHA256 为：

```text
9e6114b27cdb09736d58d03ddc6e46a02eaf396caf829c06f5d10602d1ddf542
```

强化 oracle 的四个 case 全部通过：

| case | source row pattern | high output-column weight | result |
|:--|:--|--:|:--|
| 1 | row-code `row*32+col` | 1 | exact, finite |
| 2 | row-code `row*32+col` | 2 | exact, finite |
| 3 | row 0..31=1, row 32..63=7 | 1 | exact, finite |
| 4 | row 0..31=1, row 32..63=7 | 2 | exact, finite |

四个 case 都同时检查了 BF16 debug readback、FP32 raw result 和 finite；结果文件为
`qwen_t8192_qh_wg128_closure/qh_wg128_full_tile_observation.json`，全部
`full64_expected_exact=true`、`max_abs=0`。其中 `B_high=2` 的 row-half 观察为：

```text
wave0: low=1, high=2
wave1: low=7, high=14
```

这同时区分了 source-row half 和 output-column half，证明 probe 不再只是 identity
B 下的偶然重复。默认 WG256 的 Q/H/K 与 dual-consumer regression 也重新执行，
共 20 个检查全部通过。

### 15.10 full64 机器证据与边界

新 capture 位于：

```text
qwen_t8192_qh_wg128_closure/full64_capture/
```

其中包含 pre/post block-dot MLIR、pre-opt/post-opt LLVM、LLVM pass snapshots、
link-debug bitcode/LLVM、HSACO 和 ISA。关键证据如下：

| 项目 | full64 probe |
|:--|--:|
| HSACO SHA256 | `47f2f64242c166fc164429dd6af9524bed7091dec6928ab40c77af2f85457338` |
| ISA SHA256 | `a4893da985f47cfe4fe7b8bec782a0c76644c385886954c8558ca30df5bdfd8e` |
| MFMA static | 8，两个独立链各 4 条 |
| MFMA opcode | `v_mfma_f32_32x32x8_bf16` |
| s_barrier / s_waitcnt | 2 / 23 |
| ds_read / ds_write | 45 / 68 |
| v_perm | 20 |
| VGPR / AGPR / SGPR | 72 / 32 / 20 |
| LDS / private / spill | 16384 B / 0 / 0 |

`post_block_dot_lowering.mlir` 中明确出现两个带有
`c16.wg128_qh_output_half = 0` 和 `= 1` 的内部 operand op，以及最终
`vector<32xf32>` 的 `joinColumns`。ISA 中出现 `a[0:15]` 与 `a[16:31]` 两组独立
accumulator 目的区间，对应 probe 的两个输出半区。

以上资源字段来自新 full64 HSACO 的 AMDGPU metadata，而不是从旧 probe 推断：
`full64_capture/code_object_notes.txt` 记录了 `.agpr_count=32`、`.vgpr_count=72`、
`.sgpr_count=20`、`group_segment_fixed_size=16384` 和 zero spill/private segment。

因此，本轮结论是：**实验性 two-accumulator/64x64 Q@H consumer contract
PASS**。它证明此前的第一处差异确实可以通过 AveLang 的通用 block-dot lowering
表示，不需要新增硬件 MFMA intrinsic。它还没有证明完整 chunk-o 的 Q@H、Q@K、
score@V 三个阶段都与 Triton 完成 lane-level parity，因此下一步只能把同一 contract
分别用于 Q@K 和 score@V 的只读 mapping audit；仍然不跑性能、不改 schedule、不接
production。
