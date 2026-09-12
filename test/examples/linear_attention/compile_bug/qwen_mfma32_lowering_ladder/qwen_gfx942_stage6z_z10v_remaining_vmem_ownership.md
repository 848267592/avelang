# Qwen gfx942 Stage 6Z Z10V：剩余 VMEM 的 Producer Ownership Closure

## 结论先行

本轮 **停止 Z10V，不实现 source/compiler candidate**。

严格比较 Z8W 与 selected native WG256 后，没有找到一个满足预注册 gate 的、可
删除的 global producer 重复：

- Q：Z8W 已经是一次 Q-cache producer；没有新的 Q pointer global reload。
- H：两边都是每个逻辑 K-stage 生产一次；Z8W 的差距是 packet/layout/consumer
  feeding，不是已经证明的同一 H tile 重复 global load。
- K0/K1：Z8W 的 initial/lookahead 与两个 source half 覆盖的是不同逻辑 stage，
  不是同一 K 字节的重复 producer。
- V-new：Z8W 是一次 `[64,64]` producer 后在 LDS 中复用；没有足够证据把它的
  672/448 aggregate VMEM 归因为重复 global producer。
- g：确实有 score 与 final scaling 多角色，但 Z6G-S/I 已经作为独立支线完成并
  以性能 No-Go 关闭；本轮明确不重复。
- output：是冻结的 public BF16 ABI store，不是可删除的 producer。

因此 Z8W 的 `448 VMEM/CTA` 相对 native 的 `140 VMEM/CTA` 的剩余差距，当前能
被证据支持的解释是：**typed packet granularity、producer-to-consumer physical
layout、fragment feeding、地址/ownership 辅助工作以及 aggregate PMC 无法按
operand 唯一分摊的组合**。它不是一个已经定位到 H/K/V-new 的“同一逻辑 tile 多
加载一次”问题。

机器可检查的逐张量 ledger 在
[`stage6z_z10v_producer_ownership.json`](stage6z_z10v_producer_ownership.json)。
本报告只记录审计和停止决策，没有改 kernel、compiler、allocator、selector 或
production dispatch，也没有重新跑 benchmark。

## 1. 审计问题与冻结边界

Z10V 只允许回答一个问题：

> Z8W 的 `448 VMEM/CTA` 相对 selected native WG256 的 `140 VMEM/CTA`，是否有
> 某个非 Q、非 g、非 output 的 logical tensor，在 Z8W 中被 global producer 的
> 次数明确地做多了？

冻结对照如下：

| 项目 | Z8W | selected native WG256 |
|:--|--:|--:|
| 目标 | gfx942 | gfx942 |
| BT/BV/BK | 64/64/32 | 64/64/32 |
| workgroup | 256 | 256 |
| CTA ownership | 2 CTA/chunk-head | same-shape native selection |
| MFMA | 160/CTA dynamic | 160/CTA dynamic |
| VMEM | 448/CTA | 140/CTA |
| LDS instructions | 448/CTA | 480/CTA |
| VALU | 6990/CTA | 3376/CTA |
| SALU | 840/CTA | 660/CTA |
| Z8W code-object | VGPR/AGPR/SGPR=`104/32/33`，LDS `32768 B` | selected metadata `shared=12288`；其它 capture 有 `24576` 或 readobj 为 `0` |
| private/spill | `0/0` | `0/0` |

这里的 `VMEM` 和 `LDS` 是动态 profiler 指令计数，不是字节数。`448-140=308`
是 aggregate counter gap，不能把它直接分配给某个 tensor。

本轮没有重新做 broad VALU genealogy。历史报告已经完成了 VALU 大类审计；本轮
只在 producer ledger 中记录与 H/K/V-new producer 直接绑定的 load/layout 路径，
不把 static ISA 行数当成 dynamic PMC。

## 2. 四种数字必须分开

每个 operand 同时记录四种不同概念：

1. **Unique logical bytes**：一个 CTA 在冻结 ABI 下确实需要的逻辑 tensor 字节。
2. **Logical packet count**：按 BF16x8 或 FP32x4 折算的逻辑 packet 数，只是
     数据量单位，不是硬件 transaction 数。
3. **Lexical load sites**：源码、TTGIR 或 LLVM 中静态出现的 producer region 数。
4. **Dynamic PMC**：硬件实际执行的每 CTA VMEM 指令计数。

例如，Z8W 的 K0 有两个 lexical site：一个是 stage 0 prologue，一个是
stage 1/2/3 lookahead。但这两个 site 负责不同 stage；不能因为 lexical site 是
2 就把它说成同一 K 字节加载了两遍。

### 2.1 每 CTA 的逻辑数据量

| tensor | 逻辑形状 | unique bytes/CTA | 逻辑 packet 单位 | packet-equivalent |
|:--|:--|--:|:--|--:|
| Q | `[64,128] BF16` | 16384 B | BF16x8 | 1024 |
| H | `[64,128] BF16` | 16384 B | BF16x8 | 1024 |
| K0 | `[32,128] BF16` | 8192 B | BF16x8 | 512 |
| K1 | `[32,128] BF16` | 8192 B | BF16x8 | 512 |
| V-new | `[64,64] BF16` | 8192 B | BF16x8 | 512 |
| g | `[64] FP32` | 256 B | FP32x4 | 16 |
| public output | `[64,64] BF16` | 8192 B | BF16x8 | 512 |
| 合计 | 不含临时重读 | 65792 B | 仅作逻辑单位 | 4080 |

两边在 logical tensor contract 上需要相同数据量。native 的优势不是把这些
逻辑数据从数学上删除，而是用更宽的 typed block、distributed ownership 和
consumer-compatible shared/dot layout 组织这些数据。

## 3. 工件与身份

### 3.1 Z8W

Z8W 的高层源文件是：

`vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z7ab_joint_qhk_superphase.py`

该文件同时承载 Z7AB source schedule 与 Z8W waterfall-free lowering。source SHA256
是：

```text
66b7883f6990630644b03c35d5d0f5da6b79f3b190431d25fd467e8b149c524e
```

机器工件目录：

`codex_qwen_bt64_stage6z_z8w/machine_z8w_final/`

包含 `lowered_llvm.ll`、`exact_lto/kernel_section_*.mir`、`llc_mir/stop_after_*.mir`、
`final_isa.s`、`z7ab_fixed.hsaco` 和 `machine_evidence.json`。HSACO SHA256：

```text
5029735cca8715f16ff9beed6d0cc5c18454ff064660b783edfc4a8c083514df
```

Z8W 的静态最终 ISA 代表性计数如下：

| family | count | 解释边界 |
|:--|--:|:--|
| `buffer_load_dwordx4` | 12 | H/K packet family；不是 dynamic VMEM |
| `global_load_dword` | 80 | g/其它 FP32 或辅助路径的 aggregate family |
| `global_load_ushort` | 48 | Q/V/其它 BF16 的 aggregate family |
| `global_store_short_d16_hi` | 16 | public BF16 output store family |
| `ds_read_b128` | 48 | shared fragment read family |
| `ds_write_b128` | 12 | packed/shared write family |
| `ds_write_b16` 与 `ds_write_b16_d16_hi` | 48/32 | phase materialization family |
| `s_waitcnt` / `s_barrier` | 163/24 | static lexical counts |
| `v_mfma_f32_32x32x8_bf16` | 56 | static instruction family；dynamic 为 160/CTA |

`machine_evidence.json` 中的 aggregate `global_load=128` 与上述具体 family 统计
属于同一静态机器图的不同汇总口径；不能将其反演为某个 operand 的访问字节数。

### 3.2 selected native WG256

native 工件目录：

`codex_qwen_bt64_stage6z_fixed_z2_vs_native_audit/native_T2048/selected/`

TTGIR 的核心属性是 `ttg.num-warps=4`、wave64、gfx942。shared encoding 包含：

- `#shared`：swizzled shared，`vec=4`；
- `#shared1`：H 的 `vec=1` shared；
- `#shared2`：K 的 swizzled `vec=4` shared；
- `#shared4`：rotating shared。

selected native 静态 ISA 计数：

| family | count |
|:--|--:|
| `buffer_load_dwordx4` | 14 |
| `buffer_store_dwordx2` | 4 |
| `ds_read2_b64` | 8 |
| `ds_read_b64` | 40 |
| `ds_read_u16` | 32 |
| `ds_write2st64_b64` | 4 |
| `ds_write_b128` / `ds_write_b64` / `ds_write_b32` / `ds_write_b16` | 4/8/8/16 |
| `s_waitcnt` / `s_barrier` | 48/11 |
| `v_mfma_f32_32x32x8_bf16` | 40 |

native `.json` 的 hash 是：

```text
d5c5e6b6d5ee7abce52cb10ce5d3937161f4d0f195f594f096cf9b2b4f75e1a1
```

selected HSACO SHA256 是：

```text
cc74acf84104a0900ed327fc469826c228c94fdf07fd55d8d302c3e88c04e72d
```

native 的 shared/LDS metadata 在不同采集工件中不一致：selected metadata 为
`12288 B`，其它 trace/capture 曾记录 `24576 B`，readobj 的
`.group_segment_fixed_size` 也可能为 `0`。这属于 collector/metadata 口径差异，
本轮只用 TTGIR 的 allocation/ownership 以及动态 PMC，不用这组字段证明 producer
重复。

## 4. Z8W 源码 ownership 追踪

### 4.1 Q：已关闭，不是 Z10V 候选

Z8W source lines 105-111 分配共享存储：

```text
shared = al.make_shared((2 * Q_CACHE_ROWS, BK), al.bf16)
q_cache = shared
phase = shared
```

Q cache 占低端 16 KiB。lines 113-124 的 producer：

```text
for k_stage in range(4):
    for rep in range(8):
        idx = tid + rep * WORKGROUP
        q_cache[...] = convert(convert(q[...], f32) * scale, bf16)
    syncthreads()
```

逻辑上是 4 个 `[64,32]` block，共 `[64,128]`，每个 Q element 只由这一个
producer region 从 `q_ptr` 取得一次。lines 170-171、236-237 的 Q 消费通过
`q_cache_vec`，不是新的 `q[...]` global load。

所以 Q 有一个已证明的区别：Z8W 的 source-level producer 是 scalar/narrow
ownership，而 native 是 typed `tensor<64x32xbf16>` packet；但“producer 3 次变
成 1 次”的机会已经由 Z5A/Z5B 完成，不能在 Z10V 重复登记。

### 4.2 H：四个 distinct stage，不是四次同字节重复

Z8W lines 153-164 对每个 `k_stage`：

1. 用 `h_rsrc` 形成当前 H stage 的 raw address；
2. `raw_buffer_load_x4` 取 BF16x8 packet；
3. 写入 `phase[H_STAGE_BASE + ...]`；
4. barrier 后由 Q@H MFMA 消费。

`h_row=tid//4`、`h_col_base=(tid%4)*8` 覆盖一个 `[64,32]` stage。四个 stage
合起来正好是 `[64,128]` 的 16384 B。`value_half` 只影响 consumer 读取哪一组
phase row，H global producer 本身没有一个“value_half==0/1 各自重新 load”的
第二 source region。

native TTGIR 的初始 H packet 在 lines 170-176，loop H packet 在 lines 213-226；
它同样有 4 个逻辑 stage，只是 producer/local/dot layout 由 `shared1` 和
memdesc carried loop 表达。故 H 的证据结论是：

```text
Z8W logical H producer groups = 4
native logical H producer groups = 4
同一 logical H bytes 的 multiplicity 差距 = 未证明
```

### 4.3 K0：initial 与 lookahead 是不同 stage

Z8W lines 139-146 是 K0 stage 0 的 prologue producer；lines 182-200 是
`next_stage=1..3` 的 lookahead producer。每个 active producer thread 写一个
BF16x8 packet，`tid<128` 表示两个 wave 负责 K0。

K0 的逻辑覆盖是：

- 4 个 `[32,32]` stage；
- 总 `[32,128]`；
- 8192 B；
- 512 个 BF16x8 logical packets。

关键点是：initial site 覆盖 stage 0，lookahead site 覆盖 stage 1、2、3。它们在
逻辑 index 上不重合。Z9S 已经审计并关闭了 K0 wait/consumer-point scheduling；
Z10V 不重新把 lookahead site 错误标成 duplicate。

native 的 K TTGIR 在 lines 159、174 先写初始 block，lines 206-224 在 loop 中
生产和轮换 K memdesc。native 的 K logical stage 也不是只加载一个 32x64 block
就完成全部 chunk；它在循环中轮换 distinct packet。

### 4.4 K1：第二 source half 的 distinct producer

Z8W lines 218-243 创建独立的 `score1_acc`，每个 `k_stage` 通过
`k1_offset`、`raw_buffer_load_x4` 和 K1 phase 写入后，再由 score1 MFMA 消费。

K1 的 `[32,128]` 逻辑 tile 与 K0 不同，是第二 source half；它们不能因为共享
同一 K pointer 就互相合并。源代码只有一个 K1 producer region，四个 stage index
分别覆盖不同逻辑字节。

native 的 carried K memdesc 在 TTGIR lines 206-234 体现相同的阶段化 K ownership，
区别在于 `shared2` 的 swizzled physical layout 和 dot operand local load，而不是
native 把同一 K1 stage 神奇地零成本复用。

### 4.5 V-new：一次逻辑 producer，后续 phase 复用

Z8W lines 259-268：

```text
for rep in range(16):
    idx = tid + rep * WORKGROUP
    phase[...] = vn[...]
syncthreads()
```

`idx=0..4095` 覆盖当前 `[64,64]` V-new tile，每个 logical element 一次 global
producer。lines 274-279 的两个 `source_half` 只从 `phase_vec` 读取；没有第二个
`vn[...]` producer。

native TTGIR 在 line 354 有一个 `amdg.buffer_load tensor<64x64xbf16>`，之后
lines 359-363 用 shared4/local_load 进入 update dot。这是很强的“宽 block 与
consumer layout”证据，但不是 Z8W 存在重复 V-new global producer 的证据。

## 5. Native 的真实 reuse 证据

不能只看 native ISA 的 `buffer_load_dwordx4=14`。TTGIR 才能说明哪些 packet
服务了哪些 consumer。

### 5.1 Q 同时服务 Q@H 与 Q@K

native TTGIR loop 中：

```text
%b_q_274 = amdg.buffer_load %q[...]       // next Q stage
%b_q_275 = ttg.local_load %b_q_254 -> dot_op
%b_o_295 = tt.dot %b_q_275, %b_o_294     // Q@H
%b_A_296 = tt.dot %b_q_275, %b_k_286     // Q@K
```

同一个 `%b_q_275` SSA value 同时喂给两个 dot consumer。这是 native 在 typed
operand/lifetime 上的优势，但它对应的是“同一个 producer value 被两个 consumer
复用”，不是 Z8W 的 Q 又被 global load 两次；Z8W 已经把 Q producer once 和
direct Q-cache consumer 做到了，只是它的 fragment/layout 表达不等价紧凑。

### 5.2 H/K 用 carried memdesc 与 typed dot operand

native 的 `scf.for` 从 0 到常量 3，carry 两个 accumulator 和 Q/K/H memdesc：

```text
scf.for stage = 0 .. 3
    global packet load
    local_store to rotating shared
    local_load -> dot_op
    Q@H / Q@K MFMA
    yield next Q/K/H memdesc
```

因此 native 把 producer、shared physical layout、consumer dot operand 和 lifetime
作为一条路径表达。Z8W 的 source producer 次数并没有高于这个四-stage loop；主要
差异是 AveLang 的 `raw_buffer_load_x4 -> phase -> view -> MFMA fragment` 路径在
最终机器中仍包含更多 materialization 和 address/ownership 辅助。

### 5.3 V-new 是 block load，不是 zero-load

native 的 V-new 也经过 shared/local load 和 MFMA，并有 LDS instruction；它不是
“native 没有 V global load”。native 只把 `[64,64]` V block 作为一个 typed
producer/consumer 单元，Z8W 则以当前 source-level packet/phase ownership 生成同一
逻辑 tile。

## 6. 逐张量最终 ledger

| tensor | Z8W producer groups | Z8W lexical sites | native producer groups | native lexical sites | same logical bytes duplicate proven? | 主要差距 |
|:--|--:|--:|--:|--:|:--:|:--|
| Q | 4 K-stage blocks | 1 | 4 rotating blocks | 2 | 否 | Z8W Q fill 较窄；已关闭 duplicate |
| H | 4 K-stage blocks | 1 | 4 rotating blocks | 2 | 否 | typed H packet/layout |
| K0 | 4 source stages | 2 | 4 carried stages | 2 | 否 | K producer/consumer layout |
| K1 | 4 source stages | 1 | 对应 4 carried stages | 2 | 否 | K1 phase/fragment feeding |
| V-new | 1 `[64,64]` tile | 1 | 1 `[64,64]` block | 1 | 否 | scalar/phase 到 typed block 的粒度 |
| g | 多 consumer roles | 3 | 2 logical TTGIR roles | 2 | 有 role-reuse gap，但已关闭 | Z6G 已完成，禁止重复 |
| output | 1 ABI store | 1 store | 1 ABI store | 1 store | 否 | BF16 store width；不是 producer elimination |

这张表最容易被误读的两行是 K0 和 H：

- lexical site 数是源码组织；
- producer group 数是逻辑 stage；
- dynamic PMC 是第三个独立概念。

不能用 `2 lexical sites > 1 lexical site` 直接推出 `2x logical bytes`，也不能
用 `global_load_ushort=48` 直接推出 Q/V 的字节数。

## 7. 为什么 448 vs 140 仍然没有转化成 Z10V candidate

### 7.1 不是 logical I/O 字节不一致

在冻结的 same-shape body 中，两边都需要 Q/H/K0/K1/V-new/g，并写 BF16 output。
unique logical bytes 都是约 `65792 B/CTA`。native 的 140 不是“只读了 140 字节”，
Z8W 的 448 也不是“多读了 448 字节”。这是动态 VMEM issuing counter。

### 7.2 不是已证明的 H/K/V producer duplicate

source/TTGIR 都支持四 stage H/K ownership 和一次 V-new tile producer。Z8W 与
native 的 producer group 数没有出现一个清晰的 `2 -> 1` 或 `4 -> 2` ownership
差距。

### 7.3 更可能是 packet/layout/fragment feeding

当前工件支持以下方向，但它们不是 Z10V 的 single-producer elimination：

1. native `buffer_load_dwordx4` typed packet 与 Z8W 的混合 BF16 scalar/narrow
   producer；
2. native `shared/shared2/shared4` physical encoding 与 Z8W phase/view 地址；
3. native `%b_q_275` 这类 dot operand SSA value 复用，与 Z8W `q_words -> view`
   反复形成 fragment 的差异；
4. Z8W source-level lane/row/column/phase address tuple 伴随的 VALU/SALU；
5. g 的多角色 load opportunity，但该方向已由 Z6G 关闭。

这些问题需要新的 general typed producer-layout-shared-consumer abstraction，
而不是伪造一个“单个 H/K producer 只加载一次”的 source rewrite。

## 8. Gate 判定

预注册 gate 要求同时满足：

1. 非 Q、非 g、非 output tensor；
2. Z8W 相同 logical bytes 的 global producer multiplicity 明确高于 native；
3. 可以只改 producer ownership/reuse 消除真实 global producer；
4. 不改数学、MFMA order、WG/CTA geometry、ABI。

本轮判定：

| gate | 结果 | 证据 |
|:--|:--|:--|
| Q excluded and already once | 通过，但排除 | source 113-124，Q pointer 后无 reload |
| g not reused as candidate | 通过约束 | Z6G-S/I 已关闭 |
| output not candidate | 通过约束 | ABI-required store |
| H duplicate proven | 否 | 四-stage producer multiplicity 相同 |
| K0/K1 duplicate proven | 否 | initial/lookahead/source-half 是 distinct logical stages |
| V-new duplicate proven | 否 | one `[64,64]` producer，后续 phase reads |
| clear Z10V control point | 否 | 没有可单独删除的 global producer |

最终决策为：

```text
STOP_Z10V_NO_CODE
```

## 9. 对后续研究的边界建议

本轮不登记新的 Z10V code candidate。后续若要继续接近 native，必须满足两个
条件：

1. 不再把 packet granularity 差异包装成 producer duplication；
2. 新任务应围绕通用 `block_dot_bf16_f32` 的 producer-layout-shared-consumer
   表示，证明 typed block 和 MFMA dot operand 可以跨 LLVM/MIR/ISA 保留。

这与此前 BDV2 full-scope abstraction 的方向一致，但不能在本轮把它偷偷实现成
Z10V。Z5B 仍然是 isolated performance baseline；Z8W/Z9S 保留为 correctness
与 machine-work evidence，不接 X2、selector 或 production。

## 10. 复核命令与证据位置

下面的命令只读，不会修改源码：

```bash
nl -ba test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z7ab_joint_qhk_superphase.py | sed -n '105,287p'

rg -n "buffer_load|local_load|local_store|tt.dot" \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/\
  codex_qwen_bt64_stage6z_fixed_z2_vs_native_audit/native_T2048/selected/chunk_fwd_kernel_o.ttgir

cat test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/\
  stage6z_z10v_producer_ownership.json
```

主要历史背景报告：

- [`qwen_gfx942_bt64_stage6z_z5b_remaining_vmem_operand_ledger.md`](qwen_gfx942_bt64_stage6z_z5b_remaining_vmem_operand_ledger.md)
- [`qwen_gfx942_stage6z_z8w_waterfall_free_packet_lowering.md`](qwen_gfx942_stage6z_z8w_waterfall_free_packet_lowering.md)
- [`qwen_gfx942_stage6z_z9s_critical_path_scheduler.md`](qwen_gfx942_stage6z_z9s_critical_path_scheduler.md)
- [`qwen_gfx942_bt64_stage6z_native_chunko_report.md`](qwen_gfx942_bt64_stage6z_native_chunko_report.md)

## 最终状态

Z10V 的价值是把“没有 producer ownership 消除证据”正式冻结下来。当前可以
严谨地说：

> Z8W 已把 waterfall 问题修掉，并把 aggregate VMEM 从 2464 降到 448；但在
> 448 到 native 140 的剩余部分，现有证据指向 typed packet、physical layout、
> fragment feeding 和生命周期组织，而不是某个可以单独删除的 H/K/V-new global
> producer。继续做 source-level single-producer rewrite 会把 granularity gap
> 错误地包装成 ownership gap，因此本轮停止。
