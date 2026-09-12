# Qwen gfx942 Stage 6Z：Z5B 与 Native Triton 最终机器关键路径对齐审计

## 0. 结论先行

本轮是严格的 **read-only compiler audit**。没有修改 kernel source、AveLang
compiler、allocator/RA、production selector、X2、recurrence HSACO、WG、MFMA
几何、数学或 ABI，也没有实现 P5 或新的 Qwen/chunk-o public op。

审计对象固定为：

```text
Z5B AveLang：BT64 / BV64 / BK32 / WG256 / 2 CTA per chunk-head
native Triton：同形状 WG256 selected chunk_fwd_kernel_o
gfx942 / wave64 / BF16 Q,K,H,V-new/output / FP32 g / MFMA32
```

冻结的 T=2048 per-CTA 动态 PMC 为：

| metric | Z5B | native | Z5B - native | Z5B/native |
|:--|--:|--:|--:|--:|
| MFMA | `160` | `160` | `0` | `1.00x` |
| VMEM | `672` | `140` | **`+532`** | `4.80x` |
| LDS | `672` | `480` | `+192` | `1.40x` |
| VALU | `7072` | `3376` | **`+3696`** | `2.09x` |
| SALU | `768` | `660` | `+108` | `1.16x` |

已有同口径 body diagnostic：

| T | Z5B ms | native ms | Z5B/native |
|--:|--:|--:|--:|
| 2048 | `0.066899501` | `0.042644000` | `1.57x` |
| 8192 | `0.157253496` | `0.091215502` | `1.72x` |

由这两个点得到的 endpoint slope 约为：

| kernel | slope |
|:--|--:|
| Z5B | `0.9412 us/chunk` |
| native | `0.5050 us/chunk` |

### 核心判断

1. **Z5B 的 `+532 VMEM/CTA` 不是 Q duplicate 的残留。** Z5B 已通过
   dedicated Q cache 将 Q global producer 降到一个 source/LLVM region；当前
   Q 的成本是一次仍然偏窄的 scalar BF16 fill。剩余 VMEM 主要由整个 BF16
   producer/materialization 路径、g 的多角色 global load，以及输出/阶段边界
   共同构成。由于只有 aggregate VMEM PMC，不能伪造六类 operand 的精确分摊。
2. **`+3696 VALU/CTA` 不能被一个 opcode family 精确闭账，但最终机器图已经
   收敛到两个高可信族：**
   - scalar ownership/address/select 与 phase-specific LDS address recipe；
   - `<4 x i32> -> <8 x bf16> -> extract/insert -> <4 x bf16>` 的 fragment
     reconstruction/feeding。
3. **native 的优势是 work reduction 和 overlap improvement 的组合。** native
   同一个 `scf.for` 同时持有 Q/H/K typed local blocks，Q local operand 直接
   同时喂 `Q*H` 和 `Q*K` 两个 dot；ISA 中 next buffer load 在前一批 MFMA 未结束
   前已经发起。Z5B 则把 Q fill、H/inter、score source-half、score conversion、
   V-new/intra 分成更细的 producer/consumer phase，并在窄 load 后更早等待。
4. 本轮唯一登记的下一步是 **`Z6G typed FP32 g-tile residency`**，只设计不实现：
   让当前 chunk/head 的 g tile 在 score target/source 与 final scaling 两个
   consumer role 之间复用。它由 source/LLVM 的多角色证据支持，不需要创建
   新 Qwen op，也不应和 Q/K/V-new planner 变化叠加。

机器可读 stage map：

[`stage6z_z5b_vs_native_stage_map.json`](./stage6z_z5b_vs_native_stage_map.json)

机器可读 root-cause ledger：

[`stage6z_z5b_vs_native_root_causes.json`](./stage6z_z5b_vs_native_root_causes.json)

---

## 1. 审计边界、身份和证据等级

### 1.1 Z5B identity

源码：

[`qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer.py`](../../vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer.py)

关键 source ownership：

| source 行 | 内容 |
|:--|:--|
| `93-98` | 一个 32 KiB shared allocation；低区为完整 Q cache，高区为原 phase area |
| `100-111` | `k_stage=0..3`、`rep=0..7` 的唯一 Q global producer |
| `116-138` | Phase A：H producer，Q 从 Q cache 直接取，生成 inter |
| `140-165` | Phase B：两个 source half，各四个 K32 stage，Q 继续从 Q cache 取 |
| `167-179` | causal/gating/score BF16 store |
| `181-201` | V-new producer 和 score@V-new intra MFMA |
| `203-209` | final g scaling、FP32 composition、BF16 output store |
| `212-239` | hard WG256 launch contract，无 WG128 fallback |

Z5B exact machine artifact：

```text
codex_qwen_bt64_stage6z_z5b_machine_stage1/
```

| 字段 | 值 |
|:--|:--|
| kernel | `_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z5b_direct_q_cache` |
| HSACO SHA256 | `979889c1a10ac53064cd1d2da91298b67e4ef7f2db3baadc171872cb56b0ed67` |
| code-object WG | `256` |
| code-object LDS | `32768 B` |
| code-object VGPR/AGPR/SGPR | `104/32/28` |
| private segment | `0 B` |
| VGPR/SGPR spill | `0/0` |

该目录的 `machine_evidence.json` 写有 `launch_executed=false`。这只说明该次
机器 dump capture 没有在 dump 命令中 launch kernel，不能拿它否定独立 PMC/body
capture。本文使用的动态 PMC 和 latency 来自已有 fresh-process benchmark/rocprof
工件，并在表中明确区分两者。

### 1.2 Native identity

同形状 native 工件：

```text
codex_qwen_bt64_stage6z_fixed_z2_vs_native_audit/machine/native_T2048/
```

| 字段 | 值 |
|:--|:--|
| kernel | `chunk_fwd_kernel_o` |
| HSACO SHA256 | `cc74acf84104a0900ed327fc469826c228c94fdf07fd55d8d302c3e88c04e72d` |
| WG/wave | `256 / wave64` |
| code-object VGPR/AGPR/SGPR | `132/32/89` |
| private/spill | `0/0` |
| static group segment metadata | `0`，但跨 collector/selected capture 不一致，不用它做因果结论 |

native 的 dynamic LDS `480/CTA` 是本轮使用的 LDS 工作证据。不同 capture 对
`group_segment_fixed_size` 的 metadata 不一致，因此不把一个静态 LDS 字段当成
native 没有 shared memory，也不把它用于解释 latency。

### 1.3 证据等级

| 等级 | 含义 |
|:--|:--|
| A | source/IR/final ISA 或 PMC 能闭合逻辑归属，或直接观测到机器事实 |
| B | 能闭合结构和方向，但不能恢复精确动态执行次数 |
| C | 只有静态相似性/上下界/地址形态，不能据此宣称精确贡献 |
| N/A | 当前工件不足，明确不推断 |

本报告始终遵守：

```text
static lexical instruction count != dynamic hardware count
VMEM instruction count != accessed byte count
```

---

## 2. 完整 logical stage map

### 2.1 Z5B 的高级阶段

```text
A0 Q cache fill
   ↓ barrier
A  Q @ H^T -> inter
   ↓ barrier
B  Q @ K^T -> score，source_half 0/1
   ↓
C  causal/gating/g/score BF16 conversion
   ↓ barrier
D  score @ V-new -> intra
   ↓
E  inter * exp(g_target) + intra -> BF16 output
```

Z5B 的 lowered LLVM block 是最可靠的阶段锚点：

| logical stage | lowered LLVM block | 直接证据 |
|:--|:--|:--|
| A0 | `80-152` | Q pointer `%0` 的 `load bfloat`、addrspace(3) Q store、barrier |
| A | `153-311` | H load、Q/H shared `<4 x i32>` load、两个 MFMA call site |
| B | `313-480` | K load、Q/K shared load、score MFMA，`value_half` predicate |
| C | `482-546` | score accumulator、g target/source load、`llvm.exp.f32`、score store |
| D | `548-718` | V-new load/store、score/V shared load、intra MFMA |
| E | `718-774` | final g load、exp、FP32 add、BF16 output store |

### 2.2 Z5B ISA PC 映射

这些 PC 是 final ISA 中的代表性 unrolled/CFG cluster，不是假设一条连续的
source-to-PC debug line table：

| stage | 代表性 PC | 机器模式 |
|:--|:--|:--|
| A0 | `0x1988-0x2c58` | `global_load_ushort -> s_waitcnt -> ds_write_b16_d16_hi -> barrier`，共四个 K stage producer cluster |
| A | `0x2c8c-0x30b7` | H 的窄 BF16 load/store、Q/H `ds_read_b128`、MFMA |
| B | `0x30b8-0x34c0` | K 的窄 BF16 load/store、Q/K `ds_read_b128`、score MFMA |
| C | `0x34c0-0x458c` | FP32 g loads、causal/exp/gating、score BF16 phase store |
| D | `0x45a8-0x4a5c` 与 `0x59ec-0x5ecc` | V-new producer、shared reads、intra MFMA；第二段是 CFG continuation，不是第二次算法阶段 |
| E | `0x4a98-0x5f34` 与 `0x5f34-0x67bc` | final g/output path 与 BF16 stores；部分 late continuation 无法只按线性 PC 唯一归属 |

Z5B final ISA lexical total：

| family | count |
|:--|--:|
| `global_load_ushort` | `112` |
| `global_load_dword` | `80` |
| `global_load` total | `192` |
| `global_store_short_d16_hi` | `16` |
| `ds_read_b128` | `56` |
| `ds_write_b16` + `ds_write_b16_d16_hi` | `144` |
| `v_mfma_f32_32x32x8_bf16` | `56` |
| `s_waitcnt` | `214` |
| `s_barrier` | `32` |

注意：`56` 是 lexical MFMA，不是动态 MFMA。Z5B 的动态 PMC 是 `160/CTA`。
PC 上有 conditional/CFG continuation，因此不能用一段 PC 的 MFMA 行数直接
替代 source loop 的实际 wave 执行。

### 2.3 Native 的 logical stage

native TTGIR 显示了一个与 Z5B 不同的结构：

```text
local_alloc Q/H/K
    ↓ local_store Q/H/K
一个 scf.for
    ├─ typed local_load Q
    ├─ typed local_load K
    ├─ typed local_load H + trans
    ├─ tt.dot(Q, H, inter)
    └─ tt.dot(Q, K, score)
```

直接证据在：

```text
chunk_fwd_kernel_o.ttgir:122-124   Q/H/K local_alloc
chunk_fwd_kernel_o.ttgir:172-176   Q/K/H local_store
chunk_fwd_kernel_o.ttgir:177       一个 scf.for
chunk_fwd_kernel_o.ttgir:196-217   local_load 与两个 tt.dot
```

最关键的是 `%b_q_275` 同时作为 `%b_o_295` 和 `%b_A_296` 的 Q operand。也就是
native 在同一 loop 内将一个 typed Q operand 交给 `Q*H` 与 `Q*K`，不是先完成
一个完整 Phase A，再重新建立 Phase B 的 Q consumer。

native final ISA 的 stage cluster：

| stage | 代表性 PC | static lexical evidence |
|:--|:--|:--|
| A+B | `0x1954-0x1eec` | 12 `buffer_load_dwordx4`、16 packed LDS writes、40 LDS reads、32 MFMA、7 barrier |
| C | `0x1eec-0x2aa8` | 17 FP32 `global_load_dword`，g/scale cluster |
| D | `0x2aa8-0x3360` | 2 `buffer_load_dwordx4`、24 LDS writes、40 LDS reads、8 MFMA、4 barrier |
| E | `0x3360-0x3534` | 4 `buffer_store_dwordx2` |

native total static manifest：

```text
buffer_load=14, buffer_store=4, global_load=17,
ds_read=80, ds_write=40, mfma32=40,
s_waitcnt=48, s_barrier=11
```

native 的 `40 lexical MFMA x 4 waves = 160 dynamic MFMA/CTA` 与 PMC 对得上；
Z5B 的 `56 lexical MFMA` 则不能这样直接相乘，进一步说明 Z5B 有更多
conditional/CFG/fragment materialization 路径。

---

## 3. VMEM `+532/CTA` 的闭账

### 3.1 可以确认的事实

每个 CTA 的 unique logical tile 下界为：

| operand | shape | dtype | unique logical bytes |
|:--|:--|:--|--:|
| Q | `[64,128]` | BF16 | `16384 B` |
| K | `[64,128]` | BF16 | `16384 B` |
| H | `[64,128]` | BF16 | `16384 B` |
| V-new | `[64,64]` | BF16 | `8192 B` |
| g | `[64]` | FP32 | `256 B` |
| output | `[64,64]` | BF16 | `8192 B` |

这些是逻辑字节，不是 VMEM bytes，也不等同硬件 transaction。

Z5B source/LLVM 可以直接证明：

- Q：只有 A0 一次 Q global producer；A/B 后续是 dedicated Q LDS cache 的
  `addrspace(3)` `<4 x i32>` read。Q 三次 global pass 的旧问题已被 Z5A/Z5B
  关闭。
- H：Phase A 一次 H producer pass；没有足够 debug provenance 证明两个
  value-half 把相同 H tile 完整重复生产。
- K：两个 source-half 是数学/ownership 分区；不能把两个 half 当成重复读取。
- V-new：一个 source-level producer pass；窄 BF16 load 是确定的，但重复 global
  producer 未被证明。
- g：score target/source 两个角色和 final scaling 另一个角色在 source/LLVM
  中都可见，存在跨 logical consumer 的重复 load opportunity。
- output：public BF16 store 是必需 ABI 工作，不能直接删除。

### 3.2 静态机器 work 对比

| family | Z5B | native | 机器含义 |
|:--|--:|--:|:--|
| BF16 global load family | `112 global_load_ushort` | 主要由 `14 buffer_load_dwordx4` 承担 | Z5B 是 scalar/narrow producer；native 是 typed packet producer |
| FP32 g load family | `80 global_load_dword` | `17 global_load_dword` | Z5B 有更多 score/final g load region；native g 路径更 compact |
| output store | `16 global_store_short_d16_hi` | `4 buffer_store_dwordx2` | Z5B output store 更窄；必须保留公共 BF16 输出 |

不能直接把 `112-14` 或 `80-17` 乘四后写成 dynamic VMEM 差值。它们只能构成
upper/modeling evidence，因为 Z5B 有 lane mask、conditional blocks 和 CFG
continuation。对于 native，31 个 static load family 和 4 个 packet store 与
140/CTA 的 selected capture 一致；对于 Z5B，static memory family 只提供
上界/结构模型，`672/CTA` 仍以 PMC 为准。

### 3.3 按 logical stage 的 VMEM 方向

| stage | Z5B 可观测/可建模的主要 VMEM | native 结构 | 结论 |
|:--|:--|:--|:--|
| A0/A | Q fill + H producer；Q 先独立填 cache，H 再进入 phase | Q/H/K packet 在同一 typed loop 中预取/复用 | Z5B 有独立 Q producer 和窄 H producer；Q duplicate 已消除，但 producer ownership 仍不如 native |
| B | K source-half/K32 producer，Q 只读 LDS | K typed block 与 Q typed operand 同 loop | 主要是 packet/layout/consumer grouping 差距；不能证明 K 是同一 tile 的无条件重复 producer |
| C | g target/source scalar roles | 17 个 native g load lexical cluster，多 consumer block | g 是唯一能从 source/LLVM 确认多角色重复机会且有最大模型机会的候选 |
| D | V-new 窄 BF16 producer | 2 个 `buffer_load_dwordx4` + typed local path | Z5B 有窄 load/materialization，但重复 global read 尚未证明 |
| E | final g + 16 个窄 BF16 stores | 4 个 packet stores | 输出必须保留；g final role 可通过 residency 重新审计 |

### 3.4 `+532` 能闭账到什么程度？

不能逐 operand 精确闭账。当前可以给出：

1. **确定存在的 work excess：** Z5B 采用 112 条 lexical `global_load_ushort`
   和 80 条 `global_load_dword`，native 采用更少的 typed packet family；这
   证明 producer packet/ownership 在机器上不同。
2. **确定不是当前主因的旧项：** Q cache→phase Q republish 已由 Z5B 删除；
   Z5B 的 VMEM 与 Z5A 相同，说明 Z5B 的主要收益是 LDS，不是进一步减少 Q
   global load。
3. **最大可建模单一剩余项：** g 的 score target/source + final scaling
   opportunity 约 `192` 个 FP32 scalar issuing opportunity。它不是 `672` 中的
   exact 192，但同时有 source/LLVM 多角色证据、native 对照和地址 VALU 伴随
   成本。
4. **其余 `+532`：** 由 Q-fill/H/K/V-new 窄 packet 与 phase materialization
   组成，但现有 aggregate counter 不能唯一分摊。

因此，报告不写“g 解释了全部 +532”；只写“g 是当前唯一具有最高可建模
单项机会、且可用一个独立 control variable 验证的 offender”。

---

## 4. VALU `+3696/CTA` 的最终机器归因

### 4.1 总量和静态 recipe 对照

Z5B 的 final ISA 关键 static proxy：

| family | Z5B | native | 差异方向 |
|:--|--:|--:|:--|
| `v_lshl_add_u64` | `107` | `34` | Z5B 更多 64-bit global/LDS offset recipe |
| `v_add3_u32` | `80` | `35` | Z5B 更多 affine index composition |
| `v_or_b32` | `190` | `54` | Z5B 更多 packed/index/ownership bit composition |
| `v_cndmask_b32` total | `240` | `97` | Z5B 更多 predicate/select path |
| `v_accvgpr_read_b32` | `64` | `48` | Z5B 更多 accumulator feeding read |
| `v_accvgpr_write_b32` | `32` | 未见同等 write family | Z5B 存在额外 accumulator materialization |
| `v_perm_b32` | `0` | `32` | native 也有交换指令，不能把所有 layout work 都归给 Z5B |

这些是 static lexical pattern，不是 dynamic VALU。它们不能相加后等于
`3696`，但提供了比“总 v_* 行数”更可信的机器配方证据。

### 4.2 第一大族：BF16 fragment reconstruction / operand feeding

Z5B lowered LLVM 在 A、B、D 多处都有相同结构：

```text
addrspace(3) load <4 x i32>
    -> bitcast <4 x i32> to <8 x bfloat>
    -> 8 次 extractelement / insertelement
    -> 重新组装 <4 x bfloat>
    -> MFMA intrinsic
```

具体证据：

| 区域 | lowered LLVM |
|:--|:--|
| Phase A | `293-374` |
| score | `493-577` |
| intra | `783-865` |

native TTGIR 则直接使用 `ttg.local_load` 产生 `#ttg.dot_op` typed operand，
由 `%b_q_275`、`%b_k_286`、`%b_o_294` 进入 `tt.dot`。native 不是没有 layout
工作；它把 layout/fragment contract 保留到 dot operand，而 Z5B 在 LLVM
阶段已经把它拆成多个 scalar vector element boundary。

final ISA 也看到 Z5B 的 accumulator feeding family：

```text
Z5B: v_accvgpr_read_b32 = 64, v_accvgpr_write_b32 = 32
native: v_accvgpr_read_b32 = 48，未见同等 write family
```

这不能宣布“96 条 AGPR transfer 就等于 3696 VALU”，但它确认了 fragment/
accumulator feeding 是实际 machine work，而不是 source 变量名上的猜测。

### 4.3 第二大族：address/index/ownership/select

Z5B 在每个 producer phase 重新计算：

```text
tid -> rep -> row/col
source_half/k_stage -> phase row
value_half/lane_group -> LDS word
global stride -> byte/word address
valid/causal predicate -> cndmask/select
```

这些 recipe 直接出现在：

- Q cache fill 的 `k_stage/rep` address block；
- H Phase A 的 H phase offset；
- K Phase B 的 `source_half + k_stage + rep` offset；
- C 阶段 target/source g；
- D/E 的 V-new/output address。

`v_lshl_add_u64`、`v_add3_u32`、`v_or_b32`、`v_cndmask` 的 Z5B static proxy
都高于 native。native TTGIR 虽然也有大量 arith/index，然而它先生成 blocked
tensor coordinates，再由 typed buffer/local/dot encoding 统一降低，且同一 loop
内复用 Q/H/K coordinates。

这族一定是 `+3696` 的可信组成，但无法由 aggregate VALU PMC 给出准确动态
百分比。

### 4.4 第三大族：phase-specific materialization 与 overlap 损失

它不是纯 VALU opcode family，却同时推动 VALU 和 latency：

- Z5B 有 `ds_write_b16`/`ds_write_b16_d16_hi` 共 `144` 条 lexical store；
- native 使用 packed `ds_write_b64/b128/ds_write2st64_b64` 等 family，共 `40`
  条 static LDS write；
- Z5B `s_barrier=32`，native `s_barrier=11`；
- Z5B `s_waitcnt=214`，native `s_waitcnt=48`；
- Z5B 许多地址/ownership recipe 是为 phase buffer 的当前用途重新生成的。

这里不能把 `144-40` 直接算成动态 VALU，也不能把 barrier 差直接算成微秒。
正确结论是：Z5B 的 producer/consumer boundary 更碎，导致更多 address/select/
fragment work，并缩小了 load 与 MFMA 的重叠窗口。

### 4.5 未闭账的部分

`7072-3376=3696` 的剩余部分包括：

- g/exp/causal/output 数学中必须保留的 FP32 arithmetic；
- scalar BF16 conversion/pack/unpack；
- branch/CFG continuation 中无法唯一映射到一个 logical stage 的 address/select；
- native 的 `v_perm`、packed FMA/MUL 与 Z5B scalar math 的编码差异。

本轮不把这些部分伪造为某一个单一 root cause。只要没有 per-PC dynamic
counter，就不能从 static opcode 数量反演精确动态贡献。

### 4.6 VALU 排名

| 排名 | family | 证据 | 是否解释全部 +3696 |
|--:|:--|:--|:--|
| 1 | typed BF16 fragment reconstruction / accumulator feeding | Z5B LLVM extract/insert 三处重复 + final ISA AGPR read/write；native typed dot-op | 否，但最高机器证据 |
| 2 | address/index/ownership/select recipe | Z5B `v_lshl_add_u64/v_add3/v_or/cndmask` static proxy 明显更高 | 否，只有方向和上界 |
| 3 | phase-specific materialization / synchronization-related helper work | Z5B 32 barrier/214 wait、窄 LDS stores；native 11/48 | 不是 VALU 的独立计数，但在 critical path 上 |

---

## 5. 执行依赖和关键路径对齐

### 5.1 本轮采用的 dependency proxy

对 final ISA 做了 nearest-forward lexical scan：

```text
load -> next s_waitcnt
waitcnt -> next ds_read
ds_read -> next s_barrier
s_barrier -> next MFMA
```

这是静态文本距离，不是 cycle、wave issue timestamp 或真正的 hardware stall。
它的价值是比较两种机器图的依赖组织，不是对 latency 做一条线性方程拟合。

| proxy | Z5B | native |
|:--|:--|:--|
| load→wait count | 194 | 31 |
| load→wait median | 5 instructions | 51 instructions |
| wait→ds_read median | 16 | 6.5 |
| ds_read→barrier median | 6 | 18 |
| barrier→MFMA median | 7 | 7 |
| static `s_waitcnt` | 214 | 48 |
| static `s_barrier` | 32 | 11 |
| lexical MFMA/barrier | `56/32=1.75` | `40/11=3.64` |

### 5.2 Z5B 的典型顺序

Q/H/K/V-new 的一个代表性 pattern 是：

```text
global_load_ushort
  -> s_waitcnt vmcnt(...)
  -> ds_write_b16
  -> ...
  -> s_waitcnt lgkmcnt(0)
  -> s_barrier
  -> ds_read_b128
  -> s_waitcnt lgkmcnt(...)
  -> MFMA
```

这个 pattern 有两个重要后果：

1. 每个 scalar/narrow producer 很快就被 wait 所约束，load lookahead 窗口短；
2. phase buffer 的用途切换需要多次 CTA-wide barrier，MFMA 被切成多个短 run。

Z5B 的 source-level phase separation 对 correctness 和资源控制有价值，不能
直接批量删 barrier；本轮只把它当作机器差异记录。

### 5.3 Native 的典型顺序

native ISA 在 `0x1954-0x1eec` 看到：

```text
buffer_load_dwordx4 Q/K/H
  -> packed LDS stores
  -> barrier
  -> 多组 ds_read_b64/ds_read2_b64
  -> MFMA cluster
  -> 在当前 MFMA cluster尚未完全结束时继续 buffer_load_dwordx4
  -> 下一组 packed LDS stores / barrier
```

最有用的具体 PC 证据：

| PC | 事件 |
|:--|:--|
| `0x1954/0x198c/0x199c/0x19bc` | 初始 Q/K/H packet load |
| `0x19fc-0x1a3c` | packed LDS store |
| `0x1a48` | barrier |
| `0x1a4c-0x1ae8` | local read 与 MFMA cluster |
| `0x1b00/0x1b14/0x1b44` | 下一批 packet load，在后续 MFMA/barrier 前已发起 |
| `0x1b20` | 后续 phase barrier |

TTGIR 同时证明 Q operand 的逻辑复用：

```text
%b_q_275 = ttg.local_load Q
%b_o_295 = tt.dot %b_q_275, %b_o_294, inter
%b_A_296 = tt.dot %b_q_275, %b_k_286, score
```

因此 native 少 barrier/waitcnt 不是因为“没有同步”，而是因为：

- Q/H/K 的 shared/dot operand lifetime 被统一规划；
- 一次 typed packet 可以服务多个 dot consumer；
- 一个 loop/phase 可以让 next producer 与 current MFMA 重叠；
- packed LDS layout 减少了逐元素 producer/store 和 fragment rebuild；
- barrier 只放在实际 shared ownership/lifetime 切换处。

### 5.4 work 量还是 overlap？

答案是 **二者共同作用，但 work 量是首要、overlap/serialization 是放大器**。

支持 work 量首要的事实：

- MFMA 动态工作完全相同：`160 vs 160`；
- VMEM 多 `532/CTA`；
- VALU 多 `3696/CTA`；
- LDS 多 `192/CTA`；
- 同 shape、同 ABI 下 body latency 仍为 `1.57x/1.72x`。

支持 overlap 也是真实因素的事实：

- Z5B static barrier/wait `32/214`，native `11/48`；
- native load→wait median `51`，Z5B `5`，说明 native 允许更大的 issue window；
- native 每个 barrier 间平均 lexical MFMA `3.64`，Z5B `1.75`；
- native 有明确 next buffer load 与当前 MFMA 交错的 PC 证据。

不能进一步说“work 占 70%、serialization 占 30%”，当前 artifact 没有这种
因果分解能力。准确结论是：**Z5B 先做了更多机器工作，再以更碎的 phase/wait
组织削弱了工作重叠。**

---

## 6. Ranked root-cause table

| 排名 | root cause | Z5B final-machine evidence | native evidence | estimated excess | critical path | 最小可控位置 | 需要大 block_dot semantic abstraction? |
|--:|:--|:--|:--|:--|:--|:--|:--|
| 1 | BF16 producer packet + typed fragment feeding 不紧凑 | `load bfloat`、`global_load_ushort=112`、`ds_write_b16`、LLVM extract/insert 三处、AGPR read/write | `buffer_load_dwordx4=14`、typed `local_load -> dot_op`、packed LDS | 对 VMEM/VALU/LDS 同时有高影响；精确数 unresolved | 是，Q/H/K/D 都在主 MFMA 输入链 | producer-layout/consumer-dot boundary | **若一次统一改 Q/H/K/V 的 owner/layout，需要；不能新建 Qwen op** |
| 2 | g target/source/final 多角色 global residency 缺失 | `global_load_dword=80`；LLVM C `639-648`，E `905-906`；source 两个 consumer role | native g cluster `17` 条 lexical FP32 load，block/scaling path | modeled 约 `192` issuing opportunities；不是 PMC exact | 是，C score 进入 D，E 是完成路径 | 现有 g producer/consumer lifetime；可独立 A/B | **不需要新 block_dot；可用一个 g-tile residency source/lowering control** |
| 3 | phase-specific address/ownership/select recipe | `v_lshl_add_u64=107`、`v_add3=80`、`v_or=190`、`cndmask=240` | native 对应 recipe 更 compact，且 coordinates 随 typed block 复用 | 方向明确，dynamic exact unresolved | 大量位于 load→LDS→MFMA 链 | 通用 physical layout/ownership plan | 若跨 producer/consumer 统一，复用 block_dot；单 g 实验不需要 |
| 4 | phase/barrier/wait serialization | `32 barrier/214 wait`，每个窄 load 后很快 wait | `11 barrier/48 wait`，next load 与 MFMA overlap | 不能换算成静态微秒；对 slope 有明确风险 | 是 | phase-aware scheduling/lifetime | 需要完整 phase 依赖信息，但不是单纯 RA |
| 5 | output narrow store | `16 global_store_short_d16_hi` | `4 buffer_store_dwordx2` | 有限且是 ABI 必需工作 | E completion path | output packet store lowering | 不需要 |

### 为什么没有把“Q duplicate”列为当前第一项？

因为 Z5B 已经通过 source/LLVM 证据消除了它：Q pointer `%0` 只有 A0 producer
region；A/B 的 Q consumer 都是 dedicated Q cache 的 addrspace(3) read。继续
把旧的 Q duplicate 当作当前根因会重复做已完成的 Z5A→Z5B 工作。

---

## 7. 唯一下一步设计：Z6G typed FP32 g-tile residency

本轮不实现，只登记一个控制变量。

### 7.1 为什么选 g

此前剩余 VMEM ledger 已经对六类 operand 做过 source/LLVM ownership 审计；在
本轮 final-machine 对齐中，g 仍然是唯一同时满足以下条件的单项：

1. source 中有两个以上明确 logical consumer role：score target/source 与
   final scaling；
2. LLVM 能看见两个不同 g pointer load region；
3. static ISA `global_load_dword` 为 `80`，native 为 `17`，方向和 native
   packet/ownership 对照一致；
4. modeled issuing opportunity 约 `192`，是当前单一 operand 中最大的可建模
   opportunity；
5. g load/index 直接位于 C/E 的依赖链，并伴随 address/VALU；
6. 可以只修改 g 的 producer/consumer lifetime，不触碰 Q/K/H/V-new、MFMA、
   accumulator、recurrence 或 public output contract。

### 7.2 设计边界

候选数据流：

```text
current chunk/head g global producer
        -> 一个最简单的 typed FP32 g tile residency
        -> score target/source consumer
        -> final scaling consumer
```

必须保持：

- BT64/BV64/BK32/WG256/2 CTA；
- Q cache、K/H/V-new/output 路径不变；
- MFMA 数学、causal mask、gating/exp 数学顺序不变；
- 不把 g 做成大型 private array；
- 不改 allocator/RA、selector、X2 或 production；
- stable/ideal 若同时做，必须是两个独立 arm，不能叠加；
- correctness 先于性能；没有 per-operand exact PMC 时只能比较总量变化和
  machine evidence，不能声称 g 精确占掉多少 VMEM。

### 7.3 预期可以观察的证据

不是预设“VMEM 一定下降到某个数字”，而是检查：

1. g global producer region 是否从两个/多角色 consumer path 收敛为一个；
2. C/E 是否改为从同一 residency value/cache 取 g；
3. Q/K/H/V-new/output 的 LLVM/ISA graph 是否保持不变；
4. g 相关 `v_lshl_add`/index/select 是否随之下降；
5. 是否增加新的 LDS round trip、register live range 或 occupancy cliff；
6. T=2048 和 T=8192 body latency 是否都真实改善。

### 7.4 为什么不是立即做 P5、Q-A 或 V-new planner

- **不是 P5：** P5 会同时改完整 chunk residency/多个 operand 生命周期；本轮
  已经证明“全图更宽”不能替代一个可归因的控制变量，且会重新混入 Q/K/H/
  V-new 的不确定性。
- **不是 Q-A：** Z5B 已完成 Q duplicate 消除；继续做 Q pass/phase 变体会
  重复已关闭问题，且不能解释 g 的明确多角色 load 区域。
- **不是 V-new planner：** V-new 的窄 load 确定存在，但当前工件不能证明它
  有比 g 更大的重复 producer；也不能把它从 aggregate 672 中精确分出来。
- **g residency 风险和收益更可控：** 它位于一个明确的 logical input，能在
  不改变 MFMA/accumulator/ownership 的情况下做 source-level A/B，正好符合
  “一个最大、可验证 offender”的停止条件。

这个下一步不等于宣称 g 是全部 native gap；它只是在当前证据下最干净、最值得
先验证的一刀。

---

## 8. 对任务最后八个问题的直接回答

### 1) 过去已经知道 native VALU/VMEM 少，本轮新增的机器级证明是什么？

新增的不是再次报总数，而是完成了：

```text
logical A-E stage
 -> Z5B LLVM block
 -> representative final-ISA PC cluster
 -> static wait/barrier/MFMA sequence
 -> source loop multiplicity
```

具体新增证据包括：

- Z5B 的 LLVM `153/313/482/548/718` 五组 block 与 A-E 对齐；
- native `%b_q_275` 同时喂两个 `tt.dot` 的 producer-consumer reuse 证据；
- native `0x1b00/0x1b14/0x1b44` next packet load 在 MFMA cluster 内提前发起；
- Z5B/native 的 lexical dependency proxy：load→wait median `5 vs 51`、
  barrier `32 vs 11`、MFMA/barrier `1.75 vs 3.64`；
- Z5B 的 fragment reconstruction 与 AGPR read/write 在 A/B/D 的 LLVM/ISA
  证据，而非仅凭 profiler AccVGPR 猜测。

### 2) `+532 VMEM/CTA` 主要来自哪些 logical stages 和 producer 行为？

主要来自：

1. A0/A/B/D 的窄 BF16 producer 与 phase materialization；
2. C/E 的 g 多角色 scalar FP32 load；
3. E 的窄 BF16 output store和相关地址工作。

Q 的三次重复 global producer不在 Z5B 当前图中；Q 仍有一次偏窄 fill，但不能
称为 duplicate。六类 operand 到 `672` 的 exact attribution 当前无法由 aggregate
PMC 闭合。

### 3) `+3696 VALU/CTA` 能闭账多少，前三大来源是什么？

不能逐条精确闭账。前三大可信族是：

1. typed BF16 fragment reconstruction / accumulator feeding；
2. address/index/ownership/select recipe；
3. phase-specific materialization 和同步相关 helper work。

它们分别有 LLVM extract/insert、AGPR read/write、ISA arithmetic/cndmask 与
32/214 对 11/48 barrier/wait 证据，但不能相加伪造为 3696。

### 4) native 少 barrier/waitcnt 的真实结构原因？

不是 native 没有同步，而是 Q/H/K typed block、shared layout、dot operand 和
loop lifetime 一起规划：一个 Q operand 服务两个 dot，packed store/read 承担
更多逻辑元素，next producer 可以在 current MFMA 期间提前发起，barrier 只保护
真实 phase/lifetime 切换。

### 5) latency root cause 主要是 work 量、serialization/overlap，还是两者？

两者。`+532 VMEM/+3696 VALU/+192 LDS` 说明 work 量是首要差距；`32/214` 对
`11/48`、load→wait 窗口和 MFMA/barrier 比说明 Z5B 更碎的 phase 组织又削弱
overlap。当前不能给两者做百分比因果分解。

### 6) 最大 root cause 是否真的需要大 block_dot op？

**整个 native gap 的 producer-layout-consumer 组合问题，确实跨越单条 load，
需要某种统一的 typed operand/physical plan；但不能据此新造 Qwen 专用 op。**
AveLang 已有 `block_dot`/full-scope abstraction 是合适的承载位置。

不过本轮登记的下一刀 g residency 是更窄的单 operand 实验，**不需要新建大
block_dot op**。先用它验证多角色 residency 是否能把 work 和关键路径一起收窄。

### 7) 下一刀预计能删除/隐藏哪些 final machine work？

只针对 Z6G 设计预期：

- 减少 g score target/source 与 final scaling 的重复 global load issuing；
- 减少对应的 `v_lshl_add`/index/ownership address recipe；
- 让 C→D 与 D→E 的 g value/cache consumer 更连续，减少一部分 wait/phase
  pressure。

不能预先承诺会删除多少 VMEM/VALU，也不能把 g 的 `192` 模型机会写成精确 PMC。

### 8) 为什么这比继续 P5/Q-A/V-new planner 更值得做？

因为它是唯一同时具备“可建模较大、source/LLVM 归属清楚、native 对照存在、
位于完成依赖链、改动边界单一”的剩余候选。它不会重新改变已经正确且有效的
Q cache、MFMA、accumulator 或 V-new ABI，最适合作为下一轮 controlled A/B。

---

## 9. 最终状态

```text
Z5B = current isolated performance baseline
Z5B vs native gap = work excess + phase/overlap difference
Q duplicate = closed in Z5B
exact per-operand VMEM split = unresolved by available aggregate PMC
exact per-family VALU split = unresolved by available aggregate PMC
next design = Z6G typed FP32 g-tile residency, design only
P5/Q-A/V-new planner = not started in this audit
X2/production = unchanged
```

本轮停止在审计和单一 next-step recommendation，不自行开始 Z6G 实现。
