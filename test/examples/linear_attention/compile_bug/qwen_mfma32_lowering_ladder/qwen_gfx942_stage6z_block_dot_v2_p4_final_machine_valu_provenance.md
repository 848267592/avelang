# Qwen gfx942 Stage6Z BDV2-P4：Final-Machine VALU Provenance Closure

## 1. 结论先行

本轮完成了两个连续动作：先对 Z5B、P3-S 和 native WG256 做 final-machine
VALU provenance 审计，再根据审计中最强的单一证据实现 P4。P4 没有新增
Qwen/chunk-o public op，也没有新建 Python kernel fork；它只在已有
`al.amdgpu.block_dot_bf16_f32` 的通用 full-scope lowering 中启用一个内部的
accumulator forwarding 表示。

最终结论是：

1. P3 相对 Z5B 多出来的主要可疑工作不是 global/LDS load 数，而是
   full-scope accumulator/fragment feeding：`inter_acc`、MFMA 结果、vector
   slice、`COPY/REG_SEQUENCE` 和 AGPR read/write 之间形成了重复的机器表示。
2. P4 确实去掉了高层 `vector<32xf32>` accumulator 的冗余高半部分，并让
   accumulator SSA 结果在同一 reset boundary 内向后转发。这个差异保留到了
   post-materialization MLIR、LLVM、exact-LTO MIR、final ISA 和 HSACO。
3. 但是最终 ISA 的核心 accumulator read/write 数没有下降，P4 只是把静态
   `v_mov_b32` 减少 32 条，同时增加了 32 条 `v_cndmask_b32_e64`。因此动态
   `VALU=8410/CTA` 与 P3 完全相同，不能把 P4 宣称为机器工作减少。
4. P4 资源分配变好：code-object `VGPR/AGPR` 从 P3 的 `132/48` 变成
   `104/32`，无 scratch、无 spill；但 body latency 反而比 Z5B 高约
   `5.18%`（T=2048）和 `6.95%`（T=8192）。
5. P4 是一个正确、通用、可复用的 representation/infrastructure 修复，
   但性能 No-Go。Z5B 继续是 Stage6Z isolated performance baseline；P4
   不建立 selector、不接 X2、不接 production。

机器可读的完整归因表见：

`machine-readable valu_provenance_p3_vs_z5b_vs_native.json`

## 2. 冻结边界与比较对象

所有 AveLang full-scope arms 使用同一份高层 source：

`vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_block_dot_v2_full_scope.py`

本轮没有复制 source。内部 preservation selector 只有：

```text
none                 -> Z5B 对照表示
p3_packed_reuse      -> P3 frozen control
p4_accumulator_reuse -> P4
```

冻结的逻辑和硬件契约：

| 项目 | 值 |
|:--|:--|
| target | gfx942 |
| BT/BV/BK | 64 / 64 / 32 |
| workgroup | 256 |
| CTA | 2 CTA/chunk-head |
| dtype | BF16 ABI，内部 FP32 accumulator |
| MFMA | `v_mfma_f32_32x32x8_bf16` |
| dynamic MFMA | 160/CTA |
| Q | dedicated full-Q cache，source producer pass=1 |
| K/H | BDV2 full-scope common planner |
| accumulator | phase-separated，未改变 accumulation order |
| output | caller-owned BF16 output |
| 修改对象 | 只限通用 block-dot accumulator representation |

没有改动 Q/K/H/V-new/g/output 的数学、ownership、WG、MFMA geometry、RA、
selector 或 production dispatch。

## 3. 证据等级和限制

本报告使用三类证据，避免把不同层次的数据混在一起：

| 等级 | 含义 |
|:--|:--|
| A | 直接观测：rocprof dynamic PMC、code-object metadata、最终 ISA mnemonic 计数 |
| B | 机器结构证据：MIR def/use、basic block/loop、LLVM 与 ISA 邻近关系 |
| C | 由源码和 lowering 语义推断、但不能单独推出动态执行次数 |

T=2048 的 dynamic PMC 是每个 kernel 的总值除以 512 个实际 CTA；它不能被
静态 ISA 行数替代。静态 `v_*` 行数也不能直接视为 dynamic VALU。对于没有
可用 instruction-level hardware counter 的 provenance 类别，本报告只给出
排序、上界或相对证据，不伪造每条指令的动态精确账。

AveLang 当前 binding 的初始 `get_mlir()` 路径仍会 SIGSEGV。因此 P3/P4 machine
artifact 的 `initial_mlir` 标记为 skipped；这不影响 post-materialization
MLIR、LLVM、LTO MIR、ISA、HSACO 和 correctness，但不能把 source SHA 写成
pre-branch MLIR hash。source 内容的 SHA 相同，只能证明没有新增高层 source fork。

## 4. P3 final-machine VALU 归因

### 4.1 T=2048 总量

| arm | MFMA/CTA | VMEM/CTA | LDS/CTA | VALU/CTA | SALU/CTA |
|:--|--:|--:|--:|--:|--:|
| Z5B | 160 | 672 | 672 | 7072 | 768 |
| P3-S | 160 | 448 | 464 | 8410 | 780 |
| P4-S | 160 | 448 | 464 | 8410 | 780 |
| native WG256 diagnostic | 160 | 140 | 480 | 3376 | 660 |

所以需要解释的两个差值是：

```text
P3 - Z5B    = +1338 VALU/CTA
P3 - native = +5034 VALU/CTA
```

P3 的 VMEM 和 LDS 反而比 Z5B 少 `224` 和 `208` 条/CTA。故不能把
`+1338` 简单归因成 P3 多做了 global/LDS materialization。

### 4.2 final ISA 静态族计数

下面是同一类 final ISA mnemonic 的 lexical count。它们用于 provenance
定位，不是动态计数：

| 指令族 | Z5B | P3 | P4 | P3-Z5B | 解释 |
|:--|--:|--:|--:|--:|:--|
| `v_accvgpr_write_b32` | 32 | 224 | 224 | +192 | accumulator 写入/片段物化 |
| `v_accvgpr_read_b32` | 64 | 160 | 160 | +96 | accumulator 读回/feeding |
| `v_cndmask_b32` e32 | 160 | 208 | 208 | +48 | ownership/predicate/select |
| `v_cndmask_b32` e64 | 80 | 128 | 160 | +48 | 64-bit predicate/select，P4 +32 |
| `v_mov_b32` | 75 | 108 | 76 | +33 | fragment/register copy，P4 -32 |
| `v_or_b32` | 190 | 188 | 188 | -2 | packed/index bit composition |
| `v_lshl_add_u64` | 107 | 80 | 80 | -27 | address formation |
| `v_add3_u32` | 80 | 80 | 80 | 0 | address/index affine recipe |
| `v_lshrrev_b32` | 63 | 66 | 66 | +3 | stage/lane extraction |
| `v_ashrrev_i32` | 75 | 63 | 63 | -12 | signed index/ownership |
| `v_lshlrev_b64` | 59 | 48 | 48 | -11 | byte/word offset scaling |
| `v_and_b32` | 41 | 46 | 46 | +5 | lane/stage masks |
| `v_perm_b32` | 0 | 0 | 0 | 0 | no explicit permute family |

最醒目的事实是 P3 的 accumulator read/write lexical 总增量为 `+288`，而
地址算术的一些核心族并没有增加，甚至略少。这个 `+288` 不能逐条等于
`+1338 dynamic VALU`，但它是最强的 machine-side provenance 线索。

### 4.3 九类 provenance

#### 1. tid/wave/lane ownership/index

`v_lshrrev_b32`、`v_ashrrev_i32`、`v_and_b32`、`v_or_b32` 和部分
`v_cndmask_b32` 负责把 thread id、wave、lane、value-half、row-half 和
predicate 组合成 ownership/index。它们在 P3 中确实存在，但相对于 Z5B
没有出现足以解释 +1338 的新增静态族。证据等级 B/C。

#### 2. kStage/sourceHalf/packet affine arithmetic

`v_lshrrev`、`v_lshlrev_b64`、`v_add3_u32` 与 `v_lshl_add_u64` 的相邻序列
形成 stage、source-half、packet row/column 和 feature offset。P3 的
`v_add3` 与 Z5B 相同，`v_lshl_add` 反而从 107 降到 80，因此本轮没有证据
支持“重复 udiv/urem 是主要 P3 excess”。源码中的整数表达式不能直接映射
到最终动态 VALU。证据等级 B。

#### 3. global K/H producer byte/address

P3 的 final ISA 仍有 `global_load_dword=92`、`global_load_ushort=48`、
`global_load_dwordx4=12`；Z5B 为 `80/112/0` 的对应主要族。动态 VMEM 却是
P3 `448`、Z5B `672`，说明 load width 和控制流 ownership 的关系不能用
静态行数直接相减。P3 full-scope producer 的 VMEM 优势已经实现，本类不是
P3 相对 Z5B 的 excess。相对 native 的 VMEM 差距仍然存在，是 P3-native
整体差距的重要组成部分。证据等级 A/B。

#### 4. shared/LDS physical address

地址形成通常紧邻 `ds_read_b128`、`ds_write_b16/b128`。P3/P4 的动态 LDS
都是 `464/CTA`，且 `ds_read=56`、`ds_write=92` 静态保持不变；P2 的额外
fragment read 已在 P3 消失。它不是 P3 当前 +1338 的主要新增来源，但
P3/native 的 barrier 和 phase组织仍不同。证据等级 A/B。

#### 5. ownership predicates/select/cndmask

P3 的 `v_cndmask_b32` 总数为 `336`，Z5B 为 `240`，差 `+96`；这些指令
承载合法 lane、causal/phase ownership 和 select。P4 的总数为 `368`，没有
减少这个族，说明 P4 不是 predicate 删除优化。该族是 P3 excess 的候选
组成，但没有 instruction-level dynamic counter 证明其独占 +1338。证据等级
B/C。

#### 6. MFMA operand register/fragment feeding

这是本轮选择 P4 的唯一主目标。源码在 full-scope loop 中把 MFMA 结果先
写回 `inter_acc` 的 vector，再由下一次 block-dot 读取；P3 的 post MLIR
同时出现 `vector<32xf32>` accumulator boundary、slice/rebuild 和
first-class operand materialization。exact-LTO MIR 中可见多组 AGPR
`COPY`、`REG_SEQUENCE`、subregister 组合，最终 ISA 则出现：

```text
v_accvgpr_write_b32: 224 (P3) vs 32 (Z5B)
v_accvgpr_read_b32 : 160 (P3) vs 64 (Z5B)
```

这是最直接的 P3/Z5B machine representation 差异。P3 的 `+288` lexical
AGPR transfer 由多个 MFMA result/accumulator slice 的 feeding chain 产生，
并与 `v_mov`、`COPY/REG_SEQUENCE` 共同构成高频 fragment family。它不能被
宣称为精确的 `+288 dynamic VALU`，但比地址族有更高证据等级。

#### 7. transpose/layout permutation

最终 ISA 没有 `v_perm_b32`，因此没有证据表明一个大规模显式 permute 指令族
是主要问题。布局转换仍以 vector slice、subregister、`REG_SEQUENCE` 和
select 形式存在；这些已归入第 6 类而不是重复计数。证据等级 B。

#### 8. causal/g/exp/output math

这些是冻结数学路径。没有通过 P3/P4 的 source、MFMA 数量、BF16 输出和
correctness 证据发现它们改变。不能把数学 VALU 误归入 planner overhead，
也没有足够 counter 将它们从总 VALU 中单独剥离。证据等级 A/C。

#### 9. remaining/unknown

`8410` 中剩余部分包括多个小型 address/select/copy block 的动态执行，当前
rocprof 只给出聚合 VALU，不能逐条按 logical operand 反演。该部分必须保留
为 unknown，不用它支撑额外 patch。证据等级 A（总量）/C（细分）。

## 5. P3-Z5B 和 P3-native 的差值闭账

### 5.1 P3-Z5B：+1338 VALU/CTA

能够严谨闭账的程度如下：

| 差值来源 | 证据 | 能否归入 +1338 |
|:--|:--|:--|
| global/LDS load 数 | 动态 VMEM/LDS，等级 A | 排除为“P3多做”的主因；P3反而更少 |
| address affine recipe | final ISA/MIR，等级 B | 没看到足够新增；不能作为第一目标 |
| predicate/select | static `+96 cndmask`，等级 B | 可解释一部分，但非动态精确闭账 |
| accumulator feeding | static `+288 AGPR transfer` + MIR chain，等级 B | 最强主线，但不是 1:1 dynamic VALU |
| copy/move/fragment reconstruction | static `+129` 的 cndmask+move组合，等级 B | 与 accumulator chain耦合，不能独立相加 |
| unknown | PMC 聚合限制，等级 C | 剩余未能逐条闭账 |

因此本轮的诚实结论不是“1338 条已经逐条对应”，而是：已经排除
global/LDS excess，且已把最大的可控 machine representation family 收敛到
accumulator/fragment feeding；剩余 `+1338` 的精确动态逐指令拆分需要更细粒度
的 hardware counter 或机器级 instrumentation。P4 正好验证了这条控制杆是否
能改变最终 dynamic work。

### 5.2 P3-native：+5034 VALU/CTA

native 的优势不只是某一条更宽 load。已有 same-shape native audit 显示：

- native dynamic VMEM `140/CTA`，P3 `448/CTA`，差 `308/CTA`；
- native dynamic VALU `3376/CTA`，P3 `8410/CTA`，差 `5034/CTA`；
- native dynamic LDS `480/CTA`，P3 `464/CTA`，数量接近；
- native static manifest 的 barrier/waitcnt 约 `11/48`，P3 static 为
  `44/160`，说明 phase/依赖组织也不同，但 static 数不等于动态耗时。

所以 P3-native 的主要差距排序是：

1. native 的 producer ownership 和 packet reuse 使 global issuing work 更少；
2. AveLang full-scope common planner 的 address/ownership/select recipe 更
   宽，且 fragment/AGPR feeding 更重；
3. native 的 phase lifetime 和 wait/MFMA overlap 不同；
4. LDS 数量本身不是主差距，因为两者动态 LDS 已接近。

native 的 TTIR/LLVM provenance 不能可靠恢复每个 tensor element 的全部 lane
映射，因此这里不伪造更细的 5034 逐条分配。

## 6. 唯一 P4 控制杆

### 6.1 为什么选择 accumulator forwarding

阶段 A 的排序结果是：

```text
最大可信异常：MFMA result -> vector slice -> accumulator reload
未证明的候选：重复 affine address arithmetic
已修复的旧问题：P2 B64 fragment read
```

因此 P4 没有同时修改 address、predicate、load width 或 LDS layout，而是只
增加通用的 `AccumulatorForwardingMap`：

1. 同一个 `block_dot` 的 `acc_low` 在 enclosing vector reset 之外，优先复用
   已经产生的 SSA value；
2. 遇到新的 accumulator reset boundary 时停止 forward，避免把 score half0
   错误传入 score half1；
3. full-scope P4 直接保留实际 MFMA 产生的低 tile result，不再生成源代码
   随后不会读取的 duplicated high half；
4. K/H 仍经过同一个 `LogicalBlockLayoutPlan`、同一个 full-scope lowering，
   没有 `kSpecialCase`、`hSpecialCase` 或 Qwen address table。

关键实现位置：

`lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc`

关键内部标记：

```text
avelang.block_dot.p4_ssa_forwarding
avelang.block_dot.p4_ssa_forwarding_reset_boundary
avelang.block_dot.p4_low_tile_ssa
```

这些是 compiler-internal provenance marker，不是新的 public operation。

### 6.2 P4 是否真正改变了 machine graph

是。不同层级的证据如下：

| 层级 | 证据 |
|:--|:--|
| source | source SHA 与 P3 相同，只有 runtime preservation selector 不同 |
| post MLIR | P4 出现 `p4_low_tile_ssa` 与 forwarding/reset attrs；P3 有 `vector.from_elements vector<32xf32>` boundary |
| LLVM | P3/P4 `lowered_llvm.ll` SHA 不同，P3 `886965...`，P4 `a9b3ae...` |
| pre-LTO AMDGCN | P3/P4 SHA 不同，`1d75f5...` vs `4c1813...` |
| LTO MIR | section00/01/02/18 SHA 均不同，且 exact-LTO return code=0 |
| final ISA | P3/P4 SHA 不同，`d7a2d3...` vs `fc8e5a...` |
| HSACO | P3 `d17483...`，P4 `902d6b...`，不同 |

P4 的 typed/SSA 表示确实保留到 LLVM/MIR 层；但它不是一个最终 ISA 中独立
存在的 block-dot opcode。ROCDL/AMDGPU 最终仍以普通 LDS load、MFMA、VGPR/AGPR
copy/select 指令实现，所以不能把“IR 不同”误写成“最终 dynamic work 一定少”。

## 7. P4 final ISA 的实际变化

P3 和 P4 的主要 static count：

| 指令族 | P3 | P4 | 变化 |
|:--|--:|--:|--:|
| MFMA32 | 56 | 56 | 0 |
| global load | 140 | 140 | 0 |
| global store | 16 | 16 | 0 |
| `ds_read` | 56 | 56 | 0 |
| `ds_write` | 92 | 92 | 0 |
| `s_barrier` | 44 | 44 | 0 |
| `s_waitcnt` | 160 | 160 | 0 |
| `v_accvgpr_write_b32` | 224 | 224 | 0 |
| `v_accvgpr_read_b32` | 160 | 160 | 0 |
| `v_mov_b32` | 108 | 76 | -32 |
| `v_cndmask_b32_e64` | 128 | 160 | +32 |
| `v_cndmask_b32_e32` | 208 | 208 | 0 |
| `v_lshl_add_u64` | 80 | 80 | 0 |
| `v_add3_u32` | 80 | 80 | 0 |

因此 P4 删除/减少的具体机器工作，只能准确表述为：最终 ISA lexical 中
有 32 条 `v_mov_b32` 少了；但同一变换引入了 32 条 e64 `v_cndmask`，
accumulator read/write 和所有主要 load/LDS/MFMA/barrier 族不变。按 dynamic
PMC，P4 没有删除可观测的 VALU：`8410 -> 8410/CTA`。这是本轮最重要的
负结果，不能只看 `v_mov -32` 就宣布成功。

## 8. 资源和正确性

### 8.1 code object 与 profiler resource

| arm | code VGPR | code AGPR | code SGPR | LDS | private | spill/scratch |
|:--|--:|--:|--:|--:|--:|:--|
| P3 | 132 | 48 | 30 | 32768 B | 0 | 0 |
| P4 | 104 | 32 | 30 | 32768 B | 0 | 0 |

T=2048 profile 的 resource fields：

| arm | profiler VGPR | profiler AccVGPR | SGPR | LDS | scratch | occupancy |
|:--|--:|--:|--:|--:|--:|--:|
| P3 | 88 | 88 | 112 | 32768 B | 0 | 14.940439 |
| P4 | 84 | 92 | 112 | 32768 B | 0 | 14.673293 |

`code-object AGPR`、`profiler AccVGPR` 和 MIR virtual register 数量不是同一个
指标。P4 的 code-object allocation 变小，但 profiler 的 AccVGPR 字段略升；
不能把它们混成“P4 一定少了 16 个动态 AGPR”。两者共同可靠的结论是：没有
private memory、scratch 或 spill，LDS 没变。

### 8.2 Correctness gate

P4 相对 Z5B 的结果：

| case | result |
|:--|:--|
| T=64/512/1024/2048/4096/8192/16384 | BF16 byte-exact，finite |
| T=64/8192/16384 caller-owned output | pass |
| T=64/8192/16384 zero-V + NaN-prefill | pass |
| max abs difference | 0 |
| old block-dot/direct-K64/BV32/Stage6S regressions | `22 passed` |

P4 没有通过改变 accumulation order、减少 MFMA 或放宽 tolerance 得到正确性。

## 9. Dynamic PMC 与性能

### 9.1 T=2048 dynamic PMC

P4 fresh rocprof：512 CTA，按 CTA 归一化：

| arm | MFMA | VMEM | LDS | VALU | SALU | profiler VGPR | AccVGPR | occupancy |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Z5B | 160 | 672 | 672 | 7072 | 768 | 76 | 100 | 记录于既有 capture |
| P3 | 160 | 448 | 464 | 8410 | 780 | 88 | 88 | 14.940439 |
| P4 | 160 | 448 | 464 | 8410 | 780 | 84 | 92 | 14.673293 |
| native | 160 | 140 | 480 | 3376 | 660 | diagnostic | diagnostic | diagnostic |

P4 完整保留了 P3 的 VMEM/LDS/MFMA 优势，但没有真实减少 VALU。

### 9.2 Fresh-process body benchmark

口径固定为：caller-owned isolated body、current HIP stream、no Graph、warmup=10、
repeat=50、7 个 fresh-process session、rotating order。单位为 ms，取 session
median 的 median：

| T | Z5B | P3-S | P4-S | native | P4/Z5B |
|--:|--:|--:|--:|--:|--:|
| 2048 | 0.066899501 | 0.069784001 | 0.070364498 | 0.042644000 | 1.0518x |
| 8192 | 0.157253496 | 0.167788997 | 0.168190002 | 0.091215502 | 1.0695x |

P4 对 Z5B 的 7 个 paired difference 全部为正：

```text
T=2048: +2.5035, +3.8455, +3.1640, +2.7845, +3.6245, +3.1850, +3.0445 us
T=8192: +9.7345, +11.5775, +9.3135, +11.8980, +11.1570, +7.8315, +12.6990 us
```

P4 相对 native 的 session ratio 约为 `1.61x-1.66x`（T=2048）和
`1.82x-1.87x`（T=8192）。T=16384 correctness 已通过，但按照预注册规则，
P4 在 T=2048 和 T=8192 都没有优于 Z5B，因此没有继续跑条件性的 T=16384
性能 session。

两点 slope（仅用于 body diagnostic）：

| arm | T=2048 -> 8192 |
|:--|--:|
| Z5B | 约 `0.9412 us/chunk` |
| P3 | 约 `1.0209 us/chunk` |
| P4 | 约 `1.0190 us/chunk` |
| native | 约 `0.5050 us/chunk` |

P4 相对 P3 的 resource 改善没有转成 latency 改善；P4 甚至在 T=2048 和
T=8192 都略慢于 P3。

## 10. 为什么 P4 没有把动态 VALU 降下来

P4 改变的是编译器内部的 accumulator SSA 表示：它去掉了 source-visible
但后续只使用低半部分的 duplicated high half，并在 reset boundary 内尝试
forward。可是 full-scope lowering 后仍需要：

1. 把 MFMA 的实际 accumulator fragments 送入多个后继 consumer；
2. 处理 BF16 fragment、subregister 和 vector shape 的类型边界；
3. 处理 phase predicate 与不同 consumer 的选择；
4. 让 LTO/RA 在同一段 full-scope control flow 中完成物理寄存器分配。

这些工作在 P4 final ISA 中仍表现为相同的 `v_accvgpr_read/write`、
`v_cndmask`、address/select 族和相同的 dynamic VALU。也就是说，P4 的
representation 差异在最终 machine lowering 中部分收敛了；它改善了 allocation
结果，却没有改变执行工作和关键路径。

这不是“P4 没有被编译器看到”：P4 的 LLVM/MIR/ISA/HSACO 全部不同。准确说法
是：

```text
P4 representation survives to machine graph,
but the final AMDGPU lowering still materializes an equivalent VALU/AGPR feeding path.
```

## 11. 决策

### P4 gate

| gate | result |
|:--|:--|
| correctness | PASS |
| MFMA/VMEM/LDS unchanged from P3 | PASS |
| scratch/spill/private | PASS, all zero |
| final ISA/HSACO differs | PASS |
| dynamic VALU decreases | FAIL, 8410 -> 8410 |
| T=2048 improves vs Z5B | FAIL |
| T=8192 improves vs Z5B | FAIL |

正式状态：

```text
Z5B = isolated performance baseline
P3  = generic block_dot representation/infrastructure, performance No-Go
P4  = generic accumulator forwarding infrastructure, correctness/resource PASS,
      performance No-Go
```

本轮不启动 Q A-planner、V-new 扩展、P4.1/P4.2、WG/MFMA/RA、X2 或 production
接入。若未来继续，必须先引入能在 final machine/PMC 层真正删除 feeding
instructions 的通用 physical plan；不能再只做 source/LLVM 形状变化后等待
后端自动收敛。以当前证据，最稳妥的路线是暂停 BDV2 V2 性能线，保留 P3/P4
作为 compiler regression/infrastructure evidence。

## 12. 工件和复现

### P3 machine artifact

```text
codex_qwen_bt64_stage6z_bdv2_p3_machine_specialized/
```

P3 HSACO SHA256：

```text
d17483b933a7abf68b34b0e193aa4e3e5d9f93f48a12cfa8d84b285c9091af5a
```

### P4 machine artifact

```text
codex_qwen_bt64_stage6z_bdv2_p4_machine_specialized_v3/
```

P4 HSACO SHA256：

```text
902d6ba183b80a4f8f27d68721ee916191afa4d08975e81b6eed95694ce00b6d
```

P4 source SHA256：

```text
aee02e0309152ab34893836720c38266c6519c57499ef640db3e4ff48152047e
```

P4 dynamic PMC 原始工件：

```text
codex_qwen_bt64_stage6z_bdv2_p4_rocprof/
```

P4 benchmark：

```text
codex_qwen_bt64_stage6z_bdv2_p4_bench_T2048_sessions7.json
codex_qwen_bt64_stage6z_bdv2_p4_bench_T8192_sessions7.json
```

正确性和编译器回归 driver：

```bash
python3 check_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py \
  --arm specialized --planner bdv2_p1_affine \
  --preservation p4_accumulator_reuse \
  --T 64 512 1024 2048 4096 8192 16384

python3 -m pytest -q \
  test_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py \
  test_qwen_gdn_direct_k64_block_dot_ab.py \
  test_qwen_gdn_direct_k64_bv32_typed_operand_ab.py \
  test_qwen_gdn_direct_k64_bv32_persistent_operand_c0.py \
  test_qwen_gdn_bt64_bf16_recurrence_full_stage6s.py
```

完整路径、SHA、static count、dynamic PMC、provenance category 和 benchmark
paired samples 同步记录在：

`machine-readable valu_provenance_p3_vs_z5b_vs_native.json`

