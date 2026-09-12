# Qwen gfx942 BT64 Stage 6Z：Fixed Z2 与 Native vLLM 同形状机器差距审计

## 结论摘要

本轮是 **measurement-only machine-gap audit**。没有修改 Z2 kernel source，
没有修改 native Triton 工件，没有接入 X2，没有建立 Z4，也没有修改
allocator、RA、生产 selector、recurrence HSACO 或 Stage 5/R4 路线。

审计对象在 T=2048 完全对齐到：

```text
BT64 / BV64 / BK32 / gfx942 / wave64 / WG256 / 2 CTA per chunk-head
```

fresh public capture 确认 native vLLM 本次实际选中的配置是：

```text
BK=32, BV=64, num_warps=4, num_stages=2, num_ctas=1, WG=256
```

因此本轮没有把 native 的 WG128、其他 autotune candidate 或旧配置混入比较。

在同一 current HIP stream、预分配输入/输出、no Graph、warmup=10、repeat=50、
5 个 fresh-process session、轮换执行顺序下：

| T=2048 body | 中位数 | 说明 |
|:--|--:|:--|
| fixed Z2 | `0.0783565 ms` | 5 个 session median 的 median |
| native selected WG256 | `0.0426830 ms` | fresh public 选择后 pin 的 exact body |
| fixed Z2 / native | `1.8358x` | session 比值的中位数 |
| paired Z2 - native | `35.6735 us` | session paired difference 的中位数 |

新鲜 rocprof PMC 也使用同一个 WG256 形状，网格为 131072 work-items，即
`512 CTA`。每 CTA 的动态机器工作如下：

| dynamic PMC / CTA | fixed Z2 | native | Z2/native |
|:--|--:|--:|--:|
| MFMA | `160` | `160` | `1.00x` |
| VMEM | `928` | `140` | `6.63x` |
| LDS instructions | `928` | `480` | `1.93x` |
| VALU | `11,400` | `3,376` | `3.38x` |
| SALU | `1,072` | `660` | `1.62x` |

这组数据把主要差距定位到 **Z2 的 global/LDS operand materialization、地址与
layout 变换、以及显式 phase round-trip**，而不是 MFMA 数量、scratch、spill
或单纯的 occupancy cliff：

- MFMA 动态工作完全相同；
- Z2 的 VMEM、LDS、VALU、SALU 均明显更高；
- 两边 scratch/private segment/spill 都是零；
- Z2 的 code-object VGPR 较高，但 profiler 的 occupancy 反而较高；native
  并不是靠更低寄存器资源“作弊”获得结果，而是做了更少的机器工作；
- 当前证据支持“Z2 的 source phase / producer-consumer 组织没有达到 native
  的 typed tile 数据流”，但还不足以把所有差距归因成某一个单独 compiler pass。

本轮唯一允许登记的下一候选是：

> **保持 WG256/BV64/BK32/MFMA32 不变，针对 Z2 Phase B/C 的 producer-to-consumer
> typed tile materialization 做一条 matched source/IR 数据流审计与单一替换实验，
> 优先消除 Q/K/V-new 的重复 global-to-LDS-to-fragment round-trip。**

本候选只由本轮 VMEM/LDS/VALU 证据支持；本轮不实现它。

---

## 1. 审计边界与公平性

### 1.1 固定输入和 launch contract

两边均使用：

- `T=2048`，`32 chunks`；
- `B=1, H_K=4, H_V=8, K=V=128`；
- `q/k/v_new/h=BF16`，`g=FP32`，输出为 `BF16`；
- `BT=64, BV=64, BK=32`；
- gfx942 wave64；
- WG256；
- `2 CTA/chunk-head`，总 CTA 数 `32 * 8 * 2 = 512`；
- 预分配 input/output；
- current HIP stream；
- 不使用 CUDA/HIP Graph；
- compile、module load、首次 JIT 和 native config pin 均在计时区间外；
- 每个 session 是独立 Python process；
- 5 个 session 使用 rotating `z2,native` / `native,z2` 顺序。

native worker 在第一次 body launch 之前，把 fresh public selection 得到的
`BK=32, BV=64, num_warps=4, num_stages=2` 写入 exact autotuner cache key，
避免 rocprof 或 HIP-event 区间出现其他 candidate dispatch。native 不再使用此前
直接调用 wrapper 时误选的 WG128 数据。

### 1.2 明确排除的数据

以下数据不进入本轮当前排名：

- Phase-B dead K overread 修复之前的 Z2 timing/PMC/ISA；
- 旧 Stage 6Z native WG128 direct-body 数据；
- native autotune candidate 污染的 pmc/pmc2；
- profiler trace 被当成 HIP-event body latency 的结论；
- X2 full graph 或 production Eager API 结论。

本轮有效 PMC 目录是 `codex_qwen_bt64_stage6z_fixed_z2_vs_native_audit/pmc3/`，
有效 body JSON 是 `body_T2048_pinned5.json`。

---

## 2. Correctness fresh check

correctness 是每个 arm 一个 fresh process，结果与 Stage 6W BF16 chunk-o
reference 比较。reference 用于锁定本轮 output contract，不把不同实现的
内部 FP32 累加误差误报成机器差距。

| arm | finite | max abs vs Stage6W | threshold | BF16 byte-exact |
|:--|:--:|--:|--:|:--:|
| fixed Z2 | yes | `7.62939453125e-06` | `0.0078125` | no |
| native selected WG256 | yes | `3.0517578125e-05` | `0.0078125` | no |

两者均通过 finite 和冻结的 `1/128` output threshold。两者都不是与 Stage6W
reference 的逐 bit 相等；这是 BF16/不同 dot accumulation 的正常诊断结果，不能
用 byte-exact false 反推 kernel 错误。

结果文件：

- `codex_qwen_bt64_stage6z_fixed_z2_vs_native_audit/correctness/T2048/z2.json`
- `codex_qwen_bt64_stage6z_fixed_z2_vs_native_audit/correctness/T2048/native.json`

---

## 3. Fresh body latency

### 3.1 五 session 原始数据

每个数是一个 fresh worker 内 50 次 HIP-event sample 的 median：

| session | order | fixed Z2 ms | native WG256 ms | Z2-native us | Z2/native |
|--:|:--|--:|--:|--:|--:|
| 0 | Z2 -> native | `0.0802390` | `0.0417020` | `38.5370` | `1.9241x` |
| 1 | native -> Z2 | `0.0781760` | `0.0428840` | `35.2920` | `1.8230x` |
| 2 | native -> Z2 | `0.0783565` | `0.0426830` | `35.6735` | `1.8358x` |
| 3 | Z2 -> native | `0.0796180` | `0.0423630` | `37.2550` | `1.8794x` |
| 4 | native -> Z2 | `0.0774150` | `0.0429640` | `34.4510` | `1.8019x` |

汇总：

| arm | mean of session medians | median of session medians | session范围 |
|:--|--:|--:|:--|
| fixed Z2 | `0.0787609 ms` | `0.0783565 ms` | `0.0774150–0.0802390` |
| native WG256 | `0.0425192 ms` | `0.0426830 ms` | `0.0417020–0.0429640` |

`wall_ms` 只作为辅助记录，最终 body 比较使用同一 stream 的 HIP event。它的
作用是确认没有明显 host 阻塞异常，不用 wall time 取代 device body time。

### 3.2 profiler trace 不等于 body latency

有效 pmc3 的 kernel trace median 是：

| arm | profiler trace median |
|:--|--:|
| fixed Z2 | `53.840 us` |
| native selected WG256 | `15.984 us` |

这两个值受 rocprof instrumentation 影响，不能直接与 `0.0783565 ms` 和
`0.0426830 ms` 做一一换算。它们只用于确认两边都捕获到同形状 WG256 dispatch
以及配套 PMC。

---

## 4. Dynamic PMC：按 CTA 对齐

### 4.1 总量和 CTA 数

两边 `Workgroup_Size=256`、`Grid_Size=131072`，所以：

```text
CTA = 131072 / 256 = 512
```

以下总量来自同口径 rocprof counter CSV，不由 ISA 静态计数推导：

| dynamic counter | fixed Z2 total | native total | Z2/native |
|:--|--:|--:|--:|
| `SQ_INSTS_MFMA` | `81,920` | `81,920` | `1.00x` |
| `SQ_INSTS_VMEM` | `475,136` | `71,680` | `6.63x` |
| `SQ_INSTS_LDS` | `475,136` | `245,760` | `1.93x` |
| `SQ_INSTS_VALU` | `5,836,800` | `1,728,512` | `3.38x` |
| `SQ_INSTS_SALU` | `548,864` | `337,920` | `1.62x` |
| `OccupancyPercent` | `15.863258` | `9.510814` | collector metric |

换算到每 CTA：

| dynamic counter / CTA | fixed Z2 | native |
|:--|--:|--:|
| MFMA | `160` | `160` |
| VMEM | `928` | `140` |
| LDS | `928` | `480` |
| VALU | `11,400` | `3,376` |
| SALU | `1,072` | `660` |

MFMA 的每 CTA 数相同，说明不是“Z2 少算或 native 多算 MFMA”造成的结果。
Z2 约多出每 CTA `788` 个 VMEM、`448` 个 LDS instruction、`8,024` 个 VALU
和 `412` 个 SALU。这个差距量级足以解释 `35.7 us` 的 body gap，且不需要借助
未经测量的 transient-state 假设。

### 4.2 资源与 spill

| resource | fixed Z2 code object / PMC | native selected code object / PMC |
|:--|:--|:--|
| VGPR | readobj `168` / PMC `88` | readobj `132` / PMC `100` |
| AGPR | `32` | `32` |
| SGPR | readobj `44` / PMC `112` | readobj `89` / PMC `96` |
| private segment | `0 B` | `0 B` |
| VGPR/SGPR spill | `0/0` | `0/0` |
| scratch | `0 B` | `0 B` |
| LDS metadata | `16,384 B` | selected Triton metadata `12,288 B` |

Z2 的 code-object metadata 是固定代码对象的权威资源描述；PMC 中的 VGPR/SGPR
字段是 collector 对运行 dispatch 的报告，二者不能互相替代。native 的
`llvm-readobj` 报告 `.group_segment_fixed_size: 0`，而 fresh Triton cache
JSON 报告 `shared: 12288`，rocprof 也报 `LDS_Block_Size=0`。这是本轮 native
collector/ELF metadata 的已知不一致，不能把它写成 native 没有 LDS。TTGIR 中
明确存在 shared allocation，故 native LDS 只能写成“选定工件声明 12 KiB、
readobj/collector 字段不可靠”。

两边都没有 private memory 或 MIR spill。因此当前 gap 不是 RA 把某一边 spill
到 scratch 后造成的，也不是 Z2 的 AccVGPR=172 类 resource cliff；本轮 Z2
PMC 的 `Accum_VGPR_Count=32`、native 为 `36`。

---

## 5. Static ISA lexical count

下表全部来自 final ISA 文本的 lexical instruction count。它们不是动态 PMC，
不能乘以 wave 数后当成实际执行数，也不能从 static MFMA count 反推 elapsed time。

| static ISA category | fixed Z2 | native selected |
|:--|--:|--:|
| MFMA32 | `20` | `40` |
| MFMA16 | `0` | `0` |
| `global_load*` | `136` | `17` |
| `global_store*` | `16` | `0` |
| `buffer_load*` | `2` | `14` |
| `buffer_store*` | `2` | `4` |
| `ds_read*` | `20` | `80` |
| `ds_write*` | `88` | `40` |
| `s_waitcnt` | `127` | `48` |
| `s_barrier` | `9` | `11` |
| `v_add*` | `187` | `54` |
| `v_add3*` | `0` | `0` |
| shift / `lshl*` | `166` | `42` |
| and/or/bfe/bitwise | `381` | `199` |
| permute / ds_bpermute / DPP | `0` | `0` |
| v/s mov/cndmask proxy | `105` | `15` |

Z2 的 static MFMA32 lexical count 为 20，native 为 40，但动态 MFMA 总数相同。
这是代码展开形态、wave/operand feeding 方式不同造成的 lexical-vs-dynamic
差异；不能因为 native static MFMA 是 2 倍就判断它算得更多。

Z2 的 `global_load*`/`global_store*` 文本主要是 flat/global 指令；native 的
`buffer_load*`/`buffer_store*` 是 Triton AMDGPU backend 的 buffer 形式。两类
都算 global-memory instruction family，不能只比较某一行 mnemonic。

### 5.1 Static lexical dependency distance

这是 final ISA 中“下一条匹配 mnemonic”的文本距离 proxy：

| dependency proxy | fixed Z2 count/min/median/max | native count/min/median/max |
|:--|:--|:--|
| global load -> waitcnt | `136 / 1 / 5 / 79` | `17 / 2 / 154 / 276` |
| waitcnt -> ds_read | `101 / 2 / 174 / 1034` | `44 / 2 / 8 / 692` |
| ds_read -> barrier | `12 / 5 / 774 / 817` | `72 / 2 / 21 / 648` |
| barrier -> MFMA | `9 / 1 / 23 / 281` | `11 / 1 / 7 / 148` |

这个表不是硬件 latency，也不是 dynamic waitcnt 数。它能说明机器图的组织：
Z2 有大量 global load 后快速递进的显式 wait，并在 wait/LDS/phase 之间留下
较长的静态间隔；native 的 LDS read 与 MFMA/phase barrier 更密集交错。
但因为本轮 dynamic 工作本身已经明显不同，不能把所有剩余时间差硬解释成
waitcnt placement。

---

## 6. Source -> IR -> machine 对照

### 6.1 Fixed Z2 的 source phase 与索引公式

源文件：

`test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py`

关键 mapping：

```text
program_id = block_id(0)
v_block_idx    = program_id % 2
value_head_idx = (program_id // 2) % 8
chunk_idx      = program_id // (2 * 8)
value_base     = v_block_idx * 64
chunk_start    = chunk_idx * 64
key_head_idx   = value_head_idx >> 1
```

对应 source 行：program mapping 在 60-66，shared phase/i32 view 在 68-71。

#### Phase A：Q/H staging -> inter-state MFMA

- source 75-86：每个 `k_stage` 将 Q 和 H 写入同一 `phase` shared buffer；
- source 87：CTA-wide barrier 发布 Q/H；
- source 90-96：从 `phase_vec` 取 i32 words、view 成 BF16 fragment，执行 MFMA32；
- source 98：下一次 K32 写入前的 reuse barrier。

这是一个明确的 producer -> shared -> fragment consumer round-trip。Z2 的
`phase` 是 `(256,32) BF16`，再建立 `i32` view；它不是 native TTGIR 中直接
表达的 typed dot operand。

#### Phase B：Q/K staging -> score

- source 100-124：两个 `source_half`，每个内部四个 `k_stage`；
- source 111-123：Q/K 写入 phase；
- source 125：每个 K32 stage 的 staging barrier；
- source 129-135：i32 view、BF16 fragment view、两个 MFMA32；
- source 137：下一 K32 阶段 reuse barrier；
- source 142-152：score 的 causal mask、g/exp 和 score materialization；
- source 154 附近：score publication barrier。

Z2 修复了 Phase-B dead K overread，但保留了这个 source-level phase 结构。
因此本审计不能使用修复前数据，也不能把修复本身误认为性能优化。

#### Phase C：V-new staging -> score-times-V MFMA -> output

- source 154-164：V-new 以 token/value 逻辑布局写入 phase；
- source 167-175：重新通过 i32 view / BF16 fragment view 取 score 和 V fragment；
- source 174-175：intra MFMA32；
- source 177-183：FP32 result 乘 decay 后转 BF16 store。

因此 Z2 的一个逻辑 tile 至少出现：global load、shared store、shared read/view、
MFMA consumer、再 phase reuse 或 output store。动态 PMC 中 Z2 的 LDS 928/CTA
和 VMEM 928/CTA 是这些显式 materialization 的总结果，而不是单一一条
`ds_read` 指令的结果。

### 6.2 Native vLLM TTIR/TTGIR 证据

fresh selected native 工件的 source location 是：

```text
/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fla/ops/chunk_o.py:42:0
```

native TTGIR 的关键内容：

- `%b_q_69`：`64x32 BF16` shared allocation；
- `%b_h_70`：`64x32 BF16` shared allocation；
- `%b_k_71`：`32x64 BF16` shared allocation；
- lines 142/159/170：Q/K/H typed buffer load；
- lines 172/174/176：对应 shared local_store；
- lines 195/207/214：typed local_load；
- lines 216-217：同一组 dot operand 分别进入 output dot 和 K dot；
- lines 222/224/226：loop-carried next tile 的同一 shared bank store；
- lines 229-234：尾部 local_load 后直接执行 dot；
- lines 354-363：V operand 的 `64x64` in-thread transpose、shared allocation、
  typed local_load 和最终 dot。

native TTIR 直接保留：

```text
tt.load -> ttg.local_store -> ttg.local_load(dot_op) -> tt.dot
```

而不是先变成 AveLang 的 scalar `phase` / i32 view / 多层 BF16 fragment view。
native TTGIR 没有为本审计提供可以逐一对应 11 个 ISA `s_barrier` 的显式
`ttg.barrier` 行；因此不能猜测每个 barrier 的 source provenance。可以可靠
确认的是 shared allocation、typed local_load、loop-carried rotating memdesc
和 dot operand layout 均在 TTGIR 中存在，随后在 final ISA 中形成 11 个 barrier
及大量交错 `ds_read_b64`/MFMA。

### 6.3 LLVM、MIR、ISA 产物完整性

#### Fixed Z2

目录：

`codex_qwen_bt64_stage6z_fixed_z2_vs_native_audit/machine/z2_T2048/`

包含：

- `lowered_llvm.ll`：AveLang lowering 后 LLVM；
- `pre_lto_amdgcn.s`：LTO 前 AMDGCN assembly；
- `exact_lto/linked.hsaco.0.5.precodegen.bc`、`kernel_section_*.mir`：
  exact LTO replay 与 machine pass dump；
- `llc_mir/`：独立 llc MIR pass dump；
- `final_isa.s`：最终 code object disassembly；
- `z2_fixed.hsaco`；
- `readobj.txt`：新生成的 `llvm-readobj --notes` 资源元数据；
- `machine_evidence.json`、`capture_summary.json`。

initial AveLang MLIR printer 在当前 binding 上会 core dump，本次命令明确
`--skip-initial-mlir`，因此没有伪造 initial MLIR 文件。source 级高层语义和
post-lowering LLVM/MIR/ISA 均有实物，缺失项已明确标记。

#### Native selected

目录：

`codex_qwen_bt64_stage6z_fixed_z2_vs_native_audit/machine/native_T2048/`

包含：

- `chunk_fwd_kernel_o.source`；
- `chunk_fwd_kernel_o.ttir`；
- `chunk_fwd_kernel_o.ttgir`；
- `chunk_fwd_kernel_o.llir`；
- `triton_pre_lto_amdgcn.s`；
- `llc_mir/`；
- `final_isa.s`；
- `chunk_fwd_kernel_o.hsaco`；
- `readobj.txt`；
- `machine_manifest.json`。

HSACO identity：

| arm | symbol | SHA256 |
|:--|:--|:--|
| fixed Z2 | `_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z2` | `2afaa8867da656421a9c87988ed4dcad757796c1b1d3b93e27307fb401a1c09a` |
| native | `chunk_fwd_kernel_o` | `cc74acf84104a0900ed327fc469826c228c94fdf07fd55d8d302c3e88c04e72d` |

两个机器图的 hash 不同，且 final ISA/LLVM/MIR 都不是同一个 code object。

---

## 7. 机器归因

### 7.1 不是 MFMA 数量

同形状 PMC：`81920 MFMA` 总量、`160 MFMA/CTA` 完全相同。native static
MFMA32 lexical count 是 40，Z2 是 20，但这只是展开和 operand feeding 的
表达差异。继续增加/减少 MFMA32 指令不会是本轮有证据的方向。

### 7.2 不是 scratch/spill 或 RA cliff

两边：

- private segment `0`；
- VGPR spill `0`；
- SGPR spill `0`；
- final MIR 没有 `SI_SPILL_AV32/AV64 SAVE/RESTORE`。

Z2 exact-LTO machine corpus 和 native llc MIR corpus 的 spill family 都为零。
MIR 中的 COPY/REG_SEQUENCE 统计是不同 pipeline dump 的 lexical proxy，不能当
动态执行数；但它们与 final ISA 的 copy-like/地址类指令差异方向一致。

### 7.3 主要差距是显式 operand materialization

Z2 相对 native 的 static 方向：

- `global_load*`：136 vs 17，另有 buffer load 2 vs 14；
- `global_store*`：16 vs 0，另有 buffer store 2 vs 4；
- `ds_write*`：88 vs 40；
- `s_waitcnt`：127 vs 48；
- `v_add*`：187 vs 54；
- shift：166 vs 42；
- bitwise：381 vs 199；
- copy-like：105 vs 15；
- permute：两边都是 0。

动态 PMC 与 static 方向一致：Z2 VMEM/VALU/SALU/LDS 均更高。由此可以把
机器 gap 归因到以下 producer-consumer round-trip：

```text
Q/K/V-new global load
 -> Z2 phase shared BF16 store
 -> i32 view / scalar word extraction
 -> BF16 fragment reconstruction
 -> ds_read / MFMA
 -> score/V phase publication或下一阶段重新写入
```

native 的对应路径在 TTGIR 中是 typed blocked/shared/dot-op layout，且有 rotating
memdesc 和 loop-carried local load/store。它仍然有 LDS 和 barrier，但每 CTA
做的 VMEM/LDS/VALU/SALU 明显少得多。

### 7.4 waitcnt/phase overlap 不是首要归因，但确有结构差异

如果只看依赖 proxy：

- native `waitcnt -> ds_read` median 8、`ds_read -> barrier` median 21、
  `barrier -> MFMA` median 7；
- Z2 对应 median 174、774、23。

这说明 Z2 的 ISA 中有更长的 phase-separated lexical region，native 的 LDS
read/MFMA 更紧密交错。但由于 Z2 已经多做了 6.63x VMEM 和 3.38x VALU，不能
把 gap 简化成“native 只是 waitcnt 放得更好”。正确顺序是：先消除不必要的
materialization，下一次实验再在相同机器工作量下测 overlap。

---

## 8. 是否已经证明是 AveLang compiler lowering 问题？

本轮证据可以支持一个比“某条 LLVM 指令错误”更精确的结论：

> fixed Z2 的高层 source phase 已经显式选择了 consumer-friendly shared
> `phase`、i32 view、fragment extraction 和多次同步；在 lowering 后这些结构
> 继续体现为更多 global/LDS/地址/VALU/SALU 工作。native TTGIR 则从一开始
> 拥有 typed shared/dot operand layout。因而剩余差距首先是 **high-level
> schedule/layout 与 AveLang lowering 表达能力共同造成的 producer-consumer
> 数据路径差距**，不能只称为 RA 问题。

本轮还不能做的 stronger claim：

- 不能仅凭 Z2 vs native 异源 A/B 证明“同一个 AveLang high-level IR 在 LLVM
  lowering 中必然生成了错误代码”；
- 不能据此断言一个单独 `ds_read` lowering pass 就是根因；
- 不能把 native 的 readobj LDS=0 当成无 shared memory；
- 不能把 static count 当 dynamic count。

要向 compiler team 证明纯 lowering 问题，仍需要 future same-source A/B：同一
pre-lowering AveLang IR、同一 ownership/shape/barrier/MFMA，只切 generic 与
specialized typed operand lowering。那是下一层证据，本轮没有实现。

---

## 9. 唯一下一候选（只登记，不实现）

本轮只选择一个方向：

### Matched typed Phase-B/C producer-consumer materialization

冻结：

- WG256、BV64、BK32；
- 两 CTA/chunk-head；
- MFMA32 数学与 K32 accumulation order；
- BF16 ABI、输出语义、correctness contract；
- 不改 RA、allocator、X2、production selector。

先建立 source/TTGIR 对照，让 Q/K/V-new 的一次 global load 直接进入可复用的
typed shared/dot operand encoding；只消除已经由 PMC 支持的 round-trip，不先改
barrier 计数，不先做 swizzle sweep。验收必须同时看：

1. VMEM/CTA 是否从 928 向 140 靠近；
2. LDS/CTA 是否从 928 向 480 靠近；
3. VALU/SALU 是否同步下降；
4. MFMA 仍为 160/CTA；
5. scratch/spill 仍为零；
6. T=64/512/2048/8192 correctness 不变。

这不是 Z4 实现，也不是本轮的性能 patch；它是由本次 exact same-shape ledger
支持的唯一下一候选。

---

## 10. 复现命令与产物

### Fresh correctness

```bash
cd /workspace/project/avelang
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/audit_qwen_gdn_bt64_stage6z_fixed_z2_vs_native.py \
  --arm z2 --json-out .../correctness/T2048/z2.json \
  --candidate-out .../correctness/T2048/z2_candidate.pt \
  --reference-out .../correctness/T2048/z2_reference.pt

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/audit_qwen_gdn_bt64_stage6z_fixed_z2_vs_native.py \
  --arm native --json-out .../correctness/T2048/native.json \
  --candidate-out .../correctness/T2048/native_candidate.pt \
  --reference-out .../correctness/T2048/native_reference.pt
```

### Pinned body benchmark

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_stage6z_fixed_z2_vs_native_selected.py \
  --T 2048 --sessions 5 --warmup 10 --repeat 50 \
  --out .../body_T2048_pinned5.json
```

native config pin 的具体逻辑在：

`test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_stage6z_native_selected_wg256.py`

### PMC

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/profile_qwen_gdn_bt64_stage6z_fixed_z2_vs_native.py \
  --T 2048 --warmup 2 --repeat 5 --out-dir .../pmc3 \
  --native-cache /tmp/qwen_stage6z_native_audit_cache_T2048
```

最终报告所引用的目录：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
  codex_qwen_bt64_stage6z_fixed_z2_vs_native_audit/
```

---

## 11. 审计状态

| 项目 | 状态 |
|:--|:--|
| fixed Z2/native T2048 same-shape | 完成 |
| fresh correctness | 完成 |
| 5-session body latency | 完成 |
| exact native source/TTIR/TTGIR/LLVM/ISA/HSACO/readobj | 完成 |
| exact Z2 LLVM/pre-LTO/LTO-MIR/ISA/HSACO/readobj | 完成，initial MLIR printer 明确 N/A |
| dynamic PMC 与 static ISA 分离 | 完成 |
| Z2 vs native machine attribution | 完成，结论为 materialization/dataflow 主差距 |
| kernel source 修改 | 无 |
| X2/production 接入 | 无 |
| Z4 实现 | 无 |

本轮把 Stage 6Z 的问题从“猜 barrier、猜 LDS、猜 selector”收敛成了一个可
复现的机器事实：**在相同 WG/CTA/MFMA 几何下，fixed Z2 的主要额外成本是
source-visible operand materialization 与 address/layout feeding，而不是
MFMA 数或 spill。**
