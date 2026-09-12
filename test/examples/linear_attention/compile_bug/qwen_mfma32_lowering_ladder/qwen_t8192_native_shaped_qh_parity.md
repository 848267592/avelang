# Qwen GDN T=8192 Native-Shaped Q@H Parity

## 结论

本轮完成了一个 experimental-only 的 AveLang Q@H 路径。它使用 WG128、两
个 wave、gfx942 和现有已经关闭的 WG128 Q@H full64 lowering；Z5B 没有修改，
没有触碰 Q@K、score@V、调度、barrier/waitcnt 或生产路径。

最终结论是 **B：没有达到 exact Q@H machine-chain parity，第一处剩余的具体
ISA 差异是 LDS consumer 到 MFMA fragment 的形成方式**。

两边已经确认相同的部分：

- Q@H 的逻辑操作都是 `Q[64,32] @ H[32,64] -> FP32[64,64]`；
- 最终算术指令都是 `v_mfma_f32_32x32x8_bf16`；
- 实验 AveLang 路径已经是 WG128、two-wave contract；
- 新路径的 Q@H correctness oracle 四组均 `max_abs=0`；
- 新路径 private segment、VGPR spill 和 SGPR spill 都是零。

仍然不同的是：AveLang 在 Q@H 的 LDS consumer 中使用若干窄
`ds_read_u16`，再用 `v_perm_b32` 组成 MFMA 输入；selected Triton 使用
blocked/swizzled shared encoding，经 packed `ds_read2_b64`/
`ds_read2st64_b64` 直接形成 typed dot operand。因而不能把两个路径称为
逐 lane、逐字节的物理 parity。

本轮没有 benchmark。报告中的指令数量是 static lexical count，不能解释为
dynamic hardware work 或 latency。

## 冻结对象和实验产物

实验 source：

[`repro_qwen_gdn_t8192_native_shaped_qh_wg128.py`](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/repro_qwen_gdn_t8192_native_shaped_qh_wg128.py)

AveLang artifact：

[`qwen_t8192_native_shaped_qh_wg128/`](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_t8192_native_shaped_qh_wg128/)

实验 contract：

| 项目 | 值 |
|:--|:--|
| target | gfx942 |
| logical op | Q `[64,32]` x H `[32,64]` -> FP32 `[64,64]` |
| workgroup | 128 threads，2 waves |
| MFMA | `v_mfma_f32_32x32x8_bf16` |
| Z5B | 未修改 |
| Q@K / score@V | 未触碰 |
| benchmark | 未执行 |

Triton 对照是 T=8192 public API 实际选中的 frozen object：

[`native/T8192/selected/`](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_native_chunko/native/T8192/selected/)

| Triton identity | 值 |
|:--|:--|
| kernel | `chunk_fwd_kernel_o` |
| workgroup | 128 threads，2 waves |
| BT/BV/BK | 64 / 64 / 32 |
| stages | 2 |
| HSACO SHA256 | `e201dd58c83e64565f10754789ac294df53343b9bbbe764e8f667817066ee5` |
| ISA SHA256 | `00f0647036be4632891e48ecea3b937357e4814aace57c84f4f127ef94614df5` |
| LLVM SHA256 | `9c2d3a9c142c6c8dcd4eb965abf12dfcb0d2d84aab39c1736ff1ea1aac9001c9` |

AveLang artifact identity：

| AveLang identity | 值 |
|:--|:--|
| HSACO SHA256 | `6ac7990e3f66489026ee90184e8ef0a80e4d530073f6ccdf279d618301eae070` |
| final ISA SHA256 | `8ca94f08fb3a3d9164113895f7ae446fb2a66ce1ed0ef7c3fad1348d45637590` |
| lowered LLVM SHA256 | 见 `artifact_hashes.sha256` 和 `compiler_ir/kfrag/` |
| code-object VGPR/AGPR/SGPR | 72 / 32 / 20 |
| LDS | 16384 B |
| private segment | 0 B |
| VGPR/SGPR spill | 0 / 0 |
| wavefront | 64 |

Triton selected object 的 ELF note 是 VGPR/AGPR/SGPR=`220/64/76`，private
segment 和 spill 为零。其 selected JSON 记录 shared metadata 12288 B，而
ELF note 的 fixed LDS 为 0；这两个字段的语义不同，本报告不把它们强行合并
成一个 LDS 数字。

## Correctness

新 source 在 capture 前先编译并保存 binary，然后运行已有 Q@H fragment oracle
风格的区分输入。`row_code` 和 `row_half` 都覆盖低/高输出 half，并分别使用
`b_high=1` 和 `b_high=2`。

| source pattern | b_high | max_abs | finite | 结果 |
|:--|--:|--:|:--|:--|
| `row_code` | 1 | 0.0 | pass | pass |
| `row_code` | 2 | 0.0 | pass | pass |
| `row_half` | 1 | 0.0 | pass | pass |
| `row_half` | 2 | 0.0 | pass | pass |

运行结果保存在：

[`correctness.json`](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_t8192_native_shaped_qh_wg128/correctness.json)

`all_max_abs_zero=true`。debug BF16 readback、raw FP32 output 和 finite 检查
均通过。

## 编译流水和证据等级

本轮复用已有 C16 WG128/full64 compiler gate，没有修改 compiler source：

```text
native_shaped_qh_wg128.py
  -> post_block_dot_lowering.mlir
  -> post_block_dot_operand_materialization.mlir
  -> post_late_lowering.mlir
  -> preopt/postopt LLVM
  -> exact full-LTO MIR replay
  -> final_isa.s
  -> native_shaped_qh_wg128.hsaco
```

相关文件位于：

```text
qwen_t8192_native_shaped_qh_wg128/compiler_ir/kfrag/
qwen_t8192_native_shaped_qh_wg128/exact_lto/
qwen_t8192_native_shaped_qh_wg128/final_isa.s
qwen_t8192_native_shaped_qh_wg128/native_shaped_qh_wg128.hsaco
```

`exact_lto_replay.returncode=0`。exact-LTO 目录包含 20 个 pass section，
从 greedy RA 前后到 virtual-register rewrite 和 prologue/epilogue。所有 section
的 `av32_spill_saves`、`av64_spill_saves` 和 spill virtual register 列表均为
零。该证据证明这次产物确实完成了 exact-LTO MIR replay；它不表示已获得
Triton 的相同 register allocation。

`compiler_ir/recurrence/` 是全局 dump hook 产生的 incidental persistent-
recurrence 输出，不是本 Q@H 结论的依据；Q@H 的相关 IR 以
`compiler_ir/kfrag/` 为准。

## Q@H machine ledger

下表将 AveLang 的整个 isolated Q@H probe 与 Triton 的 T=8192 selected full
`chunk_fwd_kernel_o` 分开标注。两者不是同一个完整工作量，所以 count 栏只能
用于结构观察，不能直接做性能排名。

| machine item | AveLang native-shaped Q@H | selected Triton/native | parity |
|:--|:--|:--|:--|
| program/wave ownership | WG128，2 waves；source 使用 wave/row-half，oracle 已闭合 | WG128，2 waves；TTGIR `#mma warpsPerCTA=[1,2]` | contract-level aligned |
| logical output | 两个 32-column accumulator half 覆盖 `[0,64)` | `#mma` dot operand 覆盖同一逻辑 `[64,64]` output | logical pass |
| global/buffer load | final ISA 有 2 条 `global_load_dwordx4`；整个 probe 的 combined global/buffer lexical count=4 | selected full kernel 有 `buffer_load_dwordx4`；whole-kernel lexical count=28 buffer loads、18 global loads | partial |
| source packet ownership | Q source packet 由 C16 Q-role lowering 按 WG128 plan 生产 | blocked ownership 由 TTGIR encoding 生产 | not byte-identical |
| LDS write | Q source consumer 代表序列为 `ds_write_b64`，offset 8192/8256；整个 probe `ds_write=68`，其中包含 identity-H 初始化的 `ds_write_b16` | `ds_write2st64_b64`、`ds_write_b128`；whole selected kernel `ds_write=36` | different physical path |
| LDS read | 代表序列有 `ds_read_b64`、`ds_read2_b64`，同时有多条 `ds_read_u16`；整个 probe `ds_read=45` | `ds_read2_b64`、`ds_read2st64_b64`、`ds_read_b64`；whole selected kernel `ds_read=56` | first remaining divergence |
| permutation | Q@H isolated probe `v_perm_b32=20` | selected full kernel `v_perm_b32=48`，由多个阶段共同贡献 | not count-comparable |
| MFMA | 8 static `v_mfma_f32_32x32x8_bf16` in isolated Q@H probe | 80 static MFMA32 in selected full kernel；Q@H debug region is part of full loop | opcode pass, count not comparable |
| barrier | 2 static `s_barrier` | 11 in selected full kernel | not tuned in this task |
| waitcnt | 23 static `s_waitcnt` | 63 in selected full kernel | not tuned in this task |
| registers | VGPR/AGPR/SGPR=72/32/20 | VGPR/AGPR/SGPR=220/64/76 | different allocation |
| LDS metadata | 16384 B fixed LDS | JSON shared=12288 B，ELF fixed LDS=0 | metadata schemes differ |
| private/spill | private=0，spill=0 | private=0，spill=0 | pass |

### Count scope warning

AveLang 的 `static_isa_counts.json` 是该 isolated Q@H kernel 的 lexical count：

```json
{
  "mfma32": 8,
  "global_or_buffer_load": 4,
  "ds_read": 45,
  "ds_write": 68,
  "v_perm_b32": 20,
  "s_barrier": 2,
  "s_waitcnt": 23
}
```

其中 `global_or_buffer_load=4` 是 family 汇总，不是字节数；实际 ISA 中可以看到
两条 `global_load_dwordx4` 以及其余 helper/buffer load。Triton 的 28/18/56/36/80
是 selected full kernel 的静态 lexical 统计。由于 isolated Q@H 与 full chunk-o
的 loop、阶段和展开不同，不能用 `8 < 80` 推出 AveLang 做得更少或更快，也不能
用 `20 < 48` 推出缺少 permutation。

## 代表性回溯链

### AveLang

最终 ISA 的 Q@H 代表窗口位于 `final_isa.s` 约 382 行以后：

```text
global_load_dwordx4              offset 0 / 64
  -> ds_write_b64                LDS offset 8192 / 8256
  -> s_waitcnt + s_barrier
  -> ds_read_b64                 LDS offset 8192
  -> ds_read_u16                 offsets 0, 128, 256, 384, ...
  -> v_perm_b32                  selector s4
  -> v_mfma_f32_32x32x8_bf16
```

具体可见：

```text
final_isa.s:382-383   global_load_dwordx4
final_isa.s:398-411   ds_write_b64
final_isa.s:414-432   barrier, ds_read, permutation, first MFMA
final_isa.s:437-462   more ds_read_u16, permutation and MFMA consumers
```

这里的 `ds_read_u16` 和 `v_perm_b32` 不是错误，它们是当前 AveLang source/view
表达在该 physical layout 下形成 BF16 fragment 的合法路径；但它不是 native 的
packed dot-operand chain。

### Triton/native

TTGIR 直接保存了 typed contract：

```text
chunk_fwd_kernel_o.ttgir:122-124  Q/H/K shared memdesc
chunk_fwd_kernel_o.ttgir:142      Q buffer_load
chunk_fwd_kernel_o.ttgir:195-196  Q buffer_load -> local_load dot_op opIdx=0
chunk_fwd_kernel_o.ttgir:206-207  K buffer_load -> local_load dot_op opIdx=1
chunk_fwd_kernel_o.ttgir:213-217  H local_load/transpose -> Q@H tt.dot
```

selected ISA 的对应窗口包含：

```text
buffer_load_dwordx4
  -> ds_write2st64_b64 / ds_write_b128
  -> s_barrier
  -> ds_read2_b64 / ds_read2st64_b64
  -> v_mfma_f32_32x32x8_bf16
```

代表性指令位于：

```text
chunk_fwd_kernel_o.amdgcn:162-172  packed LDS writes
chunk_fwd_kernel_o.amdgcn:192-207  vector load, LDS read and MFMA
chunk_fwd_kernel_o.amdgcn:213-235  following packed reads and MFMA
```

Triton 的 TTGIR `#shared/#shared1/#shared2` 和 `#ttg.dot_op` 保存了 shared
physical encoding 与 MFMA operand encoding；AveLang 新 probe 的 source contract
在 C16 lowering 中已经能选中 WG128 plan，但最终 Q@H consumer 仍通过普通 shared
view 和窄 element read 形成 fragment。

## Parity matrix

| parity 层次 | 结论 | 证据 |
|:--|:--|:--|
| program/wave contract | **PASS at contract level** | 两边都是 WG128、wave64、2 waves；AveLang oracle 通过 |
| global packet shape | **PARTIAL** | AveLang Q source 有 `global_load_dwordx4`，但完整 producer/lane contract 未与 Triton byte-identical |
| LDS write physical encoding | **FAIL** | native 有 packed/swizzled `ds_write2st64_b64`/`ds_write_b128`，AveLang 代表路径为 `ds_write_b64` 加 ordinary view/stage |
| LDS read/fragment formation | **FAIL, first concrete divergence** | AveLang 出现 `ds_read_u16` + `v_perm_b32`，native 是 packed `ds_read2*` 到 typed dot operand |
| MFMA opcode/dtype | **PASS** | 两边都是 `v_mfma_f32_32x32x8_bf16` |
| MFMA count | **NOT PROVEN** | 8 是 AveLang isolated Q@H count，80 是 native full-kernel count，dynamic count 未采集 |
| accumulator ownership | **logical pass only** | 两个 AveLang accumulator half 与 native 逻辑 output shape 对应，物理寄存器分配不同 |
| private/spill | **PASS** | AveLang 和 native 都是 private=0、spill=0 |

## First remaining divergence

在已经对齐 WG128/two-wave contract，并确认 MFMA opcode 相同之后，执行链中第一
个可由最终 ISA 直接确认的剩余差异是：

```text
AveLang:
  LDS address/view
    -> multiple ds_read_u16
    -> v_perm_b32 reconstruction
    -> MFMA32

Triton:
  blocked/swizzled shared encoding
    -> packed ds_read2_b64 / ds_read2st64_b64
    -> typed dot operand
    -> MFMA32
```

因此当前问题不是缺少 MFMA intrinsic，也不是 MFMA32 与 MFMA16 的选择。它是
**Q@H 的 shared physical layout 到 dot-operand fragment 的 lowering contract 尚未
完全保留**。本轮没有做 pass-by-pass convergence bisect，所以不能进一步断言
某个 LLVM pass 是唯一责任点。

## 唯一后续登记项，不执行

如果继续 Q@H-only 路线，下一次应只测试一个 source/lowering 控制杆：让 AveLang
WG128 Q@H consumer 直接使用 native-compatible 的 packed LDS/dot-operand
representation，使其数据链趋近：

```text
typed Q/H producer
  -> packed/swizzled LDS store
  -> packed LDS read
  -> dot operand
  -> existing MFMA32
```

这不是新增 MFMA intrinsic，也不是 instruction scheduling。它只能在新的
experimental Q@H path 中验证；本轮没有实现它，也没有继续 Q@K、score@V 或 full
chunk-o。

## 明确未做事项

- 没有修改 Z5B；
- 没有修改 Q@K 或 score@V；
- 没有 benchmark、rocprof 或动态 PMC；
- 没有调 barrier、waitcnt 或 scheduler；
- 没有复制 Triton HSACO、ISA 或物理寄存器编号；
- 没有新增 MFMA intrinsic；
- 没有接入生产 selector 或 X2。

---

## 14. Final addendum：packed-H v6 结果

本节是 packed-H v6 完成后的最新结论。前文第 7、8、11 节描述的是旧
ordinary-H baseline，保留作为历史对照；下面的 v6 结果覆盖同一 Q@H probe
在 packed-H arm 下的新机器证据。v6 没有修改 Z5B、Q@K、score@V、调度、
waitcnt/barrier、production selector 或 X2，也没有运行 benchmark。

### 14.1 实验身份和 correctness

v6 使用：

~~~text
AVELANG_C16_WG128_QH_PACKED_H=1
AVELANG_C16_WG128_QH=1
AVELANG_C16_WG128_QH_FULL64=1
AVELANG_BLOCK_DOT_LOWERING=specialized
~~~

编译和运行都在 ljd_qwen_vllm_avelang_rocm722 Docker 中完成。实际被 Python
import 的是 build-software-pipeline/python/_avelang_bindings.so；因此本轮
重新构建了这个实际 binding，没有把其他 build 目录的结果当成证据。

source：

[repro_qwen_gdn_t8192_native_shaped_qh_wg128.py](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/repro_qwen_gdn_t8192_native_shaped_qh_wg128.py:121)

artifacts：

[qwen_t8192_native_shaped_qh_wg128_packed_h_v6/](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_t8192_native_shaped_qh_wg128_packed_h_v6/)

四个区分输入全部 exact：

| pattern | b_high | raw FP32 max_abs | finite | 结果 |
|:--|--:|--:|:--|:--|
| row_code | 1.0 | 0.0 | pass | pass |
| row_code | 2.0 | 0.0 | pass | pass |
| row_half | 1.0 | 0.0 | pass | pass |
| row_half | 2.0 | 0.0 | pass | pass |

correctness.json 记录了 Q[64,32] @ H[32,64] -> FP32[64,64]、WG128、
two waves、v_mfma_f32_32x32x8_bf16 和 all_max_abs_zero=true。debug BF16
readback、raw FP32 output 与 finite 检查均通过。

### 14.2 v6 的 packed-H producer/consumer

v6 的 H stage 是 [64,32] BF16，source 再将其作为 [64,8,2] U32 view，
strides [16,2,1] 写入四-BF16 packet：

~~~text
repro...py:139       h_stage = al.make_shared((64, 32), al.bf16)
repro...py:151-153   U32 packet view
repro...py:154-195   packet producer
~~~

对 producer_tid，source 显式使用：

~~~text
t113 = (producer_tid << 6) & 1984
t127 = (producer_tid >> 2) & 8
t125 = (producer_tid << 5) & 2048
h_base = t113 | t127 | t125
packet_step = fragment << 4
h_byte = h_base ^ packet_step
~~~

每次写入一个包含四个 BF16 值的 packet；logical_byte 只负责计算 identity-H
的逻辑值，h_byte 负责实际物理写入。因此 v6 已经不是旧的逐元素 ds_write_b16
H producer。

late-lowering MLIR 在 Q/H MFMA consumer 处保留了 typed vector：

~~~text
post_late_lowering.mlir:546, 552, 557, 561, 566, 570, 575, 579
post_late_lowering.mlir:601, 604, 609, 613, 618, 622, 627, 631
~~~

这些位置是 llvm.load -> vector<4xbf16>，并带有 c16.native_dot_role = Q 或 H。
post-opt LLVM 的最终 Q/H MFMA 输入则是 <4 x i16> LDS loads（例如
postopt_llvm.ll:497-520 和 526-539），直接喂给 8 个
llvm.amdgcn.mfma.f32.32x32x8bf16.1k。

因此，final ISA 中约 final_isa.s:359-366 的 ds_read_u16 不能再被误归为
H MFMA operand。它位于 debug BF16 readback/output 路径；Q/H 的 MFMA operand
链已在 LLVM 中是 packed <4 x i16> load。

### 14.3 v6 资源和静态机器证据

| 项目 | v6 packed-H |
|:--|--:|
| static MFMA32 | 8 |
| global/buffer load lexical count | 4 |
| ds_read lexical count | 19 |
| ds_write lexical count | 40 |
| v_perm_b32 | 4 |
| s_barrier | 2 |
| s_waitcnt | 20 |
| code-object VGPR/AGPR/SGPR | 68 / 32 / 14 |
| fixed LDS | 12288 B |
| private segment | 0 B |
| VGPR/SGPR spill | 0 / 0 |
| exact-LTO AV32/AV64 spill saves | 0 / 0 |

以上是整个 isolated probe 的 static lexical count，不是 dynamic PMC，也不能
与 native full chunk-o 的全 kernel count 直接排名。v6 exact-LTO replay 有 20
个 section，所有 section 的 AV32/AV64 spill save/reload 都是零。

### 14.4 selected Triton 的冻结 H physical contract

selected artifact：

[native/T8192/selected/](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_native_chunko/native/T8192/selected/)

在 chunk_fwd_kernel_o.llir:103-133，native H producer 使用：

~~~text
pbase = ((tid << 4) & 2032) ^ (tid & 56)
~~~

H shared region 从 byte offset 8192 开始，producer packet locations 是：

~~~text
8192 + pbase
8192 + (pbase | 2048)
8192 + (pbase ^ 8)
8192 + ((pbase ^ 8) | 2048)
~~~

最终 ISA 的对应 producer 是 ds_write2st64_b64，可见
chunk_fwd_kernel_o.amdgcn:162-172 及后续重复 stage。

native H consumer 在 chunk_fwd_kernel_o.llir:134-162,191-198 使用：

~~~text
t113 = (tid << 6) & 1984
t126 = (tid << 2) & 56
t127 = (tid >> 2) & 8
t125 = (tid << 5) & 2048
h_native = t113 | (t126 ^ t127) | t125
~~~

然后读取 h_native ^ 0、h_native ^ 16、h_native ^ 32、h_native ^ 48。其 ISA
形态是 ds_read2_b64/ds_read2st64_b64 到 MFMA32，例如
chunk_fwd_kernel_o.amdgcn:202-220。

### 14.5 exact physical 对照

| 项目 | AveLang v6 | selected Triton | 结论 |
|:--|:--|:--|:--|
| logical H operand | identity H，Q@H raw output exact | blocked/shared1 H，Q@H logical contract | logical correctness pass |
| producer storage | compact [64,32] BF16 + U32 packet view | shared1 blocked/swizzled packet region | physical layout 不同 |
| producer base | t113 \| t127 \| t125，再 xor {0,16,32,48} | ((tid<<4)&2032) ^ (tid&56)，再组合 \|2048、^8 | 不相同 |
| consumer base | C16 packed-H branch 使用 t113 \| t127 \| t125 | t113 \| (t126 ^ t127) \| t125 | 不相同 |
| consumer type | LLVM vector<4xbf16>，post-opt <4xi16> | LLVM <4xi16> / TTGIR dot_op | packed semantics趋同 |
| LDS read opcode | 主要是 ds_read_b64 | ds_read2_b64/ds_read2st64_b64 | machine path 不同 |
| MFMA | exact v_mfma_f32_32x32x8_bf16 | exact v_mfma_f32_32x32x8_bf16 | opcode pass |

这证明了一个局部但重要的事实：AveLang 可以表达并正确执行 BF16x4 packed
LDS producer 到 typed MFMA consumer。但 v6 还没有复现 selected Triton 的
pbase -> shared1 -> h_native packet ownership；它是自洽的 compact layout，
不是 Triton 的 byte/offset-equivalent layout。

### 14.6 Updated parity decision

| parity 项 | v6 结论 |
|:--|:--|
| A. MFMA opcode/dtype | PASS |
| B. MFMA count | NOT PROVEN；只证明 isolated static=8，未采 dynamic PMC |
| C. WG128/two-wave contract | PASS |
| D. packed Q/H operand representation | PASS at LLVM semantic level |
| E. producer/consumer lane-level physical mapping | FAIL / not exact parity |
| F. exact native LDS opcode/offset chain | FAIL |
| private/spill | PASS |

本轮应停止在这里。首个剩余差异不是 MFMA intrinsic，也不是 instruction
scheduling，而是 H operand 的 lane/physical LDS mapping：v6 的 producer 和
consumer 互相匹配，所以 correctness exact；但它们与 Triton 的 pbase 和
h_native 函数不相同，因而最终使用了不同的 LDS store/read instruction path。

本轮唯一登记、但不实现的下一代码改动是：只为 H operand 建立一个
native-compatible producer/consumer physical-layout arm，让 pbase、shared
bank placement 和 h_native ^ {0,16,32,48} 在同一个 experimental contract
中一致；Q operand、MFMA、WG128、barrier/scheduler、Q@K、score@V 和 Z5B 全部
保持冻结。下一轮应先做 correctness/MLIR/LLVM/ISA parity，仍不能提前 benchmark。

最终状态：Case B。packed-H consumer 已闭合，但 exact Triton H physical
producer/consumer parity 尚未达到；首个剩余差异是 H 的 physical LDS mapping，
不是 MFMA intrinsic，也不是 instruction scheduling。

## 15. Final addendum：v8 exact H LDS physical-offset closure

本节覆盖上一节记录的 v6 之后的 v7/v8 修正。v6 的 Case B 结论仍保留为历史
状态；**v8 是本次 H physical-offset audit 的最终状态**。本轮仍然没有 benchmark，
没有触碰 Q、Q@K、score@V、Z5B、scheduler、barrier/waitcnt 或 production。

### 15.1 v8 执行和修正记录

实验使用同一个 Docker 容器 `ljd_qwen_vllm_avelang_rocm722`，并重新构建了
`_avelang_bindings`。修改只有两处：

1. packed-H source producer 使用 selected Triton 的 `pbase` packet map，并把
   四个 packet 排列为 `[pbase, pbase|2048, pbase^8, (pbase^8)|2048]`；
2. Qwen block-dot H consumer 使用 `t115 xor t116`，不再使用旧的
   `t127` 单项 inner address。

v7 曾经把 packet 的 bit-0/bit-1 顺序写反，因而出现 `max_abs=4032`。修正 packet
顺序后得到 v8；这次失败和修正被保留，避免把一次错误的 producer map 误写成
compiler 或 MFMA 问题。

对应 source 和 compiler 修改位置：

```text
repro_qwen_gdn_t8192_native_shaped_qh_wg128.py:160-205
lower_qwen_block_dot_pass.cc:5054-5151
```

v8 工件：

[v8 artifacts](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_t8192_native_shaped_qh_wg128_packed_h_v8/)

| v8 artifact | SHA256 |
|:--|:--|
| HSACO | `aea62cf6eebf0692542aa988f5c52f7c68dd5229dc14399fa02d26967ab94e12` |
| final ISA | `e31221164bacde73ee8488dc2521840e30d44e5f9032d8212aad14aeb5c2c8cb` |
| pre-opt LLVM | `87eb2cf59b699ee9fd6550c9b4a2256c0bdad0766076da1794fd3fc0f92e2c8f` |
| post-opt LLVM | `50b91ecccdec237955e9cbc1952bfb77e05dceaafac257651c17c252bf33dc94` |
| final MLIR | `fe87db91baf9c695d2072190b94a4d161a13461fa8d5167ac71fd9fb0ef623ee` |

### 15.2 Correctness and resource gate

v8 使用原有 Q@H oracle，四个区分方向全部通过：

| source pattern | `b_high` | `max_abs` |
|:--|--:|--:|
| `row_code` | 1 | 0.0 |
| `row_code` | 2 | 0.0 |
| `row_half` | 1 | 0.0 |
| `row_half` | 2 | 0.0 |

`debug` BF16 readback、raw FP32 output 和 finite 检查均通过，
`all_max_abs_zero=true`。v8 没有运行 benchmark。

| resource / static count | v8 |
|:--|--:|
| workgroup / waves | 128 / 2 |
| MFMA32 | 8 static |
| global/buffer load family | 4 static |
| `ds_read` | 17 static |
| `ds_write` | 40 static |
| `v_perm_b32` | 4 static |
| `s_barrier` / `s_waitcnt` | 2 / 18 static |
| code-object VGPR/AGPR/SGPR | 72 / 32 / 14 |
| LDS | 12288 B |
| private segment | 0 B |
| VGPR/SGPR spill | 0 / 0 |
| exact-LTO AV32/AV64 spill saves | 0 / 0 |

这些 count 是 isolated probe 的 static lexical count，不是 dynamic PMC，也不是
native full chunk-o 的性能数字。`exact_lto/summary.json` 中 20 个 replay section
均没有 AV32/AV64 spill save，最终 ELF note 的 private segment 仍为零。

### 15.3 冻结 Triton H producer/consumer contract

selected Triton H producer 的 byte-level packet base 是：

```text
pbase(p) = ((p << 4) & 2032) ^ (p & 56)

P(p, j) = (pbase(p) ^ (((j >> 1) & 1) << 3))
          | ((j & 1) << 11)
```

所以 `j=0,1,2,3` 的 packet 顺序是：

```text
pbase, pbase | 2048, pbase ^ 8, (pbase ^ 8) | 2048
```

在 Triton selected object 中，H shared region 的整体 base 是 `+8192`；下表的
地址全部先写成相对于 H region 的 byte offset，最后再加该 base。

H consumer 的实际地址链是：

```text
t113 = (t << 6) & 1984
t115 = (t << 2) & 56
t116 = (t >> 2) & 8
t125 = (t << 5) & 2048
hBase(t) = t113 | (t115 ^ t116) | t125
C(t, f) = hBase(t) ^ (f << 4)
```

其中 `t = lane | (outputHalf << 6)`，`f=0..3`。从 consumer packet 反解 producer
时，使用：

```text
p = ((t & 31) << 2) | (f & 3)
j = ((t >> 6) & 1) | (((t >> 5) & 1) << 1)
P(p, j) == C(t, f)
```

这不是只比较 source-level 变量名：v8 的 post-opt LLVM 已经保留了上述最终地址
计算。`postopt_llvm.ll:509-520` 生成 `t113/t115/t116/t125/hBase`，
`:523-549` 对 H shared pointer 进行 `<4 x i16>` loads，并对每个 packet 使用
`xor 8/16/24`（对应 byte `xor 16/32/48`）。producer 的 packed
`store <2 x i32>` 则在 `postopt_llvm.ll:200-204`、`:257-261`、`:314-318`、
`:371-375` 等重复 packet region 中落到 addrspace(3)。

### 15.4 代表性 lane/packet parity table

每个 producer tuple 是 `(producer_tid, producer_packet_j)`；`C0..C3` 是
`f=0..3` 的 consumer byte offsets。每个表项都满足 `P(p,j)==C(t,f)`。

| outputHalf | lane | `t` | `C0,C1,C2,C3` relative H bytes | producer `(p,j)` for `f=0..3` |
|--:|--:|--:|:--|:--|
| 0 | 0 | 0 | 0, 16, 32, 48 | (0,0), (1,0), (2,0), (3,0) |
| 0 | 1 | 1 | 64, 80, 96, 112 | (4,0), (5,0), (6,0), (7,0) |
| 0 | 4 | 4 | 272, 256, 304, 288 | (16,0), (17,0), (18,0), (19,0) |
| 0 | 16 | 16 | 1024, 1040, 1056, 1072 | (64,0), (65,0), (66,0), (67,0) |
| 0 | 32 | 32 | 8, 24, 40, 56 | (0,2), (1,2), (2,2), (3,2) |
| 0 | 63 | 63 | 2032, 2016, 2000, 1984 | (124,2), (125,2), (126,2), (127,2) |
| 1 | 0 | 64 | 2048, 2064, 2080, 2096 | (0,1), (1,1), (2,1), (3,1) |
| 1 | 1 | 65 | 2112, 2128, 2144, 2160 | (4,1), (5,1), (6,1), (7,1) |
| 1 | 4 | 68 | 2320, 2304, 2352, 2336 | (16,1), (17,1), (18,1), (19,1) |
| 1 | 16 | 80 | 3072, 3088, 3104, 3120 | (64,1), (65,1), (66,1), (67,1) |
| 1 | 32 | 96 | 2056, 2072, 2088, 2104 | (0,3), (1,3), (2,3), (3,3) |
| 1 | 63 | 127 | 4080, 4064, 4048, 4032 | (124,3), (125,3), (126,3), (127,3) |

### 15.5 Final ISA and machine interpretation

v8 final ISA 的 H producer 已经不是 v6 的自洽 compact shortcut。代表性 packed
producer stores 位于 `final_isa.s` 的 `0x1C74`、`0x1D20`、`0x1DA4`、`0x1E1C`，
均为 `ds_write_b64`，其地址由前面的 `and/xor/or/lshl` VGPR address chain
生成。Triton 对应的 selected ISA 使用 `ds_write2st64_b64`；这是 store packing/
encoding 的差异，不是 byte offset 公式差异。

H consumer 窗口位于 `final_isa.s:290-306` 附近：packed `ds_read_b64`/
`ds_read2st64_b64` 读取后进入 `v_mfma_f32_32x32x8_bf16`。同时，LLVM 中 H 的
MFMA source 是 `<4 x i16>`，所以后续出现的 `ds_read_u16` 不属于 H MFMA operand
链；它来自 probe 中其它 scalar/debug 路径。Triton selected ISA 的 H consumer
使用 `ds_read2_b64`/`ds_read2st64_b64`，对应 `hBase xor {0,16,32,48}`。

因此本轮得到的是两个需要明确区分的结论：

1. **H effective LDS byte offsets：Case A，已达到 parity。** 代表性 lane、
   output-half 和四个 packet 的 producer/consumer 地址全部相等，且该地址链已
   穿过 LLVM 到最终 ISA 的 packed H/MFMA 区域；
2. **完整 ISA opcode/packing parity：尚未宣称。** AveLang 当前使用的 packed
   store/read 形式与 Triton 的 `ds_write2st64_b64`/
   `ds_read2st64_b64` 仍不同。这个差异属于下一层机器指令打包/局部 lowering
   问题，本轮不继续实现。

本轮停止条件已经满足：H physical producer/read offset parity 已闭合；没有新增
MFMA intrinsic 的理由，也没有开始 scheduler、barrier/waitcnt 或 benchmark 实验。

## 16. v9 read-only audit：相同 packet/offset 为何没有收敛到 DS pair

### 16.1 本轮边界

本轮冻结 v8 的 H 地址公式、producer packet 排列、consumer `hBase ^
{0,16,32,48}`、Q 路径、MFMA、WG128、barrier/waitcnt 和所有性能路径。没有重新
跑 correctness，没有 benchmark，也没有修改 source 或 compiler。问题只限定为：

```text
相同的 H packet 和相同的有效 LDS byte offset
    -> AveLang: ds_write_b64 / 部分 ds_read2 + ds_read_b64
    -> Triton:   ds_write2st64_b64 / ds_read2*
```

v8 第 15 节已经证明了地址等价。本节审计的是地址等价之后的表示和指令形成，不能
反过来修改第 15 节的地址结论。

冻结工件：

```text
AveLang v8:
  test/examples/linear_attention/compile_bug/
  qwen_t8192_native_shaped_qh_wg128_packed_h_v8/

Triton selected T=8192:
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
  codex_qwen_bt64_stage6z_native_chunko/native/T8192/selected/
```

### 16.2 结论先行

**首个差异不在地址，而在 shared packet/layout 语义在 LLVM 边界被抹平。**

AveLang 当前路径把“这是两个具有固定 `st64` 关系的 shared packet 访问”表示成
普通的 LLVM `LoadOp`/`StoreOp` 加几个审计属性。`c16.packed_lds_store`、
`c16.native_dot_lds_load` 和 `c16.packet_width` 只记录信息，不会生成 AMDGPU 的
成对 shared-memory operation，也不会把多个 load/store 组成一个 first-class dot
operand group。

Triton 的 TTGIR 则一直保留：

```text
#ttg.swizzled_shared / #ttg.amd_rotating_shared
    -> ttg.local_store / ttg.local_load
    -> #ttg.dot_op(kWidth=4)
    -> AMDGPU shared packet lowering
```

因此，Triton 后端看见的是带有物理 layout、packet width、operand role 和 dot
consumer 关系的一组访问；AveLang 后端看见的是数条独立的 64-bit addrspace(3)
读写。两边访问的字节可以相同，但后端可利用的证明不同。

### 16.3 Store 侧：为什么 AveLang 是 `ds_write_b64`

#### Triton 的 LLVM 形态

冻结的 selected Triton `chunk_fwd_kernel_o.llir` 在 H producer 附近
(`:103-119`) 保留了四个有明显固定关系的 typed stores：

```text
store <4 x bfloat> %103, base
store <4 x bfloat> %106, base | 2048
store <4 x bfloat> %109, base ^ 8
store <4 x bfloat> %112, (base ^ 8) | 2048
```

对应的 H region 还整体位于 `+8192`。这里每个 store 都是 4 个 BF16，也就是
64-bit payload；更重要的是，四个 sibling address 的关系以直接的 GEP/常量形式
存在于同一个 producer sequence 中。Triton final ISA 将相邻的两个 64-bit
store 形成：

```text
ds_write2st64_b64 ... offset1:4
ds_write2st64_b64 ... offset0:16 offset1:20
```

证据位于 selected ISA 的 H producer 区域约 `:162-172`。这不是 Triton source
里硬编码了某个寄存器号，而是 typed shared encoding 加上可识别的 sibling store
关系穿过 LLVM 后，被 AMDGPU 后端选择为 `st64` pair。

#### AveLang 的 LLVM/MIR 形态

v8 AveLang H producer 的 `postopt_llvm.ll` 在以下位置生成独立的 store region：

```text
postopt_llvm.ll:200-204
postopt_llvm.ll:257-261
postopt_llvm.ll:314-318
postopt_llvm.ll:371-375
```

每个 packet 是：

```text
store <2 x i32> ..., ptr addrspace(3) %computed_gep, align 8
```

`<2 x i32>` 和 Triton 的 `<4 x bfloat>` 都是 64-bit，`align 8` 也满足 packed
访问。因此，**payload 宽度和对齐不是主要差异**。差异在于：AveLang 的 packet
来自 source 的 `u32` view/`packet` 构造，之后由 ordinary `vector::StoreOp`
逐个落到经过 `physical_row/physical_group` 和多个 GEP 计算的地址；LLVM 中没有
一个“这两个 store 应组成 `st64` pair”的关系。

exact-LTO MIR 直接证实了后端最终选择：

```text
kernel_section_08.mir:220-263
  H producer: DS_WRITE_B64_gfx9
  没有 DS_WRITE2ST64_B64
```

最终 v8 ISA 的 H producer 代表位置约为 `0x1C74`、`0x1D20`、`0x1DA4`、
`0x1E1C`，同样全部是 `ds_write_b64`。因此不能把结果解释为“地址错了”或
“BF16 packet 没有打包”；准确表述是：

> AveLang 在 LLVM/MIR 中只保留了独立 64-bit store；Triton 保留了可被后端识别的
> 固定 sibling packet 关系，因而形成了不同的 DS 指令编码。

当前 compiler 的对应实现也支持这个判断：

```text
lower_qwen_block_dot_pass.cc:880-918
  通过 mlir::vector::StoreOp 写 packed BF16x4，附加 c16.* 属性

lower_qwen_block_dot_pass.cc:4425-4458
  通过 LLVM::LoadOp 读 BF16x4，附加 c16.* 属性
```

代码中没有通用的 `ds_write2`/`ds_read2` intrinsic，也没有把 `c16.*` 属性降低为
target-specific paired shared-memory op。属性是证据标签，不是指令选择约束。

### 16.4 Read 侧：为什么没有完全形成 `ds_read2*`

这里要修正一个容易误读的说法：**AveLang 不是完全没有 read packing。** v8 已经
有一部分配对，说明 LLVM/MIR 后端在局部情况下能够识别兼容关系；但它没有把整个
H MFMA operand group 收敛成 Triton 同样的 `ds_read2*` 序列。

#### 两边 LLVM load 的宽度其实已经接近

Triton selected LLIR 的 H consumer (`:130-181`) 使用多个：

```text
load <4 x i16>, align 8
```

v8 AveLang `postopt_llvm.ll` 的 H consumer (`:509-549`) 也使用多个：

```text
load <4 x i16>, align 8
```

所以 read 侧不能简单归因于“AveLang 还是 BF16 scalar load”或“缺少 64-bit
vector load”。v8 的每次 H read 已经是 4 个 16-bit 元素，即 64-bit payload。

#### v8 的实际部分配对

v8 exact-LTO MIR `kernel_section_08.mir:274-293` 显示同时存在：

```text
DS_READ2ST64_B64  %ir.430, %ir.454
DS_READ_B64       %ir.426
DS_READ_B64       %ir.434
DS_READ_B64       %ir.441
DS_READ_B64       %ir.448
DS_READ2ST64_B64  %ir.437, %ir.457
DS_READ2ST64_B64  %ir.444, %ir.460
DS_READ2ST64_B64  %ir.451, %ir.463
DS_READ2_B64      %ir.311, %ir.329
```

这说明后端确实能对部分 load pair 进行合并，但仍有明确的单独
`DS_READ_B64`。v8 final ISA H/MFMA 窗口因此是 mixed form，而不是纯粹的 Triton
`ds_read2*` form。后续更晚出现的 `DS_READ_U16` 属于 probe 的其它 debug/scalar
路径，不属于已经闭合的 H MFMA operand chain。

#### 为什么同样的 offset 没能全部配对

AveLang 当前 H consumer 的 LLVM 形态是独立的 load：

```text
%130 = hBase
%131 = hBase + 2048
%133 = hBase ^ 16
%134 = (hBase ^ 16) + 2048
%136 = hBase ^ 32
%137 = (hBase ^ 32) + 2048
%139 = hBase ^ 48
%140 = (hBase ^ 48) + 2048

load <4 x i16> %130/%131/%133/%134/%136/%137/%139/%140
```

这些数值确实与冻结的 H 地址公式一致，但在 current AveLang lowering 中，load
仍然是由 byte-address `LLVM::GEPOp + LLVM::LoadOp` 逐个发出的，没有一个共同的
`dot_op`/shared encoding 对象表明：

```text
这八个 load 是同一个 H operand 的四个 packet，且应按两个 st64/read2 group
提供给同一 MFMA consumer。
```

后端只能在它能从相邻 memory op、地址 provenance、别名信息、offset 关系和调度
顺序中局部证明时配对。能够证明的部分形成 `DS_READ2ST64_B64` 或
`DS_READ2_B64`；其余部分由于 GEP 来源、packet group 边界、与另一 operand 的
交错以及普通 raw load 表示，保留为 `DS_READ_B64`。这解释了“有部分 read2，
但不是完全相同的 read2 packing”。

### 16.5 Triton 为什么更容易形成完整 packet packing

selected Triton TTGIR 在文件头和主循环中保留了机器相关但仍然是通用 layout
语义：

```text
#mma = #ttg.amd_mfma<warpsPerCTA=[1,2], instrShape=[32,32,8], isTransposed=true>
#shared1 = #ttg.swizzled_shared<vec=1, perPhase=1, maxPhase=1, order=[1,0]>
#shared2 = #ttg.swizzled_shared<vec=4, perPhase=2, maxPhase=8, order=[0,1]>

ttg.local_store  tensor -> memdesc<#shared*>
ttg.local_load   memdesc -> tensor<#ttg.dot_op<opIdx=*, kWidth=4>>
tt.trans         H linear operand -> dot_op operand
tt.dot
```

对应证据为 `chunk_fwd_kernel_o.ttgir:7-12,122-124,171-176,196-217`。这里的
`#shared*`、`#dot_op` 和 `kWidth=4` 不是注释，而是 lowering 输入的一部分。它们
让后续 AMDGPU lowering 知道 packet 的物理布局和 consumer grouping，而不是在
最后一步从普通 pointer load 猜测。

AveLang v8 的 `emitC16NativePackedLdsLoadAtByte()` 最终返回一个
`LLVM::LoadOp<vector<4xbf16>>`，producer 也最终是普通 `vector::StoreOp`。这条
路径保留了正确的地址和 payload，却没有保留 Triton 那种 first-class shared
encoding/operand grouping。因此同一地址只能得到“后端局部识别的 pair”，不能稳定
得到完整的 `ds_write2st64_b64`/`ds_read2*` contract。

### 16.6 证据分级和不能过度推断的部分

| 结论 | 证据等级 | 依据 |
|:--|:--:|:--|
| H byte offset 已相同 | A | v8 producer/consumer 逆映射表、LLVM 地址链、correctness 4/4 |
| v8 H producer 没有 DS_WRITE2ST64 | A | exact-LTO MIR 与 final ISA |
| v8 read 侧是 mixed packing | A | exact-LTO MIR 中同时出现 DS_READ2ST64/DS_READ2 和 DS_READ_B64 |
| 两边 LLVM payload 都是 64-bit packed load/store | A | v8 post-opt LLVM、Triton LLIR |
| Triton 的 typed layout/dot contract 是 packet grouping 的上游依据 | A/B | TTGIR first-class encoding，LLIR sibling GEP/store，最终 ISA |
| 某一个 LLVM pass 单独决定了 pair formation | C | 当前工件没有 AMDGPU backend pass-by-pass trace，不能伪造具体 pass 名称 |

因此，本轮可以确定 lowering **输入表示**的缺口，但不能把责任虚构成某一个
未被捕获的 LLVM pass。进一步把问题归因到具体 backend pattern/pass，需要增加
LLVM AMDGPU instruction-selection/ISel debug trace；本轮没有开启该 trace。

### 16.7 下一步只登记一个候选，不实施

唯一值得继续测试的控制杆是通用的 **paired LDS packet / dot-operand lowering
contract**：在 `block_dot_bf16_f32` 的 generic planner 中表达 producer packet
的 sibling 地址关系、64-bit element group、shared encoding、operand role 和
MFMA consumer group，让它在 LLVM 仍然可见，再由 AMDGPU lowering 选择
`DS_WRITE2ST64_B64`/`DS_READ2*`；不能修改 H 地址公式，也不能写 Qwen 专用地址
表。

这不是本轮实现，也不是建议直接手写 ISA。当前证据只支持以下顺序：

```text
first-class paired shared packet contract
    -> verify LLVM/MIR preserves sibling relation
    -> verify DS_WRITE2ST64_B64 / complete DS_READ2* formation
    -> correctness
    -> only then benchmark or scheduling audit
```

最终判断：**地址已经对；MFMA intrinsic 也已经对；当前差异是 shared packet/layout
语义没有完整保留到 AMDGPU machine lowering。** Store 侧因此没有形成
`ds_write2st64_b64`，read 侧只能局部形成 `ds_read2*`。本轮到此停止，不修改地址
公式、不修改 kernel、不跑性能。

## 17. v9 H producer sibling-store experiment：部分形成 DS_WRITE2ST64

### 17.1 实验边界和唯一 source 改动

本轮从 v8 source 直接分叉，仍冻结：

- H `pbase`、`h_byte0..3` 和 consumer `hBase ^ {0,16,32,48}` 的地址公式；
- Q、Q@K、score@V、Z5B、MFMA、WG128、2-wave ownership；
- H consumer/read path、barrier/waitcnt 和所有调度；
- 不新增 LDS intrinsic，不跑 benchmark。

唯一 source 改动在：

```text
test/examples/linear_attention/vllm_compare/
repro_qwen_gdn_t8192_native_shaped_qh_wg128.py:156+
```

旧形态是：

```text
for packet_index in range(4):
    构造一个 packet
    计算该 packet 的地址
    立刻 store
```

v9 改成：

```text
producer_tid = tid & 127
pbase = ((producer_tid << 4) & 2032) ^ (producer_tid & 56)

构造 packet0, packet1, packet2, packet3
计算 addr0, addr1, addr2, addr3

store64(addr0, packet0)
store64(addr1, packet1)
store64(addr2, packet2)
store64(addr3, packet3)
```

其中仍然是冻结的：

```text
addr0 = pbase
addr1 = pbase | 2048
addr2 = pbase ^ 8
addr3 = (pbase ^ 8) | 2048
```

因此这不是地址修复，而是把 packet construction 和 store placement 分离，给
LLVM/AMDGPU 后端一个能够识别 sibling store 的机会。

### 17.2 Correctness 和 identity

v9 在 `ljd_qwen_vllm_avelang_rocm722` 中 fresh compile 并运行了既有 4-case
Q@H oracle：

| source pattern | `b_high` | `max_abs` |
|:--|--:|--:|
| `row_code` | 1 | 0.0 |
| `row_code` | 2 | 0.0 |
| `row_half` | 1 | 0.0 |
| `row_half` | 2 | 0.0 |

`all_max_abs_zero=true`，debug BF16 readback、raw FP32 output 和 finite 检查均通过。

v9 工件：

```text
test/examples/linear_attention/compile_bug/
qwen_t8192_native_shaped_qh_wg128_packed_h_v9/
```

| artifact | SHA256 |
|:--|:--|
| HSACO | `dc7794e11fe83dcae4cc6670ba3b08ca6f50243aaaeb6ee4274b4e8fdd88b205` |
| final ISA | `4228aa139b06d92cdffb2937fcdf644b1de598e8e3955fc19d0bd9eb2a1052d4` |
| post-opt LLVM | `f62de050e03d97341eda833fe647e9a03b688c4d5b3f31a112a0ebeebf7d5ffe` |
| exact-LTO `kernel_section_08.mir` | `f83b5d5a7e52bfa39a1d30f4104f0d930f6e647cbc1dc35d34c2a0709c4bbc22` |

### 17.3 LLVM：第一组 pair 已经可见

v9 `postopt_llvm.ll:254-268` 的 store sequence 是：

```llvm
%202 = gep @shared, %.idx
%203 = gep %202, %149
store <2 x i32> %133, %203, align 8

%204 = gep %202, 2048
%205 = gep %204, %149
store <2 x i32> %148, %205, align 8

%206 = gep %202, %199       ; %199 = pbase ^ 8
store <2 x i32> %171, %206, align 8

%207 = gep @shared, %.idx11
%208 = gep %207, %200
store <2 x i32> %198, %208, align 8
```

前两条 store 具备后端需要的直接关系：同一个 `%202` base、同样的 `%149`
packet offset，第二条只增加固定的 2048 byte sibling base。它们都是 64-bit
`<2 x i32>`、`align 8`，且没有另一条 store 插入中间。v9 final ISA/MIR 已经把这
一组变成：

```text
DS_WRITE2ST64_B64  ... (store %ir.201), (store %ir.202)
```

证据：

```text
exact_lto/kernel_section_08.mir:195
final_isa.s:196
```

这证明仅仅改变 producer IR 的 store grouping，就足以让当前 generic AMDGPU
lowering 形成一个 `DS_WRITE2ST64_B64`；不需要新的 LDS intrinsic，也不需要修改
H 地址公式。

### 17.4 第二组 pair 的精确 blocker

第二组虽然在 source 中也是连续 store，但 LLVM 没有保留完整 sibling provenance：

```text
packet2:
  %206 = gep %202, %199

packet3:
  %207 = gep @shared, %.idx11
  %208 = gep %207, %200
```

数值上，`%206` 和 `%208` 仍然访问冻结地址 `pbase ^ 8` 与
`(pbase ^ 8) | 2048`；但 LLVM 表达中：

- packet2 从 `%202` 继承 base，再加 `%199`；
- packet3 重新从 shared global 和另一个 `%.idx11` 形成 base，再加 `%200`；
- 没有一个直接可见的 `(%206 base) + 2048` sibling GEP；
- `pbase ^ 8` 和 `pbase ^ 8 | 2048` 的关系被 memref/view index lowering
  拆成了不同的 pointer provenance。

exact-LTO MIR 对应为：

```text
kernel_section_08.mir:195  DS_WRITE2ST64_B64  %ir.201, %ir.202
kernel_section_08.mir:197  DS_WRITE_B64       %ir.203
kernel_section_08.mir:198  DS_WRITE_B64       %ir.205, offset 8192
```

也就是 H producer 的四个 logical 64-bit stores 变成了“一组 pair 加两条独立
store”，而不是两组 pair。这个 blocker 的类别是：

```text
pointer provenance / address expression shape
```

不是：

- store type：两边都是 64-bit packed store；
- alignment：两边都是 `align 8`；
- instruction adjacency：四条 LLVM store 已经相邻，只有 GEP 在 store 之间；
- H address formula：v8/v9 correctness 和 offset audit 均通过；
- MFMA intrinsic：MFMA32 没有变化。

因此不能把“第二组仍是 `DS_WRITE_B64`”解释成后端完全不支持 pair，也不能用
新 intrinsic 掩盖问题。当前 generic path 已经证明第一组可以 pair；未收敛的是
第二组地址在 IR 中的共同 base/provenance。

### 17.5 v8/v9 machine comparison

| 指标 | v8 | v9 | 说明 |
|:--|--:|--:|:--|
| H producer pair | 0 | 1 `DS_WRITE2ST64_B64` | 第一组 pair 已形成 |
| `kernel_section_08.mir` `DS_WRITE_B64` | 8 | 6 | total section count |
| final ISA `ds_write` family | 40 | 39 | static lexical total |
| `DS_READ2ST64_B64` | 4 | 4 | consumer unchanged in family/count |
| `DS_READ2_B64` | 1 | 1 | consumer unchanged |
| `DS_READ_B64` | 4 | 4 | consumer unchanged |
| MFMA32 | 8 | 8 | exact opcode/count unchanged |
| `v_perm_b32` | 4 | 4 | unchanged |
| barrier / waitcnt | 2 / 18 | 2 / 18 | unchanged |
| VGPR / AGPR | 72 / 32 | 72 / 32 | unchanged |
| SGPR | 14 | 18 | producer grouping增加了4个 SGPR，但没有 spill |
| LDS | 12288 B | 12288 B | unchanged |
| private / VGPR spill / SGPR spill | 0 / 0 / 0 | 0 / 0 / 0 | unchanged |

v9 的 H consumer/read 和 MFMA 没有被 source 改动；机器证据中的 read family、
MFMA、barrier 和 waitcnt 均保持不变。v9 只改变了 producer store formation，且
增加了少量 SGPR 使用，没有引入 private memory 或 spill。

### 17.6 Final decision：Case B

本实验得到的是一个有价值但未完全闭合的结果：

```text
Case B：partial DS_WRITE2ST64 formation
```

已经证明：

1. 不改 H 地址公式，只重排 source producer IR，就能让第一组 sibling store
   形成 `DS_WRITE2ST64_B64`；
2. H correctness 仍为 4/4、`max_abs=0`；
3. consumer/read、MFMA32、barrier/waitcnt、LDS 和 spill 没有回归；
4. 当前 generic LLVM/AMDGPU path 并非完全缺少 DS pair 能力。

还没有证明：

1. 第二组能否在不改变地址公式的前提下形成 pair；
2. 这一次静态指令减少能否带来性能收益；本轮明确没有 benchmark。

本轮停止。第二组的下一控制点如果继续研究，只能是保持同样地址值，把
`packet2/packet3` 的 shared pointer provenance 在 LLVM 层统一成一个共同 base；
在此之前不应改 consumer、加 intrinsic、调 scheduler 或修改 waitcnt/barrier。

## 18. v10 packet2/packet3 shared-base closure

### 18.1 唯一改动

本轮只处理上一节 Case B 的 pointer provenance blocker。没有改变 H packet 的数值
地址，也没有改变 packet payload、consumer、MFMA、Q、Q@K、score@V 或调度。

source 中将 packet3 的物理 view 索引改写为与 packet2 等价的相对表达：

```python
physical_row3 = physical_row2 + 32
physical_group3 = physical_group2
```

因为一个 shared row 是 64 bytes，所以这正好表示：

```text
addr3 = addr2 + 32 * 64
      = addr2 + 2048 bytes
```

这是冻结地址公式的代数重写，不是地址变化。packet2/packet3 仍然分别对应：

```text
addr2 = pbase ^ 8
addr3 = (pbase ^ 8) | 2048
```

### 18.2 LLVM 证据

v10 `postopt_llvm.ll` 的 H producer 现在是：

```llvm
%199 = gep @shared, %.idx       ; pair A base
%200 = gep %199, %146
store <2 x i32> %132, %200, align 8

%201 = gep %199, 2048
%202 = gep %201, %146
store <2 x i32> %168, %202, align 8

%203 = gep @shared, %.idx9      ; pair B base = shared + (pbase ^ 8)
%204 = gep %203, %197
store <2 x i32> %196, %204, align 8

%205 = gep %203, 2048
%206 = gep %205, %197
store <2 x i32> %194, %206, align 8
```

关键变化是 packet2 和 packet3 都从 `%203` 派生，packet3 明确经过
`%205 = gep %203, 2048`。因此 LLVM 已经保留了目标形式：

```text
pair_base = shared + (pbase ^ 8)
addr2 = pair_base + x
addr3 = pair_base + 2048 + x
```

四条 store 都仍然是 64-bit `<2 x i32>`、`align 8`，且没有改变 payload。

### 18.3 MIR/ISA 结果

v10 exact-LTO MIR 的 H producer 已形成两组 pair：

```text
kernel_section_08.mir:192
DS_WRITE2ST64_B64 ... (store %ir.198), (store %ir.199)

kernel_section_08.mir:194
DS_WRITE2ST64_B64 ... (store %ir.201), (store %ir.202)
```

其中第二条就是 packet2/packet3 的 pair。v10 final ISA 也保留了两条
`ds_write2st64_b64`。这满足本实验的目标，不需要新增 LDS intrinsic。

### 18.4 Correctness、consumer 和资源

v10 工件：

```text
test/examples/linear_attention/compile_bug/
qwen_t8192_native_shaped_qh_wg128_packed_h_v10/
```

四个既有 Q@H oracle 均通过：

| source pattern | `b_high` | `max_abs` |
|:--|--:|--:|
| `row_code` | 1 | 0.0 |
| `row_code` | 2 | 0.0 |
| `row_half` | 1 | 0.0 |
| `row_half` | 2 | 0.0 |

`all_max_abs_zero=true`，finite 和 BF16 debug readback 均通过。

| metric | v8 | v9 | v10 |
|:--|--:|--:|--:|
| H producer `DS_WRITE2ST64_B64` | 0 | 1 | 2 |
| final ISA `ds_write` family | 40 | 39 | 38 |
| `DS_READ2ST64_B64` | 4 | 4 | 4 |
| `DS_READ2_B64` | 1 | 1 | 1 |
| `DS_READ_B64` | 4 | 4 | 4 |
| MFMA32 | 8 | 8 | 8 |
| `v_perm_b32` | 4 | 4 | 4 |
| barrier / waitcnt | 2 / 18 | 2 / 18 | 2 / 18 |
| VGPR / AGPR | 72 / 32 | 72 / 32 | 72 / 32 |
| SGPR | 14 | 18 | 18 |
| LDS | 12288 B | 12288 B | 12288 B |
| private / spill | 0 / 0 | 0 / 0 | 0 / 0 |

read/MFMA 的指令族和数量没有变化；本轮只闭合 producer store 的 pair formation。
SGPR 相比 v8 增加 4，但没有 private memory 或 spill。所有数字都是 static
machine evidence，不是动态性能计数。

### 18.5 Final decision

本轮达到 Case A：

```text
packet2 和 packet3 在 LLVM 共享同一个 base，
packet3 = pair_base + 2048，
MIR/ISA 形成第二条 DS_WRITE2ST64_B64。
```

因此，上一轮的 pointer provenance blocker 已经通过最小 source-level address
expression rewrite 消除。没有修改 H 地址值，也没有引入新的 hardware intrinsic。
本轮仍然不运行 benchmark；性能意义需要另一个独立实验确认。

v10 hashes：

```text
HSACO:        c4c81a2c2a581a18a0a2318279a71732cb5b672bb3036df4cba34bbd9ba06949
final ISA:    696b85b1ded6cf212f9c9913dfd88c3c5096b15c315014deda74a9f65f196cc0
postopt LLVM: 4c78f452f654b3459052f422c3e7d0468d6ba8e5e6038b15050827d936decb90
exact MIR:    549d568fc13e445b3f0ac168d3b5ee05fed12d989c1b26773f877c4cd9fcd638
```

## 19. v10 H-consumer read-packing closure audit

本轮只审计 H operand 的 `DS_READ_B64` 归属和 sibling grouping。没有修改 H
producer、H 地址公式、Q、Q@K、score@V、scheduler、barrier/waitcnt，也没有运行
benchmark。审计直接复用 v10 的 source、post-opt LLVM、exact-LTO MIR、final ISA、
HSACO 和 4/4 correctness 工件：

```text
test/examples/linear_attention/compile_bug/
qwen_t8192_native_shaped_qh_wg128_packed_h_v10/
```

### 19.1 先按 shared allocation 归属读取指令

不能按 MFMA 前后的一段 ISA 文本把所有 `DS_READ_B64` 都算成 H。v10 的 LLVM
global shared symbol 给出了可靠归属：

| shared symbol | 逻辑 operand | producer 证据 |
|:--|:--|:--|
| `...packed_h_kernel_0` | H | `postopt_llvm.ll:254-266` 的四个 H packet store，以及 `:416-438` 的 H consumer load |
| `...packed_h_kernel_1` | Q | `postopt_llvm.ll:282-305` 的 Q global load/store，以及 `:412-435` 的 Q consumer load |

所有 H consumer load 都是对齐的 `<4 x i16>`，`align 8`。因此本轮把 LLVM pointer
所属的 shared allocation 作为 H/Q 的分类依据，而不是把 `DS_READ_B64` 的位置
作为分类依据。

### 19.2 v10 的四个 H packet 已全部形成 paired read

H 的四个 logical packet 是 `hBase ^ {0,16,32,48}`，每个 packet 还要读取同一
packet 的 `+2048` byte sibling。v10 的 LLVM/MIR/ISA 对应关系如下：

| H packet | LLVM H loads | 有效 byte offset | exact-LTO MIR | MFMA consumer |
|:--|:--|:--|:--|:--|
| 0 | `%334`, `%358` | `hBase ^ 0`, `(hBase ^ 0)+2048` | `DS_READ2ST64_B64`, `kernel_section_08.mir:226` | `v[4:5]`，MIR `:235` |
| 1 | `%341`, `%361` | `hBase ^ 16`, `(hBase ^ 16)+2048` | `DS_READ2ST64_B64`, `:237` | `v[8:9]`，MIR `:238` |
| 2 | `%348`, `%364` | `hBase ^ 32`, `(hBase ^ 32)+2048` | `DS_READ2ST64_B64`, `:240` | `v[12:13]`，MIR `:241` |
| 3 | `%355`, `%367` | `hBase ^ 48`, `(hBase ^ 48)+2048` | `DS_READ2ST64_B64`, `:243` | `v[16:17]`，MIR `:250` |

第一组的 `+2048` 在 LLVM 中直接表现为：

```llvm
%358 = getelementptr i8, ptr addrspace(3) %334, i32 2048
%359 = load <4 x i16>, ptr addrspace(3) %358, align 8
```

其余三组使用等价的 BF16-element index 形式：`%361` 相对 `%341`、`%364`
相对 `%348`、`%367` 相对 `%355` 都增加 1024 个 BF16 element，即 2048 bytes。
LLVM 因此保留了同一 packet base 与固定 sibling offset；后端最终把每组降成
一条 `DS_READ2ST64_B64`，MIR 的四条 memory operand 也明确列出了对应的两个
load。

v10 exact MIR 的完整 H read 序列是：

```text
226  DS_READ2ST64_B64  %ir.334, %ir.358  -> v[4:7]
237  DS_READ2ST64_B64  %ir.341, %ir.361  -> v[8:11]
240  DS_READ2ST64_B64  %ir.348, %ir.364  -> v[12:15]
243  DS_READ2ST64_B64  %ir.355, %ir.367  -> v[16:19]
```

这四组分别供 H MFMA source packet 的 `v[4:5]`、`v[8:9]`、`v[12:13]`、
`v[16:17]` 使用；对应的 Q packet 则由另一 shared allocation 的单独读取供给
另一 MFMA source。

### 19.3 剩余四条 `DS_READ_B64` 实际属于 Q，不是 H

v10 MIR 中仍然可见的四条单读是：

```text
228  DS_READ_B64  %ir.330  -> v[20:21]
232  DS_READ_B64  %ir.338  -> v[22:23]
233  DS_READ_B64  %ir.345  -> v[24:25]
234  DS_READ_B64  %ir.352  -> v[26:27]
```

它们在 LLVM 中分别来自 `...packed_h_kernel_1`（`postopt_llvm.ll:412-435`），
不是 H 的 `...packed_h_kernel_0`。它们对应 Q@H 的另一输入 operand，且 H/Q 的
源寄存器关系在 ISA 中保持为：

```text
H paired read -> v[4:5],  Q single read -> v[20:21] -> MFMA
H paired read -> v[8:9],  Q single read -> v[22:23] -> MFMA
H paired read -> v[12:13], Q single read -> v[24:25] -> MFMA
H paired read -> v[16:17], Q single read -> v[26:27] -> MFMA
```

所以 v10 聚合窗口的统计仍然是 `DS_READ2ST64_B64=4`、`DS_READ_B64=4`，但
**H-specific 统计是 `DS_READ2ST64_B64=4`、`DS_READ_B64=0`**。此前把窗口中
剩余的四条单读概括为“H read path mixed”是不精确的；按 shared pointer provenance
重新分类后，该表述由本节覆盖。

`DS_READ2_B64 %ir.215,%ir.233`（MIR `:245`）不属于上述四个 H MFMA packet
链，不能被拿来替代或重复计入 H consumer read。

### 19.4 与冻结 selected Triton H operand 的对照

selected Triton T=8192 的 LLVM 同样显式生成四组 H packet base 和固定的
`+2048` sibling：

```text
chunk_fwd_kernel_o.llir:144-154
  %130/%131, %133/%134, %136/%137, %139/%140
chunk_fwd_kernel_o.llir:179-186
  八个对齐的 <4 x i16> H loads
```

其 final ISA 的 H operand read 使用 paired LDS reads，例如：

```text
chunk_fwd_kernel_o.amdgcn:202-220
  ds_read2st64_b64 ... offset1:4
  ds_read2st64_b64 ... offset1:4
  ds_read2st64_b64 ... offset1:4
```

selected Triton 的其他 `DS_READ_B64` 出现在不同 operand/phase，不能因为它们在
同一 kernel 中出现，就要求 H 也消费这些单读。就 H packet 的有效地址关系和
paired read family 而言，v10 已达到本轮要求：

| H read property | v10 AveLang | selected Triton |
|:--|:--|:--|
| packed element type | `<4 x i16>` / BF16x4 | `<4 x i16>` / BF16x4 |
| packet sibling distance | 2048 bytes | 2048 bytes |
| H paired read family | `DS_READ2ST64_B64` | `DS_READ2ST64_B64` |
| H single `DS_READ_B64` in MFMA chain | 0 | 0 for the compared paired packets |
| H MFMA opcode | `v_mfma_f32_32x32x8_bf16` | same |

### 19.5 Correctness、producer 和资源冻结复核

本轮没有改代码，因此复用 v10 已完成的 correctness：

| gate | result |
|:--|:--|
| Q@H cases | 4/4 |
| `max_abs` | 0.0 for all cases |
| BF16 debug readback | exact |
| H producer pairs | 2 `DS_WRITE2ST64_B64` |
| H consumer pairs | 4 `DS_READ2ST64_B64` |
| H consumer singles | 0 |
| MFMA32 | 8 static instructions |
| private segment | 0 B |
| VGPR / AGPR | 72 / 32 |
| spill | 0 |

v10 identity remains unchanged：

```text
HSACO:        c4c81a2c2a581a18a0a2318279a71732cb5b672bb3036df4cba34bbd9ba06949
final ISA:    696b85b1ded6cf212f9c9913dfd88c3c5096b15c315014deda74a9f65f196cc0
postopt LLVM: 4c78f452f654b3459052f422c3e7d0468d6ba8e5e6038b15050827d936decb90
exact MIR:    549d568fc13e445b3f0ac168d3b5ee05fed12d989c1b26773f877c4cd9fcd638
```

### 19.6 Final decision

**Case A：H consumer read-packing 已经闭合。** 本轮没有 remaining H-related
`DS_READ_B64` 可以配对；如果继续改动剩余四条单读，就会进入 Q operand，而这被本
轮明确冻结。因而没有 source/compiler 修改，也没有新增 v11 binary。

如果未来要继续减少聚合窗口中的 `DS_READ_B64`，下一实验必须明确命名为 Q
consumer read-packing 实验，不能再写成 H read closure。当前 H 路径应保持 v10
不变。

## 20. Q/A operand exact physical-contract audit（v10，read-only）

本节对应 Q@H 中剩余四条 `DS_READ_B64` 的专门审计。范围严格限制为 Q/A
operand；H producer/consumer、Q@K、score@V、Z5B、scheduler、barrier/waitcnt 和
benchmark 均未修改，也没有重复运行 correctness gate。

### 20.1 冻结工件与证据等级

AveLang v10：

```text
test/examples/linear_attention/compile_bug/
  qwen_t8192_native_shaped_qh_wg128_packed_h_v10/
```

主要证据：

```text
compiler_ir/kfrag/post_block_dot_lowering.mlir
compiler_ir/kfrag/postopt_llvm.ll
exact_lto/kernel_section_08.mir
final_isa.s
code_object_notes.txt
```

selected Triton/native T=8192：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
  codex_qwen_bt64_stage6z_native_chunko/native/T8192/selected/
```

主要证据：

```text
chunk_fwd_kernel_o.ttgir
chunk_fwd_kernel_o.llir
chunk_fwd_kernel_o.amdgcn
chunk_fwd_kernel_o.hsaco
```

本节把 TTGIR/LLIR 中明确的 tensor encoding、LLVM 的 shared pointer 和 final ISA
memory instruction 作为 A 级证据；把 source helper 对 packet ownership 的描述作为
B 级辅助证据。最终判断不依赖 ISA 窗口位置，而依赖 LLVM/MIR 的 pointer provenance
和 memory operand。

### 20.2 先给结论

Q/A 的唯一第一处分歧是：**producer ownership**。

在第一个 global packet 到 LDS producer 的阶段，两个实现已经把一个 lane 的 Q
元素分给了不同的 logical row：

```text
native：  一个 lane 负责 row r 和 row r + 32
AveLang： 一个 lane 负责 row r 和 row r + 1
```

因此后续的物理地址也不同：native 的两个 row-band packet 使用 `+2048` bytes
的 sibling 关系，AveLang v10 的两个 token-row packet 在 final ISA 中使用 `+64`
bytes 的 row stride。四条 AveLang Q `DS_READ_B64` 是这条 producer ownership/layout
差异的结果，不能先把它们单独归因成一个晚期 read-packing 失败。

这不是 MFMA intrinsic 缺失，也不是本轮可以直接提出的 scheduler 修复。本轮按
要求停止在审计，不登记代码修改。

### 20.3 Native Q producer：lane、global packet 和 LDS store

selected Triton LLIR 对 Q producer 的关键计算是：

```llvm
%34 = lshr i32 %tid, 2
%35 = and i32 %34, 31
%36 = or disjoint i32 %35, 32
...
%41 = shl i32 %tid, 3
%42 = and i32 %41, 24
...
%68 = raw.ptr.buffer.load.v4i32(... %63 ...)
%72 = raw.ptr.buffer.load.v4i32(... %64 ...)
```

也就是相对于当前 Q tile 的局部坐标：

```text
r0 = (tid >> 2) & 31
r1 = r0 + 32
c0 = (tid << 3) & 24
```

两条 `v4i32` buffer load 每条是 16 bytes，即 8 个 BF16；它们分别覆盖
`Q[r0, c0:c0+8]` 和 `Q[r1, c0:c0+8]`。这是 final ISA 的：

```text
chunk_fwd_kernel_o.amdgcn:118-119
buffer_load_dwordx4 v[4:7],  ...
buffer_load_dwordx4 v[8:11], ...
```

native producer 的 packet base 是：

```text
pbase = ((tid << 4) & 2032) ^ (tid & 56)
```

一个 lane 的四个 8-byte BF16x4 store packet 为：

```text
packet 0: pbase
packet 1: pbase ^ 8
packet 2: pbase + 2048
packet 3: (pbase ^ 8) + 2048
```

final ISA 的 Q producer 是两条 `DS_WRITE2ST64_B64`：

```text
chunk_fwd_kernel_o.amdgcn:162
  ds_write2st64_b64 v42, v[4:5], v[8:9] offset1:4
chunk_fwd_kernel_o.amdgcn:168
  ds_write2st64_b64 v43, v[6:7], v[10:11] offset1:4
```

这里每一条 pair 都把两个 row-band 的同一 packet group 作为 sibling；LLVM 中对应
的是 `%101` base、`%104 = %101 | 2048`、`%107 = %101 ^ 8` 和
`%110 = %107 | 2048`（`chunk_fwd_kernel_o.llir:103-121`）。因此 `+2048` 是
producer layout 的真实 byte relation，不是从 ISA 文本位置猜出的关系。

### 20.4 AveLang v10 Q producer：source/MLIR/LLVM/ISA

v10 的 `post_block_dot_lowering.mlir:776-840` 明确给出 Q/A producer 的逻辑坐标：

```text
wave      = tid >> 6
lane      = tid & 63
laneLow   = lane & 3
laneHigh  = lane >> 2
rowBase   = (wave << 5) + (laneHigh << 1)
colBase   = laneLow << 3
```

四个 BF16x4 packet 是：

```text
packet 0: Q[rowBase,     colBase + 0]
packet 1: Q[rowBase,     colBase + 4]
packet 2: Q[rowBase + 1, colBase + 0]
packet 3: Q[rowBase + 1, colBase + 4]
```

对应的 source helper 是
`lower_qwen_block_dot_pass.cc:666-697`，producer 在
`lower_qwen_block_dot_pass.cc:866-917` 生成四个 `vector<4xbf16>` load/store。这里
的 `source` 是 row-major `[64,32]` BF16 tile，Q/A stage 的 plan-driven shared
element offset 为：

```text
phase        = (row >> 1) & 7
physicalGroup = (col >> 2) xor phase
byteOffset   = 2 * (row * 32 + physicalGroup * 4 + (col & 3))
```

这条公式来自 `emitC16NativeSharedElementOffset`，不是把 physical 地址表硬编码进
报告。因而一个 packet 的 base byte offset 是由它的 logical row/column 决定的。

v10 final ISA 的 Q global producer 是两条 16-byte load：

```text
final_isa.s:202-203
global_load_dwordx4 v[4:7],  ...
global_load_dwordx4 v[14:17], ... offset:64
```

但它们对应的 Q LDS producer 是四条独立 8-byte store：

```text
final_isa.s:214   ds_write_b64 v32, v[4:5]
final_isa.s:219   ds_write_b64 v33, v[6:7]
final_isa.s:221   ds_write_b64 v32, v[14:15] offset:64
final_isa.s:222   ds_write_b64 v33, v[16:17] offset:64
```

所以 final machine 的 global packet 宽度已经和 native 同为 16 bytes；差异不在
“AveLang 只能发 BF16 标量 global load”。差异是这些 16-byte 输入在 lane 内被拆成
哪两个 logical row，以及随后被写入哪个 shared physical row band。

### 20.5 代表 lane/packet 对照

下表只列 packet base；每个 base 对应连续 4 个 BF16，即 8 bytes。AveLang 的
`A0/A1/A2/A3` 是按 post-block MLIR 的 row-major producer packet 顺序；native 的
`N0/N1/N2/N3` 是按 native `pbase` producer 顺序。

| tid | AveLang logical rows / col | AveLang LDS packet bytes | native logical rows / col | native LDS packet bytes |
|---:|---|---|---|---|
| 0 | rows 0,1; col 0,4 | A0=0, A1=8, A2=64, A3=72 | rows 0,32; col 0 | N0=0, N1=8, N2=2048, N3=2056 |
| 1 | rows 0,1; col 8,12 | A0=16, A1=24, A2=80, A3=88 | rows 0,32; col 8 | N0=16, N1=24, N2=2064, N3=2072 |
| 4 | rows 2,3; col 0,4 | A0=136, A1=128, A2=200, A3=192 | rows 1,33; col 0 | N0=64, N1=72, N2=2112, N3=2120 |
| 16 | rows 8,9; col 0,4 | A0=544, A1=552, A2=608, A3=616 | rows 4,36; col 0 | N0=272, N1=280, N2=2320, N3=2328 |
| 32 | rows 16,17; col 0,4 | A0=1024, A1=1032, A2=1088, A3=1096 | rows 8,40; col 0 | N0=544, N1=552, N2=2592, N3=2600 |
| 63 | rows 30,31; col 24,28 | A0=1928, A1=1920, A2=1992, A3=1984 | rows 15,47; col 24 | N0=968, N1=976, N2=3016, N3=3024 |

两个重要点：

1. 两边都覆盖完整的逻辑 Q tile；差异是 lane ownership，不是缺失 Q 元素。
2. AveLang 的 A2/A3 与 A0/A1 相隔一个 row stride，即 `64` bytes；native 的
   N2/N3 与 N0/N1 相隔一个 32-row band，即 `2048` bytes。这个差异在 producer
   阶段已经存在，不能归咎于 consumer read instruction selection。

### 20.6 Q consumer read 与 MFMA packet provenance

AveLang v10 LLVM 直接给出了 Q/A consumer 的四个 Q load：

```llvm
%330 = gep bfloat addrspace(3) @..._kernel_1, i32 %329
%331 = load <4 x i16>, ptr %330, align 8
%338 = gep ... %329 xor 8
%339 = load <4 x i16>, ptr %338, align 8
%345 = gep ... %329 xor 16
%346 = load <4 x i16>, ptr %345, align 8
%352 = gep ... %329 xor 24
%353 = load <4 x i16>, ptr %352, align 8
```

证据位置为 `postopt_llvm.ll:410-439`。四个 `%331/%339/%346/%353` 被直接送入
四个 MFMA intrinsic 的 Q/A 输入；exact-LTO MIR 对应：

```text
228  DS_READ_B64  %ir.330 -> v[20:21]
232  DS_READ_B64  %ir.338 -> v[22:23]
233  DS_READ_B64  %ir.345 -> v[24:25]
234  DS_READ_B64  %ir.352 -> v[26:27]
```

这四个地址是同一个 Q shared allocation 上的 `qBase xor {0,8,16,24}`，而不是
native 的 `qBase` 与 `qBase + 2048` sibling pair。AveLang 因此形成四条独立
`DS_READ_B64`；v10 的 H operand 则保持已经闭合的四条 `DS_READ2ST64_B64`。

native selected 的 TTGIR 对 Q 的 contract 是：

```text
%b_q_69 = ttg.local_alloc !ttg.memdesc<1x64x32xbf16, #shared, #smem>
%b_q_275 = ttg.local_load ... -> #ttg.dot_op<opIdx=0, kWidth=4>
%b_o_295 = tt.dot %b_q_275, %b_o_294, ...
```

对应 `chunk_fwd_kernel_o.llir:144-154,179-186` 的 Q loads 使用
`qBase`, `qBase+2048`, `qBase^16`, `qBase^16+2048`, ... 这一类 packet base；
final ISA 在首个 Q@H window 中用 `DS_READ2_B64`（例如
`chunk_fwd_kernel_o.amdgcn:202` 和 `:215`）把这些 sibling packet 送入
`v[12:15]`、`v[24:27]`，再由 MFMA 的 A/src operand 使用。后续 loop window 重复
同一类 contract，不能把 native full kernel 的全部静态读数当成一个 Q@H isolated
计数。

### 20.7 Side-by-side ledger

| 项目 | AveLang v10 Q/A | selected Triton Q/A | parity |
|---|---|---|---|
| logical Q coverage | `[64,32]`，完整 | `[64,32]`，完整 | PASS |
| final global packet | 2 x `global_load_dwordx4`，16 B | 2 x `buffer_load_dwordx4`，16 B | PASS（final width） |
| source ownership | lane 内相邻 rows `r,r+1` | lane 内 row-band `r,r+32` | **FAIL** |
| Q LDS packet store | 4 x `DS_WRITE_B64` | 2 x `DS_WRITE2ST64_B64` | consequence of ownership/layout |
| producer sibling distance | `+64` bytes | `+2048` bytes | FAIL |
| Q LDS consumer load | 4 x `DS_READ_B64` | paired `DS_READ2_B64` for compared packets | FAIL |
| MFMA opcode | `v_mfma_f32_32x32x8_bf16` | same | PASS |
| Q fragment source | v10 `v[20:21]...v[26:27]` | native packed groups such as `v[12:15]`/`v[24:27]` | not register-identity parity |
| Q/A pointer provenance | ordinary shared GEP + `<4xi16>` load | `#shared` + `#ttg.dot_op` local load | contract differs |

### 20.8 Resource and synchronization evidence

这些字段是 code-object/final artifact 的静态资源，不是性能计数器：

| 资源 | AveLang v10 Q@H probe | selected Triton T=8192 full kernel |
|---|---:|---:|
| VGPR | 72 | 220 |
| AGPR | 32 | 64 |
| SGPR | 18 | 76 |
| LDS | 12288 B | 12288 B |
| private segment | 0 B | 0 B |
| VGPR/SGPR spill | 0 / 0 | 0 / 0 |
| workgroup | 128 | 128 |

资源表的 scope 不同：左侧是隔离 Q@H probe，右侧是 selected native full
`chunk_fwd_kernel_o`，所以不能用它推出性能优劣。它只确认本轮 Q producer ownership
差异没有伴随 v10 spill/private failure。

### 20.9 唯一第一处分歧与停止条件

最终分类只选择一项：

```text
FIRST Q/A DIVERGENCE = producer ownership
```

理由是它在链条中的位置最早：

```text
logical Q element ownership
 -> global packet row assignment
 -> LDS physical row-band (+64 vs +2048)
 -> store grouping
 -> read grouping
 -> MFMA packet registers
```

两边的 MFMA opcode、BF16/FP32 contract、最终 global packet width 和完整逻辑 Q 覆盖
都已经有证据；但在 producer ownership 处已不相同。因此本轮不把 `store packing` 或
`read packing` 登记为“第一处分歧”，它们是 ownership/layout 差异的后续机器表现。

本轮没有提出代码改动，没有 benchmark，也没有改变 H 路径。下一次若要继续，必须
先建立一个只改变 Q producer ownership 的实验，并保持 H 完全冻结；在该实验之前，
不能把四条 `DS_READ_B64` 单独重写成 paired read 来宣称达成 native parity。
