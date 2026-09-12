# Qwen gfx942 Stage 6Z：通用 `block_dot_bf16_f32` V2 的 MFMA-B same-source A/B

## 结论摘要

本轮完成了一个 experimental-only 的 compiler/high-level IR 实验：在冻结的
Stage 6Z Z5B direct-Q-cache-consumer chunk-o 上，使用同一个高层
`al.amdgpu.block_dot_bf16_f32` 语义表达 `Q @ H.T` 和 `Q @ K.T`，再通过
`AVELANG_BLOCK_DOT_LOWERING=generic|specialized` 选择两条 lowering。

两条 arm 共享同一份 Z7B kernel source、同一套 Q cache、phase-separated
accumulator 顺序、BT64/BV64/BK32、WG256、MFMA32 数学和 BF16 ABI。generic arm
在 lowered LLVM 中使用 scalar BF16 B operand materialization；specialized arm
在 lowered LLVM 中确实出现了更多 `<8 x bfloat>` typed load，证明分叉不是只改了
命令行名字。

但是，specialized 的 typed B-operand 语义没有保留到最终机器图：两条 arm 的
exact-LTO code object HSACO SHA256 完全相同，最终 ISA 的指令主体也相同，dynamic
MFMA/VMEM/LDS/VALU/SALU 完全相同。specialized 没有降低 VMEM、没有降低 LDS
materialization、没有降低 VALU，也没有带来延迟收益；它在两个正式长度上都略慢于
Z5B。

因此本轮的正式判断是：

```text
correctness                  PASS
通用 block-dot source contract PASS
K/H 共用 MFMA-B lowering 接口   PASS（到 lowered LLVM）
typed B operand 保留到 ISA      FAIL
specialized 的机器工作减少       FAIL
specialized 的 latency 收益       FAIL
Z7B 晋级                        NO-GO
当前 isolated baseline           继续保持 Z5B
```

这不是“AveLang 完全不能表达 block dot”的证据。它更精确地证明：当前新增的
specialized B-operand 表示虽然能够改变 AveLang lowering 产出的 LLVM，但在后续
AMDGPU/LTO codegen 中仍被规范化成与 generic arm 相同的机器图。下一步不应直接
扩展 Q 或 V-new；应先定位 LLVM/AMDGPU/LTO 中 generic 和 specialized 第一次收敛
的具体 pass，或者建立能够真正保留到 machine lowering 的 target-specific typed
operand 表示。

---

## 1. 实验问题和边界

### 1.1 要回答的问题

此前 Z5B 已经证明 dedicated Q cache 和 direct Q-cache consumer 可以把 Q 的
global producer pass 从 3 次降到 1 次，并显著删除 Q republish 的 LDS 工作。但
Z5B 相对 native vLLM 仍有明显的 VMEM/VALU/LDS 差距。前面的 BF16 operand-feeding
audit 指出，AveLang 的 K/H feeding 仍容易退化为：

```text
BF16 producer
  -> generic shared buffer
  -> i32 view
  -> extract/insert
  -> MFMA operand
```

native Triton 则保留了更完整的 blocked/shared/dot-operand 语义。本轮只研究
MFMA 的 B operand，也就是：

```text
K -> score = Q @ K.T
H -> inter = Q @ H.T
```

不同时重写 Q、V-new、g、output 或 accumulator schedule。这样即使实验失败，
也可以把结果归因到 K/H B-operand lowering，而不是把多个结构变化混在一起。

### 1.2 冻结的 Z5B contract

| 项目 | 冻结值 |
|:--|:--|
| baseline | Stage 6Z Z5B direct-Q-cache-consumer |
| target | gfx942 / wave64 |
| tile | BT64 / BV64 / BK32 |
| launch | WG256，2 CTA/chunk-head |
| dtype | Q/K/H/V-new/out=BF16，g=FP32，accumulator=FP32 |
| Q | dedicated full LDS cache，global producer 只做一次 |
| accumulator | `inter_acc -> score_half0 -> score_half1` phase-separated |
| math | causal mask、MFMA32、K32 reduction order、输出数学不变 |
| output | caller-owned BF16 output，zero-V/NaN-prefill contract 不变 |
| 禁止修改 | production、X2、recurrence HSACO、allocator/RA、WG、MFMA geometry、Q/V-new/g/output |

本轮没有设置静态资源 No-Go。VGPR、AGPR、LDS、occupancy、barrier、scratch 和
spill 只作记录；arm 只要 correctness-safe 且能运行，就进入实际 latency 测试。
最终是否晋级只由 correctness 和 fresh-process body latency 决定。

---

## 2. 旧 `block_dot` V1 能否复用

### 2.1 可以复用的部分

现有 `al.amdgpu.block_dot_bf16_f32` 已经提供了几个正确的基础契约：

- BF16 输入块和 FP32 accumulator 的结果语义；
- MFMA32 的数学结果类型；
- K32 reduction 的调用边界；
- 现有 verifier 对 workgroup BF16 staging 和 accumulator 形状的检查；
- 旧 Qwen/direct-K64 路径已经验证过的 MFMA32 lowering 基础设施。

因此本轮没有创建新的 `qwen_chunk_o_dot` 一次性专用 op，也没有把整个 chunk-o
schedule 藏进 intrinsic。source 仍然显式创建 Q/H/K block、显式控制 loop、phase
和 accumulator。

### 2.2 不能直接照搬的部分

旧 lowering 中仍有 Qwen-specific 的指针、phase 和 buffer 约束，不能把它们当成
通用 block-dot contract。为此本轮做了最小泛化：

- `AveLangOps.td`：把 block-dot 的描述和属性契约从 Qwen-specific 语义扩展为
  通用 logical block-dot 语义；
- `AveLangOps.cc`：为 operand-mode 增加 rank-2 workgroup BF16 block、result
  vector32xf32、integer anchor 和 accumulator verifier；同时保持旧用户兼容；
- `amdgpu_module.cc`：增加同一 operation 的 generic source-facing helper：
  `block_dot_bf16_f32_operand` 和
  `block_dot_bf16_f32_operand_transposed`；
- `lower_qwen_block_dot_pass.cc`：增加 operand-role/source-role/transpose
  metadata 读取和通用 B-operand lowering。K/H 不复制两套 Qwen 特例，而是进入
  同一个 `lowerMfmaBOperand` 等价的内部 abstraction，H 只通过 transpose/role
  metadata 表达逻辑转置。

旧路径仍保留，因此旧 recurrence correctness 不依赖本轮实验。

---

## 3. 高层 source 的实际变化

新 source：

[`qwen_gdn_bt64_native_chunko_stage6z_z7b_dot_v2.py`](../vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z7b_dot_v2.py)

它从 Z5B 最终 source 分叉，仍保留：

1. Q full-cache fill；
2. Phase A 的 H staging 和 `inter_acc`；
3. Phase B 的两个 score half；
4. V-new/intra/output 逻辑；
5. 旧的 launch grid、shape guard、caller-owned output。

唯一新的高层计算表达是两个 block-dot 调用：

```text
inter = block_dot_bf16_f32_operand_transposed(Q_cache, H_phase, ...)
score = block_dot_bf16_f32_operand(Q_cache, K_phase, ...)
```

`transposed` 只记录 H 的逻辑 transpose 语义；它不是第二个 kernel，也不是新的
Qwen-specific primitive。source 不出现 gfx942 lane ID、LDS byte offset、
`ds_write`、MFMA register 或硬件寄存器编号。

### 3.1 两臂的分叉位置

两臂调用完全相同的 source function：

```text
Z7BDOT-G: AVELANG_BLOCK_DOT_LOWERING=generic
Z7BDOT-B: AVELANG_BLOCK_DOT_LOWERING=specialized
```

source file SHA256：

```text
0a73d0ca01d97d4a49b9d7eed118d7bcaa775b1ac9ab5dbd7f80dadde605873c
```

两臂的 source function、shape、ownership、loop 和 operand metadata 相同；唯一
控制变量是 compiler lowering selector。高层 source identity 因此成立。

### 3.2 pre-branch MLIR 的限制

本轮尝试用当前 Docker binding 的 `get_mlir()` 保存 pre-branch MLIR。该 binding
在打印这份 kernel 的 initial MLIR 时发生 SIGSEGV，进程退出码为 `139`。这与此前
Stage 6Z 的 MLIR printer 问题一致。由于没有拿到有效 MLIR 文件，本报告不伪造
pre-branch MLIR hash，也不把 source hash冒充 MLIR hash。

严格证据等级如下：

| 层次 | 证据 | 结论 |
|:--|:--|:--|
| source function | 同一文件、同一 function hash | A：完全相同 |
| initial MLIR | printer SIGSEGV | C：未取得 |
| lowered LLVM | 两份文件和 hash 不同 | A：分叉确实发生 |
| exact-LTO MIR | 两份 replay 均成功，均无 spill；最终结构一致 | B：未保留可见收益 |
| final ISA | 指令主体一致，diff 只有 HSACO 文件路径头 | B：机器工作一致 |
| HSACO | SHA256 完全相同 | A：最终 code object 相同 |

因此这是一个有效的 source/LLVM same-source A/B，但不是一个拥有完整
pre-branch MLIR hash 的完美 A/B。这个工具限制必须保留在证据链中。

---

## 4. Generic 与 specialized 的 lowering 差异

### 4.1 Generic arm

generic B operand 使用当前合法的 generic shared/view/fragment 路径，核心特征是
scalar BF16 load 后再组成 fragment。lowered LLVM 的文本统计为：

| lowered LLVM lexical item | generic |
|:--|--:|
| `load bfloat` | 36 |
| `load <8 x bfloat>` | 4 |
| `extractelement <8 x bfloat>` | 48 |
| MFMA call marker | 11 |

这些是 LLVM 文本中的静态模式计数，不是动态 VMEM 或动态 MFMA 数。MFMA marker
包含函数内的 operand lowering 片段，不能直接替代 rocprof 计数。

### 4.2 Specialized arm

specialized B operand 通过同一个通用接口选择 gfx942 typed/packed B path，尽量
保留 BF16x8 operand 语义。lowered LLVM 的文本统计为：

| lowered LLVM lexical item | specialized |
|:--|--:|
| `load bfloat` | 4 |
| `load <8 x bfloat>` | 8 |
| `extractelement <8 x bfloat>` | 80 |
| MFMA call marker | 11 |

这说明 specialized 不是空分支：scalar B load 大幅减少，vector BF16 load 增加，
并且 LLVM 文件与 generic 不同。

### 4.3 LLVM hash

| 文件 | SHA256 |
|:--|:--|
| generic lowered LLVM | `824343c97ef4f1f6dd907d7234e1eeec0af1d94480f3f533ef7693fc557c4f37` |
| specialized lowered LLVM | `60a6d662ee02d2f3845d758c5c5e25676d9b076c44da161e5ff630407813cf2d` |

到此为止，compiler-side A/B 证明是成立的：同一高层 source 进入了不同的
lowered LLVM。

---

## 5. Exact-LTO MIR、ISA 与 HSACO

### 5.1 工件位置

generic：

[`machine/generic_final`](codex_qwen_bt64_stage6z_z7b_mfma_b_same_source_ab/machine/generic_final/)

specialized：

[`machine/specialized_final`](codex_qwen_bt64_z7b_mfma_b_same_source_ab/machine/specialized_final/)

两边都保存了：

- `lowered_llvm.ll`；
- `pre_lto_amdgcn.s`；
- `exact_lto/kernel_section_00.mir` 到 `kernel_section_19.mir`；
- pre/post-greedy 与 virtregrewriter MIR；
- `final_isa.s`；
- `code_object_notes.txt`；
- `machine_evidence.json`；
- `llc_mir/`；
- exact-LTO replay stdout 和 replay argv。

两边 exact-LTO replay return code 都是 `0`。pre-greedy、post-greedy 和
virtregrewriter summary 均没有 `SI_SPILL_AV32/AV64` save/reload。

### 5.2 Final ISA 静态计数

| 指令类别 | generic | specialized |
|:--|--:|--:|
| `v_mfma_f32_32x32x8_bf16` | 56 | 56 |
| `s_barrier` | 32 | 32 |
| global load | 192 | 192 |
| global store | 16 | 16 |
| `ds_read*` | 56 | 56 |
| `ds_write*` | 144 | 144 |
| `ds_bpermute*` | 0 | 0 |

`final_isa.s` 的 SHA256 分别是：

```text
generic      11db030d667172aec26585166d27bc22162697663a6af10206eba8e55cd61706
specialized  9f5b47e8d71289c326400e25c1feee7198804e87f3d39a3165edce225626229
```

两份文本 hash 不同的唯一 diff 是 objdump 第一行中的 HSACO 文件路径，指令主体
没有差异。不能把两个文本文件 hash 不同误读成 machine instruction 不同。

### 5.3 HSACO 和 code-object metadata

| 字段 | generic | specialized |
|:--|--:|--:|
| HSACO SHA256 | `da2b8532146a1c74b1ff99a00596272543bfd68ce57ba4a7301c5eb922f028ac` | 同左 |
| code-object VGPR | 108 | 108 |
| code-object AGPR | 32 | 32 |
| code-object SGPR | 28 | 28 |
| LDS fixed | 32768 B | 32768 B |
| private segment | 0 B | 0 B |
| VGPR spill | 0 | 0 |
| SGPR spill | 0 | 0 |

这是本轮最强的失败证据：specialized 的 LLVM 差异没有穿透到最终 HSACO。

### 5.4 第一次收敛位置

可以确认的层次是：

```text
same source
  -> different lowered LLVM
  -> exact-LTO/MIR structural convergence
  -> same final ISA body
  -> same HSACO
```

当前工件还没有把 LLVM 中间 pass 的每个名字逐一切片，因此不能把收敛精确归因
到某一个 LLVM pass。最严谨的结论是：**收敛发生在 lowered LLVM 之后、最终
AMDGPU machine code/HSACO 形成之前；具体 pass 尚未定位。**

这意味着下一次 compiler 实验的控制点应是 pass-by-pass LLVM/AMDGPU convergence
bisect，而不是继续在 source 中改一点 load 宽度。

---

## 6. Correctness gate

每个长度都使用独立 fresh process。generic 和 specialized 均与冻结的 Z5B
BF16 输出做 byte-exact 比较，并检查 finite。这里的比较是同源改写的语义回归，
不是把 Z5B 当作独立数学 reference；Z5B 的 correctness/contract 已在前一阶段
通过。

### 6.1 普通长度

| T | generic finite | generic vs Z5B max abs | specialized finite | specialized vs Z5B max abs |
|---:|:--:|---:|:--:|---:|
| 64 | pass | 0 | pass | 0 |
| 512 | pass | 0 | pass | 0 |
| 1024 | pass | 0 | pass | 0 |
| 2048 | pass | 0 | pass | 0 |
| 4096 | pass | 0 | pass | 0 |
| 8192 | pass | 0 | pass | 0 |
| 16384 | pass | 0 | pass | 0 |

### 6.2 caller-owned output / zero-V-new / NaN prefill

以下组合全部通过：

| T | generic | specialized | 检查 |
|---:|:--:|:--:|:--|
| 64 | pass | pass | zero-V-new、NaN-prefilled output、finite、byte-exact |
| 8192 | pass | pass | zero-V-new、NaN-prefilled output、finite、byte-exact |
| 16384 | pass | pass | zero-V-new、NaN-prefilled output、finite、byte-exact |

没有因为放宽 tolerance、跳过错误、跳过 NaN 或改变 output ownership 而通过。

---

## 7. T=2048 dynamic PMC

下表来自 fresh rocprofv3 PMC，不是从 static ISA 推导。T=2048 的总 grid 为
131072 work-items，WG256，因此为 512 CTA；表中数值已除以 512 CTA。

| arm | MFMA/CTA | VMEM/CTA | LDS/CTA | VALU/CTA | SALU/CTA | occupancy | profiler VGPR | profiler AccumVGPR | profiler SGPR | LDS metadata | scratch |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Z5B | 160 | 672 | 672 | 7072 | 768 | 14.77894% | 76 | 100 | 112 | 32768 B | 0 |
| Z7B-G | 160 | 672 | 688 | 7178 | 704 | 14.605566% | 84 | 92 | 112 | 32768 B | 0 |
| Z7B-B | 160 | 672 | 688 | 7178 | 704 | 14.789358% | 84 | 92 | 112 | 32768 B | 0 |
| native WG256 | 160 | 140 | 480 | 3376 | 660 | 9.357701% | 100 | 36 | 96 | 0* | 0 |

native 的 `LDS_Block_Size=0` 是当前 collector 对外部/native code object 的
metadata 限制；不能解释为 native 没有 LDS。native 的 dynamic LDS 指令数 480
仍然是有效 PMC 证据。

### 7.1 机器工作解释

- Z7B-G 和 Z7B-B 的 MFMA 保持 160/CTA，没有通过删数学工作获得收益；
- 两条新 arm 的 VMEM 都保持 672/CTA，没有把 LLVM 的 vector load 变成更少的
  dynamic global load issuing instructions；
- 两条新 arm 的 LDS 都从 Z5B 的 672/CTA 变成 688/CTA，反而增加 16；
- VALU 从 7072 增至 7178，说明 specialized LLVM 表示没有消除最终 fragment
  feeding；
- SALU 从 768 降到 704，但不足以抵消 LDS/VALU 和整体 codegen/schedule 成本；
- Z7B 的 code object 没有 private memory，也没有 spill；本轮不是由 spill 失败。

### 7.2 static 与 dynamic 的区别

final ISA 的 56 条 MFMA32 是 lexical static instruction count；rocprof 的
160 MFMA/CTA 是动态执行计数。两者不相等是循环、lane 和 CTA 执行次数造成的，
不能用 56 代替 160，也不能从 56 推 VMEM/LDS 字节数。性能结论只使用上表的
dynamic PMC 和 HIP-event body timing。

---

## 8. Fresh-process body benchmark

口径：

- caller-owned preallocated output；
- current HIP stream；
- no CUDA Graph；
- warmup=10、repeat=50；
- 7 个 fresh-process session；
- 四臂 rotating order：Z5B、Z7B-G、Z7B-B、native；
- HIP event body timing，不包含 compile/module load/allocator；
- native 使用同形状 WG256 selected body，作为 diagnostic，不是 public full API 排名。

### 8.1 正式结果

| T | Z5B ms | Z7B-G ms | Z7B-B ms | native WG256 ms | Z7B-G / Z5B | Z7B-B / Z5B | Z5B / native |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 0.067360 | 0.069162 | 0.069283 | 0.042583 | 1.0268x | 1.0285x | 1.5818x |
| 8192 | 0.158055 | 0.164845 | 0.165146 | 0.091196 | 1.0430x | 1.0449x | 1.7331x |

specialized 与 Z5B 的 paired difference：

```text
T=2048: +2.424, +1.803, +2.043, +1.343, +2.384, +1.663, +1.542 us
T=8192: +5.408, +9.213, +7.171, +5.348, +7.091, +8.112, +4.227 us
```

generic 与 Z5B 的 paired difference：

```text
T=2048: +2.785, +1.282, +2.263, +1.502, +1.883, +1.743, +1.542 us
T=8192: +6.469, +7.831, +6.990, +5.829, +6.870, +6.550, +5.969 us
```

7 个 session 中两条新 arm 相对 Z5B 的每个 paired difference 都为正。它们不是
小幅随机“可能更快”，而是本次口径下稳定更慢；但不能把这解释为新 lowering
引入了 spill，因为机器资源和 HSACO 证明是同一个 final code object。

### 8.2 endpoint slope

用 T=2048（32 chunks）和 T=8192（128 chunks）两点计算：

```text
latency = intercept + slope * chunks
```

| arm | intercept | slope |
|:--|--:|--:|
| Z5B | 0.037128 ms | 0.944740 us/chunk |
| Z7B-G | 0.037268 ms | 0.996693 us/chunk |
| Z7B-B | 0.037329 ms | 0.998568 us/chunk |
| native WG256 | 0.026379 ms | 0.506375 us/chunk |

specialized slope 比 Z5B 高约 5.70%，比 native 高约 97.2%。按照本轮“只有
T=2048/T=8192 都有稳定正收益才继续 T=16384”的规则，没有再运行 T=16384
性能测试；correctness 的 T=16384 已经通过。

---

## 9. 四个核心问题的回答

### 9.1 旧 block-dot V1 是否能安全复用到 chunk-o

**可以复用基础 contract，不能原样复用 Qwen-specific staging。**

本轮没有破坏旧路径，且新 source 使用同一个 block-dot operation 和同一 MFMA32
结果语义。verifier、logical operand metadata 和 lowering helper 的泛化方式
可以服务 chunk-o。可是旧路径本身不会自动带来 native 的 blocked/shared/dot-op
producer-consumer layout；如果不继续保留 typed semantic 到 machine lowering，
最终仍会退化回 generic path。

### 9.2 K/H 是否共享通用 MFMA-B lowering abstraction

**在 source 和 lowered LLVM 层面是。** K 和 H 都进入同一个 operand-mode、同一套
generic/specialized B helper；H 的转置通过显式 metadata 表达，没有复制一套独立
Qwen pass。

**在最终机器层面尚未证明有收益。** 因为 generic/specialized 最终收敛为同一
HSACO，不能说“通用 abstraction 已经生成了 native-style B operand”。它只证明
抽象边界是可实现、可验证的；下一步还要让该 abstraction 影响 AMDGPU machine
selection 或后续 lowering。

### 9.3 typed B lowering 是否减少 VMEM/VALU/LDS 并转化为 latency

**没有。** lowered LLVM 中 scalar load 确实减少，但最终：

```text
VMEM: 672 -> 672 / CTA
LDS : 672 -> 688 / CTA
VALU: 7072 -> 7178 / CTA
MFMA: 160 -> 160 / CTA
```

body latency 也从 Z5B 的 0.067360 ms 增至 specialized 的 0.069283 ms（T=2048），
从 0.158055 ms 增至 0.165146 ms（T=8192）。所以不能用 LLVM 的 vector load
文本变化宣称性能改进。

### 9.4 失败究竟属于哪一层

当前证据支持以下精确表述：

1. 不是 correctness failure；所有长度和 caller-owned/NaN 检查通过；
2. 不是 allocator/RA spill；private segment 和 spill 都为 0；
3. 不是 MFMA 数学工作改变；dynamic MFMA 完全相同；
4. specialized source/metadata 到 lowered LLVM 确实生效；
5. typed B operand 在 lowered LLVM 之后被 AMDGPU/LTO 规范化，最终回到与 generic
   相同的 machine graph；
6. 具体是哪个 LLVM/AMDGPU pass 首次收敛，当前工件尚未逐 pass 定位。

因此当前最佳失败分类是：

```text
late typed-operand semantic did not survive post-LLVM AMDGPU/LTO codegen;
exact convergence pass unresolved.
```

这比“某一条 load 写得不够宽”更准确，也比“编译器一定有 bug”更谨慎。

---

## 10. 晋级判断

| gate | Z7B-G | Z7B-B |
|:--|:--:|:--:|
| T=64/512/1024/2048/4096/8192/16384 correctness | PASS | PASS |
| T64/8192/16384 zero-V + NaN output | PASS | PASS |
| MFMA/CTA 数学工作保持 | PASS | PASS |
| scratch/spill | PASS，0 | PASS，0 |
| lowered LLVM 出现预期 A/B 分叉 | PASS | PASS |
| final HSACO 与 Z5B 不同 | N/A | N/A |
| T2048 latency 优于 Z5B | FAIL | FAIL |
| T8192 latency 优于 Z5B | FAIL | FAIL |

Z7B-B 不晋级，Z7B-G 只作为 generic control 的证据保留。Z5B 继续作为 Stage 6Z
isolated research baseline；没有接入 X2，也没有修改 production selector、
recurrence HSACO 或 public dispatch。

---

## 11. 复现命令和工件

### Correctness

```bash
python3 test/examples/linear_attention/vllm_compare/check_qwen_gdn_bt64_stage6z_z7b_dot_v2.py \
  --T 2048 --arm generic

python3 test/examples/linear_attention/vllm_compare/check_qwen_gdn_bt64_stage6z_z7b_dot_v2.py \
  --T 2048 --arm specialized
```

全长度和 zero-V/NaN matrix 已由该脚本在 Docker fresh process 中完成。

### Machine capture

```bash
python3 test/examples/linear_attention/vllm_compare/dump_qwen_gdn_bt64_stage6z_z7b_dot_v2_machine_artifacts.py \
  --variant generic --T 2048 --skip-initial-mlir \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/\
codex_qwen_bt64_stage6z_z7b_mfma_b_same_source_ab/machine/generic_final

python3 test/examples/linear_attention/vllm_compare/dump_qwen_gdn_bt64_stage6z_z7b_dot_v2_machine_artifacts.py \
  --variant specialized --T 2048 --skip-initial-mlir \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/\
codex_qwen_bt64_z7b_mfma_b_same_source_ab/machine/specialized_final
```

### Fresh-process body benchmark

```bash
python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_stage6z_z7b_dot_v2.py \
  --T 2048 --sessions 7 --warmup 10 --repeat 50 \
  --out test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/\
codex_qwen_bt64_stage6z_z7b_mfma_b_same_source_ab/bench_T2048_sessions7.json
```

T8192 使用同一 harness，输出为 `bench_T8192_sessions7.json`。原始 JSON、PMC
CSV、machine evidence 和 LTO MIR 都保留在本报告列出的目录中。

### Compiler build

本轮 compiler 改动在 gfx942 Docker 中通过 clean build，绑定和 build 目录为：

```text
container: ljd_qwen_vllm_avelang_rocm722
image: qwen_vllm_avelang_rocm722_backup:latest
build: /tmp/avelang-z7b-build3
python bindings: /tmp/avelang-z7b-build3/python
```

---

## 12. 与 Stage 6Z 的关系和下一步边界

Stage 6Z 的有效状态更新为：

```text
Z2  = fixed WG256 historical/phase-aware candidate，已有 barrier/历史数据约束
Z3  = WG128 correctness pass，但 long-text 性能 No-Go
Z5A = dedicated full-Q LDS cache，Q producer pass=1
Z5B = direct Q-cache consumer，删除 Q republish，当前 isolated baseline
Z6G = g-residency arms，performance No-Go
Z7B = block-dot MFMA-B same-source A/B，LLVM 分叉但 machine 收敛，No-Go
```

本轮之后不应立即做 Q typed operand 或 V-new typed operand，因为当前证据显示
specialized K/H B operand 还不能影响最终 machine graph。唯一合理的 compiler-side
后续是：

1. 对 generic/specialized 的 LLVM、LTO bitcode、pre-greedy MIR 做 pass-by-pass
   convergence bisect；
2. 找到首次把 typed B operand 规范化成 generic shared/vector path 的具体 pass；
3. 只有该控制点明确后，才决定是扩展 IR attribute、LLVM intrinsic、AMDGPU
   lowering，还是承认当前表示不足。

若不做这个控制点审计而继续增加 Q/V-new 变体，无法区分是 operand lowering
失败还是 full schedule/lifetime 代价，也会重复 Z7B 的实验逻辑。

---

## 13. 最终学习结论

这轮实验最有价值的结果不是“specialized 没变快”，而是把证据链分成了两段：

```text
高层逻辑 block-dot 表达能力：有
同源 lowering 分叉能力：有
LLVM typed/vector 形态：有
保留到最终 AMDGPU machine graph：没有
```

因此以后判断“是不是 AveLang 高级代码没有写对”时，不能只看 source 或 LLVM
中是否出现了 vector load。必须继续检查：

- exact-LTO pre/post-greedy MIR；
- final ISA；
- HSACO hash；
- dynamic PMC；
- fresh-process latency。

只有当 typed semantic 在这些层次仍然存在，并且同一数学工作下 dynamic machine
work 和 latency 都下降，才可以把它登记为真正的 compiler/lowering 性能收益。

