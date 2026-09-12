# Qwen GDN gfx942 BT64 Stage 4 优化实验复盘

这份文档记录本次 BT64 Stage 4 优化的完整过程，重点解释：为什么要做
这个阶段、之前的瓶颈是什么、每个 kernel 尝试了哪些方案、为什么选择
这些方案、代码结构发生了什么变化、哪些实验失败了，以及最后得到的
收益和倒退。

英文结果报告和机器可读数据仍然保留在：

- [英文总报告](</home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_nonrecurrence_stage4_report.md>)
- [Stage 4 实验目录](</home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_nonrecurrence_stage4>)
- [最终决策 JSON](</home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_nonrecurrence_stage4/final_decision.json>)

本文适合作为学习材料和后续复盘记录。

---

## 1. 先理解整个 Qwen GDN pipeline

这条路径不是一个单独 kernel，而是多个阶段串起来：

```text
g cumsum
  -> KKT
  -> solve
  -> W/U
  -> chunk_gdr recurrence
  -> chunk_o
  -> BF16 public output
```

本次 Stage 4 的核心原则是：

```text
只优化非 recurrence 的高级 Avelang 代码
保持 gfx942 asm recurrence 完全不变
```

固定目标形状：

```text
B=1, Hk=4, Hv=8
K=128, V=128
q/k/v: BF16
g/beta/intermediate/final_state: FP32
chunk_size = BT = 64
layout = [B,T,H,D]
T % 64 == 0
target = gfx942 / MI300
```

最终使用的 Stage 4 源文件是：

[qwen_gdn_bt64_nonrecurrence_mfma_v2.py](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_nonrecurrence_mfma_v2.py:1)

---

## 2. 为什么要做 Stage 4

### 2.1 Stage 2 的情况：功能正确，但上游和下游是 scalar fallback

Stage 2 的 BT64 full pipeline 已经能跑通完整接口，也能连接冻结的 asm
recurrence，但是 KKT、W/U、chunk-o 仍然使用通用 scalar 风格实现。

T=2048 时，Stage 2 大致是：

| 阶段 | Stage 2 延迟 |
|:--|--:|
| full | `16.0413 ms` |
| W/U | `约 4.0051 ms` |
| chunk-o | `约 12.1517 ms` |

rocprof 显示 Stage 2 的典型问题：

- workgroup 只有 1 个 thread；
- MFMA count 为 0；
- VALU/VMEM 数量巨大；
- chunk-o 还有 private scratch；
- 大量循环完全在单线程中展开或串行执行。

所以第一步很明确：先把 scalar fallback 消掉。

### 2.2 Stage 3 的情况：已经 MFMA 化，但 BT64 ownership 还不够好

Stage 3 复用了 v14/v24 的 MFMA16 microkernel，把 W/U 和 chunk-o 改成了
可用的 MFMA 版本：

| 阶段 | Stage 3 T=2048 |
|:--|--:|
| full | `1.0922 ms` |
| KKT | `约 0.3502 ms` |
| solve | `约 0.109~0.127 ms` |
| W/U | `约 0.2949 ms` |
| asm recurrence | `约 0.205 ms`，未插桩 HIP event |
| chunk-o | `约 0.2562 ms` |

Stage 3 的进步很大，但仍然明显慢于：

```text
v24 BT16: 约 0.5882 ms
vLLM:     约 0.3613 ms
```

Stage 3 的关键问题不是“有没有 MFMA”，而是：

- KKT 仍然是 BT64 scalar 单线程版本；
- W/U 的 ownership 仍然接近机械执行多个 token16 小块；
- chunk-o 的跨 token16 复用不够充分；
- scalar correction、重复 load/store 和 kernel ownership 仍有优化空间。

因此 Stage 4 的假设是：

```text
Stage 3 的主要剩余损失来自高级代码的 tile/ownership/data reuse，
而不是先去改 compiler、LLVM RA 或 asm recurrence。
```

这个假设必须通过 kernel-level benchmark、rocprof 和 ISA 证据验证。

---

## 3. 实验总策略

本次没有一开始直接写一个“大而全”的新 full kernel，而是采用增量接入：

```text
Stage 3 full
  -> 只替换 KKT
  -> KKT + W/U
  -> KKT + W/U + chunk-o
```

对应的 incremental full T=2048 结果：

| 版本 | full latency | 相对前一阶段 |
|:--|--:|--:|
| Stage 3 | `1.089079 ms` | baseline |
| Stage 4 KKT-S0 | `0.795162 ms` | `1.3696x` |
| Stage 4 KKT + WU-S1 | `0.610087 ms` | `1.3034x` |
| Stage 4 KKT + WU-S1 + chunk-o-S0 | `0.475867 ms` | `1.2821x` |

这样做的好处是每一步都能回答：

1. 单 kernel 是否正确；
2. 单 kernel 是否变快；
3. 接回 recurrence 后收益是否传递到 full；
4. 如果 full 失败，究竟是哪一个新增阶段造成问题。

---

## 4. KKT：从单线程 scalar 到 BT64 native MFMA

### 4.1 原来的代码和问题

原来的 BT64 KKT 走 v6 风格的通用 kernel：

```text
grid  = B * T * Hv
workgroup = 1 thread
MFMA = 0
```

数学上，KKT 是 chunk 内的严格下三角 token-token 矩阵：

```text
a[t,h,s] = beta[t,h]
           * dot(k[t, key_head], k[chunk_start+s, key_head])
           * exp(g[t,h] - g[chunk_start+s,h])
```

当 `s >= local_token_position` 时输出 0。

也就是说，KKT 不是 solve，也不是一个任意的大矩阵；它本质上是：

```text
一个 BT64 chunk 内部的 K[64,128] @ K[64,128].T
然后施加 causal mask / beta / decay
```

这正适合复用 v24 已验证的 `mfma_16x16x16_bf16_f32`。

### 4.2 Stage 4 KKT-S0 的代码变化

新增 kernel：

```text
_qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0
```

新的 ownership：

```text
每个 CTA = 一个 (chunk, value_head, row_tile, col_tile)
row_tile = 0..3
col_tile = 0..3
每个 tile = 16x16
workgroup = 64 threads
```

BT64 不再由一个线程处理，而是拆成 4x4 个 token16 tile：

```text
BT64 matrix
  -> 4 rows of 16 tokens
  -> 4 cols of 16 source tokens
  -> 16 independent CTA tiles
```

每个 lower/diagonal tile：

1. 把两侧的 `[16,128]` K tile staging 到 LDS；
2. 用 8 次 MFMA16 覆盖 K=128；
3. 在 writeback 时应用 strict-lower mask；
4. 同时处理 FP32 beta 和 decay；
5. 写回与 solve 完全兼容的 `[1,T,8,64]` layout。

### 4.3 为什么先选这个方案

原因不是单纯“MFMA 越多越好”，而是：

- v24 BT16 KKT 已经证明数学和 layout 是可靠的；
- v20 BT32 KKT 已经证明 token matrix 可以 MFMA 化；
- 16x16 tile 的 fragment mapping 在仓库中已有稳定范例；
- 不需要引入新的 32x32 fragment layout 风险；
- causal mask 可以在 MFMA 后安全处理。

### 4.4 结果

| T | 旧 Stage 3 KKT | 新 KKT-S0 | speedup |
|--:|--:|--:|--:|
| 512 | `0.145897 ms` | `0.034731 ms` | `4.2007x` |
| 1024 | `0.213637 ms` | `0.035513 ms` | `6.0157x` |
| 2048 | `0.351423 ms` | `0.046589 ms` | `7.5430x` |

T=2048 rocprof：

| 指标 | KKT-S0 |
|:--|--:|
| workgroup | 64 |
| MFMA | 20,480 |
| VGPR | 20 |
| AccVGPR | 4 |
| LDS | 8,192 B |
| scratch | 0 B |
| VGPR/SGPR spill | 0 / 0 |
| trace | 40.941 us |

KKT correctness：

- T=64/128/512 对比旧 KKT，max abs `4.47e-8`；
- KKT 后接 solve，max abs `3.73e-8`；
- causal mask 和跨 token16 tile 均通过。

结论：KKT-S0 是明确成功的方案，保留并接入 full。

---

## 5. W/U：从重复 token16 MFMA 到 BT64 四 wave ownership

### 5.1 W/U 的数学

W 和 U 的共同形式是一个 chunk-local 小矩阵乘：

```text
A_w[t,s] = a_solved[t,s] * beta[s] * exp(g[s])
A_u[t,s] = a_solved[t,s] * beta[s]

W = A_w @ K
U = A_u @ V
```

输出为 FP32，供 asm recurrence 使用。

### 5.2 Stage 3 之前的主要问题

历史 v14/v24 的 MFMA16 microkernel 是为 BT16 设计的。直接把它扩展到
BT64，容易形成：

```text
4 个独立 token16 ownership
每个 tile 都重复读取 A / K / V / g / beta
每个 tile 都单独做 correction 和 store
```

这种方案虽然出现了 MFMA，却没有充分利用 BT64 内部的共享数据。

### 5.3 W/U-S0：先保守地改 ownership

新增 kernel：

```text
_qwen_gdn_w_bf16_kernel_bt64_mfma_v2_s0
_qwen_gdn_u_bf16_kernel_bt64_mfma_v2_s0
```

新的 launch：

```text
grid = num_chunks * 8 value_heads * 8 column_tiles
workgroup = 256 threads
wave0..wave3 = 四个 token16 row tile
```

和 Stage 3 相比，Stage 4 让一个 CTA 处理完整 BT64 row extent，而不是让
多个 CTA 分别处理各个 token16 row。这样可以：

- 共享 A staging；
- 共享 K/V source tile；
- 减少 kernel program 数；
- 减少重复的 address generation；
- 保留 MFMA16 的稳定 fragment mapping。

### 5.4 W/U-S0 的第一版：保留 scalar residual correction

为了保证 FP32 correctness，主 BF16 MFMA 后保留了原来的 residual：

```python
main = bf16(A) @ B
correction = (A_fp32 - bf16(A)) @ B
result = main + correction
```

但第一版 correction 仍然是 scalar loop：

```text
for source_offset in range(64):
    correction += (coeff_fp32 - coeff_bf16) * B[source_offset]
```

结果：正确，但收益不够强。

T=2048：

```text
Stage 3 W/U       0.248470 ms
W/U-S0            0.229160 ms
```

只有约 `1.084x`，说明 ownership 改善存在，但 scalar residual 仍然是明显
负担。

### 5.5 失败尝试：删除 correction

下一步尝试是直接删除 correction，只保留 BF16 main MFMA：

```python
result = bf16(A) @ B
```

这个版本非常快，所以它是一个有价值的性能上界诊断。但 correctness
完全不能接受，尤其是在：

- nonzero gate；
- high dynamic range；
- cancellation；
- 多 chunk recurrence。

失败原因是：FP32 A 先转 BF16 后，量化误差会被 W/U 计算和 recurrence 放大。

这个函数被保留为：

```text
qwen_gdn_w_u_bt64_mfma_v2_no_correction_failed
```

但 final path 永远不调用它。

### 5.6 W/U-S1：把 residual correction 也 MFMA 化

最终采用的方案是：

```text
第一次 MFMA：bf16(A) @ B
第二次 MFMA：bf16(A_fp32 - bf16(A)) @ B
最后相加
```

代码层面新增了 constexpr 控制：

```python
use_residual_correction = False
use_mfma_residual = True
```

residual A 的生成：

```python
coeff_staged = float(bf16(coeff_residual))
residual = coeff_residual - coeff_staged
a_bf16[...] = bf16(residual)
```

然后使用和主路径相同的 MFMA fragment mapping。

这一步的逻辑是：

```text
不牺牲 correction 的数学作用，
但把 correction 的逐元素串行计算改成矩阵 tile 计算。
```

### 5.7 W/U-S1 结果

| T | Stage 3 | W/U-S0 | W/U-S1 residual MFMA |
|--:|--:|--:|--:|
| 512 | `0.123203` | `0.095522` | `0.060971` |
| 1024 | `0.149723` | `0.146318` | `0.064196` |
| 2048 | `0.248470` | `0.229160` | `0.084165` |

T=2048 的 S1 相对 Stage 3 为 `2.9522x`。

S1 和 exact S0 的误差：

```text
W/U max abs <= 1.8775e-6
```

T=2048 rocprof：

| 指标 | W | U |
|:--|--:|--:|
| workgroup | 256 | 256 |
| MFMA | 262,144 | 262,144 |
| VGPR | 52 | 48 |
| AccVGPR | 4 | 8 |
| LDS | 2,560 B | 2,560 B |
| scratch | 0 B | 0 B |
| trace | 27.722 us | 24.797 us |

相对于 Stage 3，MFMA count 因为加入 residual MFMA 而增加，但 VALU、VMEM、
SALU 和 trace 显著下降。这说明本次收益来自 ownership、数据复用和消除
scalar correction，而不是单纯减少 MFMA 数量。

结论：保留 W/U-S1，不保留 no-correction 版本。

---

## 6. chunk-o：从多个小 tile 到一个 BT64 cooperative CTA

### 6.1 chunk-o 的计算结构

chunk-o 需要计算两类贡献：

```text
inter-state contribution:
    q @ H

intra-chunk causal contribution:
    score(q,k,g) @ V_new
```

然后把两者相加，得到输出。

### 6.2 Stage 2/Stage 3 的问题

Stage 2 的 generic chunk-o 是单线程 scalar fallback：

- workgroup=1；
- MFMA=0；
- 需要大量 q/k/vn/h global load；
- private scratch；
- T=2048 trace 超过 11 ms。

Stage 3 已经使用 MFMA16，但仍然保留比较分散的 token16 tile ownership，
跨 tile 的 Q/K/H/V-new 复用不充分。

### 6.3 chunk-o-S0 的代码变化

新增 kernel：

```text
_qwen_gdn_chunk_o_bf16_kernel_bt64_mfma_v2_s0
```

launch：

```text
workgroup = 256 threads
四个 wave = 四个输出 token16 tile
一个 CTA = 一个 chunk/value-head/V16 block
```

shared staging：

```text
q_scaled_bf16
h_bf16
k_bf16
score_decay_bf16
vn_t_bf16
```

目标是让一个 CTA 之内的四个 token16 tile 共享：

- Q scaled；
- H state；
- K source tile；
- V_new；
- intra score/decay 中间结果。

### 6.4 失败尝试：合并 inter/intra accumulator

从减少寄存器和最终合并指令的角度，曾经尝试把：

```text
inter_acc
intra_acc
```

合并成同一个 accumulator。

结果是数值误差约 `2e-2 ~ 3.4e-2`，超过 frozen contract。原因是 MFMA
累积顺序改变后，FP32 累加的舍入路径发生变化，尤其会影响 chunk 内 causal
贡献和 recurrence 后的最终 state/output。

这个方案没有继续“调容差”，而是直接放弃，保留两个 accumulator：

```text
inter_acc: q @ H
intra_acc: score_decay @ V_new
final = inter_acc + intra_acc
```

它们仍然在 register/accumulator 中保持到最终写回，只是在最后一步相加。

### 6.5 chunk-o-S0 结果

| T | Stage 3 | chunk-o-S0 | speedup |
|--:|--:|--:|--:|
| 512 | `0.087570` | `0.044406` | `1.9720x` |
| 1024 | `0.141130` | `0.062573` | `2.2554x` |
| 2048 | `0.228620` | `0.095342` | `2.3979x` |

T=2048 rocprof：

| 指标 | chunk-o-S0 |
|:--|--:|
| workgroup | 256 |
| MFMA | 458,752 |
| VGPR | 112 |
| AccVGPR | 64 |
| LDS | 27,136 B |
| scratch | 0 B |
| trace | 68.622 us |

chunk-o correctness：

- T=64/128/512 对比 Stage 3，max abs 为 0；
- inter-only、intra-only、source0/1/2 均通过；
- cross-token16 causal tile 通过。

结论：保留 separate inter/intra accumulator 版本。

---

## 7. solve：为什么审计了但没有重写

### 7.1 当前 solve 是什么

v18 solve 的数学是 lower-triangular recurrence：

```text
M[row,col] = M0[row,col]                         if col >= row
M[row,col] = M0[row,col]
                + sum_i<row M0[row,i] * M[i,col]  if col < row
a_solved[row,col] = M[row,col] + diagonal_identity
```

v18 使用 128-thread workgroup，把一个 row 内的 column/update 并行化，
但 row dependency 仍然必须顺序推进。

### 7.2 审计结果

| T | v18 FP32 | vLLM FP32 | v18/vLLM |
|--:|--:|--:|--:|
| 512 | `0.121140 ms` | `0.053359 ms` | `2.270x` |
| 2048 | `0.126888 ms` | `0.053779 ms` | `2.359x` |

这个差距足以允许进行一个 solve v2 实验，但没有足够理由在本轮直接改：

- vLLM 使用 hierarchical block inverse/dot schedule；
- 它不是对 v18 row recurrence 的简单局部修改；
- 当前 solve 已经 correctness 近似 exact；
- KKT/W/U/chunk-o 已经通过完整 37-case gate；
- 一个不成熟 solve 改动可能把所有下游误差放大。

因此本轮决定：

```text
solve 只做 audit，不修改实现。
下一阶段单独设计 hierarchical FP32 BT64 solve。
```

---

## 8. full pipeline 的最终结果

### 8.1 完整延迟

三组独立 session，每组 warmup=10、repeat=50，取 session median 后再取三组
中位数：

| T | Stage 4 | Stage 3 | v24 | vLLM |
|--:|--:|--:|--:|--:|
| 512 | `0.313706` | `0.490168` | `0.300726` | `0.304873` |
| 2048 | `0.475867` | `1.089079` | `0.587173` | `0.365043` |
| 8192 | `1.399619` | `3.655211` | `2.122734` | `0.725157` |
| 16384 | `2.632552` | `7.213319` | `4.252098` | `1.268545` |

T=512 时 Stage 4 稍慢于 v24/vLLM，说明 kernel launch 和固定开销仍然重要。
从 T=2048 开始，BT64 的收益足以超过 v24。

### 8.2 frozen correctness

37-case full matrix 覆盖：

- T=64/128/512/2048 随机输入；
- zero/nonzero initial state；
- neutral gate；
- high dynamic range；
- cancellation；
- small-value；
- T=8192 smoke。

最终结果：

| 检查项 | 最大误差 | 门槛 | 结果 |
|:--|--:|--:|:--|
| public BF16 output | `0.001953125` | `0.0078125` | 通过 |
| FP32 final state | `0.013475478` | `0.020000000` | 通过 |

另外：

```text
Stage 4 pytest: 29 passed
immutable regressions: 80 passed, 1 skipped
```

### 8.3 资源和 ISA

四个新 kernel 都满足：

- Scratch = 0；
- VGPR spill = 0；
- SGPR spill = 0；
- 没有 v29 那种 high-AGPR/resource cliff。

静态 ISA 中的 MFMA16 数量：

| kernel | MFMA16 指令数 | MFMA32 |
|:--|--:|--:|
| KKT | 8 | 0 |
| W | 8 | 0 |
| U | 8 | 0 |
| chunk-o | 56 | 0 |

---

## 9. 哪些方案最终保留，哪些方案放弃

### 保留

1. KKT：BT64 拆成 4x4 token16 tile，64-thread MFMA16。
2. W/U：256-thread BT64 ownership，主 MFMA + residual MFMA。
3. chunk-o：256-thread cooperative CTA，共享 Q/K/H/V-new，inter/intra accumulator 分离。
4. v18 solve：不改，作为当前正确 baseline。
5. asm recurrence：不改，直接复用冻结版本。

### 放弃

1. W/U 只保留 BF16 main MFMA：速度快但数值错误。
2. W/U 保留 scalar residual：正确，但 T=2048 只有约 `1.084x`，不够好。
3. chunk-o 合并 inter/intra accumulator：误差约 `2e-2~3.4e-2`。
4. 继续改 v29 accumulator/lifetime/compiler lowering：不属于本轮 BT64 高级代码目标，且已有 resource cliff 证据。
5. 修改 asm recurrence：当前 asm 是稳定且高效的冻结组件，没有理由在本轮冒险。
6. 直接用 vLLM full wrapper：违反实验边界，不能作为 Avelang 候选实现。

---

## 10. 最终瓶颈和下一步

需要区分两个概念：

### 整体最大 stage

T=2048 时，冻结 asm recurrence 约 `0.2015 ms`，是单个 stage 中最大的。
但它是本轮明确禁止修改的 asm kernel，而且已经是稳定实现。

### 当前可继续优化的最大高级 stage

solve 约 `0.1257 ms`，并且相对 vLLM FP32 solve 约慢 `2.36x`。所以当前
真正适合作为下一步的方向是：

```text
设计一个独立的 hierarchical FP32 BT64 solve v2
```

下一步建议顺序：

1. 先构造 solve block inverse/dot 的 isolated repro；
2. 对比 v18 row recurrence 的 correctness；
3. 检查 workgroup、LDS、VGPR、scratch；
4. 只有 solve standalone 通过后才接回 full pipeline。

目前没有证据表明需要继续修改 compiler 或 AMDGPU RA，也没有必要马上写
新的 assembly。Stage 4 的结果已经证明：通过正确的高级 ownership 和 data
reuse，BT64 非 recurrence 路径可以从 `1.089 ms` 降到 `0.476 ms`。

---

## 11. 如何重现实验

在 `ljd_qwen_vllm_avelang_rocm722` 容器中执行，必须使用历史 Avelang Python
环境，避免误导入 `/opt/avelang`：

```bash
cd /workspace/project/avelang
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention/vllm_compare:/workspace/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2
```

Stage 4 correctness：

```bash
python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_nonrecurrence_mfma_v2.py \
  -s --tb=short
```

Full frozen matrix：

```bash
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_nonrecurrence_stage4/stage4_runner.py \
  --random-cases 30
```

Full benchmark：

```bash
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_nonrecurrence_stage4/bench_stage4.py \
  --T 512 2048 8192 16384 --warmup 10 --repeat 50 --session a
```

Microbenchmark：

```bash
python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_nonrecurrence_mfma_v2.py \
  --T 512 1024 2048 --warmup 10 --repeat 50
```

rocprof、HSACO、ISA 和汇总命令已经集中在：

[commands.sh](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_nonrecurrence_stage4/commands.sh)

---

## 12. 一句话总结

这次真正有效的不是“把 kernel 改得更复杂”，而是逐级找到正确的 ownership：

```text
KKT：把 BT64 变成 4x4 个 MFMA16 tile
W/U：让一个 256-thread CTA 拥有完整 BT64 row，并把 residual 也 MFMA 化
chunk-o：让一个 256-thread CTA 复用 Q/K/H/V-new，保持数值安全的双 accumulator
solve：先审计，暂不冒险重写
recurrence：保持冻结不动
```

最终得到：

```text
Stage 3 BT64: 1.089079 ms
Stage 4 BT64: 0.475867 ms
```

并且所有 correctness、MFMA、scratch/spill、full-sequence 门槛均通过。
