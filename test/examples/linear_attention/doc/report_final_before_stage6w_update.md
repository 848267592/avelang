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

---

# 续篇：Stage 5 到 Stage 6S 的完整实验链、失败原因与推理方法

> 本续篇从上文 Stage 4 结束的位置继续记录。  
> 它不仅汇总“做成了什么”，也保留所有重要的负结果、停止条件和测量口径变化。
>
> 阅读时必须区分三种结论：
>
> 1. **kernel standalone 结论**：只说明某个 kernel 本体；
> 2. **full graph 结论**：说明它接入上下游以后是否仍有收益；
> 3. **production/public API 结论**：说明真实公开接口下的端到端表现。
>
> 三者不能互相替代。后续曾经因为混用不同 harness、CUDA Graph 与 Eager
> public API 数字，造成“差距怎么反而变大”的观感。本文会把这些口径明确拆开。

---

## 13. Stage 4 之后，我们为什么先优化 solve

Stage 4 完成后，非 recurrence 的灾难性 scalar fallback 已经消失：

- KKT 已使用 MFMA；
- W/U 已使用 BT64 ownership 和 residual MFMA；
- chunk-o 已使用 cooperative CTA；
- scratch 和 spill 均为 0；
- full 从 Stage 3 的约 `1.089 ms` 降至约 `0.476 ms`。

此时不能再用“哪个 kernel 看起来大就直接改哪个”的方式推进。我们重新审计了
各阶段，发现当前仍可修改的阶段里，`solve` 是最明显的候选：

```text
v18 solve T=2048：约 0.126 ms
vLLM FP32 solve：  约 0.054 ms
差距：约 2.36x
```

但这并不意味着应立即重写 solve。首先要回答：

1. v18 慢在数学工作量，还是慢在 lowering/寄存器？
2. vLLM 使用的是同一种逐行 recurrence，还是完全不同的算法 DAG？
3. 是否能在不修改 compiler 和 assembly 的前提下复现其结构？

因此 Stage 5A 被定义成 **root-cause audit-only**，而不是直接开始写新 kernel。

---

## 14. Stage 5A：solve 根因审计

### 14.1 为什么做这个实验

v18 solve 正确、无 scratch、无 spill，但 body 明显慢。这里有两种完全不同的可能：

```text
可能 A：算法结构本身串行，应该换算法；
可能 B：算法合理，只是编译器 lowering 或寄存器分配不好。
```

如果是 A，修改 compiler 没有意义；如果是 B，贸然换数学结构又会增加正确性风险。

所以 Stage 5A 的目标不是“让 solve 变快”，而是把这两个可能分开。

### 14.2 审计发现

v18 的核心结构是：

```text
对 row = 1..63 顺序推进
每一行内部并行更新 columns
每一行依赖前面所有行
```

它具有以下特征：

- 一个 chunk/head 一个 CTA；
- workgroup 128，2 waves；
- 63 个 row 依次推进；
- recurrence 区域约 189 个 barrier，加初始 barrier 共约 190；
- MFMA count 为 0；
- 主要依赖 LDS 和标量/向量更新；
- 输入即使是对角矩阵，固定 barrier 结构也基本不变。

当前 vLLM 使用的不是同一种逐行算法，而是 **4×4 个 16×16 block 的层次化下三角逆**：

```text
先求四个 16x16 对角块的局部逆
再按 block DAG 计算 X21、X32、X43
然后计算 X31、X42
最后计算 X41
```

典型公式是：

```text
X21 = -X22 A21 X11
X32 = -X33 A32 X22
X43 = -X44 A43 X33

X31 = -X33 (A31 X11 + A32 X21)
X42 = -X44 (A42 X22 + A43 X32)

X41 = -X44 (A41 X11 + A42 X21 + A43 X31)
```

最终机器代码使用 FP32 MFMA，而不是 63 行顺序 recurrence。

### 14.3 关键测量

同输入 body 测量中，v18 相比 vLLM 的差距稳定存在：

| T | v18 body | vLLM body | v18/vLLM |
|--:|--:|--:|--:|
| 512 | `0.116694 ms` | `0.035372 ms` | `3.30x` |
| 2048 | `0.122422 ms` | `0.035613 ms` | `3.44x` |
| 8192 | `0.229501 ms` | `0.043484 ms` | `5.28x` |

T=2048 的 profiler 结构也明显不同：

| 指标 | v18 | vLLM |
|:--|--:|--:|
| WG | 128 | 256 |
| MFMA | 0 | 65,536 |
| LDS 指令 | 约 1.178M | 约 0.397M |
| Scratch | 0 | 0 |

### 14.4 结论

Stage 5A 的结论是：

> solve 的主因是算法调度结构，不是 AMDGPU RA 或普通 lowering 失误。

因此唯一合理的下一步是：

```text
高层 AveLang 实现
FP32 BT64 4×16 hierarchical block inverse
一个 256-thread / 4-wave CTA
使用现有 FP32 MFMA primitive
```

### 14.5 这一步教会我们的东西

遇到慢 kernel 时，不要先问“怎样减少几条指令”，而要先问：

```text
当前依赖图是否天然串行？
参考实现是否使用了不同的数学分解？
```

Stage 5A 避免了一次错误的 compiler/assembly 优化支线。

---

## 15. Stage 5B：hierarchical FP32 solve v1

### 15.1 为什么做

Stage 5A 已经证明 v18 的问题是 63-row 串行依赖。因此 Stage 5B 不再微调 v18，
而是实现一个独立的层次化 FP32 solve：

- BT64 拆成 4×4 个 16×16 block；
- 一个 chunk/head 对应一个 CTA；
- workgroup 256，4 waves；
- block product 使用 `mfma_16x16x4_f32_f32`；
- accumulator 保持 FP32；
- 明确限制 LDS、scratch 和 spill。

### 15.2 实现原则

最重要的不是“使用 MFMA”本身，而是缩短依赖链：

```text
v18：
row0 -> row1 -> row2 -> ... -> row63

hierarchical：
4个对角块并行/局部处理
-> 三层有限的block DAG
```

这把 63 层依赖压缩成少量 block-level 阶段。

### 15.3 结果

新的 hierarchical solve：

- 正确性通过；
- 资源正常；
- scratch = 0；
- spill = 0；
- LDS 为 8 KiB；
- T=2048 集成 wrapper 中约 `0.029 ms`；
- rocprof kernel trace median 约 `13.701 us`；
- v18 同口径 wrapper 约 `0.126 ms`。

因此 solve 本体大约获得 `4.3x` 级别加速。

### 15.4 为什么这次算成功

它同时满足三项：

1. 数学结构改变有明确依据；
2. 正确性没有依赖放宽阈值；
3. 性能和资源都朝正确方向变化。

它没有修改 compiler 或 assembly，说明高级代码足以表达这个 block DAG。

---

## 16. Stage 5C：把新 solve 接回 Stage 4 full graph

### 16.1 为什么 standalone 成功后还必须做 full integration

一个 kernel standalone 快，不代表 full 一定按相同幅度变快。上下游会影响：

- 数据布局；
- 中间张量；
- kernel launch 顺序；
- cache/设备状态；
- dtype；
- 分配与 wrapper 开销。

所以 Stage 5C 只做一件事：

```text
在 Stage 4 full path 中
把 v18 solve 替换为 hierarchical_fp32_v1
其它 stage 完全不变
```

并且保持：

- 默认 selector 仍是 `v18`；
- 新 solve 只通过显式 `solve_impl="hierarchical_fp32_v1"` 选择；
- 没有 silent fallback；
- production 不变。

### 16.2 正确性

- 新集成测试：`6 passed`；
- 原 Stage 4 默认路径：`29 passed`；
- public output 和 final state 均在冻结阈值内；
- v1 solve 资源为 `VGPR=44`、`AccVGPR=4`、`Scratch=0`。

### 16.3 性能结果

T=2048 三个 session：

```text
Stage 4 + v18：约 0.474906 ms
Stage 4 + v1： 约 0.454215 ms
full gain：    约 20.13 us
```

但 solve 本体大约节省：

```text
0.126 ms -> 0.029 ms
约 97 us
```

于是出现了一个新问题：

> 为什么 solve 本体节省约 97 us，full 只节省约 20 us？

### 16.4 第一轮分段 event 诊断

在每个 stage 之间插 HIP event 后，观察到：

```text
solve：约省 96.6 us
W/U：  约多 16.9 us
recurrence：约多 75.4 us
```

但这张表不能直接相加，因为 stage event 会改变：

- dispatch 边界；
- 同步；
- cache 驻留；
- queue pacing；
- 中间张量状态。

因此 Stage 5C 没有武断地写成“L2 cache 导致 75 us”，而只得出：

> 新 solve 已正确接入；其后连续 kernel 的组合执行状态发生了变化。

这就是 Stage 5D 的来源。

---

## 17. Stage 5D：downstream state-coupling 审计

### 17.1 为什么做

Stage 5C 的核心矛盾是：

```text
solve-only gain：约 97 us
public full gain：约 19~20 us
```

若直接继续优化 solve，会忽略已经发生的 full-path 抵消；若直接改 recurrence，
又没有证据证明 recurrence 代码本身有问题。

Stage 5D 因此是 audit-only，目标是排除以下变量：

- solved tensor 数值；
- consumer pointer；
- alignment；
- allocator；
- 隐藏 dispatch；
- stage-event instrumentation；
- cache/执行状态。

### 17.2 做过的控制实验

#### A. 冻结两条执行图

两图都是 8 个 dispatch，只有 solve 不同：

```text
cumsum
-> KKT
-> solve
-> W
-> U
-> recurrence
-> chunk-o
-> cast
```

W/U、recurrence、chunk-o 的代码和 launch 不变。

#### B. 连续 tail 单 event

只用一个 event 包住：

```text
W -> U -> recurrence -> chunk-o -> cast
```

solve 在 event 前执行，tail 内不插 event。

T=2048：

```text
tail after v18：约 0.2679 ms
tail after v1： 约 0.3322 ms
v1 penalty：   约 64.3 us
```

这证明 downstream 抵消并不是分段 event 完全伪造的。

#### C. Canonical-data 控制

两种 solve 都执行，但丢弃输出；downstream 始终消费同一份 bitwise-identical
canonical tensor。

T=2048：

```text
after v18：约 0.2675 ms
after v1： 约 0.3321 ms
差距：     约 64.6 us
```

因此 solved 数值差异不是主因。

#### D. 同 consumer pointer 控制

两种 solve 的结果 copy 到同一个 canonical consumer buffer，copy 放在计时区间外。

差距仍约 `68 us`，说明 downstream 可见 pointer 和 alignment 不是主因。

但这一步仍未关闭“两个 solve 自身写入不同物理输出地址”这一变量，因此后来还需要
Stage 5E。

#### E. Warm 与大工作集扰动

T=2048：

```text
正常连续执行：差距约 68.6 us
先运行一次相同 tail：差距约 0.12 us
512 MiB workload 扰动：差距约 -0.08 us
reduction prime：差距仍约 75 us
```

这说明差距属于一种可被较长 GPU workload 重置的瞬态状态。

### 17.3 得到什么结论

高置信度结论：

> v18 solve 会留下一个对后续 tail 有利的瞬态执行状态；v1、no-solve 和短 dummy
> predecessor 不会。

但没有足够证据把它唯一归因于：

- L2/TCC/TCP cache；
- 短时频率/功耗爬升；
- runtime queue pacing；
- CU/wave 状态。

### 17.4 为什么这个“没有精确根因”的实验仍有价值

它排除了很多看似合理但错误的修复：

- 不应因为输出误差小就怪数值；
- 不应随意修改 W/U；
- 不应修改 recurrence 汇编；
- 不应增加 dummy warmup；
- 不应根据插 event 的分段数字做加法。

负结果的价值在于收缩问题空间。

---

## 18. Stage 5E：direct common-output pointer

### 18.1 为什么做

Stage 5D 虽然让 downstream 读取同一 pointer，但两个 solve 自身仍先写入不同 allocation。
这留下一个可能：

```text
solve store 的物理地址/cache-set 不同
-> 影响后续 GPU 状态
```

所以 Stage 5E 要求：

```text
v18 solve ------------------\
                             -> exact same solved_common.data_ptr()
hierarchical solve ---------/
```

不允许中间 copy、fill 或额外 dispatch。

### 18.2 实现和正确性

新增 direct-out wrapper：

```text
qwen_gdn_solve_v18_bt64_direct_out_audit(a, out)
qwen_gdn_solve_hierarchical_fp32_v1_direct_out_audit(a, out)
```

结果：

- 两种 solve 直接写 exact same `data_ptr`；
- original wrapper 与 direct-out 均 bitwise 相同；
- NaN prefill、upper triangle、重复复用、residual 全部通过；
- 共测试 80 个 allocation session、35 个不同地址；
- 合并回归 `53 passed`。

### 18.3 关键结果

T=2048：

```text
Stage 5D 原始 tail penalty：64.255 us
Stage 5E direct-common：    64.035 us
变化：                     -0.220 us
```

95% bootstrap 区间约：

```text
[63.254, 64.336] us
```

不同 ABAB/BABA/ABBA 顺序、不同 allocation 都稳定复现。

### 18.4 资源和 profiler 结果

- downstream 动态指令数相同；
- TCP 总访问数相同；
- TCC 最大相对差约 `0.232%`；
- 没有发现可解释 64 us 的 cache 流量差；
- 粗粒度 clock/power telemetry 没有稳定分叉；
- profiler 本身会放大 dispatch gap。

### 18.5 结论

> output pointer/address 被排除为主因。

此时只剩“前驱诱发的瞬态 GPU 状态”这一操作层结论，但精确硬件机制仍未识别。

这一步失败在“没有找到可修复的 pointer 问题”，但成功关闭了一个重要变量。

---

## 19. Stage 5F：低扰动可观测性 Go/No-Go

### 19.1 为什么做

Stage 5D/5E 已经证明现象真实，但没有分离：

- cache residency；
- runtime pacing；
- clock/power；
- CU/wave 状态。

继续 profiler 之前必须先确认：

> profiler 本身是否足够低扰动，能观察一个约 63 us 的效应？

如果 profiler 会改变该效应，任何 timestamp 和 counter 因果分析都不可信。

### 19.2 预注册 gate

只有同时满足以下条件才允许继续：

- 不修改 graph；
- graph 内不插 event；
- latency 扰动足够小；
- A/B penalty 扭曲足够小；
- 能提供 cache、dispatch 或短时状态信息。

### 19.3 实测结果

T=2048 tail：

```text
HIP eager/event baseline penalty：62.772 us
rocprofv3 trace-only penalty：    87.230 us
penalty distortion：              24.457 us
```

trace 还带来约 `28.402 us` 的 latency distortion。

因此 `rocprofv3 --kernel-trace` 未通过 gate。

没有任何 profiler/counter/telemetry 模式能同时满足：

```text
可观察
+
足够低扰动
```

### 19.4 决策

Stage 5F 正式宣布：

```text
Stage 5 transient-state root-cause branch closed
```

没有继续 PMC replay、PC sampling 或更多 profiler 花样，因为这会违反预注册 stop rule。

### 19.5 这一步最重要的学习点

优化研究中必须允许：

> “现有工具无法可靠归因，因此停止。”

这不是放弃，而是防止无限追逐不可观测机制。一个严格的 No-Go
通常比一个听起来合理但无证据的“L2 根因”更有科研价值。

---

## 20. 为什么后来感觉“离 vLLM 越来越远”

这里必须解释测量口径变化。

### 20.1 Avelang 实际一直在变快

按当时各阶段自己的测量环境：

```text
Stage 2：约 16.04 ms
Stage 3：约 1.09 ms
Stage 4：约 0.476 ms
Stage 5C：约 0.454 ms
Stage 6A CUDA Graph：约 0.336 ms
```

Avelang 并没有因为优化而变慢。

### 20.2 为什么 gap 看起来扩大

旧测量中，大量固定开销仍在：

- Python wrapper；
- allocation；
- launch；
- module/dispatch 固定成本。

Stage 6A 使用 CUDA/HIP Graph replay 后，两边固定成本都下降，但 vLLM 下降更多：

```text
旧口径附近：
Avelang 约 0.455 ms
vLLM    约 0.365 ms
gap     约 90 us

Stage 6A graph replay：
Avelang 0.335539 ms
vLLM    0.188900 ms
gap     146.639 us
```

这不代表 Avelang 退化，而是 graph replay 暴露了 vLLM 更低的每-chunk执行成本。

### 20.3 Stage 6A 的关键 slope

```text
Avelang：8.771643 us/chunk
vLLM：   4.365784 us/chunk
gap：    4.405859 us/chunk
```

所以真正问题是：

> Avelang 每处理一个 BT64 chunk 仍做了更多结构性工作。

### 20.4 后续统一规则：Eager public API

从后续 Stage 6T 开始，权威测量规则改为：

```text
Avelang 与 vLLM 都调用各自 public API
每个 timed sample 都是完整 Eager public API 调用
public API 内部 allocation/cast/dispatch 均计时
CUDA/HIP Graph 不再作为权威性能结论
private body 只用于诊断
rocprof trace latency 只用于结构，不用于性能排名
```

因此：

- Stage 6A/6S 的 CUDA Graph 数字仍有结构诊断价值；
- 但未来不能拿它们和 Eager public API 数字直接相减；
- 最终“SOTA/超过 vLLM”必须基于同口径 Eager public API。

---

## 21. Stage 6A：严格 full-graph gap audit

### 21.1 为什么做

Stage 5 关闭后，我们不再追不可观测的 transient state，而是回到主目标：

```text
Avelang full forward 超过 vLLM
```

Stage 6A 的目标是把 full gap 分解成：

- 哪些逻辑 stage；
- 哪些 dispatch；
- 哪些中间 tensor；
- 哪些 dtype；
- 哪些随 chunk 线性增长的成本。

### 21.2 同口径 full 结果

当时采用 CUDA Graph replay 作为结构化对比口径：

| T | Avelang | vLLM | gap |
|--:|--:|--:|--:|
| 512 | `0.122823 ms` | `0.100930 ms` | `21.893 us` |
| 1024 | `0.191845 ms` | `0.128952 ms` | `62.893 us` |
| 2048 | `0.335539 ms` | `0.188900 ms` | `146.639 us` |
| 4096 | `0.607022 ms` | `0.324142 ms` | `282.880 us` |
| 8192 | `1.156198 ms` | `0.602416 ms` | `553.783 us` |
| 16384 | `2.302501 ms` | `1.179092 ms` | `1123.409 us` |

### 21.3 实际 dispatch 图

Avelang 8 个 dispatch：

```text
cumsum
KKT
solve
W
U
recurrence
chunk-o
final cast
```

vLLM 7 个 dispatch：

```text
cumsum
KKT
solve fill
solve merge
fused W/U
recurrence
chunk-o直接写BF16
```

注意：vLLM solve 虽然逻辑上是一个 stage，但机器图中包含 fill 和 merge；
而 W/U 是一个 fused dispatch，chunk-o 直接写 public BF16 output。

### 21.4 中间 dtype 差异

T=2048 可见 contract：

| intermediate | Avelang | vLLM |
|:--|:--|:--|
| solved | FP32 4 MiB | BF16 2 MiB |
| W | FP32 8 MiB | BF16 4 MiB |
| U | FP32 8 MiB | BF16 4 MiB |
| v_new | FP32 8 MiB | BF16 4 MiB |
| output staging | FP32 8 MiB | 无 |
| public output | BF16 4 MiB | BF16 4 MiB |

Avelang 至少额外物化约 22 MiB 中间 storage，尚未计算 consumer reread。

### 21.5 各 stage body gap

| stage | T=2048 gap | gap slope |
|:--|--:|--:|
| cumsum | `+6.850 us` | `+0.001 us/chunk` |
| KKT | `+18.828 us` | `+0.878 us/chunk` |
| solve | `-14.902 us` | `-0.100 us/chunk` |
| W/U | `+36.214 us` | `+0.990 us/chunk` |
| recurrence | `+39.980 us` | `+1.261 us/chunk` |
| chunk-o | `+46.169 us` | `+1.196 us/chunk` |
| cast | `+15.483 us` | `+0.069 us/chunk` |

### 21.6 为什么当时先选 chunk-o

`chunk-o` 是最大的可修改 body gap，并且资源差异巨大：

```text
CTA：Avelang 2048，vLLM 512
MFMA：约 5.6x
VMEM：约 11.9x
LDS inst：约 5.5x
```

所以 Stage 6B 被设计为只改 ownership，不同时修改 BF16 output 或 cast。

这是一种正确的实验隔离：

```text
先验证“重复工作”假设
再验证“dtype/store”假设
```

---

## 22. Stage 6B：chunk-o ownership O0/O1

### 22.1 当前 ownership

Stage 4 current：

```text
一个 CTA = token16 × V16
一个 chunk-head 共 8 个 V16 CTA
T=2048 总 CTA = 2048
```

当前 vLLM：

```text
一个 CTA = token64 × V64
一个 chunk-head 共 2 个 CTA
T=2048 总 CTA = 512
```

Avelang 会在八个 V16 CTA 中重复读取/变换 Q/K 和 score；
vLLM 在 V64 CTA 内复用这些数据。

### 22.2 O0：token64 × V64

#### 为什么做

目标是把 CTA 降到与 vLLM 一致，同时让同一 CTA 复用 lower score。

O0：

- T=2048 CTA `2048 -> 512`；
- 4 waves 管 4 个 token16 row；
- CTA 内缓存 4×4 的 lower score tiles；
- output FP32 staging 和 final cast 不变。

#### 正确性

O0 对 current Stage 4：

- random、zero-H、zero-V_new；
- inter-only、intra-only；
- high/small/cancellation；
- cross-token16；
- 所有 V16 boundary；
- T=8192；

均 bit-exact。

#### 工作量改善

T=2048：

| metric | current | O0 |
|:--|--:|--:|
| CTA | 2048 | 512 |
| MFMA | 458,752 | 188,416 |
| VMEM | 851,968 | 335,872 |
| LDS inst | 1,343,488 | 520,192 |

这证明 ownership 假设成立：O0 真实减少了重复工作。

#### 为什么仍然失败

为了跨 V16 复用 score，O0 引入了更长的 score LDS lifetime：

| resource | current | O0 |
|:--|--:|--:|
| AccVGPR | 64 | 188 |
| LDS block | 27,136 B | 33,280 B |
| occupancy | 17.90% | 8.86% |
| static barriers | 13 | 19 |

T=2048 body：

```text
current：0.076073 ms
O0：     0.062012 ms
speedup：1.227x
要求：   >=1.5x
```

它只回收约 `14.061 us`，未达到晋级门槛。

#### 这个失败告诉了我们什么

> 减少理论工作量不等于减少 latency。

必须同时检查：

- fragment lifetime；
- AccVGPR；
- LDS 容量；
- barrier；
- occupancy。

O0 不是“方向完全错误”，而是“工作复用成功，但资源组织不足以把收益完全兑现”。

### 22.3 O1：token64 × V32

#### 为什么做

O1 是唯一允许的第二个变量，只把 V64 改为 V32，试图：

- 减小单 CTA accumulator；
- 缩短部分 lifetime；
- 提高 occupancy。

#### 结果

O1 反而产生：

- 1024 CTA；
- 更多 MFMA；
- 更多 LDS；
- 更多 barrier；
- T=2048 慢于 current；
- 长序列 slope 更差。

因此 O1 是明确负结果。

### 22.4 最终决策

- O0/O1 都不接 full；
- Stage 4 current 保持不变；
- 不继续 O2/O3 tile sweep；
- 不修改 compiler 或 assembly；
- 暂不进入 direct-BF16 chunk-o。

### 22.5 学习总结

vLLM 和 O0 表面上都是：

```text
token64 × V64
WG256 / 4 waves
512 CTA
```

但资源差异仍然非常大。说明：

> 几何参数相同，不代表 fragment lowering、寄存器生命周期和 block-dot
> lowering 相同。

这是 GPU kernel 优化中非常常见的陷阱。

---

## 23. Stage 6R：recurrence baseline reconciliation

### 23.1 为什么突然回头核对 recurrence

历史上 asm-v0 曾与当时提取的 Triton recurrence 等速：

```text
约 0.195~0.197 ms
```

但 Stage 6A 显示：

```text
Avelang recurrence 比当前 vLLM 慢约 40 us
动态 MFMA：196608 vs 65536
```

这两条结论互相矛盾。继续优化 chunk-o 之前，必须先判断：

1. profiler 聚合是否算错；
2. 当前 vLLM 是否换了 specialization；
3. dtype/ABI 是否不同；
4. 旧 asm-v0 是否真的退化。

### 23.2 Counter aggregation 审计

结果确认 `196608 / 65536` 是同口径、每个 recurrence dispatch 的动态 MFMA：

```text
按 chunk：     6144 / 2048
按 chunk-head：768 / 256
```

因此 3x 差异真实存在，不是 replay 聚合错误。

### 23.3 当前两份机器代码

旧 asm-v0：

```text
HSACO: eede...c226
W/U/v_new：FP32
WG256
dynamic LDS 57344 B
VGPR/AccVGPR 128/192
```

当前 vLLM actual：

```text
HSACO: 6320...077e
W/U/v_new：BF16
BV=32
num_warps=2
num_stages=2
WG128
dynamic LDS 40960 B
VGPR/AccVGPR 104/160
```

两者 kernarg segment 都是 88 B，但只是物理 slot 布局类似，语义 ABI 不兼容。

### 23.4 同口径 body

| T | asm-v0 FP32 | current vLLM BF16 | gap |
|--:|--:|--:|--:|
| 512 | `0.049554 ms` | `0.039199 ms` | `10.356 us` |
| 2048 | `0.154670 ms` | `0.114570 ms` | `40.100 us` |
| 8192 | `0.574333 ms` | `0.413274 ms` | `161.059 us` |
| 16384 | `1.172622 ms` | `0.841410 ms` | `331.212 us` |

### 23.5 Isolated actual-vLLM bridge

Codex 提取当前 vLLM HSACO，建立 isolated external bridge：

- T=512/2048/8192/16384；
- h/v_new/final_state 对 native vLLM bit-exact；
- T=2048 性能差约 `-0.052%`；
- hash、grid、WG、LDS 和 ABI 都有 guard。

### 23.6 结论

CASE B：

> asm-v0 没有退化，它仍忠实对应历史 FP32/WG256 specialization；当前 vLLM
> 选择了新的 BF16/WG128 specialization。

因此正确下一步不是重写 asm，而是测试：

```text
当前更快的 BF16 recurrence bridge
能否接入 Avelang FP32 上下游
```

---

## 24. Stage 6S：BF16 recurrence full-contract integration

### 24.1 为什么不能直接替换 recurrence

当前 Avelang 周边 contract 是：

```text
W FP32
U FP32
old recurrence consumes FP32
v_new FP32
```

新的 vLLM recurrence 要求：

```text
W BF16
U BF16
recurrence outputs v_new BF16
```

所以 Stage 6S 使用显式、可见、可测的 boundary conversion：

```text
W FP32 -> BF16
U FP32 -> BF16
-> current-vLLM recurrence bridge
v_new BF16 -> FP32
-> current chunk-o
```

没有 reinterpret，没有 silent fallback。

### 24.2 三条图

Graph A：旧 Avelang，8 dispatch。

Graph B：Stage 6S，11 dispatch：

```text
cumsum
KKT
solve
W FP32
U FP32
W cast
U cast
new BF16 recurrence
v_new cast FP32
chunk-o
final cast
```

Graph C：native vLLM，7 dispatch。

### 24.3 Correctness

- recurrence bridge 对 native recurrence：
  `h/v_new/final_state` bit-exact；
- public output max abs：`0.001953125`；
- final state max abs：`0.0172200203`；
- 均低于冻结阈值；
- non-default stream、graph replay、invalid guard 均通过。

注意 final state 已接近 `0.02` 门槛，所以后续 BF16 propagation 必须扩大 seed 稳定性测试，
不能随意放宽阈值。

### 24.4 Body 成本

T=2048：

```text
old recurrence：154.669 us
new recurrence：114.570 us
body gain：      40.099 us

W/U/v_new 三个边界 cast 合计：20.750 us
```

这些 isolated body 数不能简单相减当作 full gain，但能解释为什么 full 只能保留部分收益。

### 24.5 Full 结果

同一 CUDA Graph harness 下：

| T | old A | Stage6S B | vLLM C | B gain |
|--:|--:|--:|--:|--:|
| 512 | `0.121721` | `0.120579` | `0.100450` | `1.370 us` |
| 1024 | `0.191685` | `0.181350` | `0.129493` | `10.319 us` |
| 2048 | `0.335679` | `0.309019` | `0.188800` | `26.647 us` |
| 4096 | `0.608985` | `0.550138` | `0.324282` | `58.960 us` |
| 8192 | `1.159743` | `1.033295` | `0.604138` | `126.288 us` |
| 16384 | `2.302501` | `2.024268` | `1.182217` | `278.926 us` |

T=2048 gain 的 95% CI：

```text
[26.595, 26.700] us
```

gap slope：

```text
old A - vLLM：4.396726 us/chunk
Stage6S - vLLM：3.284716 us/chunk
改善：1.112009 us/chunk
```

### 24.6 决策

Stage 6S 是成功候选：

- 保留为 opt-in；
- 不改默认路径；
- 不改 recurrence HSACO；
- 不改 compiler/assembly；
- 下一步传播 BF16 storage boundary，消除三次显式 cast。

### 24.7 学习总结

真正的性能边界不只是 kernel 本体，还包括 ABI：

```text
更快 kernel
+
不匹配的 dtype boundary
=
收益被 cast/materialization 吃掉
```

Stage 6S 说明：先用显式 cast 验证算法和性能，再逐步把 producer/consumer contract
改成原生 BF16，是一种低风险推进方式。

---

## 25. 为什么下一步改成 fused W/U，而不是分别改两个 store

Stage 6S 后，最直观的想法是：

```text
让 W kernel 原生写 BF16
让 U kernel 原生写 BF16
删除两个 cast
```

这个实验能隔离 dtype 收益，但它没有解决一个更大的结构差异：

```text
Avelang：W 和 U 两个 kernel
vLLM：   一个 fused recompute_w_u_fwd_kernel
```

Stage 6A 已经测到 W/U 是重要 gap：

```text
T=2048 body gap：约 36.2 us
gap slope：      约 0.99 us/chunk
```

因此下一阶段修正为：

### F0：只融合，仍输出 FP32

```text
一个 fused W/U kernel
-> FP32 W
-> FP32 U
-> 保留两个外部 BF16 cast
```

F0 单独验证：

- 少一个 dispatch；
- 是否共享 solved/K/V/beta/decay；
- 是否减少 CTA/VMEM/LDS/MFMA；
- fusion 是否产生资源 cliff。

### F1：相同 fused schedule，原生输出 BF16

```text
同一个 fused kernel
-> BF16 W
-> BF16 U
-> 无 W/U cast
-> 无 FP32 W/U materialization
```

F0 和 F1 的 ownership、tile、MFMA、LDS 和 accumulator 顺序必须完全相同；
唯一差异是 final store dtype。

### 25.1 为什么必须先 F0 再 F1

否则性能变快时无法区分：

```text
来自 fusion/复用
还是
来自 BF16 store/cast 删除
```

这体现了实验设计中的单变量原则。

### 25.2 最大风险：resource cliff

不能简单把 W 和 U 两个 kernel 文本拼起来。需要控制 accumulator lifetime：

```text
共享输入准备
-> 计算 W
-> W 完整写回
-> 结束 W accumulator lifetime
-> 计算 U
-> U 完整写回
```

若两套 accumulator 同时长期活跃，可能重现：

- 高 AccVGPR；
- scratch/spill；
- occupancy 崩塌；
- v29 式资源 cliff。

---

## 26. 从现在起的权威测量合同：Eager public API

后续所有 Codex 任务都必须遵守：

### 26.1 什么算权威测试

Avelang、实验候选和 vLLM 都必须通过明确 public API：

```text
start event
-> public_api(...)
-> end event
```

每个 sample 都是真实 Eager 调用。

### 26.2 哪些成本必须计入

public API 内部的：

- output allocation；
- intermediate allocation；
- cast；
- contiguous/copy；
- kernel dispatch；
- wrapper glue；
- 返回对象构造；

均属于端到端性能，不能由 harness 私自绕过。

### 26.3 哪些可以预热

正式计时前可以预热：

- JIT；
- Triton autotune；
- module load；
- import；
- 首次 runtime 初始化。

### 26.4 哪些不能作为主结论

- CUDA/HIP Graph capture/replay；
- private kernel body latency；
- profiler trace duration；
- caller-owned private out-buffer 快捷路径。

它们仍可用于：

- correctness first divergence；
- 资源；
- ISA；
- dispatch 结构；
- 定位瓶颈。

### 26.5 为什么切换到 Eager public API

目标是“真实 public API 超过 vLLM”，而不是：

```text
某个预捕获内部图
在固定buffer下
比另一个内部图快
```

CUDA Graph 会去除很多真实调用中的动态成本。它适合研究 GPU steady-state
结构，但不能代替最终端到端接口结论。

---

## 27. 尚未执行的下一阶段：Stage 6T-Eager fused W/U

截至本文完成时，Stage 6T-Eager 是**计划**，不是已完成实验。

目标图：

### Stage 6S Eager baseline

```text
W FP32
U FP32
W cast BF16
U cast BF16
BF16 recurrence
v_new cast FP32
chunk-o
final cast
```

### F0

```text
fused W/U FP32
W cast BF16
U cast BF16
BF16 recurrence
v_new cast FP32
chunk-o
final cast
```

### F1

```text
fused W/U BF16
BF16 recurrence
v_new cast FP32
chunk-o
final cast
```

所有权威 correctness 和 performance 都要通过 Eager public API 完成。

### 27.1 成功标准

F0：

- FP32 W/U 正确；
- scratch/spill/private = 0；
- 不发生资源 cliff；
- Eager full 不明显退化。

F1：

- native BF16 W/U 与 F0 FP32->BF16 尽量 bit-exact；
- public output/final state 通过；
- 无 W/U cast；
- 无 FP32 W/U materialization；
- T=2048 Eager full 至少稳定回收 `10 us`；
- 长文本 slope 改善；
- wall-clock 与 HIP-event 方向一致。

### 27.2 若 F1 成功，下一步是什么

Stage 6U-Eager：

```text
让 current chunk-o 原生读取 BF16 v_new
在 load 后进入现有 FP32 计算路径
删除 v_new BF16->FP32 显式 cast
```

仍然暂时不改：

- chunk-o ownership；
- direct BF16 public output；
- final cast；
- recurrence。

这样继续保持单变量推进。

---

## 28. 完整实验账本

| 阶段 | 当时为什么做 | 做了什么 | 结果 | 状态 | 告诉我们什么 |
|:--|:--|:--|:--|:--|:--|
| Stage 5A | solve 是最大可修改瓶颈 | 审计 v18 与 vLLM solve DAG | v18 是 63-row 串行；vLLM 是 4×16 block inverse | 成功审计 | 根因是算法结构，不是 RA |
| Stage 5B | 验证 block inverse 能否高层表达 | 实现 FP32 hierarchical solve | 约 4.3x，scratch/spill 0 | 成功 | 高层 MFMA 足以解决 solve |
| Stage 5C | standalone 收益是否传入 full | opt-in 接入 Stage 4 | full 只快约 20 us | 部分成功 | standalone gain 不能直接当 full gain |
| Stage 5D | 解释约 77 us 未传递收益 | tail/canonical/pointer/warm 控制 | 发现约 64 us transient tail penalty | 成功审计 | 数值和 consumer pointer不是主因 |
| Stage 5E | 关闭 solve-store 地址变量 | 两 solve 直接写同一 pointer | penalty 仍约 64 us | 负结果但关键 | pointer/address被排除 |
| Stage 5F | 是否能继续用 profiler 归因 | 低扰动 gate | trace 把 penalty 扭曲约 24.5 us | No-Go | 工具不够低扰动，关闭支线 |
| Stage 6A | 回到超过 vLLM 主线 | 全图、dispatch、dtype、slope 审计 | gap 主要 chunk-linear | 成功审计 | 结构性工作量是主问题 |
| Stage 6B O0 | chunk-o CTA/重复读取过多 | token64×V64 ownership | 工作量大降，但资源 cliff，1.227x | 未晋级 | 几何相同不等于 lowering 相同 |
| Stage 6B O1 | 尝试 V32 减轻资源 | token64×V32 | 更慢、更多 CTA/MFMA/LDS | 失败 | V32 不是解决方案 |
| Stage 6R | 解释 recurrence 历史等速与当前 40 us gap | 对齐 HSACO/ABI/counter | 当前 vLLM 是新 BF16/WG128 specialization | 成功 | asm-v0没退化，只是旧 specialization |
| Stage 6R bridge | 验证新 vLLM kernel能否复现 | isolated HSACO bridge | bit-exact，性能差约 -0.052% | 成功 | 可作为 full 候选 |
| Stage 6S | 测试新 recurrence 在 FP32外围的净收益 | W/U cast BF16，v_new cast FP32 | T2048 full 快26.647 us | 成功候选 | ABI propagation 是下一主线 |
| Stage 6T-Eager | 计划消除 W/U 结构和 dtype gap | F0 fused FP32，F1 fused BF16 | 尚未执行 | 下一步 | 必须以 Eager public API 为准 |

---

## 29. 如何从实验结果推导下一步

这条优化路线不是随机试错，而是以下循环：

### 29.1 第一步：建立可证伪假设

例如：

```text
假设：chunk-o 慢是因为 V16 ownership 重复 Q/K/score。
```

### 29.2 第二步：只改一个主要变量

O0 只改 ownership，不改 BF16 output。

### 29.3 第三步：同时看三类证据

```text
Correctness
Resource/work count
Latency/full propagation
```

只看其中一个都可能误判。

O0 就是典型案例：

```text
Correctness：通过
Work count：明显下降
Latency：改善不足
Resource：AccVGPR/LDS/occupancy恶化
```

所以结论不是简单的“成功”或“失败”，而是：

> ownership 假设正确，但当前实现的资源组织不足。

### 29.4 第四步：设置停止条件

例如：

- 最多两个 variant；
- profiler 扰动不通过就停止；
- standalone 未过 gate 不接 full；
- full 未过 gate 不改 default。

这避免为了证明自己正确而无限调整参数。

### 29.5 第五步：区分“没有收益”和“没有知识”

一个失败实验仍可能产生高价值知识：

```text
Stage 5E：没有恢复性能
但排除了 pointer/address

Stage 5F：没有找到硬件机制
但证明当前 profiler 不适合该因果问题

Stage 6B O0：没有晋级
但证明重复 ownership 确实存在，并暴露 AccVGPR/LDS trade-off
```

真正低价值的是没有控制变量、没有停止条件、没有保留负结果的试错。

---

## 30. 当前保留和未保留的实现

### 30.1 Production/default

保持不变：

- v24 BT16 production baseline；
- 默认 Stage 4 selector；
- 原 production dispatch。

### 30.2 保留的 opt-in 实验组件

- Stage 4 BT64 KKT/W/U/chunk-o；
- Stage 5B hierarchical FP32 solve bridge；
- Stage 6R current-vLLM BF16 recurrence bridge；
- Stage 6S BF16 recurrence full candidate。

### 30.3 明确保留为证据但不接入

- Stage 6B O0；
- Stage 6B O1；
- Stage 5E direct-out audit wrappers；
- Stage 5D/5F profiler与状态控制 harness。

### 30.4 明确放弃的方向

- 继续追 Stage 5 transient state；
- chunk-o O2/O3 tile sweep；
- 把 O0 静默接 full；
- 修改旧 asm-v0 以“修复退化”；
- 根据 profiler trace latency 做性能结论；
- 加 dummy warmup 伪造状态；
- 一次同时改 fusion、dtype、ownership 和 compiler。

---

## 31. 当前性能状态应该怎样表述

### 31.1 历史 CUDA Graph 结构结果

Stage 6S 当时的 graph-replay T=2048：

```text
Stage 6S Avelang：0.309019 ms
vLLM：            0.188800 ms
```

这个结果证明 BF16 recurrence integration 有结构性收益，但不能作为今后
Eager public API SOTA 声明。

### 31.2 当前尚缺少的权威数字

仍需重新测量：

```text
Eager public API Stage 6S
Eager public API fused F0
Eager public API fused BF16 F1
Eager public API native vLLM
```

只有这四者同口径，才能回答：

> 当前真实端到端还差多少？

因此不能再使用旧 Graph 数字来判断最终是否超过 vLLM。

---

## 32. 后续路线图

```text
Stage 6T-Eager
fused W/U public API
F0隔离fusion
F1隔离BF16 store
        ↓
若成功：
Stage 6U-Eager
chunk-o原生读取BF16 v_new
删除v_new cast
        ↓
再审计：
chunk-o direct BF16 public output
删除output_fp32和final cast
        ↓
重新做Eager full gap audit
        ↓
再排序：
KKT
chunk-o ownership的新表达方式
recurrence外围
        ↓
最后做：
BT16/BT64 crossover
production dispatch policy
稳定性和长序列回归
```

每一步都必须：

- 使用 Eager public API 做权威 full；
- 只改变一个主要结构；
- 保留 correctness/resource/full 三层证据；
- 失败就记录并停止，不用下一项掩盖失败。

---

## 33. 最终学习总结

从 Stage 4 到 Stage 6S，最重要的成长不是记住某个 kernel 的参数，而是形成以下能力。

### 33.1 区分数学瓶颈与编译器瓶颈

Stage 5A 证明 solve 是算法 DAG 问题，不是 RA。

### 33.2 区分工作量减少与真实 latency 改善

Stage 6B O0 工作量下降约 60%，但因为 AccVGPR/LDS/occupancy，只获得 1.227x。

### 33.3 区分 kernel 本体与 ABI/边界成本

Stage 6R recurrence 本体快约 40 us，Stage 6S 的三次 cast 吃掉约一半收益。

### 33.4 区分 standalone、full graph 与 public API

一个 kernel 快，不代表 full 按同样幅度快；一个 CUDA Graph 快，也不代表 Eager public API
端到端同样快。

### 33.5 正确对待失败实验

失败实验不是“白做”，前提是它：

- 有明确假设；
- 控制变量；
- 有可量化 gate；
- 能排除某条路径；
- 不被后续结果篡改。

### 33.6 知道何时停止

Stage 5F 的 No-Go 是整条路线中非常重要的科研决策。无法低扰动观测时，继续追根因
只会产生越来越不可信的故事。

### 33.7 最终目标始终不变

主线仍然是：

```text
在相同 Qwen TP4 contract、gfx942/MI300、
严格 Eager public API 口径下，
让 Avelang full forward 超过 native vLLM。
```

当前已经完成的不是最终胜利，而是把问题从：

```text
scalar fallback
资源 cliff
错误的 solve DAG
旧 recurrence specialization
不匹配的 FP32/BF16 boundary
```

逐步缩小到更明确的结构差距。

下一步不是“再随便试一个 kernel”，而是执行已经定义好的 Stage 6T-Eager：

```text
F0：fused W/U FP32
F1：同schedule fused W/U BF16
```

并以 Eager public API 结果决定是否继续传播 BF16 `v_new` boundary。

---

## 34. 环境失败、工具限制与测量陷阱补充

这一节专门记录那些容易在“只看最终性能表”时被遗漏，但会直接影响实验可信度的失败。

### 34.1 Stage 5C 首次集成失败：活动 Python binding 过旧

hierarchical solve 第一次接入 full path 时，source call 报错：

```text
Symbol not found: al.amdgpu.mfma_16x16x4_f32_f32
```

这不是 solve 数学错误，也不是新 kernel lowering 失败。实际原因是容器 import 的：

```text
/opt/avelang/python/_avelang_bindings...so
```

仍然是旧 binding，而宿主源码已经包含新的 registry 和 ROCDL wrapper。

当时没有删除用户已有 build，而是在独立 `/tmp` 目录做干净 build，再原子替换活动 binding，
并用最小 probe 验证 `max_abs=0`。

这个问题教会我们：

> 当源码里存在 intrinsic、运行时却提示 symbol missing，首先核对实际 import 的二进制，
> 不要立即修改 kernel 或 compiler source。

### 34.2 Stage 5D 的 profiler/telemetry 未完整执行

Stage 5D 在完成主要 benchmark 和控制实验后，平台拒绝继续 Docker execution。
因此以下项目当时明确标记为 N/A：

- 新一轮 whole-graph rocprof timeline；
- cache/TCC/TCP counter；
- dispatch gap；
- CU/wave distribution；
- clock/power telemetry；
- 部分回归复跑。

报告没有把旧数据冒充新数据。这一点非常重要：

> N/A 比伪造完整表格更可信。

Stage 5E 后来补做了部分 counter 和 telemetry，但仍不足以唯一定位硬件机制。

### 34.3 Stage 5F 的失败不是“工具不存在”，而是“工具扰动太大”

环境中确实存在：

- `rocprofv3`；
- PMC；
- PC/thread trace 入口；
- `amd-smi/rocm-smi`。

但“工具存在”不代表“工具适合该问题”。`rocprofv3 --kernel-trace` 把约 `62.8 us`
的原生 penalty 扭曲到约 `87.2 us`，因此它的 timestamp 不能用于该因果问题。

这说明 profiler 的第一步不是采数据，而是校准：

```text
profiler是否改变了我要观察的现象？
```

### 34.4 Stage 6A standalone body profiler 出现不同 autotune specialization

Stage 6A 的独立 body profile 在新进程中，部分 vLLM kernel 选择了不同的 autotune config。
因此报告没有把这些 body counter 与 full graph counter 混合。

最终采用：

```text
full graph 实际 specialization
作为 end-to-end resource 的权威结构
```

这个细节说明：Triton kernel 的性能和资源不能只按函数名比较，必须同时记录：

- specialization；
- constexpr；
- autotune config；
- process/cache 环境；
- launch geometry。

### 34.5 Stage 6S 的 source-JIT hierarchical solve 回归失败

Stage 6S 期间，直接 source-JIT hierarchical solve 的若干测试仍因活动 binding 缺少：

```text
al.amdgpu.mfma_16x16x4_f32_f32
```

而失败。

这不是 Stage 6S BF16 recurrence 集成的 correctness failure。Stage 6S full path 使用的是：

```text
已经验证
不可变
hash-guard
的 Stage 5B solve HSACO bridge
```

因此需要把两件事分开：

```text
A. 当前运行环境的source-JIT feature export不完整；
B. 本轮实际full graph使用的冻结solve code object正确。
```

未来若要恢复 source-JIT solve 测试，应单独修复/统一活动 binding，不应顺手修改
Stage 6S 算法路径。

### 34.6 不同 benchmark harness 的数字不能交叉相减

曾出现以下几类测量：

- 普通 public/wrapper Eager；
- fixed preallocation audit harness；
- HIP event full；
- per-stage event；
- CUDA/HIP Graph replay；
- rocprof trace。

它们回答的问题不同。

例如 Stage 5E fixed-buffer full 的收益比 Stage 5C public full 更大，但不能直接声称
production 又多快了几十微秒，因为 harness 改变了 allocation 和执行环境。

后续报告必须为每个数字写清：

```text
timed callable
allocation是否计入
是否public API
是否graph replay
是否profiler
warmup/repeat/session
```

### 34.7 “测试失败”必须分成三类

以后记录失败时，应明确属于哪一种：

1. **算法/正确性失败**
   - 例如 W/U no-correction；
   - chunk-o merged accumulator。

2. **性能/资源失败**
   - 例如 Stage 6B O0 未过速度 gate；
   - O1 明确退化。

3. **环境/工具失败**
   - 例如 binding 缺失；
   - Docker quota；
   - profiler 扰动；
   - autotune specialization 漂移。

三类失败对应的下一步完全不同，不能混为“kernel 写错了”。


