# Qwen gfx942 BT64 Stage 6Z Z5B：Direct Q-Cache Consumer

## 结论摘要

本轮实现并完成了唯一允许的 Z5B 变化：从 Z5A 的 dedicated Q LDS cache
直接形成 Q fragment，删除每个 K32 stage 中的

```text
Q cache -> old phase Q rows -> phase_vec -> MFMA
```

中间复制。实验没有改 WG、MFMA 几何、K/H/V-new/g/output producer、数学、
BF16 ABI、Q cache 容量或 accumulator phase 顺序。

结论为 **Case A：Z5B 晋级为新的 isolated Stage 6Z research baseline**。

证据是三层一致的：

1. correctness：T=64/512/1024/2048/4096/8192/16384 全部与 Z5A 和 fixed
   Z2 BF16 byte-exact；caller-owned output、zero-V-new、NaN 预填充和 finite
   检查全部通过。
2. machine graph：Z5B 的动态 MFMA 保持 `160/CTA`，VMEM 保持
   `672/CTA`，LDS 从 Z5A 的 `1440/CTA` 降至 `672/CTA`；静态 ISA 的
   `ds_read/ds_write` 也从 `152/240` 降至 `56/144`。没有 private segment、
   spill 或 scratch。
3. timing：同一 fresh-process、current stream、no Graph、预分配输出、
   warmup=10/repeat=50 的 5-session paired benchmark 中，Z5B 在 T=2048
   相对 Z5A 的 session 中位数均值约快 `0.66 us`；T=8192 的 5 个 session
   全部更快，均值约快 `4.36 us`。两点 endpoint slope 从 Z5A 的约
   `0.980 us/chunk` 降为 Z5B 的约 `0.946 us/chunk`。

Z5B 仍然是 **isolated chunk-o research baseline**，没有接入 X2、没有接入
production selector，也没有替换 recurrence HSACO。它相对 native vLLM
同形状 WG256 diagnostic 仍慢约 `1.54x`（T=2048）和 `1.72x`（T=8192），
所以本报告不能宣称已经达到 native vLLM 性能。

## 1. 实验边界与冻结项

### 1.1 基线和实验对象

| 名称 | 含义 |
|:--|:--|
| fixed Z2 | Stage 6Z 修复后的 WG256/BV64/BK32 native-style chunk-o 参考 |
| Z5A | dedicated full-Q LDS cache；Q source producer pass=1，但 Q 仍被复制到旧 phase Q 区域 |
| Z5B | 从最终 Z5A 源码分叉；Q cache 直接作为 Phase A/B0/B1 的 Q consumer |
| native | current-vLLM selected chunk-o 的同形状 WG256 diagnostic reference |

Z5B 源码是：

```text
test/examples/linear_attention/vllm_compare/
qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer.py
```

Z5B 没有从 Z4C 或 fixed Z2 重新组织 accumulator。它保留了 Z5A 已经验证
正确的 full-Q cache 结构和 phase-separated accumulator 顺序。

### 1.2 冻结的硬件和数学 contract

- gfx942，wave64；
- BT64、BV64、BK32；
- WG256，2 CTA/chunk-head；
- BF16 Q/K/V-new/H，FP32 g；
- MFMA32 geometry 和 K32 accumulation order 不变；
- causal mask 不变；
- Q cache 逻辑容量为 `64 x 128 BF16 = 16 KiB`；
- 总 shared allocation 仍为 `512 x 128 BF16 = 32 KiB`；
- Q global producer 仍只有一个 source region；
- H、K、V-new、g、output 的 producer 不改；
- Q cache 在两个 score half 结束前不覆盖；
- inter、score half 0、score half 1、intra accumulator 仍 phase-separated；
- caller-owned BF16 output contract 不变；
- 不改 allocator/RA、selector、X2、production dispatch 或 recurrence HSACO；
- 不使用 Graph replay，性能只采用 current HIP stream body timing。

## 2. Z5A 的问题和 Z5B 的唯一变化

Z5A 已经把 Q 的 global producer 从三次降到一次，但仍有两类额外操作：

```text
Q global once
    -> dedicated Q cache
    -> Phase A/B 的旧 phase Q rows
    -> phase_vec
    -> MFMA32
```

这解释了 Z5A 的动态 LDS 增长：它少读了 global Q，却把当前 K32 slice
再次写入 phase 区域，然后再由 MFMA consumer 读取。

Z5B 只删除这次 Q republish：

```text
Q global once
    -> dedicated Q cache
    -> q_cache_vec 当前 K32 slice
    -> MFMA32
```

H、K 和 V-new 仍使用原 phase 区域。也就是说，本轮没有把所有 operand
layout 一起重做，没有引入新的 swizzle，也没有把三套 accumulator 合并进
同一个 K32 loop。

## 3. 源码证据

Z5B 关键位置见源码：

| 源码位置 | 证据 |
|:--|:--|
| `qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer.py:93-114` | 32 KiB shared allocation、Q cache 与 phase 的逻辑分区和 i32 view |
| `:100-111` | 唯一 Q global producer；保留与 Z5A/Z2 相同的 FP32 scaling 和 BF16 rounding |
| `:119-138` | Phase A 直接从 `q_cache_vec` 取 Q；只将 H 写入 phase 区域 |
| `:140-165` | Phase B 两个 source half 序列执行；只将 K 写入 phase 区域，Q 不再 republish |
| `:132` | Phase A 的 Q fragment 来源为 `q_cache_vec[...]` |
| `:158-164` | Phase B 的 Q fragment 来源为 `q_cache_vec[...]` |
| `:116-117`、`:143-145`、Phase C | accumulator phase 顺序保持 Z5A，不形成 Z4C 式共同 live loop |
| `:212-239` | hard WG256 launch contract，无 WG128 fallback |

关键源码形态如下：

```python
q_cache_vec = al.view(q_cache, al.Tensor((2 * Q_CACHE_ROWS, 4, 4), al.i32))

# Phase A
q_words = q_cache_vec[k_stage * BT + row_half * 32 + lane_col, word]

# Phase B
q_words = q_cache_vec[k_stage * BT + row_half * 32 + lane_col, word]
k_words = phase_vec[Q_CACHE_ROWS + score_stage_base + 64 + lane_col, word]
```

Z5A 对应位置还包含：

```python
phase[Q_CACHE_ROWS + row, col] = q_cache[k_stage * BT + row, col]
...
phase[Q_CACHE_ROWS + score_stage_base + row, col] = q_cache[k_stage * BT + row, col]
```

Z5B 中这两类 Q-to-phase store 已删除；K/H/V-new 的 phase store 保留。

## 4. Correctness gate

测试文件：

```text
test/examples/linear_attention/vllm_compare/
test_qwen_gdn_bt64_native_chunko_stage6z_z5b.py
```

测试结果：

```text
11 passed
```

### 4.1 全长度 BF16 byte-exact

测试 T 为 `64/512/1024/2048/4096/8192/16384`。每个长度都执行：

```text
Z5A -> Z5B
fixed Z2 -> Z5B
finite(Z5B)
```

结果：

| T | Z5B vs Z5A BF16 | Z5B vs Z2 BF16 | finite |
|--:|:--:|:--:|:--:|
| 64 | exact | exact | pass |
| 512 | exact | exact | pass |
| 1024 | exact | exact | pass |
| 2048 | exact | exact | pass |
| 4096 | exact | exact | pass |
| 8192 | exact | exact | pass |
| 16384 | exact | exact | pass |

这里使用的是 `torch.equal`，不是仅比较一个宽松的 abs tolerance。因此直接
Q-cache consumer 没有改变 BF16 输出语义。

### 4.2 caller-owned output / zero-V-new / NaN prefill

T 为 `64/8192/16384`，测试步骤为：

```text
V-new.zero_()
output = full_like(V-new, NaN)
Z5B writes caller-owned output
```

结果：输出没有 NaN，并且与 fixed Z2 逐元素 BF16 exact。这个 gate 排除了
“Q cache 变化只是因为输出未完全覆盖”这类假阳性。

### 4.3 shape contract

测试明确断言：

```text
WORKGROUP == Z5B_WORKGROUP_CONTRACT == 256
```

Z5B launch wrapper 也在运行时检查 output dtype、shape、device、contiguity，
并拒绝任何不满足 contract 的输入。

## 5. Source -> LLVM -> MIR -> ISA 证据

### 5.1 机器工件

Z5B 工件目录：

```text
codex_qwen_bt64_stage6z_z5b_machine_stage1/
```

其中包含：

- `lowered_llvm.ll`；
- `pre_lto_amdgcn.s`；
- `exact_lto/linked.hsaco.0.5.precodegen.bc`；
- `exact_lto/kernel_section_*.mir`；
- `llc_mir/stop_after_greedy.mir`；
- `llc_mir/stop_after_virtregrewriter.mir`；
- `llc_mir/stop_after_prologepilog.mir`；
- `final_isa.s`；
- `z5b_fixed.hsaco`；
- `code_object_notes.txt`、`capture_summary.json`、`machine_evidence.json`。

机器 capture 的初始 high-level MLIR 按命令显式 `--skip-initial-mlir`，原因是
当前 Docker 的既有 MLIR printer 会 segfault。报告不把缺失的初始 MLIR
伪装成证据；源码、lowered LLVM、LTO MIR、ISA 和 HSACO 都已实际保存。

### 5.2 LLVM：Q global producer 只有一个 source region

在 Z5B `lowered_llvm.ll` 中，Q pointer `%0` 的静态 Q producer GEP/load
出现在第 `168-169` 行：

```llvm
%136 = getelementptr inbounds nuw bfloat, ptr %0, i64 %135
%137 = load bfloat, ptr %136, align 2
```

后续 global pointer load 的 `%3/%1/%2/%5` 分别对应 H/K/V-new/output
路径；没有第二个来自 `%0` 的 Q producer region。注意这里说的是静态 source
region，不是把一个循环体内的 load 错当作只执行一次。循环展开后的动态次数
仍由 kernel 线程和 stage 共同决定。

Z5B Phase A/B 的 shared Q consumers 是：

```llvm
%229 = getelementptr i32, ptr addrspace(3) @__wg__...z5b..._0, i64 %228
%230 = load <4 x i32>, ptr addrspace(3) %229, align 4

%238 = getelementptr i32, ptr addrspace(3) @__wg__...z5b..._0, i64 %237
%239 = load <4 x i32>, ptr addrspace(3) %238, align 4
```

这两处在 `lowered_llvm.ll:293-303`，属于 dedicated shared allocation 的
`addrspace(3)` read，随后 bitcast 为 BF16 vector 并提取 fragment。

相反，Z5A 的 LLVM 在对应 Phase A/B 区域存在：

```llvm
load bfloat, addrspace(3) q_cache_address
store bfloat, addrspace(3) phase_q_address
```

也就是 `lowered_llvm.ll:190-196` 一类的 Q cache -> phase Q copy。Z5B 的
对应 LLVM 区域没有这条 Q source-to-phase store；其 phase store 只服务于
H/K/V-new producer。

### 5.3 Exact-LTO MIR

Z5B exact-LTO summary 的主要结果：

| MIR 阶段 | 结果 |
|:--|:--|
| pre-greedy | `av32_spill_saves=0`，`av64_spill_saves=0` |
| post-greedy | 无 spill virtual register |
| post-virtregrewriter | 无 spill virtual register |
| post-prologepilog | 无 spill virtual register |

这不是“没有大 accumulator”的证明，而是证明 Z5B 在当前 phase-separated
结构下没有触发 private spill。它也不把 virtual register 数量等同为硬件
AGPR 数量。

### 5.4 Final ISA / HSACO

Z5B final ISA 静态 lexical count：

| 指令类别 | Z5A | Z5B | 变化 |
|:--|--:|--:|--:|
| `global_load` | 192 | 192 | 0 |
| `global_store` | 16 | 16 | 0 |
| `ds_read` | 152 | 56 | -96 |
| `ds_write` | 240 | 144 | -96 |
| `v_mfma_f32_32x32x8_bf16` | 56 | 56 | 0 |
| `s_barrier` | 32 | 32 | 0 |
| `v_add` | 203 | 202 | -1 |
| `v_lshl_add` | 138 | 130 | -8 |

`mfma=56` 是静态 ISA 指令条数，不能直接写成动态 MFMA=56。动态 PMC
显示两臂都是 `160 MFMA/CTA`，说明 Z5B 没有通过删除数学工作取得收益。

HSACO 和 code-object metadata：

| 字段 | Z5A | Z5B |
|:--|:--|:--|
| HSACO SHA256 | `976857f75eca9cb1a8351277529b33e576686511a347d1c3e0ce0317cf4aaa65` | `979889c1a10ac53064cd1d2da91298b67e4ef7f2db3baadc171872cb56b0ed67` |
| code-object VGPR | 132 | 104 |
| code-object AGPR | 32 | 32 |
| code-object SGPR | 28 | 28 |
| LDS | 32768 B | 32768 B |
| private segment | 0 | 0 |
| VGPR spill | 0 | 0 |
| SGPR spill | 0 | 0 |
| workgroup | 256 | 256 |

HSACO hash 不同，说明 Z5B 不是运行时重命名或 benchmark alias；它生成了
不同机器代码。

## 6. T=2048 dynamic PMC

Z5A 和 Z5B 使用同一个 `rocprofv3 --kernel-trace --pmc` wrapper，均为
T=2048、WG256、512 CTA。动态数据来自 counter CSV，下面所有 `/CTA` 都是
动态总数除以实际 512 CTA，不是由静态 ISA 推出来的。

| metric | fixed Z2 | Z5A | Z5B | Z5B vs Z5A |
|:--|--:|--:|--:|--:|
| dynamic MFMA/CTA | 160 | 160 | 160 | 0 |
| dynamic VMEM/CTA | 928 | 672 | 672 | 0 |
| dynamic LDS/CTA | 928 | 1440 | 672 | -768 (-53.3%) |
| dynamic VALU/CTA | 11400 | 7136 | 7072 | -64 (-0.9%) |
| dynamic SALU/CTA | 1072 | 768 | 768 | 0 |
| code/profiler LDS | 16384 B | 32768 B | 32768 B | 0 |
| profiler VGPR field | 88 | 100 | 76 | -24 |
| profiler Accum_VGPR field | 32 | 76 | 100 | +24 |
| profiler SGPR field | 112 | 112 | 112 | 0 |
| Scratch | 0 | 0 | 0 | 0 |
| OccupancyPercent | 15.86 | 14.57 | 14.54 | -0.03 |

### 6.1 如何解释 VGPR/Accum_VGPR 字段

rocprof CSV 的 `VGPR_Count` 和 `Accum_VGPR_Count` 是 collector resource
字段；它们不能与 HSACO metadata 的 `VGPR=104/132`、`AGPR=32` 直接一一
替代。Z5B 的报告同时保留三种证据：

- HSACO metadata：Z5B `VGPR=104, AGPR=32`；
- rocprof resource row：Z5B `VGPR_Count=76, Accum_VGPR_Count=100`；
- MIR：无 spill save/reload。

因此，本轮只结论化为：Z5B 没有 spill，code-object VGPR 下降，AGPR metadata
不变；collector 的 VGPR/Accum 字段发生了资源分类重排。不能把 `100` 写成
“Z5B 使用了 100 个 AGPR”，也不能把 code-object `AGPR=32` 写成 profiler
`Accum_VGPR=32`。

### 6.2 机器工作归因

Z5B 相对 Z5A 的核心变化正好出现在 LDS：

```text
Z5A: 1440 LDS instructions / CTA
Z5B:  672 LDS instructions / CTA
```

同时：

- VMEM 不变，说明 Z5B 没有改变 Q/K/H/V-new 的 global IO；
- MFMA 不变，说明数学工作不变；
- SALU 不变，说明没有用新的标量索引链换掉 Q copy；
- VALU 略降，说明删除 Q phase copy 后也减少了少量 packet/layout 搬运；
- LDS allocation 不变，说明 Z5B 不是靠缩小 shared capacity 获益；
- occupancy 基本相同，说明没有因为 32 KiB LDS 发生新的 residency cliff。

这组数据与“删除 Q cache -> phase Q republish”这一单一变化吻合。

## 7. T=2048 paired body benchmark

原始工件：

```text
codex_qwen_bt64_stage6z_z5b_t2048_bench.json
```

计时契约：

- 5 个 independent fresh processes；
- 每 session 50 个 HIP-event samples；
- warmup=10；
- current HIP stream；
- no Graph；
- caller-owned preallocated output；
- native public call 只用于预先选定 WG256/BK32/BV64/stage2，之后计时 direct
  preallocated body；
- 每个 session 轮换四臂顺序：Z2、Z5A、Z5B、native。

| arm | median of session medians |
|:--|--:|
| fixed Z2 | `0.077555 ms` |
| Z5A | `0.066138 ms` |
| Z5B | `0.065618 ms` |
| native WG256 | `0.042643 ms` |

Z5B 的 paired 差值：

```text
Z5B - Z5A: -0.6605, -0.1800, -2.8650, +1.1420, -0.5610 us
Z5B - Z2 : -14.9020, -11.0770, -11.6580, -10.7350, -12.1580 us
```

结论：

- 相对 Z5A：4/5 session 更快，session median 中位数约快 `0.79%`；
- 相对 Z2：5/5 session 更快，约快 `15.39%`；
- 相对 native：Z5B 是 `1.538x`，仍有明显差距；
- Z5B 的绝对收益很小，但与动态 LDS 下降方向一致，因此不是只看一个
  noisy latency sample 的晋级。

## 8. T=8192 paired body benchmark

原始工件：

```text
codex_qwen_bt64_stage6z_z5b_t8192_bench.json
```

同样是 5 sessions、每 session 50 samples、warmup=10、current stream、no
Graph、caller-owned output。

| arm | median of session medians |
|:--|--:|
| fixed Z2 | `0.177143 ms` |
| Z5A | `0.160178 ms` |
| Z5B | `0.156473 ms` |
| native WG256 | `0.090755 ms` |

Z5B 相对 Z5A 的 paired 差值：

```text
-3.8055, -3.7050, -6.8300, -3.0645, -4.4065 us
```

5/5 session 均为正收益。相对 Z5A 约快 `2.31%`；相对 Z2 约快
`11.67%`。相对 native 的比值为约 `1.72x`。

### 8.1 endpoint slope

这里用 T=2048 和 T=8192 的 fresh body 中位数做两点 endpoint slope，单位为
`us/chunk`：

```text
slope = (latency_T8192 - latency_T2048) / (128 - 32)
```

| arm | T=2048 | T=8192 | endpoint slope |
|:--|--:|--:|--:|
| fixed Z2 | 0.077555 ms | 0.177143 ms | `1.037 us/chunk` |
| Z5A | 0.066138 ms | 0.160178 ms | `0.980 us/chunk` |
| Z5B | 0.065618 ms | 0.156473 ms | `0.946 us/chunk` |
| native | 0.042643 ms | 0.090755 ms | `0.501 us/chunk` |

Z5B 相对 Z5A 的 slope 下降约 `3.4%`；相对 fixed Z2 下降约 `8.8%`。
它没有消除 native 的主要 slope 差距，但说明 Q phase republish 是随 chunk
增长的真实机器工作，而不是只影响固定 launch intercept。

## 9. Native WG256 diagnostic 对照

native 工件来自已完成的同形状 WG256 capture：

```text
codex_qwen_bt64_stage6z_fixed_z2_vs_native_audit/
```

T=2048 native 的 dynamic PMC（512 CTA）为：

| metric | native total | native / CTA |
|:--|--:|--:|
| MFMA | 81920 | 160 |
| VMEM | 71680 | 140 |
| LDS | 245760 | 480 |
| VALU | 1728512 | 3376 |
| SALU | 337920 | 660 |
| Workgroup | 256 | 256 |
| profiler VGPR | 100 | 100 |
| profiler Accum_VGPR | 36 | 36 |
| profiler SGPR | 96 | 96 |
| Scratch | 0 | 0 |

native 的 LDS metadata 在该 rocprof collector 中显示为 0，不能据此判断
native 没有 LDS；其动态 `SQ_INSTS_LDS=480/CTA` 才是本表的 LDS 工作证据。

对比 Z5B：

| metric / CTA | Z5B | native | Z5B/native |
|:--|--:|--:|--:|
| MFMA | 160 | 160 | 1.00x |
| VMEM | 672 | 140 | 4.80x |
| LDS | 672 | 480 | 1.40x |
| VALU | 7072 | 3376 | 2.09x |
| SALU | 768 | 660 | 1.16x |

这说明 Z5B 的主要剩余差距已不是 MFMA 数量，也不是 Q republish。Z5B
已把 Z5A 的重复 LDS 工作删除，但仍比 native 多很多 VMEM 和约一半以上
VALU。下一步若继续 Stage 6Z，应从完整 operand ownership/数据流 ledger
中选择一个最大剩余 offender，而不是继续对 Q phase copy 做微调。

## 10. 晋级判断

预注册成功标准逐项检查：

| gate | 结果 |
|:--|:--|
| T=64/512/1024/2048/4096/8192/16384 correctness | PASS |
| BF16 byte-exact vs Z5A/Z2 | PASS |
| finite | PASS |
| caller-owned zero-V NaN output | PASS |
| MFMA=160/CTA | PASS |
| VMEM 不高于 Z5A 672/CTA | PASS，保持 672 |
| LDS 相对 Z5A 显著下降 | PASS，1440 -> 672 |
| 新的 VALU/SALU 爆炸 | PASS，无爆炸，VALU 略降，SALU不变 |
| scratch/private/spill | PASS，均为 0 |
| T=2048 稳定优于 Z5A | 按 5-session 方向 gate PASS，4/5 paired session 更快，约0.79%；小样本 bootstrap 95% CI 含 0 |
| T=8192 稳定优于 Z5A | PASS，5/5 paired session 更快，约2.31% |

因此选择：

```text
Case A:
删除 Q republish 后机器工作下降，T=2048 和 T=8192 均改善。
晋级 Z5B 为新的 isolated Stage 6Z research baseline。
```

需要保留一个统计上的边界：T=2048 的五个 cluster-level paired 差值均值为
`-0.625 us`，重采样 95% 区间约为 `[-1.887, +0.441] us`。所以这里的
“PASS”是本轮预注册的 5-session 方向 gate，不是已经足以支撑极小收益的
高置信度统计声明。T=8192 的五个差值全部为负，均值为 `-4.362 us`，同样
重采样区间约为 `[-5.620, -3.461] us`。LDS/ISA/PMC 的变化是确定性的，
T=2048 的 latency 收益则应理解为小幅、方向支持；后续若要把 Z5B 作为
长期性能基准，仍应补更多独立 session。

这不是 production-ready 结论，原因是：

1. Z5B 仍是 isolated chunk-o；
2. 它没有接入 X2 full graph；
3. 它仍约 `1.54x` native（T=2048）和 `1.72x` native（T=8192）；
4. native 的更低 VMEM/VALU 仍未解释和复现；
5. Z5B 仍占用 32 KiB LDS，code-object occupancy 略低于 fixed Z2；
6. 初始 MLIR printer 工件仍需另行修复，不能声称本轮拿到了完整的
   high-level MLIR A/B。

## 11. 复现命令和原始工件

### correctness

```bash
python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_native_chunko_stage6z_z5b.py -s
```

### paired body

```bash
python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_stage6z_z5b.py \
  --T 2048 --sessions 5 --warmup 10 --repeat 50 \
  --out codex_qwen_bt64_stage6z_z5b_t2048_bench.json

python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_stage6z_z5b.py \
  --T 8192 --sessions 5 --warmup 10 --repeat 50 \
  --out codex_qwen_bt64_stage6z_z5b_t8192_bench.json
```

### PMC

```bash
python3 test/examples/linear_attention/vllm_compare/profile_qwen_gdn_bt64_native_chunko_stage6z_z5b.py \
  --arm z5b --T 2048 --warmup 2 --repeat 5 \
  --out-dir codex_qwen_bt64_stage6z_z5b_rocprof
```

### machine capture

```bash
python3 test/examples/linear_attention/vllm_compare/dump_qwen_gdn_bt64_stage6z_z5b_machine_artifacts.py \
  --variant z5b --T 2048 \
  --out-dir codex_qwen_bt64_stage6z_z5b_machine_stage1 \
  --skip-initial-mlir
```

原始工件：

```text
codex_qwen_bt64_stage6z_z5b_t2048_bench.json
codex_qwen_bt64_stage6z_z5b_t8192_bench.json
codex_qwen_bt64_stage6z_z5b_rocprof/
codex_qwen_bt64_stage6z_z5b_machine_stage1/
```

## 12. 下一步边界

本轮唯一允许的 Z5B 任务已经完成，不应再做 Z5C 或重复的 Q phase copy
变体。Z5B 已经证明：在当前 AveLang source-level shared/view 表达能力下，
可以把 dedicated Q cache 直接接到 MFMA consumer，并真实删除重复 LDS
materialization。

如果继续追 native 差距，下一轮必须重新做完整的 per-operand machine-work
ledger，重点解释 Z5B 相对 native 的 `672 vs 140 VMEM/CTA` 和
`7072 vs 3376 VALU/CTA`。不能把剩余差距继续归因到已经删除的 Q republish，
也不能因为 isolated Z5B 变快就接入 X2。
