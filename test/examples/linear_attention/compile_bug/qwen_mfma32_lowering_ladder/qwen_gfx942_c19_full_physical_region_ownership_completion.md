# Qwen gfx942 C19-FPRO：Full Physical Region Ownership Completion

## 1. 结论先行

本轮完成了 C18-FPR 之后唯一允许的完整 ownership candidate：

`C19-FPRO = FullPhysicalRegionPlan full physical region ownership`

最终决策为：

```text
C19_FPRO_FULL_OWNERSHIP_GO_FOR_PERFORMANCE
```

这个结论只表示 C19 通过了本轮的 full-ownership、actual allocation、correctness
和 diagnostic machine gate，允许下一轮做正式性能测试。它**不是** production promotion，
也不是已经完成的 Eager public API 排名。本轮严格没有运行正式 7-session body benchmark、
没有运行 Public Eager、没有修改 selector、recurrence HSACO 或 production dispatch。

C19 的关键结果：

| 项目 | C19 结果 |
|:--|--:|
| Q/H/K/V ownership | 全部由 `FullPhysicalRegionPlan` 接管 |
| Q producer | 每个逻辑 Q K32 stage 一次，单 producer，双 consumer（Q@H/Q@K） |
| V ownership | 保留并接续 C18 的 plan-owned V producer/score@V consumer |
| shared arena | 一个 `384 x 32 BF16` arena |
| planned LDS | `24576 B` |
| actual HSACO LDS | `24576 B`，不是 C18 的 `32768 B` |
| correctness | T=64/128/512/1024/2048/4096/8192/16384 全部 BF16 byte-exact、finite |
| edge | T=64/8192/16384 caller-owned + zero-V-new + NaN-prefill 全部通过 |
| exact code object | VGPR=136、AGPR=48、SGPR=28、private=0、spill=0 |
| T=2048 diagnostic trace | `41.341 us`，仅作 profiler 诊断，不作正式排名 |
| T=2048 dynamic PMC / CTA | MFMA=160、VMEM=304、LDS=688、VALU=6418、SALU=610 |

C19 的下一步只有一个：在保持 C19 机器图不变的前提下，做正式的
caller-owned isolated body 和 Eager public API benchmark。不要在 C19 后再拆出
C19-Q、C19-H、C19-K、C19-V 或 packet/barrier/LDS 局部变体。

---

## 2. 本轮范围和严格冻结项

C19 是 C18-FPR 的一次性 full-region completion，不是 C18-Q/H/K 三个独立实验。
以下内容保持不变：

- gfx942、wave64；
- BT64、BV64、BK32；
- WG256，每个 chunk-head 两个 CTA；
- `v_mfma_f32_32x32x8_bf16` 和 K32 accumulation order；
- Q/K/H/V-new/output BF16，g 和 accumulator 的 FP32 语义；
- causal mask、global layout、caller-owned output contract；
- recurrence、allocator/RA、X2、production selector；
- 不使用 ds_bpermute generic transpose；
- 不做 packet width、LDS swizzle、barrier/waitcnt、scheduler、g-residency 或
  pipeline sweep；
- 不运行正式性能排名。

C19 只做一件事：让同一个 `FullPhysicalRegionPlan` 同时拥有 Q/H/K/V 的 producer、
shared placement、consumer feeding、phase lifetime、shared allocation 和 phase
barrier boundary。

源文件是：

[`qwen_gdn_bt64_native_chunko_stage6z_c19_full_physical_region.py`](../../vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_c19_full_physical_region.py)

它仍然只使用已有的 logical block-dot source operation；没有新建 Qwen/chunk-o
专用 public op。

---

## 3. C18 的冻结起点

本节只引用 C18 已完成的事实，不重新审计 C18。

C18 已经证明：

1. `FullPhysicalRegionPlan` 能真正改变 V producer、shared placement 和 score@V
   consumer 的机器图。
2. C18 与 P2 的 LLVM、MIR、ISA、HSACO 不同。
3. C18 全长度 correctness 通过，且没有 spill/private。
4. C18 相对 P2 的 T=2048 diagnostic PMC 为：

| 指标 / CTA | P2 | C18 |
|:--|--:|--:|
| MFMA | 160 | 160 |
| VMEM | 448 | 416 |
| LDS | 592 | 800 |
| VALU | 8474 | 8010 |
| SALU | 780 | 776 |
| code-object VGPR/AGPR | 132/48 | 116/32 |
| LDS | 32768 B | 32768 B |
| diagnostic trace | 44.947 us | 43.745 us |

C18 没有完成的部分是：

- Q 仍由 Z5B dedicated Q cache source owner 控制；
- H/K producer 和 consumer 仍依赖旧 P2 full-scope machinery；
- shared lifetime 只有 metadata 影响，没有改变物理 allocation；
- actual LDS 仍然是 32768 B。

因此 C18 的决策是 `NO-GO`，C19 的任务就是一次性删除这些剩余 ownership
依赖。C18 的完整原始报告是：

[`qwen_gfx942_c18_full_physical_region_ownership_lowering.md`](qwen_gfx942_c18_full_physical_region_ownership_lowering.md)

---

## 4. Legacy dependency map

机械依赖表已保存为：

[`stage6z_c19_legacy_dependency_map.json`](stage6z_c19_legacy_dependency_map.json)

它按 component 记录了 current C18 owner、exact function/pattern、target C19 owner、
replacement status。下面是人类可读的摘要：

| component | C18 legacy owner / pattern | C19 owner | 状态 |
|:--|:--|:--|:--|
| Q global producer | C18 source 的 `q_cache[...]` loop | `emitC19FullPhysicalProducer(role=Q)` | 完成 |
| Q shared placement | `2 * Q_CACHE_ROWS` 的 512-row arena | plan 的 Q rows `0..255` | 完成 |
| Q@H | C18/P2 full-scope operand construction | C19 role=H packed LDS consumer | 完成 |
| Q@K | C18/P2 source-half offset branch | C19 role=K packed LDS consumer | 完成 |
| H producer | old `emitFullScopeProducer`/P2 path | `emitC19FullPhysicalProducer(role=H)` | 完成 |
| H placement | C18 `phaseBase=320` | C19 `c19PhaseBase=256` | 完成 |
| H consumer | P2 fragment feeding | C19 role=H first-class consumer | 完成 |
| K producer | old P2 source-half/stage formula | `emitC19FullPhysicalProducer(role=K)` | 完成 |
| K placement | C18 historical source-half placement | C19 phase band `256..319` | 完成 |
| K consumer | P2 fragment feeding | C19 role=K first-class consumer | 完成 |
| V producer/score@V | C18 FullPhysicalRegionPlan | C19 role=V，offset-only adaptation | 保留并回归通过 |
| score shared | C18 historical score/V region | plan `scoreBase=0`、`scoreVBase=128` | 完成 |
| backing allocation | C18 512 x 32 BF16 memref | C19 384 x 32 BF16 arena | 完成 |
| lifetime/reuse | C18 metadata-only | plan lifetime + actual 24 KiB shape | 完成 |
| barriers | C18 mixed source/legacy boundaries | C19 plan-owned producer/consumer boundary | 完成 |

这里的“完成”不是指所有 source-level `al.syncthreads()` 被删除。跨 wave RAW/WAR
仍然需要同步；完成的含义是 C19 path 不再静默调用旧 owner，也不再依赖旧 owner
计算 physical row/word/allocation。必要的同步仍作为计划 phase boundary 的机器
实现保留。

---

## 5. FullPhysicalRegionPlan 的实际内容

### 5.1 Shared arena

C19 plan 使用如下固定区域：

```text
rows 0..255   : Q physical cache，四个 [64,32] K32 stage
rows 256..319 : H/K source phase band
rows 0..127   : Q last-use 之后复用为 score half 0
rows 128..255  : Q last-use 之后复用为 score-V band
```

物理 backing 只有一个：

```text
384 rows * 32 columns * sizeof(bfloat16) = 384 * 32 * 2 = 24576 B
```

C19 source 只出现一次：

```python
shared = al.make_shared((C19_SHARED_ROWS, BK), al.bf16)
```

没有第二个完整 Q buffer、没有 `q_cache` alias、没有 `phase_vec`、没有 private
Q array。物理 allocation 结果在 code object 中验证为 24576 B。

### 5.2 Q：single producer，dual consumer

Q 的逻辑 mapping 复用已经在 C16 关闭的 exact mapping：

```text
blocked2
sizePerThread = [1, 8]
threadsPerWave = [16, 4]
wavesPerCTA = [4, 1]
order = [1, 0]
shared vector = 4
perPhase = 2
maxPhase = 8
dot operand opIdx = 0
kWidth = 4
```

但 C16 mapping 只是 physical recipe utility，C19 的 ownership 是新的：

```text
real Q global ABI
  -> emitC19FullPhysicalProducer(role=Q)
  -> plan-owned Q rows 0..255
  -> Q@H
  -> Q@K source-half 0 and 1
```

每个逻辑 Q K32 stage 的 global producer pass 是 1。Q cache 必须跨过 Q@H、
Q@K source-half=0、Q@K source-half=1 的最后一次消费，之后才可以复用为 score
storage。

### 5.3 H：真实 ABI 到 plan-owned producer/consumer

H 由 rank-5 BF16 ABI 进入 C19。`emitC19FullPhysicalProducer(role=H)` 生成
BF16x8 packet producer，写入 C19 H/K phase band；C19 role=H consumer 直接从
plan row/word 公式构造 packed LDS operand，再进入 MFMA32。

没有调用旧 `emitFullScopeProducer` 作为 H owner，也没有使用旧 generic transpose
或第二块完整 transpose LDS tile。C16 的 H physical formula 可以被复用，因为
它是 encoding utility；但 owner、shared slot 和 consumer branch 都由 C19 plan
决定。

### 5.4 K：真实 ABI、合法 packet 和 wave ownership

K 由 rank-4 BF16 ABI进入 C19。producer 的关键 ownership 是：

```text
wave       = tid >> 6
lane       = tid & 63
wavePair   = wave >> 1
linear     = wavePair * 64 + lane
```

这里 `wavePair=wave>>1` 是 C19 修正后的公式；此前错误的 `wave>>2` 会漏掉一组
K producer ownership，已不在 C19 源码中。

K 仍保持真实合法的 BF16x8 producer 和 C16 的 physical utility recipe，但不会把
K diagnostic scalar gather 当成 performance implementation。K producer、K shared
placement 和 Q@K consumer 都通过 C19 plan role=K 进入。

### 5.5 V：C18 正向结果的保持

V 不重新设计。C19 只把 C18 的 V producer、score-V shared placement 和 score@V
consumer 放入同一个 384-row arena，并调整 offset：

```text
V producer -> score-V rows 128..255 -> score@V -> MFMA32
```

`V_source_level_consumer_owned_by_c19=true`，V mapping、packet width、rotating
shared、transform 和数学均冻结。对应回归证据在
[`stage6z_c19_v_ownership_regression.json`](stage6z_c19_v_ownership_regression.json)。

### 5.6 Score lifetime 修复

C19 第一次 correctness 反例暴露了一个真实的 lifetime alias：如果 score half 0
在 source-half 0 后立即写入 rows `0..127`，它会覆盖 Q cache，而 source-half 1
还需要读取同一 Q cache。错误首先出现在 token 32、34、... 的输出路径。

修复没有创建新候选，而是完成 C19 的必要 phase ordering：

```text
完成 score source-half 0 -> 保留 score_acc0
完成 score source-half 1 -> 保留 score_acc1
两个 Q@K source-half 都完成
-> 才写 score half 0/1 到可复用 rows
-> score@V
```

这解释了 C19 source 中同时存在 `score_acc0` 和 `score_acc1`。它不是把
`inter_acc + score_acc0 + score_acc1` 融进同一个 K32 loop 的新优化，也不是另一个
candidate；它是保持 Q dual-consumer lifetime 正确所必需的 C19 phase commit。

---

## 6. Legacy owner hard-fail

C19 不允许 mixed architecture。

在 compiler pass 中，C19 enabled 时：

1. 如果发现 `c18.full_physical_region` 或 `c17.full_physical_plan`，直接
   `emitError` 并使 pass failure；
2. 如果没有 plan-owned Q/H/K/V role，直接失败；
3. Q producer owner 不是恰好一个时失败；
4. V producer owner 不是恰好一个时失败；
5. C19 需要 specialized block-dot 和 first-class MFMA operand，否则失败；
6. 每个 C19 logical operation 都被标记为
   `c19.full_physical_region`、`c19.owner=FullPhysicalRegionPlan`、
   `c19.legacy_owner=false`。

静态测试同时检查 C19 source：

- 没有 `q_cache`；
- 没有 `phase_vec`；
- 没有 `c18` owner path；
- 只有一个 C19 shared allocation；
- Q/H/K/V logical block-dot 均存在。

因此 C19 不会出现“Q 已切换但 H/K 仍偷偷走 P2”的静默 fallback。

---

## 7. C18 的 32768 B 根因与 C19 的实际 allocation

完整机器可读分析见：

[`stage6z_c19_shared_allocation_root_cause.json`](stage6z_c19_shared_allocation_root_cause.json)

### 7.1 C18 为什么是 32768 B

C18 source 的 backing shape 是：

```python
al.make_shared((2 * Q_CACHE_ROWS, BK), al.bf16)
```

其中 `Q_CACHE_ROWS=256`、`BK=32`，所以 GPU outlining 看到的是：

```text
512 rows * 32 columns * 2 bytes = 32768 B
```

C18 的 lifetime attr 没有重写 memref shape，也没有让 workgroup allocation pass
把 Q cache、H/K phase、score/V band放进同一 physical base。因此 C18 的
`planned reusable bytes=24576` 只是 metadata 目标，不能改变 final HSACO。

### 7.2 C19 为什么实际变成 24576 B

C19 直接使用与 plan 一致的一个 `384 x 32 BF16` shared arena：

```text
384 * 32 * 2 = 24576 B
```

这一次 allocation shape 本身已经反映了 lifetime coalescing，LLVM workgroup
global 没有多出第二个 backing arena，最终 HSACO `.group_segment_fixed_size`
也是 `24576`。本轮不需要 generic allocator redesign，也没有添加 autotuner 或
全局 graph coloring。

因此 allocation gate 是真实 PASS，而不是“有 reuse attr 所以宣称成功”。

---

## 8. Machine artifact 和 pipeline identity

机器工件目录：

`codex_qwen_gfx942_c19_full_physical_region_t2048/machine/`

包括：

- runtime MLIR snapshots；
- `lowered_llvm.ll`；
- `preopt_llvm.ll` / `postopt_llvm.ll`；
- `llc_mir/stop_after_*.mir`；
- exact full-LTO `kernel_section_00.mir` 到 `kernel_section_19.mir`；
- `final_isa.s`；
- `c19_full_physical_region.hsaco`；
- `machine_evidence.json`；
- link argv 和 replay output。

### 8.1 Hash 和资源

| artifact | SHA256 / value |
|:--|:--|
| C19 source | `e3d5c81800fa8f8e1e4c8ca2f9148c0befff46c5bfcfc8e49c4531fd5ee92975` |
| lowered LLVM | `a8579c7ad73de7daf3f0e6ae3b18bf5914c5c5807e60b9f7089e24bc90395407` |
| final ISA | `c9279f37f79665db2173677e1c7ce3d5ff6294463cec1df41145618d4cfe5ddb` |
| HSACO | `14ffa78200ff995448c8abbe1c5484375aae1a2f75c8412a937b33d4aa628907` |
| exact LTO return code | `0` |

上表的 source SHA、LLVM/ISA SHA 是机器生成的值；报告中的路径和 hash 同时保存在
[`stage6z_c19_machine_evidence.json`](stage6z_c19_machine_evidence.json)。

注意：source SHA 的完整值以 JSON 为准。报告正文中的长 hash 仅作可读索引，
复现时应使用 JSON 和 `sha256sum` 重新核对。

### 8.2 Static ISA

| static final ISA family | C19 |
|:--|--:|
| MFMA32 | 56 |
| `ds_read_b64` | 112 |
| `ds_write_b128` | 16 |
| `ds_write_b16` | 64 |
| `ds_write_b16_d16_hi` | 32 |
| `global_load_dwordx4` | 24 |
| `global_load_dword` | 80 |
| global load total | 104 |
| `global_store_short_d16_hi` | 16 |
| `ds_bpermute_b32` | 0 |
| `s_waitcnt` | 147 |
| `s_barrier` | 46 |

这些是 final ISA 的 lexical counts。它们不是 dynamic PMC，也不是 memory bytes。
例如 `global_load_dwordx4` 和 `global_load_dword` 的 byte width 不同，不能把
104 直接解释成 104 个 transaction bytes。

### 8.3 Exact-LTO MIR

exact LTO 中：

- pre-greedy：`kernel_section_00.mir`；
- post-greedy：`kernel_section_01.mir`；
- post-virtregrewriter：`kernel_section_08.mir`；
- post-prologepilog：`kernel_section_09.mir`；
- `SI_SPILL_AV32/AV64` save/reload：0；
- private segment：0。

C19 的 code-object resource 是 `VGPR=136、AGPR=48、SGPR=28`；rocprof resource
metadata 另报 `VGPR_Count=96、Accum_VGPR_Count=80、SGPR_Count=112`。二者来源
和语义不同，不能互相替代。本报告把 code object 和 profiler resource 分开记录。

初次尝试打印 initial MLIR 时触发了 compiler printer segfault，因此该次 capture
显式使用 `--skip-initial-mlir --skip-pre-lto-assembly` 重跑。这个限制没有被隐藏：
runtime JIT dump 的 MLIR snapshots、lowered LLVM、llc MIR、exact-LTO MIR、ISA 和
HSACO 均已保存；初始 printer 的失败只影响那一份初始文本打印，不影响 final
machine graph 审计。

---

## 9. Correctness gate

自动回归入口：

[`test_qwen_gdn_bt64_stage6z_c19_runtime.py`](../../vllm_compare/test_qwen_gdn_bt64_stage6z_c19_runtime.py)

执行命令：

```bash
docker exec ljd_qwen_vllm_avelang_rocm722 bash -lc \
  'cd /workspace/project/avelang && \
   PYTHONDONTWRITEBYTECODE=1 \
   PYTHONPATH=/workspace/project/avelang/build-vllm-rocm722/python:/workspace/project/avelang/python \
   /opt/venv/bin/python3 -m pytest -q \
   test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_stage6z_c19_full_physical_region.py \
   test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_stage6z_c19_runtime.py -s'
```

结果：

```text
14 passed in 13.09s
```

### 9.1 Base matrix

| T | C19 vs C18 BF16 | finite | max abs |
|--:|:--:|:--:|--:|
| 64 | byte-exact | pass | 0 |
| 128 | byte-exact | pass | 0 |
| 512 | byte-exact | pass | 0 |
| 1024 | byte-exact | pass | 0 |
| 2048 | byte-exact | pass | 0 |
| 4096 | byte-exact | pass | 0 |
| 8192 | byte-exact | pass | 0 |
| 16384 | byte-exact | pass | 0 |

C18 已经通过 frozen Z5B reference chain，因此 C19 与 C18 的 byte-exact 保持了
这条冻结 reference chain。但这里不把 C19/C18 equality 误写成重新运行了一个独立
数学 oracle；这个边界在 `stage6z_c19_full_correctness.json` 中明确记录。

### 9.2 Caller-owned edge matrix

| T | zero V-new | NaN-prefilled caller-owned output | finite | byte-exact |
|--:|:--:|:--:|:--:|:--:|
| 64 | pass | pass | pass | pass |
| 8192 | pass | pass | pass | pass |
| 16384 | pass | pass | pass | pass |

这三组检查确认 C19 真的覆盖 caller-owned output，没有用 output 初值掩盖边界错误。

---

## 10. T=2048 dynamic PMC

本轮只有在 full ownership、actual allocation 和 correctness 都通过后才采集 PMC。
原始 CSV 在：

`codex_qwen_gfx942_c19_full_physical_region_t2048/pmc/`

采集方式是 current HIP stream、no Graph、warmup=2、repeat=5、rocprof kernel trace
加 `SQ_INSTS_MFMA/VALU/SALU/VMEM/LDS` 和 `OccupancyPercent`。

### 10.1 C19 raw and per-CTA

实际 grid work-items 为 131072，WG256，所以 CTA 数为 512。raw median 除以 512
得到 per-CTA issue count：

| metric | raw median | per CTA |
|:--|--:|--:|
| MFMA | 81920 | 160 |
| VMEM | 155648 | 304 |
| LDS | 352256 | 688 |
| VALU | 3286016 | 6418 |
| SALU | 312320 | 610 |
| OccupancyPercent | 15.054814 | not divided |
| trace median | 41.341 us | not additive |

`SQ_INSTS_*` 是动态 issue counts，不是 static lexical count。VMEM/LDS counter 也
不能直接当成访问字节数。

### 10.2 Frozen comparison

| arm | MFMA/CTA | VMEM/CTA | LDS/CTA | VALU/CTA | SALU/CTA | source of value |
|:--|--:|--:|--:|--:|--:|:--|
| Z5B | 160 | 672 | 672 | 7072 | 768 | frozen Z5B PMC |
| P2 | 160 | 448 | 592 | 8474 | 780 | C18 frozen capture |
| C18 | 160 | 416 | 800 | 8010 | 776 | C18 fresh capture |
| C19 | 160 | 304 | 688 | 6418 | 610 | this task |
| native WG256 | 160 | 140 | 480 | 3376 | same-shape diagnostic |

C19 相对 C18：

- VMEM `416 -> 304`，下降 `26.92%`；
- LDS `800 -> 688`，下降 `14.00%`；
- VALU `8010 -> 6418`，下降 `19.88%`；
- SALU `776 -> 610`，下降 `21.39%`；
- MFMA 保持 `160`，数学工作没有减少。

C19 距离 native WG256 的倍数为：

| metric | C19 / native |
|:--|--:|
| VMEM | `2.171x` |
| LDS | `1.433x` |
| VALU | `1.901x` |
| SALU | `0.924x` |

这说明 C19 不是“所有机器工作已经等于 native”，但整体已从 C18 的
`416/800/8010/776` 向 native 的 `140/480/3376/660` 移动。最明显的收益来自
完整 producer/consumer ownership 和 24 KiB arena，而不是减少 MFMA。

### 10.3 Diagnostic trace 的边界

C19 trace median 是 `41.341 us`，C18 记录为 `43.745 us`，但二者不是正式
body benchmark 的替代品。trace 包含 profiler 的观测扰动，且本轮预先禁止 7-session
正式性能。因此本报告只使用它证明“机器图没有退化到 C18 的旧工作量”，不使用
`41.341 us` 宣布 C19 性能晋级。

---

## 11. Regression 和 build 状态

机器可读汇总在：

[`stage6z_c19_regression_results.json`](stage6z_c19_regression_results.json)

本轮实际完成：

| 检查 | 结果 |
|:--|:--|
| C19 source static ownership contract | pass |
| compiler legacy-owner hard-fail contract | pass |
| C19 correctness base matrix | pass |
| C19 caller-owned edge matrix | pass |
| exact-LTO replay | return code 0 |
| spill/private check | pass, 0/0 |
| C19 PMC capture | pass |
| JSON schema/syntax | pass |
| production selector | unchanged |
| external recurrence HSACO | unchanged |
| Public Eager | intentionally not run |
| formal 7-session performance | intentionally not run |

C13/C14/C15/C16/C17 的数值回归沿用前序冻结报告；C19 没有修改它们的 oracle，
也没有把历史 PASS 伪装成当前重新运行。C18 compatibility 则由本轮 14 个 runtime
tests 和 static ownership test 覆盖。

---

## 12. 十四个问题的直接回答

### 1. C18 中 Q/H/K 还依赖哪些 legacy owner？

- Q：C18 source 的 Z5B dedicated Q cache loop。
- H：old `emitFullScopeProducer` 和 P2 full-scope consumer formula。
- K：old P2 source-half/stage producer formula和 consumer feeding。
- 另外 C18 backing memref 仍然是旧 512-row allocation。

### 2. C19 是否删除这些 legacy ownership？

是。C19 compiler pass 对 C18/C17 owner attr 建立 hard failure；C19 source 没有
`q_cache`、`phase_vec` 或 C18 owner path。utility formula 可以复用，ownership
不能复用。

### 3. Q 是否 plan-owned single producer dual consumer？

是。Q producer 由 C19 plan 生成，global Q 每个 K32 stage 一次，Q cache 后续同时
服务 Q@H 与 Q@K，且 source-half=1 完成前不复用 Q rows。

### 4. H 是否完全脱离 P2 producer/consumer？

是。H producer 和 role=H packed LDS/MFMA32 consumer均由 C19 plan branch 生成。

### 5. K 是否完全脱离 P2 producer/consumer？

是。K producer 的 ownership 和 row/word feeding均由 C19 role=K branch生成。

### 6. V 是否保持 C18 full-region ownership？

是。V 只做 C19 arena offset adaptation，C18 的 V mapping 和 score@V ownership
保持不变。

### 7. C18 actual 32768 B 的原因？

C18 source memref 是 `512 x 32 BF16`，GPU allocation pass按物理 shape分配；
metadata-only lifetime 没有改变它。

### 8. C19 lifetime reuse 是否真正改变 LLVM/HSACO LDS？

是。C19 LLVM/HSACO只有一个 384-row arena，code object group segment 从 32768 B
变成 24576 B。

### 9. 最终 actual LDS 是多少？

`24576 B`。

### 10. C19 LLVM/MIR/ISA 相对 C18/P2 如何变化？

三层均有真实差异，且不是只换 symbol/hash：C19 有 c19 role-specific producer/
consumer lowering、Q/H/K/V physical row变化、score delayed commit 和 24 KiB
workgroup allocation。C19 final ISA 的静态 counts 和 exact-LTO files 已保存。

### 11. full correctness 是否全部 PASS？

本轮执行的 base 全长度和 T64/T8192/T16384 edge 全部 PASS，`14 passed`。

### 12. code-object resources？

`VGPR=136、AGPR=48、SGPR=28、private=0、VGPR spill=0、SGPR spill=0、LDS=24576 B`。
Profiler 的 `VGPR_Count=96、Accum_VGPR_Count=80、SGPR_Count=112` 单独记录，
不能和 code object 字段混写。

### 13. T2048 PMC 相对 Z5B/P2/C18/native？

见第 10 节。C19 为 `160/304/688/6418/610`，明显优于 C18 的
`160/416/800/8010/776`，但仍高于 native 的 VMEM/LDS/VALU。

### 14. 是否达到 `C19_FPRO_FULL_OWNERSHIP_GO_FOR_PERFORMANCE`？

是。ownership、allocation、correctness、machine-distinct、no spill/private 和
PMC movement 全部满足。本结论只开放正式性能阶段，不开放 production。

---

## 13. Case decision

| Case | 条件 | C19 |
|:--|:--|:--|
| A | full ownership + actual allocation + correctness + no spill/private + PMC 向 native 移动 | **满足** |
| B | 不能脱离 legacy owner 保持 correctness | 不满足 |
| C | lifetime 仍不能控制 actual allocation | 不满足 |
| D | ownership/allocation/correctness成功但机器工作不优于 C18/P2 | 不满足 |

最终状态：

```text
C19_FPRO_FULL_OWNERSHIP_GO_FOR_PERFORMANCE
```

后续正式性能阶段的输入必须冻结为当前 C19 source、current compiler build、
current BF16 ABI 和 current WG256。下一轮才能测：

- caller-owned isolated body，T=512/1024/2048/4096/8192/16384；
- current stream、no Graph、fresh process、paired order；
- C19 对 Z5B/P2/native selected WG256；
- 最终需要时再做 Eager public API。

本轮到此停止，不自动开始上述 benchmark。

---

## 14. Artifact index

### C19 机器可读文件

- [`stage6z_c19_legacy_dependency_map.json`](stage6z_c19_legacy_dependency_map.json)
- [`stage6z_c19_full_physical_region.json`](stage6z_c19_full_physical_region.json)
- [`stage6z_c19_q_ownership.json`](stage6z_c19_q_ownership.json)
- [`stage6z_c19_h_ownership.json`](stage6z_c19_h_ownership.json)
- [`stage6z_c19_k_ownership.json`](stage6z_c19_k_ownership.json)
- [`stage6z_c19_v_ownership_regression.json`](stage6z_c19_v_ownership_regression.json)
- [`stage6z_c19_shared_allocation_root_cause.json`](stage6z_c19_shared_allocation_root_cause.json)
- [`stage6z_c19_shared_lifetime_machine.json`](stage6z_c19_shared_lifetime_machine.json)
- [`stage6z_c19_full_correctness.json`](stage6z_c19_full_correctness.json)
- [`stage6z_c19_machine_evidence.json`](stage6z_c19_machine_evidence.json)
- [`stage6z_c19_pmc_t2048.json`](stage6z_c19_pmc_t2048.json)
- [`stage6z_c19_regression_results.json`](stage6z_c19_regression_results.json)

### Source and tests

- [`qwen_gdn_bt64_native_chunko_stage6z_c19_full_physical_region.py`](../../vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_c19_full_physical_region.py)
- [`test_qwen_gdn_bt64_stage6z_c19_full_physical_region.py`](../../vllm_compare/test_qwen_gdn_bt64_stage6z_c19_full_physical_region.py)
- [`test_qwen_gdn_bt64_stage6z_c19_runtime.py`](../../vllm_compare/test_qwen_gdn_bt64_stage6z_c19_runtime.py)
- [`dump_qwen_gdn_bt64_stage6z_c19_machine_artifacts.py`](../../vllm_compare/dump_qwen_gdn_bt64_stage6z_c19_machine_artifacts.py)
- [`bench_qwen_gdn_bt64_stage6z_c19_machine_pmc.py`](../../vllm_compare/bench_qwen_gdn_bt64_stage6z_c19_machine_pmc.py)

### Capture directory

`codex_qwen_gfx942_c19_full_physical_region_t2048/`

其中 `machine/` 保存 LLVM、MIR、ISA、HSACO，`pmc/` 保存 raw rocprof CSV。初始
MLIR printer segfault 的限制和跳过参数已记录在 machine evidence JSON，不影响
这些 final artifacts 的复核。
