# Qwen gfx942 BT64 Stage 6Z Z6G：g Residency Stable/Ideal 实验

## 1. 实验结论

本轮从冻结的 Z5B `direct-Q-cache-consumer` 直接分叉，分别实现了两个互不
叠加的 g-residency 版本：

- **Z6G-S stable**：每个 CTA 将当前 64-token 的 FP32 `g` tile 写入一块
  简单的 CTA-local shared cache，后续 score target/source 和 final scaling
  只从该 cache 读取。
- **Z6G-I ideal**：同样只建立一次逻辑 g tile，但 producer 使用合法的
  `raw_buffer_load_x4`，每个 token 读取连续的四个 value-head FP32 值，再
  选出当前 value head 写入同一个 shared cache。

两条版本都通过了完整 correctness gate，但两条都没有带来真实性能收益。
在 T=2048、8192、16384 的 fresh-process、current-stream、no-Graph、caller-owned
body 测试中，Z6G-S 和 Z6G-I 都稳定慢于 Z5B。最终保持 **Z5B 为当前
Stage 6Z isolated research baseline**，不晋级 Z6G-S/Z6G-I，不接入 X2、selector
或 production。

这里的“不晋级”是性能结论，不是因为 VGPR、AGPR、LDS 或 occupancy 预先触发了
No-Go。两条 correctness 通过的 arm 都确实进入了性能测试；资源只用于解释机器
图和延迟结果。

## 2. 冻结边界

本轮没有修改：

- `BT64/BV64/BK32`；
- `WG256`、两个 CTA/chunk-head 的 ownership；
- BF16 Q/K/H/V-new/output ABI 和 FP32 g；
- MFMA32 geometry、K32 accumulation order、causal mask 和数学；
- Z5B Q full-cache residency 以及 direct-Q consumer；
- K、H、V-new、output 的 producer；
- allocator/RA、X2、selector、production dispatch。

新增的只有一块 `64 x f32` 的 `g_cache`，逻辑大小为 `256 B`。两条 arm 使用同一
个高层 kernel body，通过 `g_mode` constexpr 分别编译，绝不在同一次 kernel 中
叠加 stable 和 ideal 两套路径。

| 项目 | Z5B | Z6G-S | Z6G-I |
|:--|:--|:--|:--|
| Q/K/H/V-new/output | 冻结 | 相同 | 相同 |
| g producer | score/final 多个 consumer role | 一个 scalar tile fill | 一个 x4 packet tile fill |
| g cache | 无 | 64 x FP32 shared | 64 x FP32 shared |
| g consumer | global g pointer | shared g cache | shared g cache |
| WG | 256 | 256 | 256 |
| CTA/chunk-head | 2 | 2 | 2 |
| MFMA geometry | MFMA32 | MFMA32 | MFMA32 |
| 版本叠加 | 无 | 不与 Z6G-I 叠加 | 不与 Z6G-S 叠加 |

## 3. 高层源码变化

实验源码为：

`test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z6g_g_residency.py`

关键位置如下：

| 源码位置 | 内容 |
|:--|:--|
| 99-105 | 复用 Z5B 的 32 KiB BF16 shared 区，并新增 `g_cache = al.make_shared((BT,), al.f32)` |
| 109-119 | 完全保留 Z5B Q-cache producer，不在 Q/K/H/V-new producer 中混入 g |
| 121-140 | Z6G-S scalar FP32 fill；Z6G-I `raw_buffer_load_x4` fill |
| 142-167 | 保留 Z5B Phase A、H producer 和 inter accumulator 顺序 |
| 169-174 | 从 g cache 得到 target g，供后续 score/final 使用 |
| 175-208 | 保留两个串行 score half；source g 从 g cache 读取 |
| 210-236 | 保留 V-new、intra MFMA 和 BF16 output，final scaling 使用 cache 中的 target g |

Z6G-S 的 producer 是：

```text
if tid < 64:
    g_cache[tid] = g[chunk_start + tid, value_head_idx]
barrier
```

Z6G-I 的 producer 是：

```text
if tid < 64:
    packet = raw_buffer_load_x4(g, (chunk_start + tid, head_base))
    g_cache[tid] = packet[head_in_packet]
barrier
```

后续逻辑统一为：

```text
g_target = g_cache[token_offset]
score = score_acc * exp(g_target - g_cache[source_offset])
result = inter_acc * exp(g_target) + intra_acc
```

这验证了本轮改变确实位于 g 的 producer/consumer lifetime，而不是把某条
global load 单独替换成更宽指令。

## 4. Correctness

测试文件：

`test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_native_chunko_stage6z_z6g.py`

比较对象是同一输入下的 Z5B BF16 output。所有测试均以独立 Docker Python
进程执行，先 correctness，后 benchmark。

### 4.1 全长度 byte-exact/finite

| T | Z6G-S vs Z5B | Z6G-I vs Z5B | finite |
|--:|:--:|:--:|:--:|
| 64 | pass | pass | pass |
| 512 | pass | pass | pass |
| 1024 | pass | pass | pass |
| 2048 | pass | pass | pass |
| 4096 | pass | pass | pass |
| 8192 | pass | pass | pass |
| 16384 | pass | pass | pass |

### 4.2 caller-owned output / zero-V-new / NaN prefill

| T | Z6G-S | Z6G-I |
|--:|:--:|:--:|
| 64 | pass | pass |
| 8192 | pass | pass |
| 16384 | pass | pass |

这些测试使用 caller-owned BF16 output，先用 NaN 预填，再用 zero-V-new 输入，
没有依靠默认 allocator 内容或放宽容差隐藏错误。T=64、T=512、T=2048、T=4096、
T=8192、T=16384 的 wrapper 测试均返回 `1 passed`。

## 5. 性能口径

benchmark 文件：

`test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_stage6z_z6g.py`

固定口径：

- caller-owned preallocated output；
- current HIP stream；
- eager body，不使用 CUDA Graph；
- compile、module load 和 allocation 不在 sample 内；
- warmup=10、repeat=50；
- T=2048、T=8192 使用 7 个 fresh-process paired sessions；
- T=16384 使用 5 个 fresh-process paired sessions；
- 每个 session 轮换 arm 顺序；
- latency 为 HIP-event body median；wall-clock 只作为旁证，不作为主要排名。

native 是同形状 WG256 的 selected `chunk_fwd_kernel_o` direct body，只作诊断
参照，不改变 native selector。

### 5.1 Session median 结果

单位为 ms，表中是每个 arm 的 session medians 中位数。

| T | Z5B | Z6G-S | Z6G-I | native WG256 |
|--:|--:|--:|--:|--:|
| 2048 | 0.065618 | 0.068422 | 0.072968 | 0.042464 |
| 8192 | 0.157354 | 0.168230 | 0.183172 | 0.090835 |
| 16384 | 0.274929 | 0.306896 | 0.337021 | 0.141610 |

相对于同一 session 中的 native：

| T | Z5B/native | Z6G-S/native | Z6G-I/native |
|--:|--:|--:|--:|
| 2048 | 1.545x | 1.611x | 1.718x |
| 8192 | 1.732x | 1.852x | 2.017x |
| 16384 | 1.941x | 2.167x | 2.380x |

### 5.2 Paired difference

正数表示 Z6G 比 Z5B 慢。

| T | Z6G-S - Z5B mean | bootstrap 95% CI | Z6G-I - Z5B mean | bootstrap 95% CI |
|--:|--:|:--|--:|:--|
| 2048 | +3.165 us | [+2.020, +4.504] us | +7.253 us | [+6.049, +8.558] us |
| 8192 | +11.045 us | [+10.438, +11.634] us | +26.222 us | [+25.478, +27.015] us |
| 16384 | +30.950 us | [+30.021, +31.879] us | +61.295 us | [+60.150, +62.196] us |

T=2048 和 T=8192 的 7 个 session 差值均为正；T=16384 的 5 个差值也均为正。
因此不是单个长度或执行顺序造成的偶然负样本。

### 5.3 Endpoint slope

使用 T=2048、8192、16384 的 session-median 数据，以 chunks 为自变量做
简单线性拟合。该 slope 是本轮长度趋势诊断，不代替完整 public API 排名。

| arm | slope (us/chunk) | intercept (us) |
|:--|--:|--:|
| Z5B | 0.934 | 36.512 |
| Z6G-S | 1.066 | 33.416 |
| Z6G-I | 1.180 | 34.086 |
| native | 0.440 | 30.604 |

Z6G-S 比 Z5B 多约 `0.132 us/chunk`，Z6G-I 比 Z5B 多约 `0.246 us/chunk`。
g residency 没有改善长文本 slope，反而扩大了 slope 差距。

原始数据：

```text
codex_qwen_bt64_stage6z_z6g/bench_T2048.json
codex_qwen_bt64_stage6z_z6g/bench_T8192.json
codex_qwen_bt64_stage6z_z6g/bench_T16384.json
```

## 6. T=2048 dynamic PMC

PMC 工件：

```text
codex_qwen_bt64_stage6z_z6g/pmc/stage6z_z6g_s_T2048_rocprof.json
codex_qwen_bt64_stage6z_z6g/pmc/stage6z_z6g_i_T2048_rocprof.json
```

每个 kernel 的 `Grid_Size=131072`，`WG=256`，因此是 `512 CTA`。下表把总量
除以 512，得到每 CTA 动态 PMC。这里的 MFMA/VMEM/LDS/VALU/SALU 是硬件动态
计数，不是从静态 ISA 行数推导出来的。

| arm | MFMA/CTA | VMEM/CTA | LDS/CTA | VALU/CTA | SALU/CTA | Occupancy |
|:--|--:|--:|--:|--:|--:|--:|
| Z5B | 160 | 672 | 672 | 7072 | 768 | 14.540% |
| Z6G-S | 160 | 469 | 725 | 6003 | 757 | 8.222% |
| Z6G-I | 160 | 532 | 725 | 6106 | 911 | 8.326% |
| native WG256 | 160 | 140 | 480 | 3376 | 660 | 9.511% |

### 6.1 Resource fields

| arm | profiler VGPR | profiler Accum_VGPR | profiler SGPR | LDS block | scratch |
|:--|--:|--:|--:|--:|--:|
| Z5B | 76 | 100 | 112 | 32768 B | 0 |
| Z6G-S | 80 | 184 | 112 | 33280 B | 0 |
| Z6G-I | 76 | 188 | 112 | 33280 B | 0 |
| native WG256 | 100 | 36 | 96 | 0* | 0 |

`native` 的 `LDS_Block_Size=0` 是 external/native collector metadata 限制，
不能解释为 native 没有 LDS；它的 dynamic LDS PMC 仍为 `480/CTA`。同理，
profiler 的 `Accum_VGPR_Count` 与 code-object note 中的 `agpr_count` 不是
同一字段，不能直接一一对应。

## 7. Source/LLVM/MIR/ISA 证据

### 7.1 编译工件

Z6G-S：

```text
codex_qwen_bt64_stage6z_z6g/machine/z6g_s/lowered_llvm.ll
codex_qwen_bt64_stage6z_z6g/machine/z6g_s/exact_lto/
codex_qwen_bt64_stage6z_z6g/machine/z6g_s/llc_mir/
codex_qwen_bt64_stage6z_z6g/machine/z6g_s/final_isa.s
codex_qwen_bt64_stage6z_z6g/machine/z6g_s/z6g_s_fixed.hsaco
codex_qwen_bt64_stage6z_z6g/machine/z6g_s/machine_evidence.json
```

Z6G-I：

```text
codex_qwen_bt64_stage6z_z6g/machine/z6g_i/lowered_llvm.ll
codex_qwen_bt64_stage6z_z6g/machine/z6g_i/exact_lto/
codex_qwen_bt64_stage6z_z6g/machine/z6g_i/llc_mir/
codex_qwen_bt64_stage6z_z6g/machine/z6g_i/final_isa.s
codex_qwen_bt64_stage6z_z6g/machine/z6g_i/z6g_i_fixed.hsaco
codex_qwen_bt64_stage6z_z6g/machine/z6g_i/machine_evidence.json
```

两个 HSACO hash 不同：

| arm | HSACO SHA256 | code-object note AGPR | code-object note VGPR | code-object note SGPR | group segment |
|:--|:--|--:|--:|--:|--:|
| Z5B | `979889c1a10ac53064cd1d2da91298b67e4ef7f2db3baadc171872cb56b0ed67` | 32 | 104 | 28 | 32768 B |
| Z6G-S | `851043fbafc58a29a61d7b7b5d860e0db0aaa8569baad4126bf289bef8237ad4` | 32 | 108 | 31 | 33024 B |
| Z6G-I | `6914616828369f0881127d47900372d80519050ae4e20ef4b11e52a3cc1a0ca8` | 32 | 104 | 33 | 33024 B |

Z6G-S/Z6G-I 均为 private segment 0、VGPR spill 0、SGPR spill 0。两条 arm
确实生成了不同于 Z5B 的 code object，不是同一 binary 的运行时别名。

### 7.2 Lowered LLVM 的 g producer/consumer

Z6G-S 的 lowered LLVM 在 `lowered_llvm.ll` 中体现为：

- g pointer 的 FP32 global load 在行 213-214；
- 结果写入独立的 addrspace(3) 256-byte g cache 在行 215-216；
- 后续三个 consumer region 分别以 addrspace(3) g-cache GEP 出现于行 421、668
  附近，而不是重新从 kernel g pointer 读取 FP32。

Z6G-I 的 lowered LLVM 中：

- 行 201-235 构造 resource descriptor 和 `llvm.amdgcn.raw.buffer.load.v4i32`；
- 行 235 是一次 typed x4 packet load，行 236 做 bitcast，行 239 选出当前
  head；
- 行 241-242 将所选值写入 addrspace(3) g cache；
- 后续 g consumer 仍落到同一个 shared cache，见行 447、694 附近。

这证明两条 arm 都改变了 producer/consumer graph：global g pointer 不再出现在
score source 和 final scaling 的后续 consumer 路径中。I 的 x4 packet 也确实
保留到 ISA，final ISA 出现：

```text
buffer_load_dwordx4 v[2:5], off, s[16:19], s12
```

### 7.3 Static ISA lexical counts

静态 ISA 只回答 code graph 中出现了多少条 lexical instruction，不能代替动态
PMC。特别是 56 条静态 MFMA 在循环展开/执行次数下对应 160 MFMA/CTA，不能把
56 写成动态 MFMA。

| arm | static MFMA32 | s_barrier | global_load_* | buffer_load_dwordx4 | global_store | ds_read | ds_write | v_add | v_lshl_add |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Z5B | 56 | 32 | 192 | 0 | 16 | 56 | 144 | 202 | 130 |
| Z6G-S | 56 | 33 | 113 | 0 | 4 | 89 | 145 | 171 | 100 |
| Z6G-I | 56 | 33 | 112 | 1 | 4 | 89 | 145 | 171 | 97 |

g residency 的机器图差异是明确的：

1. MFMA 数学工作不变，静态和动态都与 Z5B 对齐；
2. g producer 区域增加一个 shared write 和一个 phase barrier；
3. 后续 g consumer 从 global scalar load 变成 shared g-cache read；
4. `ds_read` 从 56 增到 89，动态 LDS 从 672 增到 725；
5. g-cache 的 256 B 使 code-object group segment 从 32768 B 增为 33024 B，
   profiler 按粒度报告为 33280 B；
6. Z6G-S 的 profiler Accum_VGPR 从 100 增至 184，Z6G-I 增至 188，occupancy
   从 14.540% 降至 8.222%/8.326%。

### 7.4 Initial MLIR 工具边界

本轮尝试用 `MLIRGenerator.get_mlir()` 直接导出 initial MLIR。Z6G-S 在
`dump_qwen_gdn_bt64_stage6z_z6g_machine_artifacts.py` 的 initial-MLIR probe
处触发了后端 native segmentation fault；它不是 kernel launch fault，也不是
correctness fault。使用 `--skip-initial-mlir --skip-pre-lto-assembly` 后，
正常完成：

```text
lowered LLVM
HSACO link
exact LTO replay
post/pre-RA MIR stop points
final ISA
```

因此本轮的 source 证据来自新增 experimental source，LLVM/MIR/ISA/HSACO
证据来自上述成功工件。Initial MLIR 的“不可导出”本身也已记录为当前调试链
限制，不能把缺失文件伪装成已捕获。

## 8. 为什么 VMEM 降低却延迟变差

### 8.1 Z6G-S

Z6G-S 是更干净的 g residency 版本：

- VMEM：`672 -> 469/CTA`，下降 203，约 30.2%；
- VALU：`7072 -> 6003/CTA`，下降 1069；
- SALU：`768 -> 757/CTA`，略降；
- 但 LDS：`672 -> 725/CTA`，增加 53；
- static barrier：`32 -> 33`；
- profiler Accum_VGPR：`100 -> 184`；
- occupancy：`14.540% -> 8.222%`；
- T=2048：`65.618 -> 68.422 us`；
- T=8192：`157.354 -> 168.230 us`；
- T=16384：`274.929 -> 306.896 us`。

所以 g 的重复 global read 确实被消掉了一部分，但它不是当前总延迟的唯一
瓶颈。shared cache 的 barrier、cache read、cache lifetime 与现有 Q/H/K/score
phase 组合后，产生了更差的 residency/调度状态；在这个 kernel 上，减少 VMEM
没有抵消 LDS 和 live/resource 代价。

这里“Accum_VGPR 增长导致延迟回退”是由 PMC/resource 与 latency 同时观察得到的
强相关解释，不把它写成单一已形式化证明的因果定理。要精确拆出每个 g cache
read 的时间，还需要一个更小的 same-shape g-only control；本轮不再增加新的
实验 arm。

### 8.2 Z6G-I

Z6G-I 试图用更宽的 producer packet 改善 stable 的 scalar g fill，但当前
逻辑每个 token 实际只需要一个 value-head 的 g 值，因此 x4 packet 会把同一
token 的四个 head 一起读入，再 extract 当前 head。

相对 Z5B：

- VMEM：`672 -> 532/CTA`，只下降 140；
- LDS：同样增加到 725；
- VALU：下降到 6106，但比 Z6G-S 高 103；
- SALU：增加到 911，比 Z6G-S 高 154；
- profiler Accum_VGPR：188，比 stable 再高 4；
- T=2048 慢 `7.253 us`；
- T=8192 慢 `26.222 us`；
- T=16384 慢 `61.295 us`。

因此“更宽 load”不是自动等价于更快。它减少了 static global-load lexical
region，但 packet 携带了当前 consumer 不需要的三个 head，并增加了 resource
descriptor、extract 和地址/包处理，最终没有改善 latency。

### 8.3 与 native 的差距

Z6G-S 在 T=2048 已经把 VMEM 从 672 降到 469，但仍是 native 140 的约 3.35x；
LDS 从 native 480 多到 725，VALU 从 3376 多到 6003。Z6G-I 的 VMEM 为 532，
VALU 为 6106，反而更远。

这说明本轮的 g logical residency 假设是成立的，但不是剩余性能差距的主导
结构。native 的优势不是单独“少读 g”，而是其 W/K/H/V-new/score operand
ownership、LDS layout、MFMA feeding 和 phase schedule 同时减少了更多机器工作。

## 9. 结论与下一步边界

结论选择：

**Z6G-S 和 Z6G-I 均 correctness-safe，但性能 No-Go；保持 Z5B。**

具体回答：

1. **g global producer 是否减少？** 是。两条 arm 都只有一个 g tile fill region；
   LLVM downstream g consumers 已改为 addrspace(3) cache path。
2. **VMEM 是否减少？** 是。S 为 672 -> 469/CTA，I 为 672 -> 532/CTA。
3. **是否转化成延迟收益？** 否。两条 arm 在三种长文本长度均慢于 Z5B，且
   paired CI 全部为正。
4. **stable 与 ideal 谁更好？** stable 始终优于 ideal；但 stable 也不优于
   Z5B，所以不建立 Z6G baseline。
5. **是否修改了生产路径？** 否。没有 X2、selector、production、RA 或
   recurrence HSACO 修改。

当前 isolated 排名：

```text
Z5B  <  Z6G-S  <  Z6G-I       （延迟从低到高）
```

下一轮若继续，只能基于已有机器证据重新选择一个单一数据流控制杆；不应再
仅围绕 g cache 增加第三种 residency 变体。当前最大已确认的剩余差距仍是
完整 W/K/H/V-new operand feeding 与 native ownership/layout 的组合，而不是
一个尚未缓存的 g scalar。Z5B 继续作为 Stage 6Z isolated research baseline，
本轮不接入 X2 或 production。

## 10. 复现命令

### Correctness

```bash
/opt/venv/bin/pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_native_chunko_stage6z_z6g.py -s
```

### Body benchmark

```bash
/opt/venv/bin/python \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_stage6z_z6g.py \
  --T 2048 --sessions 7 --warmup 10 --repeat 50 \
  --out test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_z6g/bench_T2048.json

/opt/venv/bin/python \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_stage6z_z6g.py \
  --T 8192 --sessions 7 --warmup 10 --repeat 50 \
  --out test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_z6g/bench_T8192.json

/opt/venv/bin/python \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_stage6z_z6g.py \
  --T 16384 --sessions 5 --warmup 10 --repeat 50 \
  --out test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_z6g/bench_T16384.json
```

### PMC

```bash
/opt/venv/bin/python \
  test/examples/linear_attention/vllm_compare/profile_qwen_gdn_bt64_stage6z_z6g.py \
  --arm z6g_s --T 2048 --warmup 2 --repeat 5 \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_z6g/pmc

/opt/venv/bin/python \
  test/examples/linear_attention/vllm_compare/profile_qwen_gdn_bt64_stage6z_z6g.py \
  --arm z6g_i --T 2048 --warmup 2 --repeat 5 \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_z6g/pmc
```

### Machine artifacts

```bash
/opt/venv/bin/python -u \
  test/examples/linear_attention/vllm_compare/dump_qwen_gdn_bt64_stage6z_z6g_machine_artifacts.py \
  --arm z6g_s --T 2048 --skip-initial-mlir --skip-pre-lto-assembly \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_z6g/machine/z6g_s

/opt/venv/bin/python -u \
  test/examples/linear_attention/vllm_compare/dump_qwen_gdn_bt64_stage6z_z6g_machine_artifacts.py \
  --arm z6g_i --T 2048 --skip-initial-mlir --skip-pre-lto-assembly \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_z6g/machine/z6g_i
```
