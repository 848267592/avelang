# Qwen gfx942 BT64 Stage 6Z Z4：Q producer residency ladder

## 结论先行

本轮严格从修复后的 fixed Z2 分叉，分别测试了三个彼此独立的 Q producer
方案。三条 arm 都保持 `BT64/BV64/BK32/WG256/2 CTA per chunk-head`、BF16
ABI、MFMA32 geometry、K32 reduction order、数学和 output contract 不变，也
没有接入 X2、production selector、recurrence HSACO、allocator 或 RA。

最终结论是：

| arm | T=2048 | T=8192 | 结论 |
|:--|:--|:--|:--|
| fixed Z2 | baseline | baseline | 保持当前 baseline |
| Z4A vector-Q-only | 慢 `38.2%` | 未运行，T=2048 已失败 | No-Go |
| Z4B partial-Q-residency | 中位数快 `1.3%`，但 paired 有一个负样本 | 慢 `8.4%` | No-Go |
| Z4C full-K32-Q-fusion | 中位数快 `2.9%`，5/5 paired 为正 | 慢 `8.6%` | No-Go |

因此本轮没有 arm 同时满足“降低机器工作、T=2048 稳定收益、长文本不回退”。
**Pareto 选择仍是 fixed Z2**。Z4C 保留为有价值的诊断结果：Q producer pass
从 3 次降到 1 次确实可以减少动态 VMEM/LDS/VALU/SALU，但把三个 FP32
accumulator 的 lifetime 叠到同一 K32 loop 后，资源和长文本行为变差。这说明
“Q 重复读取”是真实成本，但“简单地把三个 consumer 放进一个 loop”不是可以
直接晋级的修复。

本轮没有建立 selector，也没有运行 X2 full graph 或 public Eager promotion。

## 1. 实验边界

固定项：

- gfx942，wave64；
- `BT=64, BV=64, BK=32`；
- `WG=256`，每个 chunk-head 两个 CTA，每个 CTA 负责一个 `[64,64]` output tile；
- `q/k/v_new/h/out=BF16`，`g=FP32`，FP32 accumulator；
- `v_mfma_f32_32x32x8_bf16`，K32 accumulation order 不变；
- 相同 global layout、caller-owned BF16 output、数学和 causal mask；
- no Graph、current HIP stream、输入和输出在计时外预分配；
- correctness 先于性能；任何 arm 的 correctness 失败只停止该 arm；
- 没有修改 K、H、V-new、g、output producer，也没有加入 Q/K double buffer。

三个变量严格分开：

### Z4A：vector-Q-only

固定 Z2 的三个 Q producer pass：

```text
Phase A Q*H       -> Q pass 1
Phase B score 0   -> Q pass 2
Phase B score 1   -> Q pass 3
```

只把 `q[...]` scalar BF16 load 改成：

```python
al.amdgpu.make_rsrc(...)
al.amdgpu.raw_buffer_load_x4(...)
al.view(packed_q, al.Tensor((8,), al.bf16))
```

对应源码是 [qwen_gdn_bt64_native_chunko_stage6z_z4_q_ladder.py](../../vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z4_q_ladder.py)，
Phase-A packet producer 位于约 `136-145` 行，Phase-B 两个 packet producer
位于约 `160-173` 行。K/H/V-new/g/output 没有同步改动。

### Z4B：partial-Q-residency

固定 scalar Q load width。把 Phase A 和 score-half 0 交错：

```text
load Q[K32] + H[K32]
barrier
inter MFMA
overwrite H rows with K-half-0
barrier
score-half-0 MFMA consumes the same Q[K32] rows
barrier
```

score-half 1 仍执行 Z2 的 Q global reload。这样 Q producer pass 从 3 次降为 2
次，但不新增第二个完整 16 KiB Q buffer，也不把 Q 保存成 private array。Z4B
的实现约在源码 `301-406` 行。

第一次实现尝试让 `q_resident_vec` 和 `phase_vec` 在 loop 分支中合并，当前 JIT
报 `Failed to generate memref argument for view()`。这不是数值失败，而是
AveLang view type merge 限制。随后把它改成“Phase-A Q slice 在 phase 中仍然
存活，直接改写 H 行为 K 行并消费”的表达，T=64 通过且最终全矩阵通过。

### Z4C：full-K32-Q-fusion

固定 scalar Q load width，使用一个外层 K32 loop：

```text
Q[K32] + H[K32]
  -> inter accumulator
  -> K-half-0 + Q[K32] -> score accumulator 0
  -> K-half-1 + Q[K32] -> score accumulator 1
```

Q 每个 K32 slice 在三个 consumer 完成前不被覆盖；不增加第二个完整 Q LDS
buffer。`inter_acc`、`score_acc0`、`score_acc1` 被有意同时保留，用于测量
requested lifetime effect。实现约在源码 `502-590` 行。

## 2. correctness gate

完整测试命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_native_chunko_stage6z_z4_q_ladder.py -s
```

结果：

```text
31 passed in 20.49s
```

覆盖内容：

| 检查 | 结果 |
|:--|:--|
| Z4A/Z4B/Z4C vs fixed Z2 BF16 byte-exact | T=64/512/1024/2048/4096/8192/16384 全通过 |
| finite | 三个 arm 全通过 |
| caller-owned output | T=64/8192/16384 全通过 |
| zero V-new + NaN-prefilled output | 三个 arm 全通过 |
| output reuse / no NaN 泄漏 | 全通过 |
| MFMA mathematical work | 动态 MFMA/CTA 均为 160，未减少数学工作 |

因此下面的性能差异不是 correctness 放宽、NaN 预填充漏写或改 MFMA 数量造成的。

## 3. 机器工件和 identity

三条 Z4 arm 使用同一个 compile-only capture driver，T=2048，`num_warps=4`，
`WG=256`。Z2 也用同一 driver 新鲜重编译，避免混入旧 Z2 metadata。

capture 脚本：

```text
vllm_compare/dump_qwen_gdn_bt64_stage6z_z4_q_machine_artifacts.py
vllm_compare/dump_qwen_gdn_bt64_stage6z_z3_machine_artifacts.py
```

工件目录：

```text
compile_bug/qwen_mfma32_lowering_ladder/
  codex_qwen_bt64_stage6z_z4_q_machine/z2/
  codex_qwen_bt64_stage6z_z4_q_machine/z4a/
  codex_qwen_bt64_stage6z_z4_q_machine/z4b/
  codex_qwen_bt64_stage6z_z4_q_machine/z4c/
```

每个目录均包含：

- `lowered_llvm.ll`；
- `pre_lto_amdgcn.s`；
- `llc_mir/stop_after_*.mir`；
- `exact_lto/kernel_section_*.mir`；
- `exact_lto/summary.json`；
- `final_isa.s`；
- `*.hsaco`；
- `code_object_notes.txt`；
- `machine_evidence.json`；
- `link_debug/amdgpu-link-0.argv.txt`。

当前 Docker 的 initial MLIR printer 会在 `get_mlir()` 处直接 segmentation fault。
因此本轮没有伪造 initial MLIR 文件；capture 以 `--skip-initial-mlir` 继续，保留
完整 source、lowered LLVM、pre-LTO、exact-LTO MIR、ISA 和 HSACO。这个限制与
此前 Stage 6Z machine capture 的限制相同，不能把 LLVM artifact 冒充成 initial
MLIR。

| arm | kernel | HSACO SHA256 | private | VGPR | AGPR | SGPR | LDS |
|:--|:--|:--|--:|--:|--:|--:|--:|
| Z2 | `_...stage6z_z2` | `2afaa8867da656421a9c87988ed4dcad757796c1b1d3b93e27307fb401a1c09a` | 0 B | 168 | 32 | 44 | 16384 B |
| Z4A | `_...z4a_vector_q` | `44c958d03d169da3a21e0057b3fc9c8ada287e828921d38d1add32d1f5e6dff4` | 0 B | 92 | 16 | 32 | 16384 B |
| Z4B | `_...z4b_partial_q_residency` | `6998fc3e16d37e1cf9e6e596f44a75ea06b580888bb92109f9bd72df7674f6b2` | 0 B | 188 | 48 | 42 | 16384 B |
| Z4C | `_...z4c_full_k32_q_fusion` | `f9d12fbce1d82be7a6ba10abc48ca4de13ae98093dcb9f1d978a297b6a68f307` | 0 B | 220 | 64 | 36 | 16384 B |

四个 exact-LTO `summary.json` 的 `SI_SPILL_AV32/AV64 SAVE` 合计均为 0，
`vgpr_spill_count=0`、`sgpr_spill_count=0`，HSACO private segment 也均为 0。

注意：code-object metadata 的 VGPR/AGPR 与 rocprof 的 `VGPR_Count`/
`Accum_VGPR_Count` 是不同观测量，下面分开报告，不能互换。

## 4. static ISA 对比

这是 final ISA 的 lexical count，不是动态执行次数；循环展开、不同路径合并和
LTO 生成的 lexical instruction path 会影响这些数字。每个 arm 的动态数字单独
来自 rocprof PMC。

| static T=2048 | Z2 | Z4A | Z4B | Z4C |
|:--|--:|--:|--:|--:|
| MFMA32 mnemonic | 20 | 56 | 20 | 20 |
| global/buffer load 总数 | 136 | 132 | 128 | 120 |
| global/buffer store | 16 | 16 | 16 | 16 |
| ds_read 总数 | 20 | 56 | 20 | 20 |
| ds_write 总数 | 88 | 96 | 80 | 72 |
| `s_waitcnt` | 127 | 164 | 118 | 108 |
| `s_barrier` | 9 | 27 | 9 | 8 |
| `v_lshl_add` | 159 | 102 | 152 | 138 |
| `v_add*` | 187 | 332 | 173 | 152 |
| tracked permute | 0 | 48 | 0 | 0 |
| tracked move/copy | 122 | 159 | 117 | 112 |

宽度 family 的 ISA 证据：

| static family | Z2 | Z4A | Z4B | Z4C |
|:--|--:|--:|--:|--:|
| `global_load_ushort` | 56 | 48 | 48 | 40 |
| `global_load_dword` | 80 | 80 | 80 | 80 |
| `global_load_dwordx4` | 0 | 4 | 0 | 0 |
| `ds_read_b128` | 20 | 56 | 20 | 20 |
| `ds_write_b16` | 64 | 80 | 64 | 64 |
| `ds_write_b128` | 0 | 16 | 0 | 0 |

Z4A 的 source-level packet 确实保留到 ISA，`final_isa.s` 中可见
`buffer_load_dwordx4`/`global_load_dwordx4` 和 `ds_write_b128`。但它同时产生
了更多 packet/view/fragment lexical path；静态出现宽 load 不等于动态 VMEM
一定下降，必须看下一节 PMC。

Z4B/Z4C 的 Q source load 仍是 `q[...]` 标量 BF16 producer，ISA 对应
`global_load_ushort` family。它们的收益来自 Q pass/lifetime，而不是 load width。

## 5. T=2048 dynamic PMC

每次 rocprof 都是独立进程，`Grid_Size=131072`，`Workgroup_Size=256`，因此：

```text
CTA = 131072 / 256 = 512
per-CTA value = grid total / 512
```

下面全部是动态 PMC 除以 512 得到的每 CTA 数；不是从 ISA 推导。

| dynamic per CTA | Z2 | Z4A | Z4B | Z4C |
|:--|--:|--:|--:|--:|
| MFMA | 160 | 160 | 160 | 160 |
| VMEM instructions | 928 | 3504 | 800 | 672 |
| LDS instructions | 928 | 480 | 800 | 672 |
| VALU | 11400 | 17684 | 9746 | 7514 |
| SALU | 1072 | 7464 | 964 | 778 |
| profiler `Accum_VGPR_Count` | 32 | 20 | 36 | 52 |
| profiler `VGPR_Count` | 88 | 84 | 108 | 100 |
| profiler `SGPR_Count` | 112 | 112 | 112 | 112 |
| LDS block | 16384 B | 16384 B | 16384 B | 16384 B |
| Scratch | 0 | 0 | 0 | 0 |
| OccupancyPercent | 15.57503 | 16.873998 | 16.090089 | 15.861459 |
| trace median | 55.482 us | 85.607 us | 54.801 us | 52.598 us |

相对 fixed Z2：

| arm | VMEM | LDS | VALU | SALU | AccVGPR |
|:--|--:|--:|--:|--:|--:|
| Z4A | `3.78x` | `0.52x` | `1.55x` | `6.96x` | `-12` |
| Z4B | `0.86x` | `0.86x` | `0.85x` | `0.90x` | `+4` |
| Z4C | `0.72x` | `0.72x` | `0.66x` | `0.73x` | `+20` |

这组证据回答了三个实验问题：

1. Q scalar load width：Z4A 没有带来收益，反而把动态 VMEM 从 928 提到
   3504，SALU 从 1072 提到 7464。当前 `raw_buffer_load_x4` 能合法表达
   packet，但不能据此声称当前 lowering 已把整个 Q path 变成高效 native
   packet dataflow。
2. Q pass 3 -> 2：Z4B 把 VMEM/LDS/VALU/SALU 分别降到
   `800/800/9746/964`，但 AccVGPR 和 code-object resource 上升，body 只得到
   边际收益，长文本回退。
3. Q pass 3 -> 1：Z4C 进一步把动态机器工作显著降到
   `672/672/7514/778`，但同时把 code-object `VGPR/AGPR` 提到 `220/64`，
   profiler AccVGPR 提到 52。它证明 pass fusion 有机器工作收益，但不证明
   当前三 accumulator lifetime 是可持续的 full baseline。

## 6. Fresh-process body benchmark

脚本：

```text
vllm_compare/bench_qwen_gdn_bt64_native_chunko_stage6z_z4_q_ladder.py
```

口径：caller-owned isolated body、current stream、no Graph、warmup=10、
repeat=50、每个 arm 独立 Python process、session 间轮换顺序。T=2048 每个
arm 5 个 session；T=8192 对 Z2/Z4B/Z4C 各 5 个 session。计时使用 HIP event，
wall time 只作为辅助记录。

### T=2048

实际五个 HIP-event session median（单位 ms）：

| arm | s1 | s2 | s3 | s4 | s5 | session median |
|:--|--:|--:|--:|--:|--:|--:|
| Z2 | 0.079719 | 0.081741 | 0.080500 | 0.080119 | 0.081441 | **0.080500** |
| Z4A | 0.110965 | 0.111105 | 0.111245 | 0.112347 | 0.112226 | **0.111245** |
| Z4B | 0.080059 | 0.079197 | 0.079719 | 0.079438 | 0.078417 | **0.079438** |
| Z4C | 0.078196 | 0.078637 | 0.076334 | 0.078977 | 0.076674 | **0.078196** |

以每个 session 的 Z2 同输入 paired：

| arm | paired `Z2-arm` range | paired median | relative to Z2 |
|:--|--:|--:|--:|
| Z4A | `-32.228 .. -29.364 us` | `-30.785 us` | `-38.2%` |
| Z4B | `-0.340 .. +3.024 us` | `+0.781 us` | `+1.3%` |
| Z4C | `+1.142 .. +4.767 us` | `+3.105 us` | `+2.9%` |

Z4C 的五个 paired session 全为正；Z4B 有一个负 paired session，不能称为
稳定收益。Z4A 虽然静态 code-object 资源小，但 dynamic VMEM 和地址工作大幅
上升，直接失败。

### T=8192

实际五个 session median：

| arm | s1 | s2 | s3 | s4 | s5 | session median |
|:--|--:|--:|--:|--:|--:|--:|
| Z2 | 0.177303 | 0.177243 | 0.177664 | 0.177823 | 0.177483 | **0.177483** |
| Z4B | 0.191624 | 0.192826 | 0.192466 | 0.193367 | 0.191804 | **0.192466** |
| Z4C | 0.193227 | 0.192726 | 0.191824 | 0.191945 | 0.192806 | **0.192726** |

paired 中位数：

| arm | paired `Z2-arm` median | relative to Z2 |
|:--|--:|--:|
| Z4B | `-14.802 us` | `-8.4%` |
| Z4C | `-15.322 us` | `-8.6%` |

Z4A 没有运行 T=8192，因为它在 T=2048 已经明确失败；这符合“正确性/性能
门槛失败的 arm 不继续扩大测试”的停止规则。

两点 body slope 仅作诊断，不冒充完整公共 API slope。用 T=2048 与 T=8192
的 chunk 数拟合：

| arm | slope |
|:--|--:|
| Z2 | `1.010 us/chunk` |
| Z4B | `1.177 us/chunk` |
| Z4C | `1.193 us/chunk` |

这说明 Z4C 的 Q pass fusion 只改善了短长度固定区间，不能改善 per-chunk 长文
成本；Z4B 也一样。它们不能作为 Stage 6Z 的下一 baseline。

## 7. 对三个问题的最终回答

### A. scalar load width 到底贡献多少？

在这份 same-shape A/B 中，Z4A 保持 Q pass=3，只改 source-level vector packet。
结果：

- source/ISA 中确实出现了 `raw_buffer_load_x4`、`buffer_load_dwordx4`、
  `global_load_dwordx4`；
- dynamic MFMA 没变；
- dynamic LDS 降低，但 VMEM、VALU、SALU 明显增加；
- T=2048 慢 `38.2%`。

因此“把 Q scalar load 写成 x4 API”本身不是当前可行优化。它没有形成与 native
相同的完整 packet ownership/consumer feeding，反而扩大了地址和 packet/view
lowering 工作。不能把 Z4A 结果解释成“硬件不适合 vector load”，更准确的说法是：
**当前 AveLang source/API 表达的局部 vector load 没有生成可持续的 native-style
Q dataflow。**

### B. Q pass 3 -> 2 的收益？

Z4B 是最小的 residency 变体：只让 Phase-A Q slice 给 score-half 0，score-half 1
仍 reload。它确实减少了动态 VMEM：`928 -> 800`，并减少 VALU/SALU；T=2048
session median 约快 `1.3%`，但 paired 有一例负值，T=8192 慢 `8.4%`。

它说明 Q duplicate pass 是实在的机器工作，但只消除一半 pass 不能跨越当前
phase/lifetime 代价。

### C. Q pass 3 -> 1 是否被 accumulator lifetime 抵消？

Z4C 的证据最清楚：

- Q pass 3 -> 1；
- VMEM `928 -> 672`，LDS `928 -> 672`，VALU `11400 -> 7514`，SALU
  `1072 -> 778`；
- dynamic MFMA 仍为 160/CTA，scratch/spill 仍为 0；
- code-object `VGPR/AGPR` 从 `168/32` 变为 `220/64`；
- profiler AccVGPR 从 32 变为 52；
- T=2048 快 `2.9%`，但 T=8192 慢 `8.6%`。

所以答案是：**是，full K32 Q fusion 的额外 accumulator lifetime 至少是
强相关的资源代价，并且足以让短文本的机器工作收益不能延伸到长文本。**
这不是“已经证明唯一根因”的声明；精确的寄存器调度因果仍应由 MIR live
interval/provenance 进一步拆分。但本轮已经足够证明不能把 Z4C 直接晋级。

## 8. Pareto 决策

预注册决策规则是：如果三个 arm 没有同时降低机器工作并提升性能，则保持
fixed Z2。结果如下：

- Z4A：机器工作部分恶化，T=2048 失败；
- Z4B：T=2048 仅边际、非稳定，T=8192 回退；
- Z4C：T=2048 稳定小收益且机器工作下降，但 T=8192 回退，resource 更高。

因此：

```text
next_baseline = fixed Z2
Z4A = No-Go
Z4B = No-Go
Z4C = diagnostic-only, not promoted
X2 integration = forbidden / not run
production selector = unchanged
```

最重要的研究结论不是“Q residency 无效”，而是：

```text
Q duplicate global work is real,
but removing it requires a lifetime-aware producer/consumer schedule.
Naive packetization or naive three-accumulator fusion is insufficient.
```

下一步若继续 Stage 6Z，只允许先做 read-only MIR live-range/consumer provenance
审计，定位 Z4C 增加的 `VGPR/AGPR` 对应哪组 `inter/score0/score1` 值；本报告本身
不实现新的 Z5，也不改变 fixed Z2。

## 9. 复现命令和 raw data

Correctness：

```bash
cd /workspace/project/avelang
PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_native_chunko_stage6z_z4_q_ladder.py -s
```

T=2048 benchmark：

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_native_chunko_stage6z_z4_q_ladder.py \
  --T 2048 --arm z4c --warmup 10 --repeat 50
```

T=2048 PMC：

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 \
  test/examples/linear_attention/vllm_compare/profile_qwen_gdn_bt64_native_chunko_stage6z_z4_q.py \
  --arm z4c --T 2048 --warmup 2 --repeat 5 \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_z4_q_pmc_T2048
```

raw benchmark JSON：

```text
codex_qwen_bt64_stage6z_z4_q_benchmark_T2048/
codex_qwen_bt64_stage6z_z4_q_benchmark_T8192/
```

raw PMC CSV/JSON：

```text
codex_qwen_bt64_stage6z_z4_q_pmc_T2048/
```

本报告引用的 static/LLVM/MIR/ISA/HSACO 工件都在：

```text
codex_qwen_bt64_stage6z_z4_q_machine/{z2,z4a,z4b,z4c}/
```

