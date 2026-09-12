# Qwen gfx942 BT64 Stage 6V: C0 Predicate-Collapse

## 结论

**V0 isolated lowering 通过，V1 full eager-public promotion 不通过。**

V0 严格只改变 C0 的一个 source scheduling 变量：原先四个
`lane_group` predicated MFMA16 区域，改为每 lane 选择已打包的 A/B
fragment，再执行一次 wave-uniform MFMA16。它成功把 static/dynamic MFMA
缩小四倍，且没有 scratch 或 spill。不过该选择链将 VGPR 从 68 提高到
100，完整 U1 图在 T=2048 的收益只有约 1--4 us；高重复 paired measurement
的置信区间跨零。因此 V1 仅保留为实验代码，**不替换 U1，不修改 default
selector，也不进入下一轮 source tuning**。

本轮没有改 solve producer、BF16 solved boundary、W/U 数学、MFMA16 geometry、
CTA/workgroup、W/U BF16 输出、recurrence、chunk-o、layout、compiler 或 assembly。

## V0 Source Change

C0 在 W 和 U 的每个 `source_tile` 内有四段同构代码：

```python
if lane_group == 0:
    acc = mfma(af[0], b[0], acc)
if lane_group == 1:
    acc = mfma(af[1], b[1], acc)
if lane_group == 2:
    acc = mfma(af_next[0], b_next[0], acc)
if lane_group == 3:
    acc = mfma(af_next[1], b_next[1], acc)
```

V0 将 fragment choice 显式变为条件表达式，并把 MFMA 放在选择后：

```python
a_operand = a_frag0[0] if lane_group == 0 else (...)
b0_operand = b0_frag0[0] if lane_group == 0 else (...)
b1_operand = b1_frag0[0] if lane_group == 0 else (...)
acc0 = mfma(a_operand, b0_operand, acc0)
acc1 = mfma(a_operand, b1_operand, acc1)
```

AveLang 将 conditional expression lower 为 `arith.select`。首次尝试使用
statement-level `if` 对临时变量赋值失败：Avelang 的 `scf.if` 分支拥有隔离
scope，赋值不能形成 if 后的 SSA result，因而所有 lane 都错误地使用默认
group-0 operand。该尝试未进入任何结果。最终 V0 使用条件表达式修复，不改
fragment layout 或 MFMA operand order。

## V0 Correctness

`test_qwen_gdn_bt64_predicate_collapse_stage6v_v0.py`：`3 passed`，覆盖
T=64/512/2048。V0 同时与 C0 output 和现有 BF16 W/U reference 对比。
四个 lane-group input fragment 均由上述 select path 消费。

V0 不与 C0 bit-exact：从四个 predicated MFMA 的 EXEC 行为改成完整 wave
MFMA 后，少量 BF16 最低位不同。冻结 gate 是 C0 delta <= `1e-3`，明显小于
W/U reference 的 `0.0078125` 接受阈值。

| T | W bit mismatch | U bit mismatch | W max abs vs C0 | U max abs vs C0 |
|---:|---:|---:|---:|---:|
| 64 | 1 | 1 | 1.4901e-08 | 1.4901e-08 |
| 512 | 8 | 8 | 1.2207e-04 | 4.8828e-04 |
| 1024 | 13 | 18 | 6.1035e-05 | 9.7656e-04 |
| 2048 | 18 | 24 | 3.0518e-05 | 9.7656e-04 |

## V0 ISA and Rocprof Gate

T=2048 has 256 W/U CTAs (`Grid_Size=65536`, `WG=256`). Both code objects
have zero private segment and zero VGPR/SGPR spills.

| metric | C0 predicated | V0 predicate-collapse |
|:--|--:|--:|
| static `v_mfma_f32_16x16x16_bf16` | 64 | 16 |
| static `v_cndmask_b32` | 56 | 200 |
| static `s_cbranch*` | 38 | 6 |
| code-object VGPR / AGPR / SGPR | 76 / 8 / 31 | 108 / 8 / 40 |
| code-object scratch / VGPR spill / SGPR spill | 0 / 0 / 0 | 0 / 0 / 0 |
| profiler VGPR / AccVGPR / SGPR | 68 / 12 / 32 | 100 / 12 / 48 |
| profiler LDS / scratch | 3072 B / 0 | 3072 B / 0 |
| occupancy | 8.0107% | 7.9765% |
| dynamic MFMA | 262144 | 65536 |
| dynamic MFMA per CTA | 1024 | 256 |
| dynamic LDS | 589824 | 393216 |
| dynamic VALU | 2087936 | 2414592 |
| dynamic SALU | 316416 | 54272 |
| dynamic VMEM | 311296 | 311296 |

同一 profiling driver 的 trace median 是 C0 `41.642 us`、V0 `38.598 us`
（-7.3%）。单独重复 V0 profile 得到 `41.842 us`，所以 trace 的绝对差只作
诊断，不作为 promotion 依据；关键事实是 dynamic MFMA 已精确达到
`256/CTA`，并且没有 resource cliff、scratch 或 spill。

未插 profiler 的 isolated body timing 在 T=2048 为 C0 `0.073129 ms`、V0
`0.071827 ms`（1.8%）。省掉的 MFMA 被 select/cndmask 和更高 VGPR 部分抵消。

## V1 Full Public Contract

V1 是唯一的 full integration：

```text
P0 BF16 solved -> V0 predicate-collapse fused W/U -> unchanged BF16 recurrence -> unchanged chunk-o
```

它复用 U1 的 cumsum/KKT/P0 solve、Stage 6S recurrence bridge、BF16-to-FP32
V-new boundary和 chunk-o。没有额外 dispatch、fallback 或 public contract change。

完整 eager-public correctness 已覆盖 T=64/512/2048/8192/16384、随机 nonzero
initial state。全部 finite 且通过原有阈值：output <= `1/128`，final state <=
`0.02`。

| T | V1 output max abs vs vLLM | V1 final-state max abs vs vLLM | V1 output max abs vs U1 | V1 state max abs vs U1 |
|---:|---:|---:|---:|---:|
| 64 | 4.8828e-04 | 5.7392e-03 | 0 | 0 |
| 512 | 4.8828e-04 | 4.6706e-03 | 0 | 0 |
| 2048 | 7.3242e-04 | 7.9160e-03 | 1.2207e-04 | 2.9802e-08 |
| 8192 | 9.7656e-04 | 4.6512e-03 | 1.2207e-04 | 0 |
| 16384 | 9.7656e-04 | 4.7188e-03 | 1.2207e-04 | 0 |

## V1 Eager Public Timing

All values below use the required eager public API: same process/input/current
stream, no CUDA graph, pre-warmup, five sessions, 20 warmups and 100 ABBA-
balanced repeats. They are not rocprof times.

| T | U1 ms | V1 ms | vLLM ms | V1-U1 us | V1/U1 | V1/vLLM |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.249551 | 0.248410 | 0.358412 | -1.141 | 0.9954x | 0.6931x |
| 1024 | 0.273627 | 0.270582 | 0.356490 | -3.045 | 0.9889x | 0.7590x |
| 2048 | 0.364962 | 0.361397 | 0.416459 | -3.565 | 0.9902x | 0.8678x |
| 4096 | 0.541285 | 0.538461 | 0.534434 | -2.824 | 0.9948x | 1.0075x |
| 8192 | 0.969402 | 0.968219 | 0.776614 | -1.182 | 0.9988x | 1.2467x |
| 16384 | 1.834768 | 1.829000 | 1.324268 | -5.768 | 0.9969x | 1.3811x |

Slope fit is U1 `6.460055 us/chunk` and V1 `6.448760 us/chunk`，只减少
`0.011295 us/chunk`。V1 的长文本 slope 确实没有变差，但下降过小。

为检验 T=2048 stability，额外运行 nine sessions、20 warmups、200 ABBA repeats：
U1 `0.362038 ms`，V1 `0.360316 ms`，aggregate 差 `-1.722 us`。但 sample-level
paired mean gain 的 bootstrap 95% CI 是 `[-2.003, 43.375] us`，包含零。因此
“相对 U1 稳定加速”这个主 gate **不成立**。

## 决策

| gate | result |
|:--|:--|
| V0 select 后 fragment 正确、冻结误差内 | pass |
| static MFMA 缩小四倍 | pass, 64 -> 16 |
| dynamic MFMA 约 256/CTA | pass, exactly 256 |
| 无 scratch/spill/resource cliff | pass; VGPR 增加但 occupancy 基本不变 |
| V1 public correctness | pass |
| T=2048 相对 U1 稳定加速 | fail |
| 长文本 slope 有实质下降 | fail; only -0.011295 us/chunk |

**No-Go for promotion.** 保留 U1 作为 Stage 6U 的实验选择，V1 保留为独立
negative/diagnostic result。不要为了这一点微小差距继续改 C0、compiler、recurrence
或 chunk-o；下一个动作应回到已有全图审计中更大的、数据支持的结构性 gap。

## Files and Commands

Source and drivers:

- `vllm_compare/qwen_gdn_bt64_predicate_collapse_stage6v.py`
- `vllm_compare/test_qwen_gdn_bt64_predicate_collapse_stage6v_v0.py`
- `vllm_compare/test_qwen_gdn_bt64_predicate_collapse_stage6v_v1.py`
- `vllm_compare/bench_qwen_gdn_bt64_predicate_collapse_stage6v_v0.py`
- `vllm_compare/profile_qwen_gdn_bt64_predicate_collapse_stage6v_v0.py`
- `vllm_compare/bench_qwen_gdn_bt64_predicate_collapse_stage6v_eager_public.py`

Raw artifacts, HSACOs, rocprof CSVs, correctness JSON and ABBA samples:

- `codex_qwen_bt64_predicate_collapse_stage6v/`

Key reproductions:

```bash
python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_predicate_collapse_stage6v_v0.py -s

python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_predicate_collapse_stage6v_v1.py -s

python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_predicate_collapse_stage6v_eager_public.py \
  --T 512 1024 2048 4096 8192 16384 --sessions 5 --warmup 20 --repeat 100 \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_predicate_collapse_stage6v
```
