# Qwen Direct-K64 Current-ABI Alignment Experiment

## 结论

这是一项 **ABI 对齐后的 direct-K 更新后缀实验**，不是完整 Qwen GDN
kernel，也不是 v29 的替代版本。

实验说明两件事：

1. 在与 current-vLLM recurrence 相同的存储边界下，即 `K` / `V-new` 为
   BF16、`g` / persistent state 为 FP32、`H` 为 BF16、final state 为 FP32，
   Avelang 可以正确运行直接 K64 更新，且 code object 为 `Scratch=0`、VGPR
   / SGPR spill 均为零。
2. 这 **不能** 单独证明 “Avelang 编译器比 Triton 差”。工作版本仍把 Triton
   的 `tt.dot` 拆成显式的 32x32 LDS staging、fragment view 和多层循环；它既不
   是相同的 high-level IR，也不拥有原生 Triton 的 `V32 x K128` CTA ownership。
   它很慢，主要是这个 source/IR 组合的地址和控制成本高，不能归因给 register
   allocator 一项。

因此，`compact/direct K` 在硬件和 ABI 上是可行方向，原 v29 的
`AccVGPR=384` / `736 B` scratch 不是 “BF16 direct K 在 gfx942 天生不能做” 的
证据；但本实验尚不是可提交给编译器团队的 compiler-only A/B 铁证。

## 问题与边界

原 v29 full rewrite 的精确 LTO/MIR 审计已经证明，`736 B` private segment 来自
`190` 个 VGPR spill words，且同时出现 `Accum_VGPR=384`。见
[`qwen_v29_full_mir_and_pred_streaming_report.md`](qwen_v29_full_mir_and_pred_streaming_report.md)。
但原 v29 的 nonzero-W recurrence reference correctness 未解决，且其输入边界与
current-vLLM BF16 recurrence 不相同。因此不能将原 v29 compact-K 的速度或资源直接与
Triton/current-vLLM recurrence 作公平比较。

本实验只固定更新后缀：

```text
BF16 K + BF16 V-new + FP32 g + FP32 state
    -> chunk-entry BF16 H + updated FP32 state
```

它刻意不包含 W/pred 到 V-new 的产生阶段，以避免把 unresolved v29 pred 语义与 K
operand 结构混为一个变量。

## 原生 Triton 的实际结构

审计的 current-vLLM Triton TTIR 是：

`codex_qwen_bt64_stage6z_native_chunko/native/T2048/triton_cache/.../chunk_gated_delta_rule_fwd_kernel_h_blockdim64.source`

关键事实：

| 项目 | 原生 TTIR 证据 |
|---|---|
| persistent state | `b_h1: tensor<32x64xf32>`、`b_h2: tensor<32x64xf32>`，第 60-61 行 |
| direct K low half | `tt.load tensor<64x64xbf16>`，第 646-647 行 |
| low-half update | `tt.dot K[64,64] * V[64,32] -> f32[64,32]`，第 650 行 |
| direct K high half | `tt.load tensor<64x64xbf16>`，第 669-670 行 |
| high-half update | 相同形状 `tt.dot`，第 673 行 |
| V-decay boundary | `b_v_588: tensor<64x32xbf16>`，第 629 行 |

所以原生实现的本质是两个 persistent FP32 state half 加两个直接的 BF16 K64
block operand。它不是 v29 的 broad `k_all_t[128,64] -> kall_vec` shared view。

## Avelang 实现

新增实验文件：

- [`repro_qwen_gdn_direct_k64_update_current_abi.py`](../../vllm_compare/repro_qwen_gdn_direct_k64_update_current_abi.py)
- [`test_qwen_gdn_direct_k64_update_current_abi.py`](../../vllm_compare/test_qwen_gdn_direct_k64_update_current_abi.py)
- [`bench_qwen_gdn_direct_k64_update_current_abi.py`](../../vllm_compare/bench_qwen_gdn_direct_k64_update_current_abi.py)

该 repro 使用 `BT=64`、`K=128`、`BV=64`、`WG=128`，一个 CTA 处理一个
`V64 x K128` state block，两个 wave 各有 `V32 x K128` 的四个 32x32 FP32
MFMA accumulator fragment。它不是原生的 `V32 x K128` CTA ownership；这是当前
可稳定验证的 Avelang MFMA32 mapping，不能被描述成 Triton 的逐项复刻。

每个 chunk 的 K source 访问只来自：

```python
k[..., k_half * 64 + col_half * 32 + b_row]
```

即 source 语义上的：

```text
K[0:64, chunk:chunk+64]
K[64:128, chunk:chunk+64]
```

随后只将下一次 MFMA 所需的 `32x32` operand stage 放到 LDS。代码中不存在
`k_all_t[128,64]`、也不存在 broad K 的 transposed `kall_vec` shared view。相关
实现见 repro 第 125-205 行；输入/输出 dtype contract 在第 53-92、226-255 行。

数学为：

```text
H_c = state_c cast to BF16
V_decay[t,v] = BF16(V_new[t,v] * exp(g_last - g[t]))
state_next[v,k] = state_c[v,k] * exp(g_last)
                  + sum_t FP32(V_decay[t,v]) * FP32(K[t,k])
```

reference 使用同一 BF16 `V_decay` 边界，避免把 BF16 rounding policy 混入这次
direct-K 审计。

## 正确性

初始 state 非零。T=64/512 使用 pytest；T=1024/2048 使用同一 reference 的 benchmark
precheck。

| T | H max abs | H mean abs | final state max abs | final state mean abs | finite |
|---:|---:|---:|---:|---:|:---:|
| 64 | 0 | 0 | `3.81469727e-06` | `1.20868663e-07` | pass |
| 512 | `0.125` | `4.79107825e-07` | `2.28881836e-05` | `7.53104359e-07` | pass |
| 1024 | `0.5` | `1.04813148e-06` | `3.05175781e-05` | `1.36176504e-06` | pass |
| 2048 | `0.5` | `1.38880455e-06` | `6.10351562e-05` | `2.54225165e-06` | pass |

H 是逐 chunk 写出的 BF16 snapshot；MFMA32 与 PyTorch matrix multiply 的 reduction
order 不同，故长序列的绝对 H max 会在大幅值位置放大。更强的 FP32 recurrence final
state 误差始终远低于 `1e-3`。pytest 固定检查 `H max <= 1.0`、final state
`max <= 1e-3`，并检查所有输出 finite：`2 passed`。

## 未插 profiler 的 body timing

这不是 full GDN，不可和 eager public API 或 native full recurrence latency 直接相减。

| T | direct-K64 update suffix median ms |
|---:|---:|
| 512 | `0.312845` |
| 1024 | `0.582867` |
| 2048 | `1.146966` |

这些数字的价值是暴露此 Avelang source 组合尚不高效，不是宣称可以替换任一
production path。

## T=2048 ROCprof 与 ISA

原始 CSV/HSACO 已固化在：

`test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_update_current_abi/`

trace 中匹配 kernel 的八个 dispatch duration 的中位数是 `1120.8685 us`。该 trace
本身会扰动短 kernel，因此不作为上节 HIP-event body latency 的替代。

| 项目 | 数值 |
|---|---:|
| workgroup | `128` |
| grid work-items | `2048`，即 16 CTA |
| LDS block | `6144 B` |
| scratch | `0 B` |
| rocprof VGPR / AccVGPR / SGPR | `24 / 144 / 112` |
| raw OccupancyPercent | `0.3241949125` |
| SQ_INSTS_MFMA | `32768` |
| SQ_INSTS_VALU | `6336416` |
| SQ_INSTS_SALU | `1087456` |
| SQ_INSTS_VMEM | `354304` |
| SQ_INSTS_LDS | `229376` |

HSACO metadata 独立报告 `.private_segment_fixed_size=0`、VGPR/SGPR spill count 都是
零。它也报告 `.vgpr_count=168` 与 `.agpr_count=16`；这与 rocprof 的
VGPR/Accum_VGPR 分类口径不同，报告不把二者混为同一个物理寄存器计数。

反汇编确认：

```text
v_mfma_f32_32x32x8_bf16
ds_write_b16 / ds_read_b128
global_load_ushort
global_store_dwordx4
```

因此 MFMA32 真正生成，最终 state store 也有向量化 global store；无 private spill。

## 与 Triton 的可比性

| 维度 | native Triton recurrence | 本 Avelang direct-K repro | 是否严格相同 |
|---|---|---|:---:|
| K / V-new storage | BF16 | BF16 | 是 |
| g / persistent state | FP32 | FP32 | 是 |
| H / final-state storage | BF16 / FP32 | BF16 / FP32 | 是 |
| K operand logical split | 2 x K64 | 2 x K64 | 是 |
| persistent state per CTA | V32 x K128 | V64 x K128 | 否 |
| high-level dot | `tt.dot(64x64,64x32)` | explicit 32x32 LDS stages + MFMA fragments | 否 |
| pred/W -> V-new | in same native kernel | deliberately excluded | 否 |
| full dispatch / full latency | complete recurrence | update suffix only | 否 |

所以可以比较 **ABI、K operand 结构、是否产生 spill/scratch**；不能比较此处的绝对
延迟、动态 instruction count，或由它直接判定 “Avelang compiler 低效”。当前 repro 的
VALU/SALU/VMEM 很高，合理的解释是手工 stage/view/loop 的 source/IR 仍显著不同于
Triton `tt.dot` 的 block-level lowering。

## 尝试过但未纳入结果的字面 CTA 对齐

曾短暂将新 repro 改成 native 的 `V32 x K128` CTA ownership（32 CTA），让两个 wave
分别持有 K0:64 和 K64:128 state half。这一手工 MFMA fragment remapping 先暴露少量
未写 H snapshot，随后在重复运行出现 GPU memory fault。该版本已撤回，没有保留为候选，
也没有使用它的 timing/counter 形成结论。

这只说明当前 Avelang source 层手写该 operand/layout mapping 还没有完成 correctness
audit；它不是硬件、Triton 或 register allocation 的负面证据。

## 对 “是不是 compiler 问题” 的判断

现在能严谨地说：

- native Triton 确实在同一 gfx942、同一 BF16 recurrence ABI 下执行 direct K64 block
  operand；因此 direct/compact K 的算法与硬件路径是存在的。
- Avelang 的已验证 direct-K 后缀也没有重现 v29 full 的 scratch/spill cliff；所以
  v29 溢出不是输入 dtype 本身或 direct K 结构必然造成。
- 现有证据仍不足以将 root cause 缩小到 “最后一条 LDS load 的 backend lowering”。已有
  same-source late-B-fragment A/B 在
  [`qwen_kfrag_same_source_lowering_ab_report.md`](qwen_kfrag_same_source_lowering_ab_report.md)
  中最终收敛到同一 LLVM/ISA，资源也同为 `VGPR=116, AccVGPR=148`。

更精确的归因是：**Avelang 目前缺少与 Triton `tt.dot` 对等的 block-operand / persistent
accumulator 表示及其 lowering，而 full v29 的 pred、address、fragment 和 update
live-region 组合又会触发 RA spill。** 这是 compiler IR/lowering 能力与目前 source 组合
共同的问题，不是已经证明的单一 RA bug。

## 唯一合理的下一实验

不要回到 broad `k_all_t`，也不要将此 repro 接入 full GDN。下一步应是一个 strict
same-source compiler A/B：新增一个可持续到 AMDGPU lowering 的 block-dot op，语义类似：

```python
state_half = al.amdgpu.block_dot_bf16(
    k_block_64x64,
    v_decay_64x32,
    state_half,
)
```

固定 source、ABI、CTA ownership、shared allocation、barrier、MFMA schedule 与数学，
只分叉 block-dot lowering。两条 lowering 均正确后才比较 scratch、spill、VGPR/AccVGPR、
MIR live intervals 和 latency。若 direct block lowering 显著降低资源而上述高层量完全
一致，才能向 compiler 团队提供接近 compiler-only 的因果证据。

## 复现命令

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:.

PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  test_qwen_gdn_direct_k64_update_current_abi.py -s

PYTHONDONTWRITEBYTECODE=1 python3 \
  bench_qwen_gdn_direct_k64_update_current_abi.py \
  --T 512 1024 2048 --warmup 5 --repeat 20

# 已执行的 T=2048 rocprof 形式。
/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex _qwen_gdn_direct_k64_update_current_abi_kernel \
  -d /tmp/qwen_direct_k64_rocprof -o direct_k64 -f csv -- \
  python3 repro_qwen_gdn_direct_k64_update_current_abi.py \
  --T 2048 --warmup 2 --repeat 5 --no-check
```
