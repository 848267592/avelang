# Qwen gfx942 BT64 Stage 6X-KS: KKT-Solve Handoff Elimination

## 状态

Stage 6X-KS 已完成 X0、X1 和 X2 的源码实现、直接正确性、预分配 body
benchmark、HSACO 检查、T=2048 rocprof，以及随后补齐的完整 Eager public-API
正式 sweep。X2 删除了 KKT 到 solve 的 FP32 全局矩阵边界，并保持相对于 Stage 6W
链的 bit-exact 语义。

**X2 已晋级为新的 Avelang BT64 experimental baseline。** 这不是从 body
benchmark 推断出来的结论：正式 Eager public API 在每个 T 独立进程中执行，5 个
session、50 个 paired Williams block、每实现 300 次调用，且 HIP event 与
wall-clock 均同向。T=2048 与 T=8192 的 event cluster CI 下界都大于零；long-text
slope 也从 W1 的 `6.341 us/chunk` 降至 X2 的 `5.694 us/chunk`。

人工正式测量的结构化汇总存于
`codex_qwen_bt64_kkt_solve_handoff_stage6x_manual_confirmation/`。旧的 Docker
产物目录由容器 `nobody` 所有，故没有覆盖其中的原始 JSON；本报告以这份明确标记为
`user_manual_formal_run` 的确认记录作为晋级证据。

本阶段没有修改 recurrence HSACO、W/U、chunk-o、vLLM 或既有 production
baseline。

## X0：源码、ownership 与 LDS 生命周期审计

### 现有 KKT

当前 BT64 KKT 是
`_qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0`：

- launch：`16 * num_chunks * 8` CTA，WG=64；
- 一个 CTA 对应一个 `(chunk, value-head, token16 row-tile, token16 col-tile)`；
- 每个 64x64 matrix 共有 16 个 tile；其中 10 个下三角或对角 tile 做 dot，6 个
  上三角 tile 只写零；
- active tile 以 8 次 `mfma_16x16x16_bf16_f32` 完成 K=128 reduction；故每个
  `(chunk, head)` 有 `10 * 8 = 80` 次动态 BF16 MFMA；
- 数学与 output layout 保持：

```text
a[t,s] = beta[t] * dot(k[t], k[s]) * exp(g[t]-g[s])  if s < t
         0                                          otherwise
```

`a` 的 global layout 是 FP32 `[1,T,8,64]`，每一个 `(chunk, head)` 对应其中
一行 64-wide matrix。Stage 6U solve 正好按同一 `(chunk, head)` 使用 WG256/4
waves 消费这一块；它只需要下三角，但现有 ABI 仍为完整 64x64 row-major matrix。

### 容量结论

每 CTA 的 A matrix 为 `64*64*4 = 16 KiB`。X1 只需一次性 stage
`K[64,128] BF16 = 16 KiB`。X2 的保守实现同时分配：

| LDS 对象 | 大小 | 生命周期 |
|---|---:|---|
| `k_all_bf16[64,128]` | 16 KiB | KKT phase |
| `a_lds[4,16,64] FP32` | 16 KiB | KKT 完成到 solve 完成 |
| Stage6U `x[7,16,16]` + `work[16,16]` | 8 KiB | solve phase |
| 合计 | 40 KiB | 保守、无 alias |

本轮没有假设 Avelang 可以安全地将 BF16 K staging 和 FP32 solve work 做类型
重解释 alias。40 KiB 在 gfx942 CTA LDS 容量内；是否值得做 32 KiB lifetime reuse
是后续独立 micro-experiment，而非本阶段的正确性前提。

## X1：1 CTA / chunk-head KKT

源码：
`vllm_compare/qwen_gdn_bt64_kkt_solve_handoff_stage6x.py`

`_qwen_gdn_kkt_bf16_kernel_bt64_one_cta_stage6x_x1` 采用 WG256。四个 waves 分别
拥有 4 个 token16 row tile，并循环四个 column tile。严格上三角不做 dot 但仍写零；
因此 a 的 layout、mask 和数值顺序保持不变。X1 仍写原 global FP32 `a`，仅验证
ownership、并行度与资源。

### X1 正确性

`test_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py` 已在 gfx942 执行：

- KKT-only：T=64/128/512/2048，random 与 high_dynamic；全部 FP32 bit-exact；
- X1 KKT 后接当前 Stage6U solve：T=64/512/2048，random 与 cancellation；全部
  BF16 bit-exact。

### X1 KKT body

预热 5、repeat 20、HIP-event body；仅供 ownership gate，不是 Eager full timing。

| T | current KKT ms | X1 KKT ms | current/X1 |
|---:|---:|---:|---:|
| 64 | 0.035473 | 0.031447 | 1.128x |
| 128 | 0.035333 | 0.032369 | 1.092x |
| 512 | 0.034832 | 0.029624 | 1.176x |
| 1024 | 0.035192 | 0.031106 | 1.131x |
| 2048 | 0.046910 | 0.032529 | 1.442x |
| 8192 | 0.170874 | 0.038117 | 4.483x |

T=2048 rocprof 显示 X1 保持 KKT 的 `20,480` MFMA，但把重复 tile staging/address
工作大幅降低。代价是 WG256 的资源和 occupancy；它没有 scratch。

| metric | current KKT | X1 KKT |
|---|---:|---:|
| grid work-items | 262,144 | 65,536 |
| workgroup | 64 | 256 |
| LDS | 8 KiB | 16 KiB |
| VGPR / AccVGPR / SGPR | 20 / 4 / 32 | 84 / 20 / 112 |
| scratch | 0 | 0 |
| MFMA | 20,480 | 20,480 |
| VALU | 1,565,696 | 639,488 |
| SALU | 187,904 | 93,184 |
| VMEM | 210,944 | 79,872 |
| LDS instructions | 184,320 | 47,104 |
| OccupancyPercent | 8.991% | 3.646% |
| median trace | 40.961 us | 9.815 us |

因此 X1 gate 通过：CTA 数从 16 降为 1 并未造成不可接受的并行度损失，且没有
scratch/spill cliff。

## X2：CTA-local KKT + hierarchical solve

### 实现

同一文件中的
`_qwen_gdn_kkt_solve_bf16_kernel_bt64_one_cta_stage6x_x2`：

```text
K/Beta/G
  -> KKT FP32 (CTA-local a_lds[4,16,64])
  -> Stage6U FP32 block-triangular solve in the same CTA
  -> BF16 a_solved global output
```

它不创建也不接收 global FP32 `a`。`a_lds` 的逻辑行布局是 `[row_block,
row_in_block,column]`；该布局与 solve 的 four-wave ownership 一致。

实现中发现了一个 Avelang fragment-layout 要点：`mfma_16x16x4_f32_f32` 的 A/B
operand 必须为 `vector<1xf32>`。初版错误地将 view 的最后一个 unit dimension
也索引掉，得到标量并触发 `MFMA operands must be vector types`。最终使用：

```python
a_lds_frag = al.view(a_lds, al.f32,
    al.make_layout((4, 16, 64, 1), (1024, 64, 1, 1)))
rhs = a_lds_frag[row_block, row_in_block, column]  # 保留最后的 1
```

这与现有 `x_frag` 的用法一致，直接将 LDS fragment 作为 FP32 MFMA operand。

### X2 直接正确性

已执行的 Stage6X KKT/solve matrix：

- current KKT -> current Stage6U solve vs X2：T=64/128/512/2048；
  random、high_dynamic、cancellation；全部 BF16 bit-exact；
- X1 KKT 与 current KKT：T=64/128/512/2048；全部 FP32 bit-exact；
- full Stage6W vs Stage6X：T=64/128/512/2048/8192，random、high_dynamic、
  cancellation、neutral_gate，且分别有/无 initial state；共 40 cases，public
  BF16 output 和 FP32 final-state 均 bit-exact。

补齐的接口回归：

- X2 caller-owned BF16 output 的 NaN prefill + 两次 reuse：通过；
- non-default stream T=64/2048：`2 passed`；
- 主 KKT/solve correctness matrix：`29 passed`。

这些测试证明 X2 相对 Stage6W 保持精确语义及接口行为；它们不把“与 Stage6W
bit-exact”偷换成“与 vLLM 完全 bit-exact”。完整输出/final-state 的跨实现接受范围仍
沿用 Stage 6S/6U 的冻结 contract。

### 预分配 KKT+solve body

直接 launch 到 caller-owned outputs，warmup=5、repeat=20。current 是原 KKT
kernel 加 Stage6U solve；X2 是一 kernel。全部 X2 BF16 bit-exact。

| T | current KKT+solve ms | X2 ms | speedup |
|---:|---:|---:|---:|
| 512 | 0.046850 | 0.034591 | 1.354x |
| 2048 | 0.056384 | 0.034691 | 1.625x |
| 8192 | 0.161080 | 0.076193 | 2.114x |

该趋势符合 handoff 消除的预期：收益随 chunk 数增长。但这仍是 standalone body，不能
替代 full Eager gate。

### X2 T=2048 rocprof / ISA

| metric | X2 |
|---|---:|
| grid work-items / WG | 65,536 / 256 |
| LDS block | 40 KiB |
| VGPR / AccVGPR / SGPR | 100 / 164 / 112 |
| scratch | 0 B |
| MFMA | 36,864 |
| VALU / SALU | 1,280,512 / 206,336 |
| VMEM / LDS inst | 90,112 / 285,696 |
| OccupancyPercent | 6.362% |
| median trace | 16.505 us |

MFMA 数为 `20,480` 次 KKT BF16 MFMA 加 `16,384` 次 solve FP32 MFMA，正好符合
融合的两阶段工作量；没有通过减少 solve 数学换取收益。HSACO 中可见：

- `v_mfma_f32_16x16x16_bf16` 静态 32 条；
- `v_mfma_f32_16x16x4_f32` 静态 56 条。

X2 的 VGPR/AccVGPR/LDS 均高于 X1，但 scratch 为零，且 trace 仍明显低于旧 KKT
单阶段 trace。资源 gate 因此为通过，但 40 KiB LDS / AccVGPR=164 意味着后续若尝试
buffer reuse，必须再次检查 occupancy，而不是假定 LDS 更少一定更快。

## Full Eager public API

全图入口为：

`vllm_compare/qwen_gdn_bt64_kkt_solve_handoff_stage6x_full.py`

```text
cumsum -> X2 fused KKT+solve -> existing BF16 W/U
       -> immutable BF16 recurrence -> existing Stage6W BF16 chunk-o
```

权威计时脚本：

`vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x_eager.py`

它固定输入、current stream、Eager public API（`cuda_graph_used=false`），对每个
block 打乱六种 Williams 三路顺序，记录 HIP event 和 wall-clock，并按 session/block
配对做 cluster bootstrap。

### 完整正式 Eager confirmation

每个 T 以独立进程运行。每个进程固定相同的 `q/k/v/g/beta/initial_state`、current
stream 和 Eager public API；compile、module load 与首次 allocation 在计时前 warmup。
随后执行 5 个 session，每个 session 运行 10 个 timed paired Williams block，block 的
起始顺序随机化。每实现共 300 个完整调用。收益为 `W1 - X2`，正值代表 X2 更快。

| T | W1 median ms | X2 median ms | paired mean gain us | event 95% CI us | X2 相对 W1 |
|---:|---:|---:|---:|:---|---:|
| 512 | 0.221349 | 0.199697 | 22.123 | [19.720, 24.647] | 快约 10.9% |
| 1024 | 0.262049 | 0.233467 | 26.070 | [23.592, 28.667] | 快约 10.9% |
| 2048 | 0.348378 | 0.319154 | 26.020 | [23.304, 28.563] | 快约 8.4% |
| 4096 | 0.518070 | 0.479532 | 40.465 | [37.439, 43.506] | 快约 7.4% |
| 8192 | 0.935690 | 0.852166 | 82.823 | [80.432, 85.154] | 快约 8.9% |
| 16384 | 1.773194 | 1.595190 | 177.540 | [176.004, 179.024] | 快约 10.0% |

T=1024/2048 的 HIP 与 wall-clock CI 都完全为正，且每个长度的多数 session 都为正；
T=4096、8192、16384 的长文本收益也稳定扩大。这满足预注册 gate：correctness、stream、
NaN/reuse、无 scratch/spill、T=2048 和 T=8192 paired CI 下界大于零、两类计时器方向
一致且 slope 不恶化。

对 T=1024--16384 的 event median 按 chunk 数拟合：W1 为 `6.341 us/chunk`，X2 为
`5.694 us/chunk`，因此回收 `0.646 us/chunk` 或约 `10.2%`。这说明收益不仅来自少一个
dispatch，也来自随 chunk 增长而消失的重复 K staging、地址计算和 FP32 `a` global
write/read。

### 同批 native vLLM 对比

| T | X2 ms | vLLM ms | X2/vLLM | 结论 |
|---:|---:|---:|---:|:---|
| 1024 | 0.233467 | 0.358813 | 0.651x | X2 快 |
| 2048 | 0.319154 | 0.401397 | 0.795x | X2 快 |
| 4096 | 0.479532 | 0.522356 | 0.918x | X2 快，gain CI [39.710, 45.394] us |
| 8192 | 0.852166 | 0.765016 | 1.114x | X2 慢 |
| 16384 | 1.595190 | 1.235115 | 1.292x | X2 慢 |

vLLM 的拟合 slope 为 `3.688 us/chunk`，仍低于 X2 的 `5.694 us/chunk`。因此当前准确
结论是：这批同口径 Eager 测试中 X2 在 `T <= 4096` 快于 vLLM，在 `T >= 8192` 稳定落后。
4096--8192 之间存在粗略 crossover，但没有把线性插值值宣称为 dispatch policy。

## 产物与复现

| 产物 | 作用 |
|---|---|
| `vllm_compare/qwen_gdn_bt64_kkt_solve_handoff_stage6x.py` | X1/X2 kernels 和 direct-out API |
| `vllm_compare/qwen_gdn_bt64_kkt_solve_handoff_stage6x_full.py` | X2 full Eager graph |
| `vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py` | KKT/solve/reuse gates |
| `vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x_full.py` | 40-case full bit-exact matrix |
| `vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x_stream.py` | non-default-stream gate |
| `vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py` | preallocated X1/X2 body benchmark and HSACO capture |
| `vllm_compare/profile_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py` | rocprof driver |
| `vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x_eager.py` | formal Williams Eager benchmark |
| `codex_qwen_bt64_kkt_solve_handoff_stage6x/` | JSON, HSACO, rocprof raw artifacts |

关键命令：

```bash
cd /workspace/project/avelang
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py -s

PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x_full.py -s

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py \
  --T 512 2048 8192 --warmup 5 --repeat 20

/opt/rocm/bin/rocprofv3 --kernel-trace --pmc \
  SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex _qwen_gdn_kkt_solve_bf16_kernel_bt64_one_cta_stage6x_x2 \
  -d test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_kkt_solve_handoff_stage6x/rocprof_x2 \
  -o stage6x_x2 -f csv -- python3 \
  test/examples/linear_attention/vllm_compare/profile_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py \
  --implementation x2 --T 2048 --warmup 2 --repeat 5

# Run once per T in a fresh process after execution access is restored.
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x_eager.py \
  --T 2048 --sessions 5 --warmup-blocks 3 --blocks 10 \
  --out-dir /tmp/stage6x_eager_t2048
```

## 决策与下一步

版本状态：

| 版本 | 状态 |
|:--|:--|
| production/default | 不变 |
| U1 / Stage 6U | 历史 experimental baseline |
| W1 / Stage 6W | 上一 experimental baseline |
| X1 | 成功的 one-CTA KKT 组件/诊断 |
| X2 / Stage 6X | **当前 Avelang BT64 experimental baseline** |

冻结 X2；不立即压缩 `40 KiB` LDS，也不做 alias/reuse、packed-lower 或
recurrence--chunk-o fusion。唯一下一步是 **Stage 6Y updated full-gap audit**：以 X2
的五-dispatch 图重新比较 cumsum、fused KKT+solve、fused W/U、recurrence 和 chunk-o
对 native vLLM 的 standalone body slope、资源、真实 dispatch identity 及 global
intermediate accounting。只有该审计确认最大的可恢复缺口后，才选择一个新的 graph 或
kernel 实验。
