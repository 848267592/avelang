# Qwen gfx942 BT64 Stage 6W: BF16 Chunk-O Boundary

## 结论

Stage 6W 完成了一个 opt-in 的全图边界优化：BT64 chunk-o 直接读取
recurrence 的 BF16 `v_new`，并将其 FP32 累加结果直接写为 BF16 public
output。没有修改 recurrence HSACO、KKT、solve、W/U、chunk-o 的 MFMA
几何/LDS tile、compiler 或默认 selector。

相对当前 U1 BT64 图，Stage 6W 的完整 **Eager public API** sweep 在全部
`T=512..16384` 点更快，T=2048 session-median 从 `0.361818 ms` 降到
`0.351862 ms`，T=16384 从 `1.829537 ms` 降到 `1.768947 ms`。不过 T=2048
的逐调用 paired bootstrap 区间跨零，所以该结果足以保留为下一轮的实验
候选，尚不足以修改默认路径。

最重要的判断是：收益来自消除全图中两个 materialized boundary，而不是
chunk-o 核心 MFMA 算术突然更快。预分配 body 测量中，新的 chunk-o 本体在
T=2048 反而约慢 `4.27 us`；完整图仍获益，是因为删除了 `v_new` 扩 FP32 和
final output cast 两个 dispatch，以及相应的 FP32 global staging。

## 冻结的改动

旧 U1 tail：

```text
BF16 recurrence V-new
  -> torch BF16-to-FP32 cast
  -> chunk-o loads FP32 then truncates to BF16 LDS/MFMA operand
  -> FP32 output staging
  -> torch FP32-to-BF16 final cast
```

Stage 6W tail：

```text
BF16 recurrence V-new
  -> chunk-o loads BF16 directly
  -> unchanged BF16 MFMA operand path with FP32 accumulators
  -> convert only at final BF16 public-output store
```

新文件：

- `vllm_compare/qwen_gdn_bt64_bf16_chunko_boundary_stage6w.py`
- `vllm_compare/test_qwen_gdn_bt64_bf16_chunko_boundary_stage6w.py`
- `vllm_compare/bench_qwen_gdn_bt64_bf16_chunko_stage6w_eager_public.py`
- `vllm_compare/bench_qwen_gdn_bt64_bf16_chunko_stage6w_body.py`
- `vllm_compare/profile_qwen_gdn_bt64_bf16_chunko_stage6w.py`

唯一新增 public entry 是
`qwen_gdn_full_bt64_stage6w_bf16_chunko_eager(...)`。它复用 U1 的
BF16 solve、C0 W/U 和 Stage 6S BF16 recurrence；只替换 chunk-o storage
boundary。任何非 BF16 `v_new`、不连续、非 BT64 的 shape 或设备不匹配都会
`ValueError`，没有 fallback。

## ISA 和资源证据

两个 HSACO 都保持相同的 WG=256、grid=2048 CTA（T=2048）、MFMA16
schedule、LDS block 和零 scratch。关键 ISA 差异符合预期：

| 位置 | U1 current chunk-o | Stage 6W |
|:--|:--|:--|
| V-new byte stride | `s_lshl_b64 ..., 2` | `s_lshl_b64 ..., 1` |
| V-new global read | `global_load_dword` | `global_load_ushort` |
| public output store | `global_store_dword` | `global_store_short_d16_hi` |
| MFMA mnemonic | `v_mfma_f32_16x16x16_bf16` | 相同 |

T=2048 rocprof 的动态 instruction counts 保持同一计算主体：

| metric | current | Stage 6W | 变化 |
|:--|--:|--:|--:|
| WG / Grid work-items | 256 / 524288 | 256 / 524288 | 相同 |
| VGPR / AccVGPR / SGPR | 112 / 64 / 112 | 112 / 64 / 112 | 相同 |
| LDS / scratch | 27136 B / 0 | 27136 B / 0 | 相同 |
| MFMA | 458752 | 458752 | 相同 |
| VALU | 12918784 | 12918784 | 相同 |
| SALU | 1495040 | 1449984 | -45056 |
| VMEM | 851968 | 851968 | instruction count 相同 |
| LDS instructions | 1343488 | 1343488 | 相同 |

VMEM 是指令数而不是传输字节数，所以 BF16 read/store 未必让该计数下降。

**rocprof trace caveat：** 在这个短 kernel 上，带 PMC 的 rocprof 把 Stage
6W 的 occupancy 报成 `0.81%`、trace 报成约 `1498 us`，而 current 为
`17.75%`、`66.86 us`。这和无 profiler、预分配 HIP-event body 的 `~0.09 ms`
以及完整 Eager 图相矛盾，且资源/动态计数并未显示 resource cliff。因此这组
trace 不满足低扰动使用门槛，只保留作 collector perturbation 证据，不能用于
性能因果结论。

## 数值正确性

独立 chunk-o 在 `T=64/512/2048` 对照旧路径
`chunk_o(v_new_bf16.float()).to(bf16)`，均为 BF16 bit-exact：

| T | int16 mismatch | max abs | mean abs |
|--:|--:|--:|--:|
| 64 | 0 | 0 | 0 |
| 512 | 0 | 0 | 0 |
| 2048 | 0 | 0 | 0 |

完整图也与 U1 public output bit-exact，final state bit-exact。相对 native
vLLM 的 full contract：

| T / case | output max abs | final-state max abs | threshold |
|:--|--:|--:|:--|
| 64 random + state | 5.24521e-4 | 4.82208e-3 | 1/128, 0.02 |
| 512 high-dynamic + state | 1.953125e-3 | 1.74136e-2 | pass |
| 2048 neutral-gate, zero state | 1.953125e-3 | 1.63818e-2 | pass |
| 8192 cancellation + state | 3.81470e-6 | 2.62512e-5 | pass |

The no-fallback guard for FP32 `v_new` also passed.

## 预分配 Chunk-O Body

这些数只测预分配 input/output 的 kernel launch，不包括两个被删除的 cast。
它们说明 direct BF16 global boundary 的 kernel body 没有形成更快的 MFMA
body，且小幅落后；这不否定全图优化。

| T | current ms | Stage 6W ms | current / W1 |
|--:|--:|--:|--:|
| 512 | 0.036154 | 0.041822 | 0.8645x |
| 1024 | 0.055202 | 0.057185 | 0.9653x |
| 2048 | 0.086869 | 0.091136 | 0.9532x |
| 4096 | 0.137304 | 0.141430 | 0.9708x |
| 8192 | 0.250192 | 0.251434 | 0.9951x |
| 16384 | 0.468576 | 0.473244 | 0.9901x |

## 权威 Eager Full Benchmark

所有下面数据均是同一 public API contract，`cuda_graph_used=false`，相同
输入/stream，ABBA order，warmup=20、repeat=100、5 session。它们是 Stage 6W
的性能判定；ISA/rocprof 只作诊断。

| T | U1 ms | Stage 6W ms | W1 gain vs U1 | W1 / vLLM |
|--:|--:|--:|--:|--:|
| 512 | 0.250191 | 0.223572 | 26.619 us, 1.1191x | 0.6190x |
| 1024 | 0.275829 | 0.267397 | 8.432 us, 1.0315x | 0.7308x |
| 2048 | 0.361818 | 0.351862 | 9.956 us, 1.0283x | 0.8581x |
| 4096 | 0.539100 | 0.523217 | 15.884 us, 1.0304x | 0.9801x |
| 8192 | 0.968759 | 0.936471 | 32.289 us, 1.0345x | 1.2119x |
| 16384 | 1.829537 | 1.768947 | 60.590 us, 1.0343x | 1.3401x |

线性拟合的 U1 slope 是 `6.439639 us/chunk`，Stage 6W 是
`6.252842 us/chunk`，回收 `0.186797 us/chunk`；native vLLM 的对应 slope
仍为 `3.926166 us/chunk`。

额外的 9-session / warmup=30 / repeat=200 confirmation：

| T | U1 ms | Stage 6W ms | aggregate gain |
|--:|--:|--:|--:|
| 2048 | 0.361998 | 0.353285 | 8.713 us (2.47%) |
| 8192 | 0.970882 | 0.940497 | 30.385 us (3.13%) |

两点的 session median 均支持正收益；不过逐调用 paired bootstrap 因 eager
调度/顺序噪声区间跨零，故不将其描述为达到默认 selector promotion gate 的统计
确定性收益。

## 决策

保留 Stage 6W 作为正确、正向但尚未提升默认的 opt-in 图候选。它完成了已知的
BF16 storage-boundary 传播方向，且不需要改 recurrence assembly。

下一步不应继续微调相同的 chunk-o arithmetic：预分配 body 已显示该本体没有
可观收益。应对完整 Eager 图做一次严格的 boundary-dispatch accounting，确认
removed cast dispatch 在公共 API 计时中的稳定贡献；若该贡献经过更强的
order-balanced confirmation 仍显著，再将 Stage 6W 作为 U1 的候选 tail，并只
选择一个新的、数据支持的 upstream global-intermediate 消除动作。

## 复现

```bash
cd /workspace/project/avelang
python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_bf16_chunko_boundary_stage6w.py -s

python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_bf16_chunko_stage6w_eager_public.py \
  --T 512 1024 2048 4096 8192 16384 --sessions 5 --warmup 20 --repeat 100 \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_bf16_chunko_boundary_stage6w

python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_bf16_chunko_stage6w_body.py \
  --T 512 1024 2048 4096 8192 16384 --warmup 20 --repeat 100
```

## 后续严格确认

本报告中的 5-session sweep 与 9-session confirmation 是候选收益线索，不是 baseline
promotion 结论。后续完成的 12-session randomized-Williams clustered Eager confirmation
在 T=2048 和 T=8192 都未通过旧的 block/session/nested CI 共同正下界门槛；GPU process
审计同时发现非独占 context、累计 eviction 和数百毫秒级 outlier。这些历史数据不能否定
早期正向收益。后续使用同一共享环境内随机 Williams block 的新鲜 8-session 重测确认：W1
相对 U1 在 T=2048 快 `8.994 us`（95% CI `[7.246,10.545]`），T=8192 快 `27.352 us`
（`[25.398,29.163]`），HIP event、wall-clock 与 nested sensitivity 一致。因此 W1 已晋级为
当前 Avelang experimental baseline；结论范围是 paired shared-environment Eager 排名。完整
数据、协议与下一候选在 `qwen_gfx942_bt64_stage6w_cluster_confirmation_and_intermediate_audit_report.md`。
