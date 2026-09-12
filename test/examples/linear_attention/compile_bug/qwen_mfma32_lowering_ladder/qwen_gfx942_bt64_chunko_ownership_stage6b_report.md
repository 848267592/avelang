# Qwen gfx942 BT64 Chunk-O Ownership Stage 6B

## 结论

**O0 是一个正确、真实减少重复工作的实验，但未通过晋级门槛；Stage 6B 不接入 full graph，也不进入 Stage 6C direct-BF16 output。**

O0 将 T=2048 的 chunk-o CTA 从 2048 降到 512，和当前 vLLM 的 CTA 数相同；它还将动态
MFMA、VMEM、LDS 指令分别降为当前的 `41.1%`、`39.4%`、`38.7%`。但它为保持跨 V16
score 复用而引入 8 KiB score LDS lifetime，使 `Accum_VGPR` 从 64 增至 188，LDS block
从 27136 B 增至 33280 B，occupancy 从 17.90% 降至 8.86%。最终 T=2048 body 仅从
`0.076073 ms` 到 `0.062012 ms`，即 `1.227x`，不足 `1.5x` gate。

O1（唯一允许的第二个试验，V64 改 V32）是负结果：更多 CTA、更多 MFMA/LDS/barrier，
T=2048 反而慢于 current。所有生产路径、FP32 output staging、独立 BF16 cast、KKT、solve、
W/U、asm recurrence、compiler 和 assembly 均未修改。

## 1. 实际 Ownership

| 实现 | T=2048 CTA | CTA/chunk-head | tile | WG / waves | 说明 |
|---|---:|---:|---|---|---|
| Stage 4 current | 2048 | 8 | token16 x V16 | 256 / 4 | 一个 wave 管一个 token16 row；Q/K score 在八个 V16 CTA 重复 |
| O0 | 512 | 2 | token64 x V64 | 256 / 4 | 四 waves 管四个 token16 row；同一 CTA 复用 lower score 到四个 V16 subtile |
| O1 | 1024 | 4 | token64 x V32 | 256 / 4 | O0 的唯一变化：两个 V16 subtile |
| vLLM | 512 | 2 | token64 x V64 | 256 / 4 | 实测 autotune: `BK=32,BV=64,num_warps=4,num_stages=2` |

当前 Avelang 的 lane mapping 和 O0/O1 相同：`lane_col=lane&15` 选择 V16/K16 列，
`lane_group=lane>>4` 与 `r=0..3` 选择四行，`wave_id` 选择 token16 row。vLLM 的精确
lane fragment permutation 由 Triton block-dot lowering 生成，不能从 Python 源码可靠恢复，
因此本报告不伪造 lane-level 映射。

vLLM 的真实 Python kernel `chunk_fwd_kernel_o` 在一个 CTA 内构造 `b_o:[64,64]` 和
`b_A:[64,64]`，完成 QH、QK、causal mask、`b_A @ V-new` 并直接写最终 output；没有
partial output global tensor、atomic 或中间 global accumulation。

## 2. 为什么 current 有 4x CTA 和大量重复工作

当前 grid 是 `NT * 8 heads * 8 V16 = 2048`。一个 chunk/head 的 64x128 Q 和 K block
在八个 CTA 被读取/变换；O0 是两个 V64 CTA，因此 Q/K 读取次数从 8 降至 2。H 和 V-new
是 V-specific，128-wide 的唯一 V16 ranges 仍必须覆盖一次，并不能随 CTA 数等比例消失。

从 source-level MFMA16 tile 模型看，数值所需的工作为 inter=256、lower-QK=80、
intra=80，共 416 个 tile calls/chunk-head。Stage 4 schedule 对每个 V16 CTA 重做完整
QK，模型为 1360；O0 用 lower 10 个 score tiles 且只在 V64 级重复，模型为 496。
该模型解释复用来源，但不能与 `SQ_INSTS_MFMA` 逐项相等：最终硬件 MFMA 指令数取决于
MFMA operand lowering、静态展开和硬件计数定义。资源决策使用下表的实际 rocprof 值。

## 3. O0/O1 实现

- O0: `_qwen_gdn_chunk_o_bf16_kernel_bt64_ownership_o0`，wrapper
  `qwen_gdn_chunk_o_bt64_ownership_o0`。
- O1: `_qwen_gdn_chunk_o_bf16_kernel_bt64_ownership_o1`，wrapper
  `qwen_gdn_chunk_o_bt64_ownership_o1`。

O0 先 stage Q `[64,128]`，只计算 ten lower `[16,16]` score tiles，并保存
`score_decay_bf16[4,4,16,16]`。之后四次顺序执行已有、已验证的 V16 inter/intra MFMA
microtile，output FP32 一次写回。没有改变 gate、scale、数值顺序、输出 dtype 或独立 cast。

O1 仅将 V64 改为 V32；它不是额外优化集合，也不是 autotune sweep。

## 4. 正确性

GPU pytest 结果：`108 passed in 31.54s`。

- O0/O1 对 Stage 4 FP32 staging 在 T=64/128/512/2048 的 random、zero-H、zero-V-new、
  inter-only、intra-only、small、high、cancellation 全部 `max_abs=0, mean_abs=0`。
- T=8192 random/cancellation/high 也全部 bit-exact。
- 四个 cross-token16 source 范围和所有 V16 boundary `[0:16,...,112:128]` 全部 bit-exact。
- Stage 4 对 vLLM `chunk_fwd_o` 的最大直接 body 差异为 `1.1281809e-05`；O0/O1 bit-exact
  于 Stage 4，因此并未放大该实现差异。
- 仅用于语义 smoke 的 O0/O1 full wrapper 在 T=64/512 对 current 的 `output_fp32`、public
  output、final state 都是零差异。

完整 Stage 6B full correctness matrix 是 `N/A_gate_not_met`：性能 gate 未过，按预注册规则
没有把候选作为 selected full graph 运行。这个 N/A 不是 correctness failure，也不能被当作
production correctness 通过。

未改动上游的回归也已单独执行：Stage 4 nonrecurrence KKT/W-U/chunk-o 与 hierarchical
solve 共 `36 passed in 30.82s`；冻结 asm-v0 bridge 共 `4 passed in 24.87s`。

## 5. Body Timing

相同输入、current stream、预分配 CUDA/HIP graph replay、warmup=20、repeat=100、5 sessions、
平衡顺序 `current,O0,O1,vLLM,vLLM,O1,O0,current`：

| T | current ms | O0 ms | O1 ms | vLLM ms | O0 gain vs current | O0-vLLM gap us |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.027080 | 0.034090 | 0.027320 | 0.017546 | -7.010 us | 16.544 |
| 1024 | 0.043624 | 0.036935 | 0.044506 | 0.020390 | 6.689 us | 16.545 |
| 2048 | 0.076073 | 0.062012 | 0.077395 | 0.029885 | 14.061 us | 32.127 |
| 4096 | 0.126629 | 0.113248 | 0.128391 | 0.052318 | 13.381 us | 60.930 |
| 8192 | 0.240077 | 0.192887 | 0.243682 | 0.088090 | 47.190 us | 104.797 |
| 16384 | 0.450830 | 0.384932 | 0.489127 | 0.178184 | 65.898 us | 206.748 |

| series | intercept ms | slope us/chunk |
|---|---:|---:|
| current | 0.017821 | 1.701165 |
| O0 | 0.017755 | 1.423759 |
| O1 | 0.013260 | 1.846943 |
| vLLM | 0.009919 | 0.648614 |
| current-vLLM | 0.007902 | 1.052551 |
| O0-vLLM | 0.007836 | 0.775145 |

O0 的长序列 slope 确实改善了 `0.277406 us/chunk`，但仍不满足 `<=0.65 us/chunk`。T=512
变慢表明更宽 CTA 的固定 LDS/barrier 成本没有被短文本摊销。

## 6. T=2048 rocprof 资源

| metric | current | O0 | O1 | vLLM Stage 6A full |
|---|---:|---:|---:|---:|
| median trace us | 67.060 | 52.919 | 68.202 | 14.662 |
| CTA | 2048 | 512 | 1024 | 512 |
| MFMA | 458752 | 188416 | 229376 | 81920 |
| VALU | 12918784 | 4054016 | 7108608 | 1728512 |
| SALU | 1495040 | 555008 | 860160 | 337920 |
| VMEM | 851968 | 335872 | 507904 | 71680 |
| LDS inst | 1343488 | 520192 | 712704 | 245760 |
| VGPR / AccVGPR / SGPR | 112 / 64 / 112 | 76 / 188 / 112 | 76 / 188 / 112 | 100 / 36 / 96 |
| LDS block B | 27136 | 33280 | 33280 | 0 |
| scratch B | 0 | 0 | 0 | 0 |
| occupancy | 17.897% | 8.864% | 9.271% | 9.473% |

O0 通过 CTA 4x、MFMA 2.435x、VMEM 2.537x、LDS 2.583x 的降低证明 ownership 改写确实生效；
但它未达到预设的 MFMA>=2.5x、VMEM>=3x，且其 AccVGPR/LDS/occupancy 退化限制了 latency。
O1 仅将 CTA 降为 2x，且 static/dynamic MFMA、DS、barrier 都变多，是明确负结果。

HSACO metadata 的 private segment、VGPR spill、SGPR spill 三者均为零。rocprof 的
`Accum_VGPR_Count` 不等同于 code-object `.agpr_count`，故本报告不将二者一一对应。

## 7. ISA/IR Evidence

每个 candidate 都保存了 ordinary JIT HSACO、AMDGCN disassembly、pre-link BC、disassembled
LLVM IR 和可 replay linker argv：

`codex_qwen_bt64_chunko_ownership_stage6b/{ir,isa}/{current,o0,o1}/`

O0 和 current 都有 56 个 static `v_mfma` text matches，O1 有 80；O0 的 static barrier
从 13 增到 19，O1 到 29。结合动态资源，这与 O0 的 score-cache reuse 和 O1 更密集的
V32 CTA 重复相一致。详见 `static_isa_analysis.md`；没有 compiler 或 assembly 改动。

## 8. Gate and Decision

| mandatory body gate | O0 | result |
|---|---:|---|
| T2048 speedup >= 1.50x | 1.227x | fail |
| T2048 body gap <= 25 us | 32.127 us | fail |
| gap slope <= 0.65 us/chunk | 0.775 us/chunk | fail |
| T8192/T16384 no regression | pass | pass |
| scratch/spill zero | pass | pass |

因此没有 selected variant，`full_integrated=false`，T=512..16384 full times、full gain 和
full gap slope 全部 **N/A**，而非估算值。不能把 14.061 us 的 body gain 直接声称为 full gain。

下一步不是 direct BF16 output/cast fusion，也不是 compiler/assembly：先关闭这个最多两
variant 的 ownership 分支并重新排序 Stage 6A 的剩余 body gaps。O0 的可观但不足收益应保留
为证据，不应静默替换 Stage 4 或 v24 production baseline。

## Evidence

- New code: `vllm_compare/qwen_gdn_bt64_chunko_ownership_stage6b.py`,
  `qwen_gdn_bt64_chunko_ownership_stage6b_o1.py`.
- Test: `vllm_compare/test_qwen_gdn_bt64_chunko_ownership_stage6b.py`.
- Benchmark: `vllm_compare/bench_qwen_gdn_bt64_chunko_ownership_stage6b.py`.
- HSACO dump helper: `vllm_compare/dump_qwen_gdn_bt64_chunko_ownership_stage6b_isa.py`.
- Reproduction commands, raw timing, profiler CSV, IR/ISA, decisions:
  `codex_qwen_bt64_chunko_ownership_stage6b/`.
