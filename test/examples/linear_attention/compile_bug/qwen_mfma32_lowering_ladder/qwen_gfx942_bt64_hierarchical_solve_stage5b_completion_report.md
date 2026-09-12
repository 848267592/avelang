# Qwen gfx942 BT64 Hierarchical Solve Stage 5B 完成报告

> 后续接入已完成：详见
> [Stage 5C 集成报告](qwen_gfx942_bt64_hierarchical_solve_stage5c_integration_report.md)。
> S0 在当前最高层 Stage 4 BT64 full path 的 T=2048 三会话中位数为
> `0.474906 ms -> 0.454215 ms`；v18 和 production dispatch 保持不变。

## 结论

Stage 5B 已完成到可接入的 standalone S0 solve：新增的 opt-in BT64
hierarchical FP32 solve 在 gfx942/MI300 上通过 source feature、数学正确性、
W/U consumer、性能、LDS、scratch 和 ISA 门。

- T=2048：v1 `0.030646 ms`，同一 benchmark harness 的 v18 `0.127149 ms`，
  speedup `4.15x`。
- v1 对 v6 的最大 solve 误差为 `3.73e-08`，最大 residual inf 为 `3.73e-08`。
- 256-thread CTA、8 KiB LDS、scratch `0 B`、VGPR `44`、AccVGPR `4`。
- ISA 实际含有 `v_mfma_f32_16x16x4_f32`，没有 BF16/F16 16x16x16 fallback。

v18 和任何 production dispatch 都没有改动。当前 v1 仍是 opt-in standalone
API，尚未改写 BT64 full pipeline；因此“可接入”不等于“已成为 production”。

## 1. 前置门与改动范围

历史 Stage 5B 被高层 API 门挡住：硬件和 Triton 已能用
`v_mfma_f32_16x16x4_f32`，但 AveLang source 没有对应 intrinsic。前一小步已
在 `amdgpu_mfma_signatures.h`、ROCDL wrapper、语言参考和 C++ test 中补齐
`al.amdgpu.mfma_16x16x4_f32_f32`。其独立 ISA/JIT 证据见
[`qwen_gfx942_fp32_mfma16_intrinsic_enablement_report.md`](qwen_gfx942_fp32_mfma16_intrinsic_enablement_report.md)。

本阶段新建而非替换基线的文件：

- `vllm_compare/qwen_gdn_solve_bt64_hierarchical_fp32_v1.py`
- `vllm_compare/test_qwen_gdn_solve_bt64_hierarchical_fp32_v1.py`
- `vllm_compare/bench_qwen_gdn_solve_bt64_hierarchical_fp32_v1.py`
- `vllm_compare/profile_qwen_gdn_solve_bt64_hierarchical_fp32_v1.py`
- `repro_fp32_mfma16x4_gemm_mapping.py`

没有改动：v18、v24、full production dispatch、AMDGPU RA、asm recurrence。

## 2. 数学与实现

输入/输出严格保持现有 solve ABI：contiguous FP32 `[1,T,8,64]`，`T > 0` 且
`T % 64 == 0`。每个 CTA 负责一个 `(chunk, value_head)`，求：

```text
X = (I + A)^-1
```

将 64x64 strict-lower matrix 切为 4x4 个 16x16 blocks。四个 diagonal block
保留 v6/v18 的 FP32 行递推；六个 lower block 用下列 DAG：

```text
X21 = -X22 A21 X11            X32 = -X33 A32 X22
X43 = -X44 A43 X33
X31 = -X33 (A31 X11 + A32 X21)
X42 = -X44 (A42 X22 + A43 X32)
X41 = -X44 (A41 X11 + A42 X21 + A43 X31)
```

每个 16x16 product 由四次 FP32 16x16x4 MFMA 累积。MFMA mapping 先由随机
16x16 GEMM repro 验证：

```text
A lane: A[lane & 15, 4*piece + (lane >> 4)]
B lane: B[4*piece + (lane >> 4), lane & 15]
C row:  4*(lane >> 4) + acc_i
C col:  lane & 15
```

该 repro 对 `torch.matmul` 的 max/mean abs 均为 `0`。

### LDS 计划

`x[7,16,16]` 保存四个 diagonal 与三个一阶 lower blocks，大小为 `7 KiB`；
`work[16,16]` 在 diagonal phase 复用为 row snapshot、在 block-DAG phase
复用为一个 FP32 matrix product workspace，大小为 `1 KiB`。总计正好 `8 KiB`。

X33、X43、X32 在其最后一个 consumer 完成后分别被 X31、X42、X41 复用，但
它们先写回全局输出。因此不会丢失最终矩阵元素，也不会扩大 LDS。

## 3. 实施中发现并修复的两个问题

1. 初版在 diagonal recurrence 前先写入单位对角。v6/v18 的递推对象是
   `M=-A`，单位阵必须在 strict-lower rows 全部完成后再加；提前加入会把第一条
   sub-diagonal 的贡献加两次。修正后 diagonal block 与 v18 对齐。
2. 初版将 `X42` 写到 0-based column block 2，即 `X43` 的全局位置，覆盖了
   正确的 X43，同时遗漏真正的 column block 1。隔离的“identity diagonal +
   only A43 nonzero”案例把它定位为 output offset bug，而不是 MFMA lane mapping
   或 wave control bug。改为 `BLOCK + lane_col` 后，完整 block DAG 通过。

这两项都记录在新测试/实现中，没有以放宽 tolerance 或 fallback 掩盖。

## 4. 正确性

完整原始输出在
[`standalone/s0_v1_pytest_results.txt`](codex_qwen_bt64_hierarchical_solve_stage5b/standalone/s0_v1_pytest_results.txt)。

| 输入 / T | max abs vs v6 | residual inf | 额外检查 |
|:--|--:|--:|:--|
| random strict-lower / 64 | `2.24e-08` | `2.24e-08` | multi-wave base case |
| random strict-lower / 128 | `1.86e-08` | `2.24e-08` | two chunks |
| random strict-lower / 512 | `3.73e-08` | `2.98e-08` | eight chunks |
| real KKT / 64 | `2.24e-08` | `2.24e-08` | also checks v18 |
| real KKT / 512 | `2.98e-08` | `2.98e-08` | also checks v18 |

下游 W/U consumer gate 也通过：T=512 时 `W` max abs `7.45e-09`、`U` max abs
`2.38e-07`。总 pytest 结果为 `7 passed in 13.53s`。

## 5. Solve-only 性能

每个数字是 HIP-event median，warmup=5、repeat=20，同一输入和 wrapper 计时
方式下比较 v1、v18、v6。原始 JSON 在
[`standalone/s0_v1_benchmark.json`](codex_qwen_bt64_hierarchical_solve_stage5b/standalone/s0_v1_benchmark.json)。

| T | v1 ms | v18 ms | v6 ms | v1 vs v18 |
|--:|--:|--:|--:|--:|
| 512 | `0.030185` | `0.122422` | `12.259758` | `4.06x` |
| 1024 | `0.028663` | `0.122141` | `18.318080` | `4.26x` |
| 2048 | `0.030646` | `0.127149` | `20.855812` | `4.15x` |

Stage 5A 的 standalone target 是 T=2048 `< 0.060 ms`；v1 以 `0.030646 ms`
通过。历史 vLLM body-only 数据为 `0.035613 ms`，但它不与本次完全同一 session，
所以这里只把它当参考，不宣称正式 end-to-end 胜过 vLLM。

## 6. T=2048 rocprof 与 ISA

Targeted rocprof 使用 7 个 dispatch，trace median 为 `13.700 us`。原始 CSV/HSACO
在 [`rocprof_s0_v1`](codex_qwen_bt64_hierarchical_solve_stage5b/rocprof_s0_v1)。

| metric | v1 S0 | Stage 5A v18 historical |
|:--|--:|--:|
| trace median | `13.700 us` | `110.224 us` |
| workgroup / waves | `256 / 4` | `128 / 2` |
| CTA count | `256` | `256` |
| LDS / scratch | `8192 B / 0 B` | `17920 B / 0 B` |
| VGPR / AccVGPR / SGPR | `44 / 4 / 32` | `96 / 128 / 112` |
| OccupancyPercent | `5.138` | `4.733` |
| SQ_INSTS_MFMA | `16384` | `0` |
| SQ_INSTS_VALU | `637440` | `2695424` |
| SQ_INSTS_SALU | `123904` | `2535680` |
| SQ_INSTS_VMEM | `40960` | `32768` |
| SQ_INSTS_LDS | `212992` | `1177856` |

v1 每 CTA 动态执行 64 条 MFMA，`64 * 256 = 16384`，与 counter 一致。静态
assembly 有 56 处 mnemonic，因为 wave0/wave1 的部分 Level-1 代码共享同一段
动态分支；这不改变动态执行数。

HSACO 反汇编明确包含：

```text
v_mfma_f32_16x16x4_f32 a[0:3], ...
```

`v_mfma_f32_16x16x16_*` 匹配数为 `0`。这证明 S0 使用的是新增的 FP32
intrinsic，而不是 BF16/F16 fallback。

## 7. 决策与下一步

S0 满足 Stage 5B standalone gate：source API 可用、正确、consumer-preserving、
scratch=0、LDS=8 KiB、T=2048 小于 0.060 ms。因此下一步可以将这一 **opt-in**
solve 接入 `qwen_gdn_full_bt64_gfx942_asm_v0_experimental.py`，随后重跑该 full
pipeline 的冻结 correctness matrix 和 stage timing。

不要修改 v18 或 production dispatch；BT64 full pipeline 仍有已知 generic KKT、W/U
和 chunk-o 瓶颈，solve 的单独成功并不等价于 full-path 性能成功。

## 8. 复现

完整命令见
[`commands_s0_v1.sh`](codex_qwen_bt64_hierarchical_solve_stage5b/commands_s0_v1.sh)。
机器可读决策见
[`s0_v1_final_decision.json`](codex_qwen_bt64_hierarchical_solve_stage5b/s0_v1_final_decision.json)。
