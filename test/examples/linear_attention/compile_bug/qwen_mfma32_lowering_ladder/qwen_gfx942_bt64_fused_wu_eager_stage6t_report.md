# Qwen gfx942 BT64 Stage 6T-Eager: Fused W/U

`timing_contract = "eager_public_api"`; `cuda_graph_used = false`.

## 结论

Stage 6T 的唯一 fused W/U schedule 已完整实现、通过 Eager public API correctness 和 expanded seed 稳定性，但**不通过性能晋级门槛**。F1 确实把 long-sequence gap slope 从 `3.455` 降到 `2.930 us/chunk`，回收 `0.525 us/chunk`；但 T=2048 相对本轮 Stage 6S 是 `-9.794 us`，即回归，而不是要求的至少 +10 us 收益。根据预注册 CASE C，不接入、不启动 Stage 6U；保留代码和证据为 experimental diagnostic。

所有权威测试均为 Eager public API；未使用 capture/replay。recurrence 仍是 hash-guard current-vLLM HSACO `632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e`。没有改 KKT、solve、chunk-o、V-new cast、final cast、compiler、assembly、production 或 v24。

## 实现

F0 kernel: `_qwen_gdn_wu_bf16_kernel_bt64_fused_stage6t_fp32`，public API `qwen_gdn_full_bt64_stage6t_fused_wu_fp32_eager`。F1 kernel: `_qwen_gdn_wu_bf16_kernel_bt64_fused_stage6t_bf16`，public API `qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager`。

两者均为 one-CTA-per-(chunk,value-head)，T=2048 为 256 CTA、WG=256；旧分离 W/U 为 4096 CTA。它们使用完全相同的 MFMA16、LDS、barrier 与 FP32 accumulation 顺序；只有最终 output store 是 F0 FP32 对 F1 BF16。F0 仍有两个 numeric W/U casts；F1 无 W/U casts 且不物化 FP32 W/U。

## Correctness

pytest `4 passed in 26.49s`。全图矩阵覆盖 T=64/128/512/1024/2048/8192、random/neutral/zero-beta/small/high-dynamic/cancellation/sparse-beta、zero/nonzero state、non-default stream；另有 T=2048 20 seeds 和 T=8192 5 seeds。99 条 public comparison 全部通过。

- 输出最大绝对误差: `0.0029296875`，阈值 `0.0078125`。
- final state 最大绝对误差: `0.015427827835083008`，阈值 `0.02`。
- F0/F1 public output 与 final state 在所有运行 case 中相等。

## 权威 Eager Full 时间

HIP event 与 wall-clock 都围绕同一次完整 public API 调用。单位 ms，前四列 HIP event，后四列 wall-clock。

| T | Stage6S | F0 | F1 | vLLM | Stage6S wall | F0 wall | F1 wall | vLLM wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.265433 | 0.255137 | 0.249769 | 0.372131 | 0.280835 | 0.270686 | 0.265022 | 0.387984 |
| 1024 | 0.292393 | 0.298161 | 0.292753 | 0.372130 | 0.307816 | 0.313804 | 0.308587 | 0.387474 |
| 2048 | 0.380383 | 0.394043 | 0.390177 | 0.426070 | 0.396642 | 0.409972 | 0.406902 | 0.442314 |
| 4096 | 0.597484 | 0.598586 | 0.588331 | 0.547390 | 0.613372 | 0.614524 | 0.604350 | 0.563694 |
| 8192 | 1.084264 | 1.054961 | 1.031687 | 0.793073 | 1.100434 | 1.071641 | 1.048126 | 0.809122 |
| 16384 | 2.079838 | 1.999439 | 1.951007 | 1.336598 | 2.097380 | 2.016750 | 1.969085 | 1.353719 |

T=2048 paired session mean gain: F0 `-13.905 us`，95% CI `[-14.718, -13.336]`; F1 `-10.780 us`，95% CI `[-12.471, -9.522]`。F1 CI 完全小于 0，明确未达门槛。wall-clock 与 HIP-event 对 F0/F1/Stage6S/vLLM 的相对方向一致。

## Slope

| implementation | intercept ms | slope us/chunk | vs-vLLM gap slope us/chunk |
|:--|--:|--:|--:|
| Stage6S | 0.160809 | 7.411 | 3.455 |
| F0 | 0.173075 | 7.067 | 3.112 |
| F1 | 0.172221 | 6.886 | 2.930 |
| vLLM | 0.308957 | 3.956 | 0.000 |

F1 的 long-T 没有回归：T8192/16384 相比 Stage6S 分别快约 52.6/128.8 us；但短中序列 T2048 回归约 9.8 us，所以全局 performance gate 失败。

## 资源诊断

| W/U path | CTA | WG | VGPR | AccVGPR | SGPR | LDS B | Scratch B | occupancy | MFMA | VALU | SALU | VMEM | LDS inst |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| old separate W+U, Stage6A context | 4096 | 256 | W52/U48 | W4/U8 | N/A | N/A | 0 | N/A | 524288 | N/A | N/A | 851968 | 1638400 |
| F0 public-profiled | 256 | 256 | 64 | 8 | 48 | 3072 | 0 | 8.523 | 524288 | 4518912 | 697344 | 491520 | 1114112 |
| F1 public-profiled | 256 | 256 | 64 | 8 | 48 | 3072 | 0 | 8.766 | 524288 | 4846592 | 696320 | 491520 | 1114112 |

F0/F1 无 scratch，未出现 v29 风格 resource cliff。F1 与 F0 MFMA/VMEM/LDS 相同，但 VALU 从 `4518912` 上升到 `4846592`；它说明单纯将 store 改为 BF16 没有把端到端 T2048 latency 转化为收益。private segment / exact spill 需要 standalone HSACO dump；本轮 Docker quota 在该补充收集前耗尽，因此标为 N/A。

## 决策

CASE C。F0/F1 都正确，F1 删除了两个 cast 和 FP32 W/U materialization，且 slope 改善超过次级 `0.30 us/chunk` 条件；但核心 T2048 gain gate 失败，不能保留为选定的 experimental public path，也不进入 Stage 6U。唯一下一动作是 **current-vLLM fused W/U golden bridge audit**，先审计 native fused W/U 的完整 ABI/中间量/ownership；本轮不创建 bridge、不修改 production。

## 证据

所有 raw samples、summary、correctness、rocprof CSV、contracts 和 commands 位于 `codex_qwen_bt64_fused_wu_eager_stage6t/`。
