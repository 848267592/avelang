# Qwen GDN Next Decision After Stage 6A Full-Graph Gap Audit

## Decision

关闭 Stage 5 transient-state root-cause 支线后，Stage 6A 已在统一的 BT64
full-graph harness 中完成 Avelang-vLLM 审计。下一步只做一个动作：

```text
Stage 6B = native BT64 chunk-o ownership 优化
```

不要同时改 W/U、cast、KKT、solve、recurrence 或 production dispatch。

## Evidence

T=2048 的 strict HIP-event graph replay：

| implementation | full ms |
|:--|--:|
| Avelang Stage 4 BT64 | 0.335539 |
| vLLM authoritative BT64 | 0.188900 |
| gap | 0.146639 |

full gap 的线性部分是 `4.406 us/chunk`。可修改 body 中：

| stage | T=2048 gap us | slope us/chunk | T=16384 gap us |
|:--|--:|--:|--:|
| chunk-o | **46.169** | **1.196** | **309.600** |
| W/U | 36.214 | 0.990 | 255.339 |
| KKT | 18.828 | 0.878 | 219.206 |
| cast | 15.483 | 0.069 | 32.688 |
| solve | -14.902 | -0.100 | -41.601 |

`chunk_o` 是第一位，因为它同时拥有最大的立即可恢复差距和长序列 slope。实际 full
trace 中 Avelang chunk-o 相比 vLLM 有 4x CTA、5.6x MFMA、约 11.9x VMEM、约 5.5x
LDS instructions。recurrence 虽也慢，但其 asm-v0 实现和 ABI 被冻结，不是本轮候选。

## Stage 6B Boundary

- 只复制当前 native BT64 `chunk_o` 到新的实验文件。
- 只重做 token/value/tile ownership 与相应的 MFMA/LDS/global-load schedule。
- 维持 FP32 output staging 和独立 BF16 cast，以隔离因果；不要在同一实验中融合 cast。
- 不修改 asm recurrence、HSACO、v18/v1 solve、Stage 4 KKT/W/U、compiler/RA 或
  production selector。
- 先做 chunk-o-only correctness 和 T sweep，再跑 full graph replay。

成功判断应同时包含：public output/final-state 阈值、T=512--16384 slope、实际 full
dispatch resource 下降和 strict ABBA full latency。若 chunk-o 未能回收明显的 slope，
下一候选才是 W/U 的共享输入读取/单 dispatch 结构，而不是重开 transient-state 调查。

完整证据见
`compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_avelang_vs_vllm_full_graph_gap_stage6a_report.md`。
