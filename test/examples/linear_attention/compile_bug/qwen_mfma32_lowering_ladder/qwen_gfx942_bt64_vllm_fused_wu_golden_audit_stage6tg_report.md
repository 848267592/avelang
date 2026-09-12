# Qwen gfx942 BT64 Stage 6T-Golden Audit

## 总结

本轮是严格 audit-only：没有修改 F1、Stage 6S、recurrence、KKT、solve、chunk-o、编译器、汇编或 production selector，也没有创建 vLLM W/U bridge。所有权威性能和正确性都来自完整 Eager public API，`cuda_graph_used=false`；IR、ISA、HSACO 和 PMC 只作诊断。

真实 native vLLM W/U 为 `recompute_w_u_fwd_kernel`。T=2048 正常 eager specialization 是 4 warps / 2 stages、grid `(32,8,1)`、256 CTA、WG=256、每 `(chunk,value-head)` 一个 CTA，HSACO 为 `d7cd8744d597dcc0a09ff0430770b6e97e23f7c8529240311f9624f1e0e0dfcd。

F1 将旧分离 W/U 的 CTA 数从 4096 降为 256 是真实的 launch/round-trip 改善，但总 MFMA 没有下降：F1 每 CTA 仍为 2048 条动态 MFMA，native vLLM 为 128。F1 的 16x 来自 16x16x16 几何 2x、main+residual 2x、每 wave 的 predicated lane-group fragment 4x。因此 256 CTA × 2048 = 524288；这不是 CTA 融合失败。

主根因是 **CASE C**：F1 需要 FP32 `a_solved`，native W/U 需要 BF16 `A`。layout 相同，但 solve-to-W/U 的数值 ABI 不兼容。次要因素是 F1 16x 动态 MFMA 和 BF16 scalar-store lowering。

## 实际 specialization

| T | config | grid | CTA | WG | HSACO |
|---|---|---|---|---|---|
| 512 | 4w/2s | (8,8,1) | 64 | 256 | d7cd8744d597dcc0 |
| 2048 | 4w/2s | (32,8,1) | 256 | 256 | d7cd8744d597dcc0 |
| 8192 | 2w/3s | (128,8,1) | 1024 | 128 | dee9be336292ad53 |
| 16384 | 2w/3s | (256,8,1) | 2048 | 128 | dee9be336292ad53 |

T=512/2048 共享一个 4w/2s specialization；T=8192/16384 共享一个 2w/3s specialization。跨完整 T sweep 不稳定，但 T=2048 重复 capture 保持同一 hash/config。

## 数学、ABI 与 ownership

- F1 W：`A_fp32 * beta * exp(g)` 先做 BF16 main MFMA16，再做 BF16 residual MFMA16；U 同理但没有 `exp(g)`。
- native W：`dot(A_bf16, BF16(K*beta*exp(g)))`；U：`dot(A_bf16, BF16(V*beta))`。
- native source 先执行并存储两个 U 的 BV=64 block，再执行并存储两个 W 的 BK=64 block；没有 residual dot，也没有可共享 W/U coefficient matrix。
- T=2048 两边都为 32 chunks、256 chunk-heads、256 CTA、每 chunk-head 一个 CTA、正常 WG=256/4 waves。
- K/V、beta/g、W/U 的 layout 对齐；决定性差异为 A：native BF16、F1 FP32。`direct_abi_compatible=false`，需要 dtype/upstream contract change，不需要 layout transform。

## 归一化工作与 store

| implementation | W main | W residual | U main | U residual | MFMA/CTA | MFMA/dispatch |
|---|---|---|---|---|---|---|
| F1 | 512 | 512 | 512 | 512 | 2048 | 524288 |
| native vLLM | 64 | 0 | 64 | 0 | 128 | 32768 |

F1 的 F0/F1 static MFMA、LDS 和 barrier 数相同。额外 327680 VALU 定位在 BF16 输出 epilogue：F0 是 `global_store_dword`；F1 是 `global_store_short_d16_hi`，前面有 `v_bfe_u32`、`v_add3_u32`、`v_or_b32`、`v_cmp_u_f32` 和 `v_cndmask_b32`。native vLLM 是 `buffer_store_dwordx2`。PMC 无法把 VALU 精确逐 opcode 分账，因此更细拆分为 N/A。

## 资源与 Eager 基线

F1 standalone HSACO 是 VGPR 72、AGPR 8、SGPR 35、LDS 3072 B、private 0、无 VGPR/SGPR spill。native normal T=2048 HSACO 是 VGPR 180、AGPR 16、SGPR 101、private 0、无 spill。F1 并非由 register/scratch 压力导致。Triton cache `shared=8192` 与 code-object fixed group segment=0 不一致，报告保留该事实，不作未证明解释。

| T | Stage6S ms | F1 ms | vLLM ms | F1-S us | F1/vLLM |
|---|---|---|---|---|---|
| 512 | 0.272184 | 0.262711 | 0.367967 | -9.474 | 0.714x |
| 2048 | 0.381487 | 0.397290 | 0.415096 | 15.803 | 0.957x |
| 8192 | 1.088638 | 1.038984 | 0.810765 | -49.654 | 1.281x |
| 16384 | 2.078308 | 1.955686 | 1.332881 | -122.622 | 1.467x |

所有表中的 baseline 是五个 session median 的 median，warmup=30、repeat=200、balanced order，HIP event 包住完整 eager public API，allocation 计入，不使用 graph。F1 在 T=2048 比 Stage6S 慢，8192/16384 更快，方向与旧 Stage6T 一致。

## Correctness 与唯一下一步

Eager public full correctness 覆盖 T=64..8192、zero/nonzero initial state、zero/sparse beta、neutral/high-dynamic/cancellation/small-value 与 non-default stream。public output 最大 abs `0.0029296875`，final-state 最大 abs `0.0102265477`，均在冻结阈值内。W/U 的中间差异来自 FP32/BF16 solve boundary，不能用最终 BF16 output 掩盖。

唯一下一步：**Stage 6U BF16 solved-boundary propagation**。只改变 solve-to-W/U storage boundary，并在同一 Eager full contract 下验证；不创建 golden bridge，不需要 compiler/assembly 修改，也不先做 tile sweep。

## 回归

`16 passed in 27.74s`。执行的集合包括 Stage 6T 完整 Eager public API 对照、non-default stream、Stage 6S BF16 recurrence hash/非法输入/solve contract，以及本轮静态审计检查。旧 `torch.cuda.graph` replay case 被刻意排除；审计 runner 与 F1 public path 均扫描确认没有 `CUDAGraph`、`torch.cuda.graph(...)` 或 `.replay()` 调用。

## 产物

- 审计目录：`test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_vllm_fused_wu_golden_audit_stage6tg`
- 最终 JSON：`test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_vllm_fused_wu_golden_audit_stage6tg/final_decision.json`
