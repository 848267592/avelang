# 事实冲突、口径差异与更正记录

## 1. 数字不是天然可互相相减

| 冲突/差异 | 正确解释 | 本报告采用的处理 |
|---|---|---|
| Stage 4/5 的 HIP event、Stage 6A Graph、Stage 6T--W Eager 数字不同 | harness、预分配、公开 API、session 和 GPU 状态不同 | 不跨表相减；每张表声明口径 |
| Stage 5B 早期报告称 source feature gate 不通过，completion report 称 S0 通过 | 中间完成了 FP32 MFMA16x4 intrinsic enablement | 按时间顺序保留两个事实，不把早期阻塞写成最终失败 |
| Stage 5C solve-only 约省 96 us、full 只省约 19 us | stage body 与 full graph 的依赖/调度状态不同；内部 event 还会扰动图 | solve body 成功不等于 full 可加和收益 |
| Stage 5E fixed buffer full 和 Stage 5C public full gain 不同 | 一个是私有预分配 audit harness，一个是公开 full path | 两者分别用于 pointer 审计与 public 结果，不互作 leaderboard |
| Stage 6S/6T/6U 多组 Eager 绝对值不单调 | 不同 session 的 clock/load 变化已在报告中记录 | 若存在同批 paired gain/CI，优先引用 paired gain |
| Stage 6W 5/9-session sweep 正向、12-session clustered run CI 跨零 | 后者被外部 context eviction 和百毫秒长尾污染 | 保留为“测量质量失败”，不写为 W1 无收益 |
| Stage 6W 附件文本要求 W1 仍为 candidate | 附件早于 2026-07-21 fresh paired shared-environment retest | 以最新机器可读 summary 和更新报告为准 |
| Stage 6W rocprof trace 约 1498 us 与无 profiler body 约 0.09 ms 相冲突 | PMC trace 对短 kernel 有严重 collector perturbation | trace 不做性能结论，只保留为反例 |
| asm-v0 与 current vLLM recurrence 的资源/时间差 | 二者 ABI、WG、BV、W/U/V-new dtype 与 HSACO hash 不同 | 不称其为同一个 kernel 的优劣比较 |

## 2. 最新 Stage 6W 更正的具体范围

权威相对排名来自
`../compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6w_paired_shared_retest/cluster_bootstrap_summary.json`：

- `timing_contract = eager_public_api`
- `cuda_graph_used = false`
- T=2048：W1 对 U1 省 `8.994 us`，session HIP 95% CI `[7.246, 10.545] us`；
- T=8192：W1 对 U1 省 `27.352 us`，session HIP 95% CI `[25.398, 29.163] us`；
- HIP 与 wall-clock 的 8/8 session paired medians 同方向；nested cluster interval 也为正；
- `promotion_scope = paired_shared_environment`，并非独占 GPU 或跨宿主绝对延迟声明；
- W1 没有成为默认 selector；它是当前 Avelang experimental baseline。

同一批中，W1 在 T=2048 是 vLLM 的 `0.8607x`（快），在 T=8192 是 `1.2259x`（慢）。因此绝不能写成“全面超过 vLLM”。

## 3. N/A 的含义

`N/A` 不是 pass，也不是 fail。它用于：未安全实现的 P1 packed store、未实施的 U2、被 profiler gate 阻止的因果采样、Docker/绑定能力限制，或尚未开始的 KKT-to-solve handoff。正文都将它们写成未完成、被阻塞或已停止，而不伪造结果。
