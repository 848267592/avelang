# `report_final.md` 修订日志（Stage 6W 更新）

本日志记录本次以历史 `report_final.md` 为原始文本、重建为 Stage 4--6W
教程的实质性修订。正式性能结论统一遵从 `timing_contract =
eager_public_api` 与 `cuda_graph_used = false`；Graph replay、private body
和 rocprof trace 只保留为诊断证据。

| 原章节/主题 | 旧内容或风险 | 修订后表述 | 原因 | 主要证据 |
|---|---|---|---|---|
| 标题与摘要 | 以 Stage 4 为主，后续内容作为“续篇” | 重组为 Stage 4--6W 连续教程 | 后续结论会反向修正 Stage 4 的瓶颈判断 | `qwen_gfx942_bt64_*_report.md` |
| Stage 5B | 早期 API gate 失败容易被理解为 solve 未完成 | 记录为“feature gate 曾失败，后续补齐 intrinsic 后 S0 standalone 成功” | 同一阶段存在早期报告和 completion report | `hierarchical_solve_stage5b_report.md`、`..._completion_report.md` |
| Stage 5C | 把分段 HIP event 数字与 full 相减解释为绝对抵消 | 标为受扰动诊断；正式结论只来自无内部 event 的 full API | stage event 改变连续执行状态 | `hierarchical_solve_stage5c_integration_report.md` |
| Stage 5D/5E | 有利前驱容易被写成确定 cache/clock 根因 | 仅称“瞬态执行状态证据”，明确机制未解析 | canonical data、same pointer、warm/perturb 排除了部分变量，未唯一定位硬件机制 | `downstream_state_coupling...`、`direct_common_out...` |
| Stage 5F | profiler 路线可被误解为尚待继续 | 记录为 stop rule 已触发，关闭 transient-state 支线 | trace-only 扭曲量超过低扰动门槛 | `transient_state_stage5f_report.md` |
| Stage 6A | Graph audit 的 full 数字可被误作权威排行榜 | 明确为结构诊断，后续所有正式排名使用 eager public API | benchmark contract 已升级 | `avelang_vs_vllm_full_graph_gap_stage6a_report.md` |
| asm-v0 | 可能被称为“当前 vLLM recurrence” | 更正为旧 Triton lineage 的 FP32 ABI；不是退化的 Avelang kernel | Stage 6R 捕获当前 vLLM BF16/WG128 specialization | `recurrence_reconciliation_stage6r_report.md` |
| Stage 6S | 把 bridge 当作 Avelang code generation | 标为 hash-guarded external HSACO bridge | 当前 vLLM HSACO 被复用，Avelang 负责显式合约边界 | `bf16_recurrence_full_contract_stage6s_report.md` |
| Stage 6T | 旧报告写为“尚未执行” | 补全 F0/F1 实际 Eager 实验与 Golden Audit | 已有完整测试、bench、source/ISA 证据 | `fused_wu_eager_stage6t_report.md`、`vllm_fused_wu_golden...` |
| Stage 6U | FP32 solved 可能被写成永久 ABI | 解释 P0 只将最终 store 改为 BF16，C0 消除 residual | BF16 producer/consumer contract 已被验证 | `bf16_solved_boundary_stage6u_report.md` |
| Stage 6V | “MFMA 四倍下降”可能被等同于 full gain | 写清 V0 指令 gate 通过、V1 Eager promotion 失败 | select/cndmask 和 VGPR 上升抵消收益 | `predicate_collapse_stage6v_report.md` |
| Stage 6W 初始 sweep | 5/9 session 正向结果曾只是 candidate | 保留为候选证据，不能单独晋级 | call-level bootstrap/后续污染样本不稳定 | `bf16_chunko_boundary_stage6w_report.md` |
| Stage 6W 历史 clustered run | “CI 跨零”被过度解释为 W1 没收益 | 更正为：历史共享环境长尾使该 run 不适合作绝对晋级否定 | 10--1000 ms outlier 和 eviction 破坏微秒级效应判定 | `stage6w_cluster_confirmation...` |
| Stage 6W 最新状态 | 附件中的旧任务仍要求把 W1 写成 candidate、U1 为 experimental baseline | 更正为 W1 是当前 **Avelang experimental baseline**，范围为 `paired_shared_environment` | 2026-07-21 8 process-isolated paired retest 的 session CI、wall 和 nested sensitivity 均为正 | `codex_qwen_bt64_stage6w_paired_shared_retest/cluster_bootstrap_summary.json` |
| production/default | experimental graph 与默认生产路径容易混淆 | 明确默认/production 仍不变；W1 未提升 selector | 本任务和相关实验都没有改 selector | Stage 6U/6W reports、源码审计 |

## 不变项

- 原报告已原样备份为 `report_final_before_stage6w_update.md`。
- 本次只修改 `doc/` 中的文档和索引；没有修改 kernel、compiler、RA、HSACO、assembly、v24 或 selector。
- `KKT FP32 a -> solve` handoff 消除只登记为下一候选，未实施。
