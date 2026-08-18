# R4 之后实验报告索引

这个目录是 R4 最小代码包的学习资料部分。报告正文均从工作区已有报告复制，
没有为了上传而改写实验结论。原始 MIR、LLVM、ISA、HSACO 和 rocprof 工件没有
复制进来；需要逐条复现时，再查看旁边的 `qwen_persistent_recurrence_r4_handoff_full`
原始档案。

## 推荐阅读顺序

| 顺序 | 文件 | 作用 | 状态 |
|---:|---|---|---|
| 1 | `00_r4_after_experiments_report_cn.md` | R4 之后完整时间线、性能、根因和 compiler 文件地图 | 总复盘 |
| 2 | `01_r4_lds_mediated_retile_report.md` | R4 正式 native full-recurrence 基线 | 晋级为 native 基线 |
| 3 | `02`--`08` | R4-tail 的 pred/layout/I-O/LDS/dual-dot 后续尝试 | 分支诊断 |
| 4 | `05_r4_tail_state_kv_pred_mn_swap_v4_io_v3_performance_report.md` | R4-tail v3 的独立性能结果 | 不能直接替代 R4 主线 |
| 5 | `09_r5_superblock_lowering_report.md` | R5 full superblock lowering | 正确但性能 No-Go |
| 6 | `10`--`13` | R4-tail 的 issue、ATT、gap budget 后续归因 | 机器工作审计 |

## 如何理解“最优”

- R4 是本包要交接的**纯 Avelang native full-recurrence 基线**。
- R5 改变了 MLIR/LLVM/MIR/ISA，但动态工作没有下降且比 R4 慢，因此没有晋级。
- R4-tail v3 在自己的分支和 Eager 口径下有更好的局部结果，但不能与 R4
  主报告的 body benchmark 直接混排；若要宣布全面替代，必须用同一 harness 重测。
- current-vLLM Triton 是机器 oracle/对照，不是本包里的 Avelang 实现。

## 报告清单

### 主线

- `00_r4_after_experiments_report_cn.md`
- `01_r4_lds_mediated_retile_report.md`
- `09_r5_superblock_lowering_report.md`

### R4-tail 后续分支

- `02_r4_tail_pred_mn_axis_swap_report.md`
- `03_r4_tail_state_kv_pred_mn_swap_report.md`
- `04_r4_tail_state_kv_pred_mn_swap_v4_io_report.md`
- `05_r4_tail_state_kv_pred_mn_swap_v4_io_v3_performance_report.md`
- `06_r4_tail_iopacket_wide_access_report.md`
- `07_r4_tail_u_vnew_lds_bridge_report.md`
- `08_r4_tail_state_kv_dual_dot_report.md`

### 后续归因

- `10_r4_tail_issue_causality_report.md`
- `11_r4_tail_production_att_attribution_report.md`
- `12_r4_tail_vs_current_vllm_full_att_bucket_report.md`
- `13_r4_tail_vs_current_vllm_gap_budget_report.md`
