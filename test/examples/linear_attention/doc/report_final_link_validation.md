# `report_final.md` 静态验证记录

执行日期：2026-07-21。此验证只读取文档和已有证据，不运行 GPU benchmark。

## 结果

| 项目 | 结果 | 说明 |
|---|---|---|
| 原报告备份 | 通过 | `report_final_before_stage6w_update.md` 存在且非空；备份 SHA256 为 `fe1a3bff31c7bbb15229e1d3c520349a7f32389ef84873e93e5ca280e84e018a` |
| 更新后正式报告 | 通过 | `report_final.md` 非空；最终扩写后验证时 SHA256 为 `4ae10b28f89dd456b1eb2fe18baba4734f9f931e7692b440e25249513687f4f1` |
| 必需章节/术语 | 通过 | 包含 Stage 4、5A--5F、6A、6B、6R、6S、6T、6T-Golden、6U、6V、6W、Stage 6W clustered confirmation、Eager public API、`cuda_graph_used = false`、`current-vLLM specialization` 与未实施的 KKT-to-solve handoff |
| 正文相对 Markdown 链接 | 通过 | 扩写后检查到 32 个，32 个目标均存在；旧版残留的两个绝对路径已转换为相对路径 |
| 证据索引路径 | 通过 | `report_final_evidence_index.json` 的 65 个仓库相对证据路径均存在；v29 historical exact-LTO raw 目录缺失已明确记录为 availability note，不伪造路径 |
| JSON | 通过 | `report_final_evidence_index.json` 可解析 |
| CSV | 通过 | `report_final_stage_timeline.csv` 为 19 条阶段记录、13 列；每行列数一致 |
| Git whitespace | 通过 | `git diff --check` 返回成功 |

## 验证边界

- 此记录不证明 Markdown 在某个远端 renderer 的视觉样式，只验证本仓库相对链接目标存在。
- 这不是性能复测。所有性能结论仍来自链接到的既有 CSV/JSON/阶段报告。
- 工作区在开始前已经包含大量用户/实验变更；本任务只在 `test/examples/linear_attention/doc/` 新增或重写文档，没有清理或回退任何既有内容。
