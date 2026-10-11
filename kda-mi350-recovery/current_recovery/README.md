# KDA / Avelang MI350 Recovery Archive

这是一个保持证据边界的 KDA 工程归档，汇集从本地会话和文件中恢复的 Avelang KDA 源码、测试、benchmark 与研究记录。仓库所有者已于 2026-10-11 明确授权将此材料发布到公开仓库的独立恢复分支。它不是可直接构建的完整项目，也没有在本机重新运行恢复代码。

## 首要源码

- `kernels/k2/A_20260927_70_BW16_TRUE_W4_REBUILD.py` 是完整恢复的 native Avelang BW16/W4 K2 源码。SHA-256 与记录的 golden 完全相同：`a496c2b2cefa4bf0ff475049876d6db34091fe64e73a586498e83a61c1174801`。
- A70 历史报告记录它在 MI300X 上的 correctness PASS；这不是本仓库重新验证的结果。源码文件头有遗留的 A69 描述，原始字节为保留哈希而未改写。
- `kernels/k1/C26/` 与 `compiler/B66_B67/` 只含截断片段；C23 和完整 B66/B67 patch 均未恢复。
- A69 保留截断源码片段、部分 patch 和 ownership 记录；这些材料不能组成可执行 kernel。

## 目录

- `kernels/`：A70 完整源码及 K1/A69 不完整材料。
- `compiler/`：B66/B67 的部分源码检查片段。
- `tests/`：标成 RECONSTRUCTED 的 K1/K2 correctness、trace 和 ownership 工具。
- `benchmarks/`：D73 两份完整历史 runner、D71 重建 runner、本地 benchmark 参考文件。
- `analysis/` 与 `reports/`：布局、ownership、资源审计与机器级分析材料。
- `references/`：从恢复报告整理的 workspace ABI 和 sweep 证据状态。
- `patches/A69/`：A69 相关 patch/diff 证据，不是完整源文件。
- `FILE_MANIFEST.json`：逐文件 SHA-256、分类及来源索引。

## 先读

1. `RECOVERY_STATUS.md`：每个版本的恢复状态。
2. `DEPENDENCIES.md`：缺少的编译器、运行时和 sweep 依赖。
3. `MI350_MIGRATION_PLAN.md`：迁移时需要重新建立的验证顺序。
4. `LICENSE_STATUS.md` 与 `SECURITY_REVIEW.md`：许可及归档筛查边界。

`EXACT_RECOVERED` 表示完整字节序列来自历史完整写入/保留源码，独立 golden 匹配会单独说明。`RECONSTRUCTED` 表示从完整历史字面量或可验证记录重建；`PARTIAL` 表示截断片段或不完整补丁链；`METADATA_ONLY` 表示只有报告/版本记录，没有源码。新写的说明文档在清单里标作 `NEW_DOCUMENT`。

## B70 machine-level layout evidence

The B70 materials are isolated four-wave MFMA ownership and data-movement probes. They do not modify the A69 production kernel and are not a complete K2 implementation. See [B70_RECOVERY_STATUS.md](reports/B70_RECOVERY_STATUS.md) for provenance, missing JSON outputs, dependencies, and MI350 validation limits.