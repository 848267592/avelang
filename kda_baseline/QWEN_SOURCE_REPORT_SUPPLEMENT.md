# Qwen 源码与报告补充归档

本文件记录 `backup/qwen-kda-repro-2026-09` 在原始 87 个核心文件和 751 个
历史文件之后新增的源码/报告补充范围。补充内容用于完整复盘 Qwen GDN 从早期
v3--v9 到 Stage 6/7 的演进；性能采样数据、模型权重和机器生成物仍不上传。

## 本次补充

本次新增 **413 个非编译器源码/报告/迁移文档文件**，约 3.91 MiB：

| 范围 | 文件数 | 内容 |
|---|---:|---|
| `test/examples/linear_attention/vllm_compare/` | 331 | v7--v9、Stage 6/7、persistent recurrence、direct-K64、Q@H、测试/bench/audit/repro 脚本和报告 |
| 早期 `test/examples/linear_attention/` | 30 | v3--v7、naive/reference、早期测试、环境报告 |
| `test/examples/linear_attention/compile_bug/` | 26 | MFMA/Qwen parity 报告、早期复现脚本、Stage 2 capture helper 和真实 Python 源码 |
| `test/examples/linear_attention/doc/` | 7 | final review/report 文档 |
| `test/examples/doc/` | 5 | Qwen 完整学习复盘、阶段原始报告附录和 v7 报告 |
| 根目录 Qwen recurrence 报告 | 12 | persistent recurrence 设计、尾部问题和调度分析 |
| benchmark / 迁移清单 | 2 | v6 standalone benchmark、旧上传清单 |

精确路径和 SHA-256 见 `QWEN_SUPPLEMENT_SHA256.txt`。该清单只覆盖本次补充，
不会替代 `QWEN_BACKUP_SHA256.txt` 或 `QWEN_HISTORY_SHA256.txt`。

## 编译器源码快照

同时保存当前 Avelang 编译器工作树的 **45 个源码/测试文件**：19 个已修改的
tracked 文件和 26 个新增的 `lib/` 源码/测试文件。它们是当前开发快照，不等同于
已经发布的稳定实现；恢复后应先查看提交说明和 `git diff`，再决定是否构建。

生成的 `_avelang_bindings*.so`、HSACO、对象文件、IR/MIR/ISA、rocprof 和采样
结果不在本补充中。

## 版本覆盖说明

原历史归档覆盖 v10--v31；本补充补上当前可读的 v3--v9 代码和报告。某些同名
`source.py` 实际是 Triton TTIR/MLIR 快照，因此仍按生成物排除；需要时由对应
capture 脚本重新生成。

## 恢复检查

在仓库根目录运行：

```bash
grep -E '^[0-9a-f]{64}  ' kda_baseline/QWEN_SUPPLEMENT_SHA256.txt \
  | sha256sum -c -
```

该补充和核心/历史清单都通过后，才认为源码与报告迁移闭包恢复完成。
