# Qwen GDN 历史演练源码与报告归档

这是 `backup/qwen-kda-repro-2026-09` 的第一段历史归档范围。它和
`QWEN_BACKUP_BRANCH_ALLOWLIST.md` 中的 Stage 6S/X2+Z5B 最小运行时闭包、以及
`QWEN_SOURCE_REPORT_SUPPLEMENT.md` 中的 v3--v9/Stage 6--7 补充同时存在：本文件
负责保存 v10--v31 的演进、实验假设和最终结论。

## 纳入范围

本次归档按实际文件快照生成，共 **751 个文件、5,901,535 bytes（约 5.63 MiB）**：

| 范围 | 文件数 | 内容 |
|---|---:|---|
| `test/examples/linear_attention/vllm_compare/` 根目录 | 146 | 文件名含 v10--v31 的 Qwen 源码、bench、test、repro、profile/audit 脚本和对应报告 |
| `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/` | 603 | 各 Stage 的 Python/C++/头文件/构建脚本、Triton/汇编源码、README、分析报告、contract 文档 |
| 同目录的手写汇编源 | 2 | `assembly/` 或明确命名为 `source.s` 的源码 |

精确文件及 SHA-256 位于
[`QWEN_HISTORY_SHA256.txt`](QWEN_HISTORY_SHA256.txt)。该文件是恢复时的权威
校验清单；不要只根据目录名猜测文件是否完整。

## 版本覆盖

历史快照包括 v10、v11、v12、v13、v14、v15、v16、v17、v18、v19、v20、v21、
v22、v23、v24、v25、v26、v27、v28、v29、v30 和 v31。这里保留的是研究过程，
不是说每个版本都适合生产部署。当前冻结的长序列性能候选仍是 **X2+Z5B**：
T=512 保留 X2，已测 T>=1024 使用 X2+Z5B；具体结论见
`qwen_gfx942_bt64_x2_z5b_chunko_full_integration_report.md`。

## Z5B 完整性说明

Z5B 的发布入口、direct-Q-cache consumer、bench、full-graph test，以及它们的
15 个本地 Python import-closure 模块已经在主 Qwen SHA 清单中；Z5B 提交清单、
X2+Z5B 集成报告和 `qwen_chunk_o_z5b_handoff/` 的源码/报告也在本次分支中。
因此恢复源码后可以重新构建和运行 Z5B，但不能把旧机器生成的 HSACO 当作源码。

## 明确排除

以下内容不属于本次源码/报告归档：性能采样 CSV/JSON、rocprof session、输入输出
张量、`.hsaco`/`.o`/`.so`/`.bin`、compiler IR/MIR/ISA dump、`machine/`、
`isa/`、`ir/`、`rocprof*/`、`golden_capture/`、`raw_sessions/`、`sessions/`
等生成目录。它们可在新服务器按脚本重新生成，不影响演练逻辑复盘。

`QWEN_BACKUP_SHA256.txt` 仍校验原来的核心 87 文件；本文件的
`QWEN_HISTORY_SHA256.txt` 负责 v10--v31 历史源码/报告集合；
`QWEN_SUPPLEMENT_SHA256.txt` 负责后续补充，三者不要互相替代。
个别 vLLM lowering snapshot 虽然文件名是 `source.py`，内容实际是 TTIR/MLIR
文本且不能作为 Python 源码解析；它们也按生成工件排除。

## 重新生成/审计范围

归档选择规则是：

```text
vllm_compare 根目录：只选 .py/.md 且 basename 含 v10--v31；
compile_bug/qwen_mfma32_lowering_ladder：选源码/脚本/报告扩展名，排除上述生成目录；
额外保留 assembly/ 和 source.s 下的手写汇编源。
```

新的性能采样应放在本地工作目录或单独压缩归档，不要把采样结果重新混入这个
快速恢复分支。
