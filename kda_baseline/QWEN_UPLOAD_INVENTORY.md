# Qwen GDN 迁移 / GitHub 上传清单

> 本清单只做整理，不执行上传、删除、移动或覆盖。后续确认清单后，再单独进行 Git commit/push。

## 1. 已确认的 GitHub 目标

- 用户 fork：`git@github.com:848267592/avelang.git`
- 上游仓库：`https://github.com/causalflow-ai/avelang.git`
- 当前本地分支：`agent/x2-z5b-submission-2026-08-18`
- 当前 HEAD：`1daeba8`
- 当前 HEAD 已存在于用户 fork 的同名远程分支。
- Qwen 历史优化分支：`qwen-gdn-optimization_ljd`
- Qwen 分支远程 HEAD：`7432ee2`

因此，当前仓库里已经提交并推送的内容不需要再次上传。`gh` 命令行工具未安装，但通过 Git SSH 的只读 `ls-remote` 已确认上述远程分支可访问。

## 2. 本地规模：不能整体上传

`test/examples/linear_attention/` 当前约 11 GB，约 2.5 万个未跟踪文件。主要来源是：

| 目录 | 约占用 | 判断 |
|---|---:|---|
| `compile_bug/` | 8.1 GB | 大量编译/捕获/回归中间产物，不能整体上传 |
| `rocprof_outputs/` | 1.9 GB | profiler 输出，不能整体上传 |
| `vllm_compare/` | 425 MB | 混有源码、日志、profile 和缓存，需要精选 |
| `current_code_6_18/` | 224 KB | 5 个小型源码文件，已经在 Qwen 远程分支 |
| `doc/` | 208 KB | 汇总报告候选，需精选 |

文件类型中有大量 `.json`、`.hsaco`、`.mir`、`.llir`、`.ttir`、`.ttgir`、`.mlir`、`.bin`、`.csv` 和 profiler 文件。这些不适合作为可重建环境的 Git 源码提交。

## 3. 第一优先级：已经在 GitHub，不要重复上传

### 3.1 Qwen 基础版本

`qwen-gdn-optimization_ljd` 分支 `7432ee2` 已包含 `current_code_6_18/` 下的 5 个版本：

- `qwen_gdn_chunked_avelang_v6_standalone.py`
- `qwen_gdn_chunked_avelang_v14_mfma_layout_fixed.py`
- `qwen_gdn_chunked_avelang_v17_mfma_layout_fixed.py`
- `qwen_gdn_chunked_avelang_v23_gdr_distributed_layout_fixed.py`
- `qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_fixed.py`

### 3.2 当前 handoff 包

当前分支 `1daeba8` 已在远程 fork，包含：

- `compile_bug/qwen_mfma32_lowering_ladder/qwen_chunk_o_z5b_handoff/`
- `compile_bug/qwen_mfma32_lowering_ladder/qwen_persistent_recurrence_r4_handoff/`
- 对应的 README、最小 Avelang/Triton kernel、benchmark、report、metadata 和必要的验证资料。

这些 handoff 包是最有迁移价值的 Qwen 代码，应作为后续重建时的主入口；不需要从服务器再次拷贝一份。

## 4. 建议后续补充上传的“小而有用”文件

下面是目前未跟踪、但有可能值得纳入仓库的文件。它们应先经过人工确认，再单独加入一个新的迁移 commit；本轮没有加入 Git。

### 4.1 推荐保留：参考实现、回归入口和最终说明

优先检查这些顶层文件是否仍被后续 handoff 覆盖：

- `test/examples/linear_attention/qwen_gdn_ref.py`
- `test/examples/linear_attention/qwen_gdn_recurrent_ref.py`
- `test/examples/linear_attention/qwen_gdn_compare_benchmark.py`
- `test/examples/linear_attention/qwen_gdn_v7_benchmark.py`
- `test/examples/linear_attention/qwen_gdn_chunk_cumsum_avelang_v3.py`
- `test/examples/linear_attention/test_qwen_gdn_chunk_cumsum_avelang_v3.py`
- `test/examples/linear_attention/qwen_gdn_chunked_avelang_v7.py`
- `test/examples/linear_attention/test_qwen_gdn_chunked_avelang_v7.py`
- `test/examples/linear_attention/qwen_vllm_container_avelang_probe_report.md`
- `test/examples/linear_attention/qwen_long_term_vllm_avelang_env_plan.md`
- `test/examples/linear_attention/qwen_long_term_vllm_avelang_env_build_report.md`
- `test/examples/linear_attention/qwen_avelang_v6_in_vllm_container_report.md`

这些文件的共同特点是体积小、能说明输入/输出 contract 或复现实验步骤。若后续确认 handoff 包已经完全替代某个旧版本，则旧版本只保留报告，不再上传重复 kernel。

### 4.2 `doc/` 报告

建议最多保留一份最终报告和少量索引：

- `test/examples/linear_attention/doc/report_final.md`
- `test/examples/linear_attention/doc/report_final_source_change_index.md`
- `test/examples/linear_attention/doc/report_final_evidence_index.json`
- `test/examples/linear_attention/doc/report_final_revision_log.md`

`report_final_before_stage6w_update.md` 属于历史快照，除非需要审计时间线，否则不建议和最终报告同时上传。

### 4.3 `my_prac/`

- `test/examples/linear_attention/my_prac/avelang_naive_prac.py`：只有在需要保留早期教学/实验入口时上传。
- `test/examples/linear_attention/my_prac/v4.py`：当前为空文件，建议排除。

## 5. 建议明确排除

以下目录/文件不应进入 GitHub 源码迁移提交：

- `test/examples/linear_attention/rocprof_outputs/`
- `test/examples/linear_attention/**/.rocprofv3/`
- `__pycache__/`、`.pyc`
- `golden_capture/`、`selected_case_tensors/`、`trace/`
- `pmc/`、`pmc_csv/`、`artifacts/`、`build/`、各类 cache
- `.hsaco`、`.bin`、`.llir`、`.ttir`、`.ttgir`、`.mir`、`.mlir`、`.amdgcn`
- 大型 profiler `.csv`、`.json` 和日志文件
- `compile_bug/` 下除已提交 handoff 包以外的自动生成 case
- `vllm_compare/` 下的 profiler/cache/中间结果，只保留经过筛选的源码和最终报告

理由：这些内容要么体积很大，要么与具体机器/ROCm/Triton 编译缓存绑定，换服务器后不能可靠复用；应改为记录生成命令、commit、镜像、GPU、ROCm/Triton/PyTorch 版本和结果摘要。

## 6. 建议的最终仓库结构

后续迁移完成后，Qwen 相关内容建议收敛为：

```text
test/examples/linear_attention/
├── current_code_6_18/                 # 已在 qwen 分支
├── compile_bug/qwen_mfma32_lowering_ladder/
│   ├── qwen_chunk_o_z5b_handoff/      # 已在当前远程分支
│   └── qwen_persistent_recurrence_r4_handoff/  # 已在当前远程分支
├── qwen_gdn_ref.py                    # 待确认是否补充
├── qwen_gdn_recurrent_ref.py          # 待确认是否补充
├── qwen_gdn_compare_benchmark.py      # 待确认是否补充
├── qwen_gdn_v7_benchmark.py           # 待确认是否补充
└── doc/                               # 精选最终报告
```

仓库中不保存 SGLang/vLLM/AITER 的整份源码；这些应记录仓库 URL、commit SHA、Docker image/tag/digest、依赖版本和重建命令。

## 7. 下一步建议

1. 先确认第 4 节的候选文件是否需要保留。
2. 对 `vllm_compare/` 和 `compile_bug/` 不做全量上传，只根据最终 Qwen 路线选择少量源码/README。
3. 增加一个环境重建清单（源码 URL/SHA、Docker 镜像 digest、ROCm/PyTorch/Triton 版本、运行命令）。
4. 最后再单独处理 Codex 聊天记录压缩归档；聊天记录不与源码混在同一个普通源码 commit 中。

本轮状态：**未上传、未删除、未移动、未修改任何既有 Qwen 代码。**

## 8. 第二轮精筛：`vllm_compare/` 和 `compile_bug/`

### 8.1 `vllm_compare/`：不再全量扩张

当前分支已经跟踪并推送了 17 个比较核心的 Python 文件，包括：

- Stage 6S BF16 recurrence bridge；
- Stage 6X KKT/solve handoff；
- Stage 6Z Z5B chunk-o；
- X2+Z5B full wrapper；
- 对应的 BF16 boundary、non-recurrence、cumsum 和 correctness test。

这部分已经在 GitHub，不需要把其余几百个历史 `v*.py`、`repro_*.py`、`profile_*.py`、`dump_*.py` 再全部上传。它们大多是已经被后续 handoff 替代的单点诊断。

### 8.2 需要补齐的 full-graph 依赖（推荐作为一个独立小 commit）

审计发现，远程已跟踪的 X2+Z5B benchmark 仍引用少量当前未跟踪依赖。若要保留“完整 Qwen BT64 full graph 可复现”能力，后续只需要考虑下面这组文件：

```text
test/examples/linear_attention/vllm_compare/
└── qwen_gdn_bt64_kkt_solve_handoff_stage6x_full.py

test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
└── codex_qwen_bt64_full_pipeline_stage2/
    ├── stage2_runner.py
    ├── bench_stage2.py
    ├── benchmark_methodology.md
    ├── commands.sh
    ├── bt64_numerical_contract.md
    ├── bt64_numerical_contract.json
    ├── full_operator_contract.json
    ├── full_operator_execution_graph.md
    ├── shape_layout_contract.md
    ├── stage_source_map.md
    ├── codex_final_summary.md
    └── final_decision.json
```

这组是 full graph 的 Python/contract/命令骨架，合计很小。但它本身还不是完整可运行闭包：`stage2_runner.py` 会导入 `qwen_gdn_bt64_gfx942_asm_v0_experimental.py`，后者需要 8.3 节的外部 asm source/bridge。不要带入同目录的 `golden_capture/` tensor、rocprof、benchmark CSV、`git_before*` 或大 JSON 结果；这些属于证据档案，不是重建依赖。

### 8.3 full graph 的外部 asm 依赖（可选，不是主推荐）

`qwen_gdn_bt64_gfx942_asm_v0_experimental.py` 通过外部 HSACO bridge 工作。若要在新服务器复现 Stage 6S/asm-v0 路线，最小源文件闭包是：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
└── codex_qwen_asm_v0_integration/
    ├── assembly/qwen_gdn_bt64_gfx942_asm_v0.s
    ├── assembly/build.sh
    ├── assembly/source_origin.md
    ├── avelang_integration/qwen_gdn_bt64_gfx942_asm_v0.py
    ├── avelang_integration/qwen_gdn_bt64_gfx942_asm_v0_bridge.cpp
    ├── avelang_integration/build.sh
    ├── avelang_integration/kernarg_abi.json
    └── tests/test_qwen_gdn_bt64_gfx942_asm_v0.py
```

`*.hsaco`、`*.o`、`*.so`、输入输出 `.bin`、rocprof 和 disassembly 不上传。新机器上用 `assembly/build.sh` 重新生成 HSACO，再用 `.py` 中的 SHA256 gate 验证。由于这是固定 gfx942 的实验性外部代码，这组应标成 **optional / experimental**，不要和便携的 Z5B/R4 handoff 混为生产实现。

### 8.4 `compile_bug/` 只保留少量最终报告

已提交的 Z5B/R4 handoff 自带报告索引；在 handoff 之外，最多补充这些最终级文档：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
├── qwen_gdn_benchmark_policy.md
├── qwen_gfx942_bt64_full_pipeline_stage2_report.md
├── qwen_gfx942_bt64_bf16_recurrence_full_contract_stage6s_report.md
├── qwen_gfx942_bt64_kkt_solve_handoff_stage6x_report.md
└── qwen_gfx942_bt64_x2_z5b_chunko_full_integration_report.md
```

这些文档分别记录统一 benchmark 规则、Stage 2 full graph、Stage 6S、Stage 6X 和 X2+Z5B 的最终 contract/结果。`qwen_gfx942_bt64_stage6z_native_chunko_report.md` 虽然信息很多，但包含历史快照和后续 addendum；同一主题已经由 Z5B handoff 的精选报告覆盖，暂不重复上传。

### 8.5 明确不选的内容

- `vllm_compare/` 中所有只做一次编译/地址/寄存器/packet 探针的脚本；
- `qwen_gdn_chunked_avelang_v10`--`v31` 的全部版本副本；稳定版本已经在 `current_code_6_18/` 或 handoff；
- `compile_bug/` 下 Stage 1--7 的完整实验目录、raw session、MIR/LLVM/ISA、HSACO 和 profiler；
- `repro_qwen_gdn_t8192_native_shaped_qh_wg128.py`、`qh_parity` 等 Q@H 单点探针。它们不是完整 KDA，也不能作为迁移后的主实现；除非以后专门复现 C16 WG128 compiler bug，否则不上传。

## 9. 当前筛选结论

后续真正需要处理的新增文件可以压缩为三档：

| 档位 | 内容 | 建议 |
|---|---|---|
| A | 已在远程的 Z5B/R4 handoff 和 17 个核心 `vllm_compare` 文件 | 保留，不重复上传；这是默认迁移方案 |
| B | full-graph Python/contract/报告 | 只有静态审阅或与 C 合并时有意义 |
| C | asm-v0 source/bridge/build | 若要真正运行 Stage 6S/X2 full graph，必须和 B 一起上传 |

因此建议按目标分两种方案：

1. **默认、最稳妥**：只依赖已经在远程的 Z5B/R4 handoff；不上传 B/C，直接在新服务器重新 clone SGLang/vLLM/AITER 并重建环境。
2. **需要精确复现旧的 X2/Z5B full graph**：成套上传 B+C；C 中只保留 `.s`、`.cpp`、`.py`、build script 和 contract，不上传 HSACO/对象文件/输入输出数据。

其余 `vllm_compare/` 和 `compile_bug/` 不按目录搬迁，留在服务器归档或只记录结果摘要。

> 重要更新：上面第 8--9 节是基于“复用旧远程分支”的早期筛选结果。现在用户决定
> 建立独立备份分支，并明确保留 Stage 6S/X2 full graph 及两个 T=8192 Q@H probe，
> 因此以同目录的
> [`QWEN_BACKUP_BRANCH_ALLOWLIST.md`](QWEN_BACKUP_BRANCH_ALLOWLIST.md) 为新的权威
> allowlist；本文件前面的旧分档不再作为最终上传依据。
