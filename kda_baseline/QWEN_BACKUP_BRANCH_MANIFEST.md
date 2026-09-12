# Qwen / KDA 完整迁移：Step 1 清单快照

> 本文件是完整迁移的第一步：只记录新独立备份分支的迁移范围和审计结果。
> 本轮不创建分支、不 `git add`、不 commit、不 push，也不修改任何 Qwen/KDA 实现。

## 1. 迁移目标

拟创建一个不依赖旧分支内容的独立备份分支：

```text
backup/qwen-kda-repro-2026-09
```

未来在新服务器上应能直接 clone 该分支，看到本次 Qwen GDN / KDA 复现所需的
源码、bridge、capture/build 脚本、contract 和结论报告；官方 SGLang、vLLM、
AITER 源码不复制到该分支，而是按固定 URL、commit/tag、Docker digest 和重建
命令重新下载。

本次明确纳入：

- Stage 6S BF16 recurrence full graph；
- Stage 6X/X2 KKT-to-solve full graph，以及 X2+Z5B chunk-o；
- 两个 `T=8192` Q@H 指标对齐 probe：
  - `repro_qwen_gdn_t8192_native_shaped_qh_wg128.py`
  - `repro_qwen_gdn_t8192_qh_parity_wg128.py`；
- 运行这些路径所需的 Stage 2、Stage 5B、Stage 6A、Stage 6R、Stage 6S、asm-v0
  的源码入口、bridge、contract 和必要报告。

权威的逐文件 allowlist 位于：

[QWEN_BACKUP_BRANCH_ALLOWLIST.md](QWEN_BACKUP_BRANCH_ALLOWLIST.md)

## 2. 审计快照

| 项目 | 结果 |
|---|---:|
| 审计时间（UTC） | 2026-09-12 02:50 左右 |
| 审计根目录 | `/home/jiandongliu/project/avelang` |
| 当前分支 | `agent/x2-z5b-submission-2026-08-18` |
| 当前 HEAD | `1daeba8fb93e430590d6dfa6019a1d0023dbb401` |
| allowlist 文件数 | 87 |
| allowlist 总大小 | 972,130 bytes（约 0.93 MiB） |
| 已跟踪文件 | 23 |
| 当前未跟踪但需纳入文件 | 64 |
| 缺失文件 | 0 |
| 当前已修改的其他 tracked 文件 | 19（保留，不在本次清单内擅自处理） |

87 个文件均已在宿主机目录中找到；本快照不包含生成的二进制和大型 profiler/compiler
工件。

## 3. 按功能分组统计

| 分组 | 文件数 | 字节数 | tracked / untracked | 作用 |
|---|---:|---:|---|---|
| `vllm_compare` | 21 | 356,814 | 17 / 4 | Stage 6S/X2 full graph、KKT/solve、chunk-o、两个 Q@H probe |
| Stage 2 full-pipeline skeleton | 15 | 42,105 | 0 / 15 | runner、benchmark、contract、execution graph |
| Stage 6A audit | 2 | 79,208 | 0 / 2 | Stage 6R capture 所需 producer-chain audit 和冻结 contract |
| Stage 6R recurrence bridge | 11 | 62,372 | 5 / 6 | current-vLLM capture、HIP bridge、ABI/launch/hash metadata |
| Stage 6S full contract | 13 | 105,209 | 0 / 13 | BF16 recurrence、correctness matrix、eager benchmark、contract |
| Stage 5B solve contract | 6 | 5,943 | 0 / 6 | hierarchical solve source contract 与重建命令 |
| asm-v0 integration | 8 | 204,741 | 0 / 8 | gfx942 assembly、Avelang adapter、HIP bridge、ABI、测试 |
| Q@H evidence | 4 | 2,454 | 0 / 4 | correctness/ISA/hash/代码对象小型证据 |
| replay 与最终报告 | 7 | 113,284 | 1 / 6 | replay 脚本、Stage 2/6S/6X/X2/Q@H 报告 |
| **合计** | **87** | **972,130** | **23 / 64** | **本次源码迁移闭包** |

## 4. 关键完整性检查

### 4.1 Stage 6S / X2 full graph

`vllm_compare` 中已确认包含以下关键链路：

- recurrence full Stage 6S；
- solved boundary / hierarchical solve；
- KKT solve handoff Stage 6X；
- `qwen_gdn_bt64_kkt_solve_handoff_stage6x_full.py`；
- X2+Z5B chunk-o full handoff；
- Stage 2 benchmark 对 `qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_fixed.py`
  的实际导入依赖；
- eager benchmark 和 full-graph test。

### 4.2 两个 Q@H 指标对齐入口

以下两个文件明确属于迁移内容，不再按“一次性 probe”排除：

```text
test/examples/linear_attention/vllm_compare/repro_qwen_gdn_t8192_native_shaped_qh_wg128.py
test/examples/linear_attention/vllm_compare/repro_qwen_gdn_t8192_qh_parity_wg128.py
```

同时保留 `qwen_t8192_native_shaped_qh_wg128/` 下的四个小型证据文件，以及
`replay_qwen_v29_lto_mir.py` 和最终 Q@H parity 报告；不迁移大型 compiler dump、
HSACO 或完整 ISA/MIR 工件。

### 4.3 Stage 6R → Stage 6A 依赖

`stage6r_capture_current_recurrence.py` 会导入 Stage 6A 的
`stage6a_full_graph_audit.py`，因此 Stage 6A 的源码入口和
`frozen_measurement_contract.json` 已列入，不能只迁移 Stage 6R 目录。

### 4.4 二进制重建策略

Stage 5B solve、Stage 6R current-vLLM recurrence、Stage 6S/asm-v0 bridge 只迁移
源码、构建脚本、ABI/launch/hash metadata 和 contract。新服务器上按固定环境重新
生成 `.hsaco`、`.o`、`.so`；不把旧机器生成物当作源码提交。

### 4.5 KDA / 容器恢复信息

为保证两个已完成的 KDA reference 能在新服务器快速重建，分支另外纳入以下小型
自有 harness、结果摘要和宿主机/容器元数据：

```text
kda_baseline/
├── docker_baseline.md
├── env_host.txt
├── KDA_CODE_PATHS.md
├── SMOKE_RESULTS.md
├── STAGE5_PERFORMANCE.md
├── STAGE5B_AITER_KIMI.md
├── bench_common.py
├── bench_kda_prefill.py
├── bench_kda_decode.py
├── bench_kda_prefill_aiter.py
├── smoke_sglang_kda.py
├── smoke_vllm_kda.py
└── results/
    ├── CORRECTNESS_SUMMARY.md
    ├── aiter_kimi_prefill.json
    ├── correctness_decode_sglang.jsonl
    ├── correctness_decode_vllm.jsonl
    ├── correctness_prefill_sglang.jsonl
    ├── correctness_prefill_vllm.jsonl
    ├── perf_decode_sglang.jsonl
    ├── perf_decode_vllm.jsonl
    ├── perf_prefill_sglang.jsonl
    └── perf_prefill_vllm.jsonl
```

这组共 22 个文件、177,096 bytes（约 173 KiB）；`KDA_BACKUP_SHA256.txt` 保存其 hash。它们记录
容器名称、镜像 tag/digest、ROCm/PyTorch/Triton、GPU、KDA 调用路径和 Stage 5
结果，但不包含官方依赖源码、模型权重或大体积日志/profiler 输出。

## 5. 明确排除项

以下内容不进入快速 clone 的源码分支：

- `.hsaco`、`.o`、`.so`、`.bin`、`.llir`、`.ttir`、`.ttgir`、`.mir`、`.mlir`；
- `rocprof_outputs/`、`.rocprofv3/`、raw sessions、`golden_capture/`；
- `compiler_ir/`、`exact_lto/`、`machine/`、`isa/` 大型 dump；
- `vllm_compare/` 中未列入核心 allowlist 或历史归档的版本，以及生成性的
  dump/地址/packet 工件；v10--v31 的源码和报告由
  `QWEN_HISTORY_ALLOWLIST.md` 额外收录；
- compile_bug 下的生成性实验输出目录；源码和报告按
  `QWEN_HISTORY_ALLOWLIST.md` 额外收录；
- Kimi/Qwen 模型权重；
- SGLang/vLLM/AITER 官方源码副本。

这些内容如有审计价值，另做服务器上的压缩归档，不混入快速恢复分支。

## 6. 当前仓库与远端记录

当前仓库远端（仅记录，不执行上传）：

```text
myfork  git@github.com:848267592/avelang.git
origin  https://github.com/causalflow-ai/avelang.git
```

本次目标是在 `myfork` 上创建新的独立备份分支；当前分支已创建并完成本地 commit，
尚未推送。不假定旧的 GitHub 分支或旧上传内容存在，也不覆盖旧分支。

## 7. 后续步骤（尚未执行）

本次完整迁移共规划 **5 步**：

1. **建立清单**：确定独立分支、allowlist、Stage 6S/X2/full graph、两个 Q@H probe
   和明确排除项；已完成。
2. **冻结恢复信息**：对 allowlist 文件计算 SHA-256，并记录官方源码 URL/SHA、
   Docker digest、ROCm/PyTorch/Triton 和重建命令；已完成。
3. **创建独立分支并暂存**：创建 `backup/qwen-kda-repro-2026-09`，只按 allowlist
   加入 Qwen 文件、KDA/容器元数据和迁移文档；已完成。
4. **本地审查并提交**：检查 staged stat、路径、敏感信息、大文件和 hash，然后
   创建本地迁移 commit；已完成，主 commit 为 `3fde826`。
5. **推送并验证恢复**：push 新分支到 `myfork`，再用临时目录做一次 clone/文件/hash
   验证，最后再处理 Codex 聊天记录归档；代码/环境迁移已完成，Codex 聊天记录仍
   单独保留为后续归档项。

第 2 步已生成：

- [QWEN_BACKUP_SHA256.txt](QWEN_BACKUP_SHA256.txt)
- [QWEN_REBUILD_ENV.md](QWEN_REBUILD_ENV.md)

第 4 步新增：

- [KDA_BACKUP_SHA256.txt](KDA_BACKUP_SHA256.txt)

后续得到确认后，按以下顺序继续：

1. 创建 `backup/qwen-kda-repro-2026-09`；
2. 只按 allowlist 加入迁移文件和本清单/allowlist，不触碰现有 19 个 modified tracked 文件；
3. 检查 `git diff --cached --stat`、文件路径、敏感信息和大文件；
4. 由你确认后再 commit/push 到新的 GitHub 分支。

截至 Step 4：备份分支已创建为 `backup/qwen-kda-repro-2026-09`；原 64 个
allowlist 新增文件、4 个迁移元数据文件，以及上面列出的 KDA/容器文件和 hash
已暂存。allowlist 中另外 23 个文件原本已在当前 HEAD，因此在分支工作树中已经
存在，不产生新的 staged diff。现有 19 个 modified tracked 文件仍未暂存；
`QWEN_UPLOAD_INVENTORY.md` 仍保持未跟踪且未纳入，这是有意排除的旧规划文档，不是
迁移遗漏。主迁移 commit 为 `3fde826`，状态 commit 为 `c6e3172`；远端最终 HEAD
为 `c6e31721535bca08d47f159b9633cb54d1592047`。临时 clone 已完成 hash、语法、
关键入口和容器 metadata 验证并删除；当前**未删除或移动项目文件，也未修改 Docker**。

## 8. 历史演练源码/报告补充归档

用户随后明确要求保留整个 Qwen 演练过程，而不只保留最终 full graph。因此本次
补充纳入：

- `vllm_compare/` 根目录中 v10--v31 的 146 个 Python/Markdown 文件；
- `compile_bug/qwen_mfma32_lowering_ladder/` 中 603 个源码/脚本/报告文件，另有
  2 个手写汇编源；
- 合计 751 个文件、5,901,535 bytes；精确路径和 hash 见
  `QWEN_HISTORY_SHA256.txt`。

性能采样 CSV/JSON、rocprof、输入输出张量、HSACO/对象文件以及 compiler
IR/MIR/ISA/machine 生成物明确不上传；这些内容后续可按脚本重新生成。Z5B 的
15 个本地 Python import-closure、发布入口、bench/test、提交清单和集成报告均
已包含，源码恢复闭包完整；Z5B 的生成 HSACO 仍需在新主机重建。
