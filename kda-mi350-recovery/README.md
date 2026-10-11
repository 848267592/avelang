# KDA MI350 Combined Recovery Archive

本目录把本地恢复包和旧分支 KDA 备份放在一个可拷贝、可纳入 Git 的工程目录中；来源仍分开保存，避免把旧基线误认为后期 native kernel。

## 目录

- `current_recovery/`：当前整理完成的 KDA 恢复包。公开归档版只更新了 README 与许可状态说明；恢复源码、报告、测试和证据文件保持原字节。各文件恢复等级、SHA-256、来源、依赖及安全审查见其中的 `FILE_MANIFEST.json`、`RECOVERY_STATUS.md`、`DEPENDENCIES.md`、`SECURITY_REVIEW.md`。
- `prior_backup_202609/kda_baseline/`：从 `848267592/avelang` 的 `backup/qwen-kda-repro-2026-09` 挑出的旧 KDA benchmark、smoke/correctness 脚本及结果，共 18 个文件。来源固定在提交 `462b4bb00320fc383ac627047cf0f0fc1e21173f`，逐文件保留 Git blob SHA 和本地 SHA-256。
- `COMBINED_FILE_MANIFEST.json`：逐文件路径、字节数、分类、SHA-256 和来源。清单不包含自身哈希。

## 公开归档

按仓库所有者 2026-10-11 的指示，此目录将提交到公开仓库 `848267592/avelang` 的新分支 `kda-mi350-recovery-2026-10-11`，不改写现有分支。仓库根目录现有 `LICENSE.txt` 为 Apache License 2.0；归档不新增许可证或第三方源码。归档中尚未确认归属的 MFMA/Qwen prototype/reproducer 未纳入。

## 使用边界

`current_recovery/kernels/k2/A_20260927_70_BW16_TRUE_W4_REBUILD.py` 的 SHA-256 与提供的 A70 golden 匹配。A70 是恢复出的 native Avelang BW16/W4 K2 源码。

旧分支 benchmark 和 correctness 文件是历史基线材料，包含 SGLang、vLLM、AITER 路径；它们不是 A70，也不代表 native Avelang K2。文件没有执行，也没有在 MI350 上验证。继续开发前应先按当前恢复包内的依赖说明检查构建和 correctness 条件。

## 排除项

未复制原始 Codex 会话归档、主机环境清单、含机器名或本机路径的旧报告/文档、私钥或认证配置，以及 AITER/ROCm 第三方仓库。旧分支中因包含工作站路径而未纳入的文件：`docker_baseline.md`、`KDA_BACKUP_SHA256.txt`、`STAGE5B_AITER_KIMI.md`；机器环境/性能快照也不纳入。未对这些历史材料做删改后伪装成原件。