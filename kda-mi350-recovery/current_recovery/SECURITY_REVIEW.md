# 安全与上传筛查

候选包静态扫描未发现 SSH private keys、GitHub/API credentials、AWS keys、Bearer tokens、IP 地址、SSH 目标、Codex auth/session 路径、原始 JSONL、二进制或大型缓存。未发现随包携带的 LICENSE/NOTICE；许可边界见 `LICENSE_STATUS.md`。

筛查发现 `STAGE5_PERFORMANCE.md` 含内部机器名。该文件未复制进项目，原始副本仍保留在恢复目录中，SHA-256 为 `9ecd641f8fd016d1d8efc0a167cfb957c20f372c8379406bf8e05a9e6a189b17`。D 盘完整学习复盘和 MFMA/Qwen prototype 没有纳入 bundle；后者代码归属/许可边界不清。AITER/ROCm 等第三方实现没有 vendored。

原始 Codex JSONL、auth/session 记录、凭据配置、原始取证日志、远程机器输出、`.hsaco/.o/.bin`、缓存和性能 JSONL 均未复制。静态扫描仅表示未命中所列模式，不构成法律许可结论或任意秘密检测的数学保证。

B70 addendum: staged only the report, probe/report source snapshots, the explicitly partial correlation fragment, and their related diff evidence. A targeted static scan found no credential, private-key, SSH host, private IPv4 address, or Codex session material in the staged B70 files. No raw Codex JSONL, AITER/ROCm third-party source tree, or machine binary was copied into the package.