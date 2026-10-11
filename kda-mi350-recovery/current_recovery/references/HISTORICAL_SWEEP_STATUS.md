# 历史 sweep 与结果状态

- D73 两个完整 runner 含 candidate generation、candidate identity checks 和 K1×K2 joint sweep 逻辑，源码以 EXACT_RECOVERED 保存。
- D71 runner 为 RECONSTRUCTED；最终远程追加状态不能独立验证。
- D71/D73 的 C68 `run_sweep.py`、原始 AITER/KDA baseline 目录、完整 per-candidate raw sweep JSON、完整 profiler 输出与机器二进制没有纳入恢复包。
- 本地学习复盘中保留 D70–D73 摘要表；未将其改写为可复现 raw result，也不在本文件重新声称性能数字。
- Stage 5 原始性能文档留在本地恢复目录，因含内部机器名未纳入 bundle。
- 运行这些工具前需要先恢复依赖并静态审查路径/子进程调用。恢复期间没有运行这些脚本。
