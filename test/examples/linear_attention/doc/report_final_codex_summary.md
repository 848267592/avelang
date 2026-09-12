# Stage 4--6W 报告审计工作摘要

## 完成内容

- 原样备份 `report_final.md` 到 `report_final_before_stage6w_update.md`。
- 将正式报告重写为 Stage 4--6W 的中文教程，统一权威性能口径为 Eager public API。
- 新增修订日志、事实更正、时间线 CSV、证据索引 JSON、源码改动索引和链接验证记录。
- 将最新 Stage 6W paired shared-environment retest 写为当前结论：W1 为当前 Avelang experimental baseline；默认生产 selector 未改变。

## 未做内容

没有运行 GPU benchmark，没有改 kernel、Avelang compiler、LLVM/AMDGPU RA、HSACO、assembly、v24 或 production selector；没有创建新优化 variant，也没有创建 commit。

## 仍需谨慎的结论

- W1 的晋级范围是相同共享 GPU 环境中的严格 paired public-API 相对排名，不是独占 GPU 的跨机器绝对延迟结论。
- W1 在 T=2048 的同批中快于 vLLM，但 T=8192 仍为 vLLM 的 1.2259x。
- `KKT FP32 a -> solve` global handoff 消除只是后续候选，尚未实现。
