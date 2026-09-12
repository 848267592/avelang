# Whole-Graph Trace Analysis

新 whole-graph rocprof 为 N/A：`rocprofv3 --list-avail` 的 Docker 调用被平台执行
额度策略拒绝，且策略明确禁止绕过。`profile_stage5d.py` 已冻结独立 process、无图内
event 的复现流程，并要求先保存设备实际 counter 列表。

从 source/JIT contract 可确认 W/U、asm 和 chunk-o symbol、specialization、grid/WG
相同；但本轮不能填写新的动态指令、cache counter、dispatch gap、CU 分布或 trace
duration。历史 Stage 4 counter 只证明这些 code object 无 scratch/spill，不作为本轮
A/B downstream 状态差异的伪替代。
