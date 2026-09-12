# Stage 5D Final Summary

本轮完成了 audit-only fixed-graph、连续 downstream tail、canonical data、copy-to-one
consumer pointer、alignment、allocator 和 cache/execution-state 控制实验。未修改任何
production kernel、asm 或 compiler。

## 最重要结果

- GRAPH-A/B 均为八个 dispatch，只有 solve kernel/WG 不同。
- T=2048 连续 tail：v18 predecessor 后 `0.267938 ms`，v1 后 `0.332193 ms`，
  v1 慢 `64.255 us`。
- 消费同一 bitwise canonical tensor 时仍慢 `64.555 us`，数据内容不是主因。
- copy 到同一个 consumer pointer 时仍慢 `68.181 us`，下游 pointer/alignment 不是主因；
  两 solve 直接写同一 out pointer 尚未执行。
- warm tail 后差距 `0.120 us`；512 MiB controlled perturbation 后差距
  `-0.080 us`；reduction prime 后仍为 `75.412 us`。
- 操作层根因是 predecessor-induced transient downstream execution state。
- cache residency 与 clock/power ramp 尚未由 counter/telemetry 分开。
- Stage 5C 分段 event 会把 v18/v1 full 分别增加 `15.623/29.844 us`，但连续 tail
  已独立复现抵消，因此 instrumentation 不是主因。

## 唯一下一步

Stage 5E 只做一件事：让 v18 和 hierarchical_fp32_v1 solve kernel 直接写入同一个
固定预分配 output buffer，再运行 unchanged downstream graph。这关闭最后一个明确且
低风险的 solve-store physical pointer/cache-set 变量。

## 未执行项

whole-graph rocprof、cache counter、dispatch gap、clock/power telemetry 和本轮 pytest
回归因平台在 benchmark/control 完成后拒绝新的 Docker execution 而标为 N/A。没有
伪造或沿用旧 counter 冒充本轮数据。

完整报告：

`../qwen_gfx942_bt64_downstream_state_coupling_stage5d_report.md`
