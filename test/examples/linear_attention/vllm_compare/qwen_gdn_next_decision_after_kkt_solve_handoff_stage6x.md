# Qwen GDN Next Decision After Stage 6X-KS

X1 与 X2 的 source/body gates、NaN/reuse 和 non-default-stream gates 均通过。X2 在
同一 CTA 内保留 FP32 KKT matrix，直接执行现有 hierarchical FP32 solve，删除 global
FP32 `a` 的 allocation、store、read 和一个 dispatch；它对 Stage6W 为 full bit-exact。

随后补齐的正式 Eager public-API confirmation 使用每个 T 独立进程、5 sessions、50
paired Williams blocks、每实现 300 calls、HIP event 与 wall-clock。X2 对 W1 在
T=512/1024/2048/4096/8192/16384 都稳定更快；关键 gate 为：

| T | X2 相对 W1 paired gain | 95% event CI |
|---:|---:|:---|
| 2048 | +26.020 us | [23.304, 28.563] us |
| 8192 | +82.823 us | [80.432, 85.154] us |
| 16384 | +177.540 us | [176.004, 179.024] us |

W1 的长文本 slope 为 `6.341 us/chunk`，X2 降至 `5.694 us/chunk`，回收约 10.2%。
资源没有 scratch/spill cliff。因此：

```text
X2 / Stage 6X = new Avelang BT64 experimental baseline
W1 / Stage 6W = previous experimental baseline
production/default = unchanged
```

X2 不是所有长度都击败 native vLLM：同口径 Eager 下 X2 在 T<=4096 快于 vLLM，
在 T>=8192 因 vLLM `3.688 us/chunk` 的更低 slope 而落后。故不更改 production
selector，也不从两个点插值得出精确 crossover policy。

## 唯一下一步：Stage 6Y updated full-gap audit

冻结 X2 的五-dispatch 图：

```text
cumsum -> fused KKT+solve -> fused W/U -> recurrence -> chunk-o
```

Stage 6Y 是 measurement-only：重新审计 X2 与 vLLM 的实际 dispatch graph、每个
逻辑 body 的 T/chunk slope、资源/ISA 与所有剩余 global intermediate。它必须先产生
当前图的证据，才可以在以下候选中选择**一个**动作：`a_solved_bf16 -> W/U` 的 source
native handoff，或 immutable recurrence -> chunk-o 边界的后续设计。不得根据旧 Stage
6W 的六-dispatch账本，直接开始 LDS alias、packed lower、recurrence fusion 或其它
kernel 改动。
