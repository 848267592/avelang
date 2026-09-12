# Stage 5E Decision

唯一建议选择 **A：两种 solve 直接写入同一个预分配 output buffer，并在完整 fixed
graph 中复用该 pointer**。

原因：canonical consumer 已排除数值和下游读取 pointer，但当前 same-pointer 实验
仍先让两种 solve 写各自 buffer，再 copy 到 canonical；原 solve store 地址和 cache-set
状态尚未关闭。直接复用 out pointer 是最小、低风险、可证伪的下一实验，不修改 solve
数学、W/U、asm、compiler 或 production。

Stage 5E 应同时采集只读 clock telemetry 和 whole-graph trace，但主动作仍是固定 out
pointer，不扩展成 fusion 或 kernel 重写。预期收益不能可靠填写；可恢复的理论上界是
当前 T=2048 约 64 us tail penalty，但在 direct control 前保持 `null`。

不推荐 fusion、compiler 或 assembly 修改：本轮没有任何 code-object、动态指令或
正确性证据支持它们。
