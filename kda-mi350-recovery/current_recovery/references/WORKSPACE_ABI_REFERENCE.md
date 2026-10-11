# 历史 workspace / dtype 参考（非完整 host ABI）

来源：直接恢复的本地文件 `KDA_PREFILL_V0_CORRECTNESS.md`。此处整理的是该 V0 correctness 记录中的边界，用于研究；**它不是 C23/C26 的生产 ABI，也没有对应完整 K1 源码**。

- 历史 V0 只接受 `B=1, H=12, K=V=128, C=32, BC=16`，FP32 beta 与 K-first state；不接受 `state_v_first=True`、grouped value heads 或训练/backward。
- canonical state 记录为 FP32 `[N,H,K,V]`，`state_v_first=False`；输出概念形状为 `[B,T,H,V]`，final state 为 `[N,H,K,V]`。
- 文档记录 Kd/Qd/Kr 为 BF16，`INV/Mqk` 为 FP16，beta 与 persistent H state 为 FP32；K1 中间矩阵为 FP32，只有最终 `INV/Mqk` 缩窄为 FP16。
- A70 文件暴露 Python wrapper `a70_bw16_k2(*, workspace, v, beta, initial_state, debug=False)`，取 `workspace.kd`, `workspace.qd`, `workspace.kr`, `workspace.gt`, `workspace.inv_mqk` 等成员并返回 output、final state 和可选 debug data。此 wrapper 不能替代缺失的 C23/C26 host dispatch/workspace ABI。

来源报告记录的端到端历史 tolerance 为 `atol=rtol=3e-2`。在 MI350 上需要先确认 reference 语义和容差，再决定新 gate；不得将此值视为目标平台的已验证标准。
