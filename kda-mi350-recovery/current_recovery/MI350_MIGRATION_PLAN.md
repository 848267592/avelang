# MI350 迁移计划

## 阶段 1：建立可复现工具链

1. 为目标 MI350 型号记录 CPU/OS、驱动、ROCm、LLVM/Avelang commit 与编译选项。
2. 在本地 clean checkout 恢复 Avelang intrinsic 定义和 AMD MFMA lowering；优先以 A70 所需 ordinary-V BF16 MFMA 为研究对象。现有 B66/B67 片段不足以直接应用。
3. 固定 PyTorch、NumPy、Triton、AITER 版本；验证 AITER commit `7bb44998274fa679ece0a37bf8426ab780b1837d` 是否兼容目标工具链。

## 阶段 2：先建立数学和 ABI 基线

- 从 `references/WORKSPACE_ABI_REFERENCE.md` 核对历史 dtype/layout 边界。该文档基于 V0 报告，不是 C23/C26 的 host ABI 规格。
- 实现 independent reference 与逐 workspace 对照；保留初始状态非零、尾块和多 chunk 场景。
- 恢复 K1 host dispatch/workspace ABI，或在新分支明确设计替代接口；不要把 C26 截断片段接入生产路径。
- 将 A70 源码、历史测试和新 MI350 结果分开记录，保留每次编译产物与测试输入哈希。

## 阶段 3：重新验证 MI350 映射与资源

下列 MI300X 参数只能作为待验证假设：

- gfx942 架构目标与 MFMA 指令/lowering；
- BW16、W4、256-thread workgroup、64-lane wave 与四 wave ownership；
- `C=32`、`H=12`、`K=V=128` 的数据布局和尾块/NT dispatch；
- 每 chunk 的 BF16/FP16 MFMA 数量、寄存器分配、AGPR/VGPR/SGPR、spill、LDS、occupancy 与并发行为；
- launch grid `[8,12,1]`、block `[256,1,1]`、scratch/debug 路径及历史 `atol=rtol=3e-2` 容差是否适合新 reference。

在 MI350 上重新生成 HSACO/ISA/MLIR，逐形状跑 correctness，再测延迟、资源和不同 batch/shape。任何性能结论都记录 GPU 型号、频率/功耗状态、warmup/repeat、commit 和二进制哈希。

## 阶段 4：迁移 benchmark

D71/D73 runner 依赖 C68/AITER 项目文件，目前不是独立工具。恢复依赖后先审阅路径与 shell 调用，再在 MI350 上重新跑 candidate sweep 和 K1×K2 joint sweep；旧 MI300X winner 仅作候选，不沿用为结论。

## B70 layout evidence on MI350

Use the B70 ownership equations and report as hypotheses for designing a new MI350 probe. Re-run Kd/Qd, INV/Mqk, and Kr coverage tests on MI350, then correlate new compiler IR and ISA with register ownership, operand permutations, LDS producer/consumer addresses, synchronization, and MFMA wave mapping. Revalidate every MI300X-specific slot offset, register number, opcode sequence, wait/barrier count, and timing assumption. B70 is an isolated probe set, not a complete production K2 kernel, and it did not modify A69.