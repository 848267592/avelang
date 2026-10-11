# 依赖与构建边界

## 可从恢复文件确认的依赖

- A70 kernel 导入 `torch`、`avelang`、`avelang.language`，使用 Avelang JIT 与 AMD MFMA lowering。
- D71/D73 工具使用 Python 标准库；D71 还使用 NumPy、PyTorch 和 `run_sweep`。D73 使用 NumPy、PyTorch、Triton、AITER FlashKDA、恢复目录中的 `bench_common`，以及 C68 的 `run_sweep`/candidate 模块。
- AITER 历史参考 commit：`7bb44998274fa679ece0a37bf8426ab780b1837d`。D 系列工具将此值作为预期 commit。
- 本地 V0 报告记录过源码 commit `c06c11c00e07ab3532e83bbe0b27e3968f3c4b99`；该 commit 的源码没有从已检查的备份分支中恢复。
- 早期项目素材可在原仓库 `848267592/avelang` 的 `backup/qwen-kda-repro-2026-09` 分支查阅。本归档不复制整个 AITER/ROCm 仓库。

## 未知或缺失

- Python、PyTorch、Triton、AITER、ROCm/LLVM/Avelang 的精确版本锁文件未恢复；没有 `requirements.txt`、`pyproject.toml`、conda environment 或可验证的完整 build command。
- A70 所需 B66/B67 compiler/intrinsic 支持没有完整 patch；只保存截断的源码检查片段。
- D71/D73 使用的 C68 `run_sweep.py`、完整 KDA baseline、AITER 子模块和原始项目目录结构没有一起恢复。
- 原始 HSACO、ISA、MLIR/TTGIR 与链接工件未找回。MI300X 的二进制不能作为 MI350 可用工件。
- 原 MI300X 环境为 gfx942；具体 compiler build、ROCm build、驱动和系统包版本没有充分证据，不能推测补齐。

## 构建尝试

归档前不尝试安装依赖或编译源码。要开始本地编译，先取得与目标 MI350 支持相符的 Avelang/LLVM/ROCm 工具链，并单独实现/恢复 B66/B67 intrinsic lowering；之后再按 `MI350_MIGRATION_PLAN.md` 逐步验证。benchmark runner 需要先补齐被引用的 C68/AITER 项目依赖。

## B70 probe dependencies

The recovered B70 probe and correctness sources import PyTorch and Triton. Running those probes requires a compatible ROCm/Triton environment and AMD GPU; this recovery did not execute them. The report-generation code also expects original B70 and C67 compiler artifacts under the KDA reports tree. The original JSON outputs and complete B_20260927_70_IR_ISA_CORRELATION.py module are missing, so the final compatibility writer is not standalone. No separate custom Avelang intrinsic implementation was recovered.