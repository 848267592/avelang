# X2+Z5B Fork 上传记录

## 目的

这份记录对应 `Avelang_Qwen_GDN_gfx942_X2_Z5B_2026-08-18_提交清单.md`，记录
将当前 X2+Z5B 版本整理并上传到个人 fork 的步骤、范围和检查结果。

个人 fork：`git@github.com:848267592/avelang.git`

发布候选：`X2+Z5B`，日期：`2026-08-18`。

## 先澄清 `gh` 依赖

本次上传不依赖 GitHub CLI (`gh`)。之前的上传也是直接使用 Git SSH remote，当前
仓库的 reflog 已证明上一条远端更新来自：

```text
git push -> myfork/qwen-gdn-optimization_ljd
```

因此：

- `git`、SSH key 和 `myfork` remote 才是代码上传所需依赖；
- 当前环境没有 `gh`，只意味着不能用 `gh pr create` 创建 Pull Request；
- `git push` 仍然可以把分支上传到 fork；
- recurrence bridge 的 `.so` 不是 GitHub CLI 或 Python 依赖，而是 HIP 构建产物，
  由仓库中的 `build_bridge.sh` 生成。

## 提交分组

提交只从明确的文件列表加入，不使用 `git add -A`：

1. `compiler: expose gfx942 fp32 mfma16x16x4 intrinsic`
   - `lib/IR/Intrinsics/amdgpu_mfma_signatures.h`
   - `lib/IR/Intrinsics/amdgpu_intrinsics.mlir`
   - `lib/IR/mlir_generator_test.cc`
   - `docs/content/language-reference/hardware-intrinsics.md`
2. `qwen: add X2+Z5B runtime`
   - X2+Z5B 的 15 个 Python import-closure 文件；
   - `stage6r_external_bridge.cpp`、`build_bridge.sh`；
   - current-vLLM recurrence `kernel.hsaco` 及 `abi.json`、`launch.json`、`sha256.txt`。
3. `qwen: verify X2+Z5B full Eager performance`
   - correctness test、Eager benchmark、完整集成报告；
   - 本清单与本上传记录；
   - 报告中的 Eager 汇总结果。原始 CSV/JSON 目录保留在本地，不进入 fork 代码提交。

## 明确排除

没有加入当前工作区中与 X2+Z5B 无关的后续研究：persistent recurrence R1--R5、
Stage 6Z 后续 Z6--Z11、C19--C26、block-dot V2、LTO 调试增强、实验性 layout
pass、临时 `.so`、`__pycache__`、Docker 私有文件和其他未验证的 selector。

Z5B 本身使用既有 BF16 MFMA32/shared-view lowering；本版本唯一新增的 compiler
intrinsic 是 X2 solve 所需的 `mfma_16x16x4_f32_f32`，没有修改 RA、allocator 或
手写汇编。

## 审查命令

在仓库根目录执行：

```bash
git status -sb
git remote -v
git diff --check
git diff --cached --check
git diff --cached --stat
git diff --cached --name-status
```

确认 recurrence bridge 的 HSACO：

```bash
cd test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_recurrence_reconciliation_stage6r
sha256sum current_kernels/vllm/kernel.hsaco
sh build_bridge.sh
```

桥接库生成后，运行完整 correctness：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_full.py -s
```

## 已有验证结果

- full correctness：`47 passed in 29.12s`；
- 口径：Eager public API、current HIP stream、no Graph、5 fresh sessions、
  10 Williams blocks；
- X2+Z5B 相对 X2：T=1024/2048/4096/8192/16384 更快，T=512 较慢；
- X2+Z5B slope：`4.839 us/chunk`；原 X2：`5.620 us/chunk`；本次 vLLM：
  `3.630 us/chunk`；
- X2+Z5B 不是 production default，v24 仍是 production baseline。

完整数字见同目录的集成报告；原始 CSV/JSON 目录仍保留在本地实验工作区，必要时可
作为单独 release asset 上传。

## 推送步骤

本次实际发布时使用以下顺序；每一步都保留输出供复核：

```bash
git switch -c agent/x2-z5b-submission-2026-08-18

# 只对清单中的文件执行 git add，绝不使用 git add -A。
git add <compiler-file-list>
git commit -m "compiler: expose gfx942 fp32 mfma16x16x4 intrinsic"

git add <runtime-file-list>
git commit -m "qwen: add X2+Z5B runtime"

git add <verification-file-list>
git commit -m "qwen: verify X2+Z5B full Eager performance"

git push -u myfork agent/x2-z5b-submission-2026-08-18
```

本记录在推送完成后补写实际 commit hash、远端分支和 `git push` 返回结果；若网络或
SSH 暂时不可用，只记录失败原因，不把“本地 commit 成功”误写成“远端上传成功”。
