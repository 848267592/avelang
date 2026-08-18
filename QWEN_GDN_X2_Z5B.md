# Qwen GDN gfx942: X2+Z5B

这是本 fork 中截至 2026-08-18 的 Qwen GDN experimental 性能候选。它不是
production default；v24 仍然是 production baseline。

## 目标与状态

- GPU：AMD MI300X / `gfx942`
- contract：`B=1, Hk=4, Hv=8, K=V=128, BT=64`
- 输入：`q/k/v=BF16`，`g/beta/initial_state/final_state=FP32`
- 测试接口：Eager public API，current HIP stream，no Graph
- X2：CTA-local KKT + hierarchical solve、BF16 W/U、current-vLLM BF16 recurrence bridge
- Z5B：direct-Q-cache chunk-o，只替换 X2 的尾部 chunk-o

X2+Z5B 在 T=`1024/2048/4096/8192/16384` 的本次完整 Eager 测试中快于原 X2，
T=`512` 有固定开销回退。它不是所有长度都胜过 vLLM，也没有修改 production
selector。

## 从这里开始看

完整算子入口：

- [X2+Z5B full entry](test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_full.py)
- [Z5B chunk-o](test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer.py)
- [X2 KKT + solve](test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_kkt_solve_handoff_stage6x.py)

编译器改动：

- [MFMA signatures](lib/IR/Intrinsics/amdgpu_mfma_signatures.h)
- [AMDGPU intrinsic lowering](lib/IR/Intrinsics/amdgpu_intrinsics.mlir)
- [MLIR generator regression test](lib/IR/mlir_generator_test.cc)
- [intrinsic documentation](docs/content/language-reference/hardware-intrinsics.md)

外部 recurrence bridge：

- [bridge source](test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_recurrence_reconciliation_stage6r/stage6r_external_bridge.cpp)
- [bridge build script](test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_recurrence_reconciliation_stage6r/build_bridge.sh)
- [captured recurrence HSACO](test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_recurrence_reconciliation_stage6r/current_kernels/vllm/kernel.hsaco)

验证与实验记录：

- [correctness test](test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_full.py)
- [Eager benchmark](test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_eager.py)
- [完整集成报告](test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_x2_z5b_chunko_full_integration_report.md)
- [提交清单](test/examples/doc/Avelang_Qwen_GDN_gfx942_X2_Z5B_2026-08-18_提交清单.md)
- [上传记录](test/examples/doc/Avelang_Qwen_GDN_gfx942_X2_Z5B_2026-08-18_上传记录.md)

## 关键性能摘要

单位为完整 Eager public API 的 HIP-event median，单位 ms：

| T | X2 | X2+Z5B | vLLM | X2+Z5B / vLLM |
|--:|--:|--:|--:|--:|
| 512 | 0.193427 | 0.204803 | 0.348857 | 0.587x |
| 1024 | 0.233246 | 0.227578 | 0.351882 | 0.647x |
| 2048 | 0.317913 | 0.291232 | 0.397110 | 0.733x |
| 4096 | 0.473723 | 0.433223 | 0.517688 | 0.837x |
| 8192 | 0.847997 | 0.753357 | 0.759887 | 0.991x |
| 16384 | 1.583108 | 1.392826 | 1.229464 | 1.133x |

## 运行前提

这是 AMD/HIP + Avelang 实验，不是 plain CUDA `.cu` 实现。current-vLLM recurrence
使用一个带 hash guard 的外部 Triton HSACO；bridge 的 `.so` 是本机编译产物，不能
直接假设跨机器可用。

在目标 ROCm 环境中先构建 bridge：

```bash
cd test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_recurrence_reconciliation_stage6r
sh build_bridge.sh
```

然后从仓库根目录运行 correctness：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_full.py -s
```

预先验证过的结果是 `47 passed in 29.12s`。完整 benchmark 命令见集成报告。

## 如何让别人找到它

必须先在 GitHub 的分支下拉框选择：

`agent/x2-z5b-submission-2026-08-18`

也可以直接打开这个分支的入口目录：

`https://github.com/848267592/avelang/tree/agent/x2-z5b-submission-2026-08-18/test/examples/linear_attention/vllm_compare`

在 GitHub 页面使用 **Go to file**，搜索：

```text
qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_full.py
```

查看顺序建议是：本 README -> full entry -> Z5B chunk-o -> 集成报告 -> correctness
test -> compiler intrinsic 文件。
