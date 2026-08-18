# R4 最小教学包

这是给别人继续指导优化用的最小代码包，不包含 R1/R2/R3、诊断 kernel、
大批量 MIR 复制和历史实验工件。

## 文件

```text
qwen_persistent_recurrence_r4_handoff/
├── README.md
├── avelang/r4_kernel_minimal.py
├── triton/chunk_delta_h.py
├── triton/metadata.yaml
├── benchmark/bench_r4_vs_triton_minimal.py
└── reports/
    ├── README.md
    └── 00--13 后续实验报告副本
```

`reports/` 包含完整中文复盘、R4 正式报告、R4-tail 后续分支报告和 R5
失败实验报告。报告正文是原始副本，方便别人从代码直接进入实验结论；大批量
MIR/LLVM/ISA/HSACO 原始工件仍不放进最小包，以免目录失去可读性。报告索引会
明确哪些数据可以直接比较，哪些只是不同 harness 下的诊断结果。

完整的原始证据档案仍保存在旁边的：

`../qwen_persistent_recurrence_r4_handoff_full/`

## R4 kernel 为什么仍有约 469 行

原始 R4 文件有 1319 行，但其中包含：

- correctness 和 CLI；
- BV16/BV32/BV64 diagnostic variant；
- I/O packet experimental branches；
- HSACO dump 和报告辅助代码。

`avelang/r4_kernel_minimal.py` 只保留主
`_qwen_gdn_persistent_recurrence_r4_joint_v4_kernel`，从原始文件第 40--487
行逐行抽取，并把固定 contract 常量直接写入文件。它仍然不可能只有几十
行，因为需要显式表达：tensor layout、persistent FP32 state、W/K shared
stage、pred MFMA、BF16 V-new round-trip、V-decay、两个 K-half update、
same-bank commit 和 final state store。

这个文件的目标是“去掉无关诊断代码但保留真实 kernel body”，不是伪代码。

## 它不是旧 v29 broad-K / compact-K

R4 应该称为：

`Direct-K64 MFMA32 + LDS-mediated retile`

它使用两个 K-half stage：

```text
k_stage[2, BT=64, K-half=64]
K[0:64, 64 tokens] + K[64:128, 64 tokens]
```

这和旧 v29 的 broad-K/compact-K 实验不是同一个命名体系。若一定要比较
形状，R4 在“不建立 broad-K 的 transposed `k_all_t/kall_vec` shared view、
按两个 K64 half 直接消费”这一点上更接近 compact/direct-K 思路；但 R4 使用
的是 MFMA32、BV32、WG128 和 LDS-mediated retile，不是旧 v29 compact-K 的
MFMA16 kernel。不要把 R4 的结果直接标成“v29 compact-K 性能”。

## 运行最小 benchmark

```bash
cd /home/jiandongliu/project/avelang
export PYTHONPATH="$PWD/python:$PWD/test/examples/linear_attention/vllm_compare"
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_persistent_recurrence_r4_handoff/benchmark/bench_r4_vs_triton_minimal.py \
  --T 512 1024 2048 8192 \
  --sessions 3 --warmup 5 --repeat 20
```

它只比较两个 arm：

1. `r4_kernel_minimal.py` 的纯 Avelang R4；
2. current-vLLM Triton recurrence body。

计时是当前 HIP stream 上的 HIP event body timing，不使用 Graph；输入/输出
在计时前分配。脚本会先做 finite/reference 检查，然后才输出时间。完整的
七臂 benchmark 仍在 full archive 的 `avelang/bench_...r4.py`。

## Triton 文件的身份

`triton/chunk_delta_h.py` 是可读 Triton Python kernel；它不是 Avelang 改写
版本。Triton 的 exact TTIR/TTGIR/LLVM/ISA/HSACO 只在 full archive 中保留，
最小包只保留教学所需的源代码和 metadata。

## 结论入口

从 R4 之后的完整实验、数字、No-Go 原因和最终 compiler/lowering 根因，阅读：

[报告索引](reports/README.md) → [完整中文复盘](reports/00_r4_after_experiments_report_cn.md)
