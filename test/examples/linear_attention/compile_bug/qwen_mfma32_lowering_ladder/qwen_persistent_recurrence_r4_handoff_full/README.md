# R4 Chunk-GDR 完整审计档案

这是给后续审计和继续优化使用的完整档案。最小教学包在同级
`qwen_persistent_recurrence_r4_handoff/`；本目录保留全部 source helper、历史
机器工件和中文复盘。它把 R4 的纯 Avelang
full-recurrence kernel、current-vLLM Triton recurrence 实现、以及同口径
body benchmark 放在一起。它不改变 production selector，也不替换仓库中
任何默认路径。

## 先看什么

1. 阅读同目录的
   `qwen_persistent_recurrence_r4_after_experiments_report_cn.md`，了解从 R4
   之后每个实验做了什么、为什么做、结果是什么，以及最后的根因判断。
2. 阅读 `avelang/repro_qwen_gdn_persistent_recurrence_r4_joint_v4.py`。
   主 kernel 是
   `_qwen_gdn_persistent_recurrence_r4_joint_v4_kernel`，约在第 40 行开始；
   文件后面还保留了诊断变体。主 kernel 的 source body 约为第 40--487 行。
3. 阅读 `triton/chunk_delta_h.py` 中的
   `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`。
4. 最后看 `benchmark/` 章节和
   `avelang/bench_qwen_gdn_persistent_recurrence_r4.py`。

## 目录

```text
qwen_persistent_recurrence_r4_handoff_full/
├── README.md
├── manifest.sha256                    # 交付包内文件校验
├── qwen_persistent_recurrence_r4_after_experiments_report_cn.md
├── benchmark/
│   └── README.md                       # test-bench口径和复现命令
├── avelang/
│   ├── repro_qwen_gdn_persistent_recurrence_r4_joint_v4.py  # R4 主 kernel
│   ├── repro_qwen_gdn_persistent_recurrence_r4.py            # R4 correctness wrapper
│   ├── bench_qwen_gdn_persistent_recurrence_r4.py           # 七臂 body benchmark
│   ├── run_qwen_gdn_persistent_recurrence_r4_body.py        # rocprof body launcher
│   ├── bench_qwen_gdn_direct_k64_bv32_full_sequence_b0.py  # Triton/bridge helper
│   ├── repro_qwen_gdn_direct_k64_bv32_full_sequence_b0.py
│   ├── repro_qwen_gdn_direct_k64_bv32_full_recurrence_p{0,1,2}.py
│   ├── repro_qwen_gdn_persistent_recurrence_r{0,1,2,3}.py
│   └── repro_qwen_gdn_persistent_recurrence_r{1,2,3}_joint_v{1,2,3}.py
└── triton/
    ├── chunk_delta_h.py            # 可读 Triton Python kernel
    ├── kernel.ttir                 # Stage 6R exact compiled artifact
    ├── kernel.ttgir
    ├── kernel.llir
    ├── kernel.amdgcn
    ├── kernel.hsaco
    ├── metadata.yaml / abi.json / elf_metadata.yaml / launch.json
    ├── disassembly.txt
    └── sha256.txt
└── artifacts/
    └── r4_machine/                     # R4 MLIR/LLVM/MIR/ISA/HSACO证据
```

## 运行方式

这些命令测的是 recurrence body，不是最终 Eager public API 排名。编译、模块
加载、分配都在 HIP event 计时区间外；不使用 CUDA/HIP Graph capture。

```bash
cd /home/jiandongliu/project/avelang
export PYTHONPATH="$PWD/python:$PWD/test/examples/linear_attention/vllm_compare"

# R4 correctness，默认 T=64/128/512/2048
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_persistent_recurrence_r4_handoff_full/avelang/repro_qwen_gdn_persistent_recurrence_r4.py

# R4、R1/R2/R3、直接 Triton body、external Stage-6R bridge
# 默认 T=512/1024/2048/8192，warmup=10、repeat=50、5 sessions
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_persistent_recurrence_r4_handoff_full/avelang/bench_qwen_gdn_persistent_recurrence_r4.py

# 例如只测 T=2048，增加 session 数
python3 .../bench_qwen_gdn_persistent_recurrence_r4.py \
  --T 2048 --sessions 7 --warmup 10 --repeat 50
```

如果只想让别人看代码，不需要先编译。先看报告，再沿着 README 中的文件名
跳转即可。运行 benchmark 需要 gfx942/MI300、ROCm、Avelang Python 包和当前
仓库已有的 Stage 6R external bridge 工件。没有 GPU 时仍然可以阅读源码、
TTIR/TTGIR/LLVM/ISA；不能据此声称重新跑过 GPU 结果。

## 两个 kernel 的身份不要混淆

| 项目 | R4 | current-vLLM Triton |
|:--|:--|:--|
| 实现来源 | Avelang 高级源码 + Avelang compiler lowering | vLLM/Flash-Linear-Attention Triton |
| 文件 | `avelang/repro_qwen_gdn_persistent_recurrence_r4_joint_v4.py` | `triton/chunk_delta_h.py` |
| 运行入口 | R4 wrapper 设 `gfx942_bt64_bv32_joint_v4` | Stage 6R current-vLLM HSACO |
| 关键 symbol | `_qwen_gdn_persistent_recurrence_r4_joint_v4_kernel` | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` |
| ABI | BF16 K/W/U/H/V-new，FP32 g/state/final | 同一 current-vLLM BF16 recurrence ABI |
| 角色 | 自己写的 native experimental baseline | 诊断 golden/control；X2+Z5B 复用它的 HSACO |

`triton/chunk_delta_h.py` 是从当前 vLLM capture 保存的可读 Triton 源码；
`triton/kernel.*` 是 Stage 6R 另外保存的 exact gfx942 编译工件。两组工件
使用同一个 kernel symbol、源码位置和 ABI 语义，但历史上是分开捕获的，不能
把它们描述成一个由同一 SHA 直接证明的 source-to-HSACO 配对。

## R4 的编译器入口

R4 wrapper 中的核心选择是：

```python
os.environ["AVELANG_PERSISTENT_RECURRENCE_LOWERING"] = \
    "gfx942_bt64_bv32_joint_v4"
os.environ["AVELANG_QWEN_K64_PIPELINE_LOWERING"] = "distributed"
```

R4 复用了 persistent recurrence op、joint planner、typed BF16x8 producer、
LDS-mediated K retile 和 Direct-K64 MFMA32。后续报告里出现的 compiler
修改，主要是 persistent recurrence planner/lowering 和 block-dot 的 typed
operand/lowering 能力，不是手写 Triton ISA，也不是改 register allocator。

## benchmark 的边界

`bench_qwen_gdn_persistent_recurrence_r4.py` 中的 `direct_triton` 是 current
vLLM recurrence body；`external_bridge` 是加载 Stage 6R HSACO 的 HIP bridge。
它们的数值和机器图用途不同：前者是可读实现对应的 control，后者是复用
Triton 已编译 code object 的外部 kernel 集成。报告会分别列出，不能把
external bridge 的时间写成 Avelang 自己编译出的 Triton 时间。

## 结果的正确解读

R4 的 isolated body 在 T=2048 为约 `0.501926 ms`，direct Triton 为约
`0.156192 ms`，约 `3.21x`；T=8192 为约 `2.007665 ms` 对 `0.472783 ms`，
约 `4.25x`。R4 没有 scratch/spill，这说明问题不是简单的寄存器溢出。
详细解释和所有后续实验见中文总报告。
