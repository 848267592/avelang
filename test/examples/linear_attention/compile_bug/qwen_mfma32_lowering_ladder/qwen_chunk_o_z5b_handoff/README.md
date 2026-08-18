# Standalone chunk-o Z5B handoff

这是给别人继续学习和优化的最小版本包。它只包含 **单独 chunk-o**，不包含
X2 完整算子、recurrence bridge、production selector，也不包含 26 MiB 的原始
MIR/LLVM/ISA 工件。

## 结论先看

当前已经实际测过的纯 AveLang standalone chunk-o 最优版本是：

```text
Stage 6Z Z5B: direct-Q-cache-consumer
```

源文件是 `avelang/chunko_z5b_minimal.py`。它从 Z5A 的 dedicated full-Q
LDS cache 出发，唯一关键变化是删除：

```text
Q cache -> old phase Q rows -> phase_vec -> MFMA
```

变成：

```text
global Q once -> dedicated Q LDS cache -> Q fragment -> MFMA
```

K/H/V-new/g/output、MFMA32、BT64/BV64/BK32、WG256、BF16 ABI、K32 累加顺序和
数学都保持不变。

## 目录

```text
qwen_chunk_o_z5b_handoff/
├── README.md
├── avelang/
│   └── chunko_z5b_minimal.py       # 真实 AveLang Z5B kernel + launch wrapper
├── triton/
│   ├── chunk_o.py                  # 捕获的 current-vLLM Triton Python kernel
│   └── metadata.yaml               # 对照版本和 dtype/shape 说明
├── benchmark/
│   ├── bench_chunko_z5b_vs_triton_minimal.py
│   └── smoke_T64_not_for_ranking.json
└── reports/
    ├── README.md
    ├── 00_chunk_o_optimization_complete_report_cn.md
    └── 01--15 关键阶段原始报告副本
```

报告副本只选了能解释 chunk-o 演进和最终决策的文件；完整 raw capture 仍在
仓库旁边的 `codex_qwen_bt64_stage6z_*` 目录中。

## 输入输出 contract

固定目标是 gfx942/MI300：

| tensor | dtype | shape |
|:--|:--|:--|
| `q`, `k` | BF16 | `[1,T,4,128]` |
| `v_new` | BF16 | `[1,T,8,128]` |
| `h` | BF16 | `[1,T/64,8,128,128]` |
| `g` | FP32 | `[1,T,8]` |
| output | BF16 | `[1,T,8,128]` |

要求 `T >= 64` 且 `T % 64 == 0`，所有 tensor 在同一张 GPU 上连续存储。
Z5B 固定 `BT=64, BV=64, BK=32, WG=256`，每个 chunk-head 使用两个 CTA。

## 如何运行 benchmark

在 Avelang 仓库根目录执行：

```bash
cd /home/jiandongliu/project/avelang
export PYTHONPATH="$PWD/python"

python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_chunk_o_z5b_handoff/benchmark/bench_chunko_z5b_vs_triton_minimal.py \
  --T 512 1024 2048 8192 \
  --sessions 3 --warmup 5 --repeat 20 \
  --json-out /tmp/chunko_z5b_vs_triton.json
```

脚本会为每个 `T/session` 启动一个 fresh Python worker，并且：

- 先完成 AveLang/Triton compile、module load、Triton autotune 和 output allocation；
- 在当前 HIP stream 上用 HIP event 计时；
- 两个 arm 使用 caller-owned preallocated BF16 output；
- 每个 repeat 按轮换顺序执行 Z5B/Triton，减少顺序偏差；
- 不使用 CUDA/HIP Graph；
- 记录 native Triton 实际 selected 的 `BK/BV/num_warps/num_stages`；
- 只把输出 finite 和两臂 BF16 输出差异作为诊断，不把两种不同累加路径强行宣称 bit-exact。

这里的 latency 是 **standalone kernel body**，不是 full Qwen GDN public API latency。
它不计 CPU allocation，但包含 kernel launch 在 HIP event 所覆盖的 GPU-side body
边界。若要测 Eager full operator，应使用 X2+Z5B 的完整 harness，不能拿这个
数字替代 full graph。

## 如何读 Triton 对照

`triton/chunk_o.py` 是实际捕获的 vLLM `chunk_fwd_kernel_o` 源码，不是手写汇编。
它使用：

```python
b_o = q @ h.T
b_A = q @ k.T
b_A = causal_mask(exp(g_target - g_source) * b_A)
out = exp(g_target) * b_o + b_A @ v_new
```

并在 kernel 内直接把 FP32 accumulator 转成 BF16 写公共输出。Triton 编译器再把
`tl.dot` lower 成 gfx942 的 `v_mfma_f32_32x32x8_bf16` 等机器指令。配套的
TTIR/TTGIR/LLVM/ISA/HSACO 没有放入最小包，路径和统计在总报告中给出。

## 各目录和报告在哪里

这个包不是把整个实验工作区搬过来，而是按用途分成四个目录：

| 目录 | 内容和用途 |
|:--|:--|
| `avelang/` | 纯 Avelang standalone chunk-o。`chunko_z5b_minimal.py` 是当前最佳 Z5B kernel 和 caller-owned launch wrapper。 |
| `triton/` | current-vLLM Triton 对照。`chunk_o.py` 是实际捕获的 Triton Python 源码，`metadata.yaml` 记录 shape、dtype 和 selector 约束。 |
| `benchmark/` | 同口径 body benchmark、T=2048/T=8192 原始 session JSON，以及 T=64 Docker smoke 记录。 |
| `reports/` | chunk-o 优化过程、各候选版本结果、机器审计和最终总结。 |

最重要的总报告在：

```text
reports/00_chunk_o_optimization_complete_report_cn.md
```

Z5B 的正式报告在：

```text
reports/06_z5b_direct_q_cache.md
```

Z5B 与 Triton 的剩余 VMEM/VALU/LDS 归因在：

```text
reports/07_z5b_remaining_vmem_ledger.md
```

`reports/README.md` 是报告目录索引，说明每份副本对应哪个实验阶段。

如果需要看完整 LLVM、MIR、ISA、HSACO 和 rocprof 原始文件，不在这个最小包里找，
请回到仓库原始证据目录：

```text
codex_qwen_bt64_stage6z_native_chunko/
codex_qwen_bt64_stage6z_z5b_machine_stage1/
codex_qwen_bt64_stage6z_z5b_rocprof/
```

这些目录与本包的对应关系已经在总报告的“文件地图”章节中列出。

## 重要边界

- Z5B 是 standalone isolated performance baseline，不是 production kernel。
- “Triton 比 Z5B 快”是同一类 chunk-o body 的诊断比较，不等于完整模型 API 的端到端倍率。
- Z5B 与 native 的 T=2048 same-shape 结果最可比；长文本 native 可能切换 WG/warp/stage，
  benchmark 会把实际 selector 记录下来，不能把不同 shape 的结果混写。
- 不能只看静态 ISA 指令数推导动态 VMEM/LDS/MFMA；动态数必须来自同口径 PMC。
- 不要把 Z5B 的 shared Q cache 误认为 recurrence 的 H/state cache；这个包只处理 chunk-o。
