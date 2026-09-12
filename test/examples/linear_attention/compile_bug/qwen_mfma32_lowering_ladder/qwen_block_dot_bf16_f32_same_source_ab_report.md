# Qwen Direct-K64 `block_dot_bf16_f32` 同源 Lowering A/B

## 1. 结论

本实验通过了预先冻结的 same-source A/B gate。一个新的实验性
`al.amdgpu.block_dot_bf16_f32` 保持为专用 AveLang op，直到
AveLang-to-memref 之后、intrinsic implementation linking 之前的 late
lowering pass 才展开。两条编译路径共享完全相同的高层源码、pre-branch
MLIR、launch、BT64/BV64/WG128 ownership、LDS allocation shape、数学、K32
MFMA32 schedule 和 global ABI；唯一差别是该专用 op 的展开次序。

在 T=2048，gfx942 specialized lowering 相比 generic lowering：

- 正式 body benchmark：`0.724316 ms -> 0.519632 ms`，`1.394x`，或降低
  `28.26%`；五个独立 JIT process 的 session 样本没有重叠。
- rocprof matching-kernel trace median：`699.339 us -> 504.009 us`，降低
  `27.93%`。
- 动态 MFMA 完全不变：总计 `32768`，即 `2048/CTA`。
- VMEM 降低 `31.19%`，VALU 降低 `40.11%`，SALU 降低 `46.89%`，LDS
  指令降低 `28.57%`。
- 资源没有 cliff：scratch 为零、MIR 中没有 spill save；VGPR
  `116 -> 104`，AccVGPR `164 -> 160`。
- generic 与 specialized 在 T=2048 的 H 和 final state 均逐元素
  bit-exact。

这是一条明确的 **Avelang lowering/scheduling 正结果**：在相同高层程序和
同一 MFMA 工作量下，晚期专用 block-dot 展开能删除重复的 V-decay staging 与
随之而来的 global/LDS/address 工作。它不证明 full-v29 broad/compact-K
寄存器问题已经解决，也不能单独证明与 Triton 的全部差距都是编译器问题。

specialized 仍为 current-Triton W=0 control 的 `4.338x`（T=2048）。该
native control 仍带有 fused-pred 工作，且 native 为 BV32/32 CTA ownership，
本实验为 BV64/16 CTA；因此该倍数是同 ABI 下的诊断指标，不能当作纯
update-only 的 compiler-only 比较。下一步应先做 **BV32/CTA ownership 对齐的
direct-K64 block-dot 诊断**，而不是改 allocator/RA、broad-K、旧 compact-K
或 full-v29。

本工作为 experimental-only。v23/v24/v26/v27/v28、production dispatch、
current-vLLM recurrence bridge、allocator/RA、full-v29 nonzero-W correctness
均未修改。

## 2. 问题、ABI 与冻结边界

目标是隔离 direct-K64 recurrence-update suffix 中手写 scalar fill、`al.view(i32)`、
fragment extract 和逐 fragment MFMA loop 的 lowering 开销。输入和状态契约如下：

| 项目 | 固定值 |
|:--|:--|
| K / V-new | BF16 `[1,T,4,128]` / BF16 `[1,T,8,128]` |
| G / initial state | FP32 G、persistent FP32 state |
| 输出 | BF16 H snapshot、FP32 final state |
| geometry | BT64、BV64、WG128、2 waves、direct K64 |
| state layout | 每 wave 的 existing H1/H2，四个 32x32 FP32 fragments |
| MFMA | 仅 gfx942 `v_mfma_f32_32x32x8_bf16` |
| T=2048 | grid `16` CTA，必须 `2048 MFMA/CTA` |

保持不变的内容：producer、K/V-new/G global ABI、CTA ownership、shared
allocation shape、barrier 数、persistent H1/H2 mapping、K32 accumulation
order、输出 layout 和 launch。没有修改 MFMA intrinsic、dtype、数学或
register allocator。

## 3. 高层源码与两条 Lowering

实验源码
`test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_block_dot_ab.py`
在每一个 K64 half 只表达一次：

```python
update_pair = al.amdgpu.block_dot_bf16_f32(
    a_stage, b_stage, k, v_new, g, tid, chunk_start,
    value_head_idx, key_head_idx, value_base, k_half, g_last,
    persistent_low, persistent_high,
)
```

该 op 返回两个更新后的 K32 persistent FP32 fragments。高层 Qwen 文件没有
explicit scalar operand fill、i32 view、fragment extraction 或 MFMA loop。

编译器新增的 op 与 lowering 位于：

| 文件 | 作用 |
|:--|:--|
| `lib/Dialect/AveLang/IR/AveLangOps.td` | `amdgpu_block_dot_bf16_f32` IR op |
| `lib/Dialect/AveLang/IR/AveLangOps.cc` | BF16/F32 shape、address space 和 fragment verifier |
| `lib/IR/Intrinsics/amdgpu_module.cc` | `al.amdgpu.block_dot_bf16_f32` source API export/check |
| `lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc` | generic / gfx942-specialized late expansion |
| `lib/Target/GPU/lower_to_llvm.cc` | 在 AveLang-to-memref 后、intrinsic link 前插入 pass |

这里的“late”是相对 Python/frontend 和通用 memref lowering 而言，并非把自定义
op 原样保存到 LLVM。专用 op 在 `LowerQwenBlockDotPass` 内消失；但两种展开的
差异在 pre-opt LLVM、post-opt LLVM、post-RA MIR 和最终 ISA 中均继续存在。

两条路径的唯一算法外差异是 V-decay staging 的消费顺序：

```text
generic:
  token0: stage A, stage K-col0, MFMA; stage A, stage K-col1, MFMA
  token1: stage A, stage K-col0, MFMA; stage A, stage K-col1, MFMA

specialized:
  token0: stage A once, stage K-col0/MFMA, stage K-col1/MFMA
  token1: stage A once, stage K-col0/MFMA, stage K-col1/MFMA
```

两条路径均保持四个 MFMA32 per static tile，且不跨两个独立 K64-half op
复用 A，避免借由增大 persistent accumulator/live range 制造新的混杂变量。

## 4. Same-Source 证明链

每个 lowering 在 fresh JIT process 编译。`AVELANG_BLOCK_DOT_LOWERING` 只被
`LowerQwenBlockDotPass` 读取。进入该分叉之前的 snapshot 相同：

| 层次 | generic SHA-256 | specialized SHA-256 | 是否相同 |
|:--|:--|:--|:--|
| pre-kfrag branch MLIR | `060f9f7660e09438404ccc5a05260779303938a007b016548789387d696ae10a` | 同左 | 是 |
| post-kfrag rewrite MLIR | 同上 | 同上 | 是 |
| post-block-dot lowering MLIR | `3c324818...a98ea4e` | `6e4ab484...3a7adf9` | 否，预期分叉 |
| pre-opt LLVM | `b77ddb24...d40ae267` | `ed884fc8...3837cb7b85` | 否 |
| post-opt LLVM | `99a052f5...8cfd6f33` | `e00db4e5...b76a11118` | 否 |

`post_gpu_outlining.mlir` 在当前 snapshot hook 中未捕获，标记为 N/A，不能据此
声称精确的 machine-pipeline first-convergence pass。但 post-opt LLVM hash 不同，
且最终 HSACO、MIR、ISA 也不同，因此没有发生“LLVM 优化后两臂重新收敛”。

post-block-dot MLIR 中，两臂都有 `33` 个静态 MFMA32 call 和 `17` 个
`gpu.barrier`；仅 marker 不同：generic 有两个
`avelang.block_dot.generic`，specialized 有两个
`avelang.block_dot.gfx942_specialized`。这表明变的是专用 lowering 的 schedule，
不是高层 Qwen 代码、工作量或同步拓扑。

## 5. 正确性

测试使用 direct-K64 reference；这不是 full-v29 nonzero-W reference gate。

| T | lowering | H max abs | H mean abs | final max abs | final mean abs | 状态 |
|--:|:--|--:|--:|--:|--:|:--|
| 64 | generic / specialized | `0` | `0` | `5.72204590e-06` | `1.26569716e-07` | pass |
| 512 | generic / specialized | `0.25` | `5.022e-07` | `1.52587891e-05` | `7.634e-07` | pass |
| 2048 | generic / specialized | `0.5` | `1.517369e-06` | `4.57763672e-05` | `2.498069e-06` | pass |

最终 rebuild 后，`pytest -q test_qwen_gdn_direct_k64_block_dot_ab.py -s`：
`2 passed in 25.27s`。T=2048 另以两个独立 JIT process 保存输出后比较：

```json
{"f_equal": true, "f_max_abs": 0.0, "h_equal": true, "h_max_abs": 0.0}
```

所以 specialized 不是以放宽 BF16 误差换取速度；它与 generic 的观测输出
bit-exact。

## 6. 正式 Body Benchmark

每臂每个 session 是 fresh process，避免一个 lowering 的 JIT code object 被另一臂
复用。warmup=5、repeat=20、五个 session，起始顺序 rotating。native 是现有
current-Triton W=0 control；它具有同一 BF16 K/V-new boundary，但包含 fused-pred
control 工作，因此只作诊断参照。

| T | chunks | generic ms | specialized ms | generic / specialized | specialized / native W=0 | native W=0 ms |
|--:|--:|--:|--:|--:|--:|--:|
| 512 | 8 | `0.203883` | `0.153648` | `1.327x` | `4.089x` | `0.037576` |
| 1024 | 16 | `0.368687` | `0.266916` | `1.381x` | `3.928x` | `0.067941` |
| 2048 | 32 | `0.724316` | `0.519632` | `1.394x` | `4.338x` | `0.119777` |

从 T=512 到 T=2048 的端点斜率：generic `21.685 us/chunk`，specialized
`15.249 us/chunk`，native W=0 control `3.425 us/chunk`。specialized 收回约
`6.435 us/chunk`，但这还不能归因成“只差 CTA ownership”，因为 native 不是
pure update-only 且 BV/CTA 映射不同。

## 7. T=2048 Rocprof

下表是八个 matching kernel dispatch 的 median。总 grid work-items `2048`，
WG=128，因此每次 kernel launch 有 16 CTA。PMC 的 MFMA/VMEM/VALU/SALU/LDS
数为整次 dispatch；括号内为 MFMA 的 CTA 归一化值。

| 指标 | generic | specialized | 变化 |
|:--|--:|--:|--:|
| trace median | `699.339 us` | `504.009 us` | `-27.93%` |
| grid / WG | `2048 / 128` | `2048 / 128` | 相同 |
| LDS block | `6144 B` | `6144 B` | 相同 |
| scratch | `0 B` | `0 B` | 相同 |
| VGPR | `116` | `104` | `-10.34%` |
| AccVGPR | `164` | `160` | `-2.44%` |
| SGPR | `112` | `112` | 相同 |
| occupancy percent | `0.323302` | `0.321294` | 基本相同 |
| SQ_INSTS_MFMA | `32768` (`2048/CTA`) | `32768` (`2048/CTA`) | 相同 |
| SQ_INSTS_VALU | `3321312` | `1989088` | `-40.11%` |
| SQ_INSTS_SALU | `428032` | `227328` | `-46.89%` |
| SQ_INSTS_VMEM | `223232` | `153600` | `-31.19%` |
| SQ_INSTS_LDS | `229376` | `163840` | `-28.57%` |

这些变化的方向与专用 schedule 一致：A/V-decay staging 不再为两个相邻 K32
consumer 重复进行，从而连带减少 global address arithmetic、global traffic 和 LDS
store/load。MFMA、barrier、LDS allocation 和 occupancy 没有靠降低计算语义来换取
收益。

## 8. ISA 与 MIR

T=2048 HSACO 使用 `/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx942` 解码。

| 静态 ISA 项 | generic | specialized |
|:--|--:|--:|
| `v_mfma_f32_32x32x8_bf16` | `32` | `32` |
| `v_mfma_f32_16x16x16_bf16` | `0` | `0` |
| `s_barrier` | `17` | `17` |
| `ds_read*` | `32` | `32` |
| `ds_write*` | `96` | `80` |
| `buffer_load*` / `global_load*` | `126` | `106` |
| `buffer_store*` / `global_store*` | `38` | `38` |
| HSACO bytes | `18696` | `15584` |

两臂都保留正确的 MFMA32，没有退化到 MFMA16。specialized 的静态 global-load 和
LDS-write 数分别少 `20` 与 `16`；这与 PMC 的动态 VMEM/LDS 降幅相互印证。

通过 `AVELANG_AMDGPU_LINK_DEBUG_DIR` 捕获真实 linker argv 后，以
`replay_qwen_v29_lto_mir.py` 重放 LTO，并在 greedy / virtregrewriter /
prologepilog 处 dump MIR：

| MIR 项 | generic | specialized |
|:--|--:|--:|
| 代表性 pre-greedy section 行数 | `2108` | `1584` |
| `SI_SPILL_AV32_SAVE` / `SI_SPILL_AV64_SAVE` | `0 / 0` | `0 / 0` |
| post-RA scratch evidence | 无 | 无 |

因此这条 direct-K64 实验没有把资源问题隐藏成 spill。它的收益来自更少的机器
指令和地址/内存工作，不是 register allocator 的补救或 physical-register
hard-coding。

## 9. 与先前 direct-K64 三方控制的关系

此前同 ABI 测得：

| T=2048 arm | median ms | MFMA / CTA | VGPR | AccVGPR | VMEM / CTA |
|:--|--:|--:|--:|--:|
| current Triton W=0 control | `0.116653` | `2048` | `104` | `160` | `1824` |
| 旧 Avelang direct-K64 MFMA32 | `1.133085` | `2048` | `24` | `144` | `22144` |
| 旧 Avelang direct-K64 MFMA16 | `2.163236` | `16384` | `52` | `212` | `42112` |

本次 specialized 的 T=2048 fresh-process benchmark 是 `0.519632 ms`，比旧手写
MFMA32 诊断值明显低，但它与上述数值不是同一轮 harness，不能把跨轮绝对差写成
正式 speedup。可信的本轮结论是 generic/specialized 同源 A/B 的 `1.394x`。同时，
它显示相同 MFMA/CTA 并不意味着相同 performance：Avelang 的显式 staging、address
work、BV64 ownership 和 native pipeline 仍是主要的剩余结构差异候选。

## 10. 结论与下一步

1. **specialized lowering 是否生效并保留？** 是。pre-branch MLIR hash 完全相同；
   分叉后 post-opt LLVM hash 仍不同，post-RA MIR、HSACO 字节数、静态 ISA 和
   rocprof PMCs 均不同。专用 op 本身按设计在 late pass 展开，而展开差异没有被
   LLVM 消除。
2. **减少了什么机器工作？** 在 MFMA、barrier、LDS allocation、CTA/WG 不变时，
   specialized 删除重复 V-decay staging，降低 VMEM、VALU、SALU、LDS 指令和 VGPR，
   没有 scratch/spill。
3. **距 Triton 还多远？** T=2048 specialized 是同轮 native W=0 control 的
   `4.338x`。该差距不能全部归咎于 compiler，因为 native 包含不同的 BV32/32-CTA
   ownership 与 fused pipeline；也不能将 native control 当作 pure-update latency。
4. **是否可以继续 block-dot lowering 微调？** 此轮已证明这一级的 source/late
   lowering 可回收大约 28% 时间。下一项应是独立的 **BV32 / CTA ownership 对齐
   direct-K64 experiment**，再判断剩余差距有多少属于 ownership/pipeline。不要在
   没有该对齐控制前修改 RA、重新尝试旧 broad/compact-K，或将此结果接入 full-v29。

## 11. 复现命令与证据目录

```bash
# Build
cmake --build /tmp/avelang-build-kfrag-qwen-rocm722 \
  --target _avelang_bindings -j 16

export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:.
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare

PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  test_qwen_gdn_direct_k64_block_dot_ab.py -s
PYTHONDONTWRITEBYTECODE=1 python3 \
  bench_qwen_gdn_direct_k64_block_dot_ab.py \
  --T 512 1024 2048 --warmup 5 --repeat 20 --sessions 5 --json

# Per-arm PMCs, run once for generic and once for specialized.
AVELANG_BLOCK_DOT_LOWERING=<mode> /opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM \
        SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex direct_k64_block_dot_ab -d <out> -o counters -f csv -- \
  python3 profile_qwen_gdn_direct_k64_block_dot_ab.py --lowering <mode> \
  --T 2048 --warmup 2 --repeat 5
```

主要文件：

- `test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_block_dot_ab.py`
- `test/examples/linear_attention/vllm_compare/test_qwen_gdn_direct_k64_block_dot_ab.py`
- `test/examples/linear_attention/vllm_compare/bench_qwen_gdn_direct_k64_block_dot_ab.py`
- `test/examples/linear_attention/vllm_compare/profile_qwen_gdn_direct_k64_block_dot_ab.py`
- `test/examples/linear_attention/vllm_compare/audit_qwen_gdn_direct_k64_block_dot_ab.py`
- `test/examples/linear_attention/rocprof_outputs/qwen_block_dot_bf16_f32_ab/`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_block_dot_bf16_f32_same_source_ab_report.md`
