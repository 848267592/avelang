# Qwen Direct-K64 S0: gfx942 Compiler-Owned Distributed K64 Stage

## 结论

S0 完成了预先限定的**编译器能力验证**，但没有通过性能晋级门槛。

新的 `amdgpu_qwen_k64_pipeline_stage_load/commit` 语义确实让同一份高层源码在
GPU-outlining 之后保持不同，并最终在 gfx942 ISA 中形成了真实的
`next K global load -> current MFMA -> late LDS commit` 顺序。它同时避免了 B2
的 `k_next[64]` thread-local 数组、scratch 和 spill。然而，在这个最小的单 K64
lookahead 中，15 个 fresh-process session 的中位延迟没有可检出的正收益：

| arm | 中位 body 时间 | 相对 immediate |
|:--|--:|--:|
| immediate | `0.024917 ms` | `1.0000x` |
| distributed | `0.024917 ms` | `1.0000x` |

session-paired `distributed - immediate` 的均值为 `-0.057 us`，bootstrap 95% CI
为 `[-0.328, +0.235] us`。区间跨过零，不能声称加速。因此 S0 不晋级，**不进入
S1（K0+K1 lookahead），也不接回 B0/full recurrence**。

这不是 B2 那种资源失败。S0 证明了“compiler-owned、分布式 vector packet、同 LDS
bank 的延迟 commit”可以被 AveLang 表达并保留到机器码；失败的是这个最小窗口没有
带来可测的 latency hiding。

## 范围与冻结条件

新实验文件：

- `vllm_compare/repro_qwen_gdn_direct_k64_pipeline_stage_s0.py`
- `vllm_compare/bench_qwen_gdn_direct_k64_pipeline_stage_s0.py`

它只复用正确的 C0 `preloaded_k` Direct-K64 MFMA32 consumer，验证一个 K half 的
lookahead。它不是 full recurrence，也不改变 B0、B1、B2、production selector、
allocator/RA、MFMA geometry、BV32 ownership 或 LDS layout。

冻结的条件为：BF16 K/V-new、FP32 g/state、K tile `64x64` BF16、WG128、两个 wave、
一套 `[2, 64, 64]` BF16 LDS K bank，以及两段各 16 条
`v_mfma_f32_32x32x8_bf16` 的 C0 consumer。所有输入、consumer、输出和 barrier
phase 都相同；唯一变量是 opaque stage token 的 late placement。

## 新的编译器边界

新增 target-specific experimental op：

```text
amdgpu_qwen_k64_pipeline_stage_load(source_k, tid, chunk_start, key_head, k_half)
    -> opaque i64 stage token
amdgpu_qwen_k64_pipeline_stage_commit(stage token, shared_k_bank)
```

实现位于：

- `lib/Dialect/AveLang/IR/AveLangOps.td`
- `lib/IR/Intrinsics/amdgpu_module.cc`
- `lib/Dialect/AveLang/Transforms/lower_qwen_k64_pipeline_stage_pass.cc`
- `lib/Target/GPU/lower_to_llvm.cc`

高层 token 不提供索引或 tensor 访问。JIT 前端会临时把 scalar 返回值包进 private
memref store/load；late pass 只识别这个唯一的传输包装，并在重新连回 stage producer
和 commit consumer 后删除它。没有 `k_next[64]` 或任何 source-level next-tile 数组。

每个 lane 仅产生四个 `vector<8xbf16>` packet：

```text
linear       = lane_id + 128 * packet, packet in [0, 3]
token        = chunk_start + linear / 8
k_packet     = linear % 8
source       = K[0, token, key_head, 64*k_half + 8*k_packet : +8]
LDS target   = K_bank[k_half, 8*k_packet + element, linear / 8]
```

128 lanes x 4 packets x 8 BF16 正好覆盖一个 `64x64` K half。当前实现保留 vector
global load，但 commit 仍按 BF16 scalar store 展开；它没有虚构 Triton 的完整
in-thread transpose 或 typed dot-fragment encoding。

`AVELANG_QWEN_K64_PIPELINE_LOWERING=immediate|distributed` 是唯一的 compiler
分叉。`immediate` 在 commit 点创建 load 和 store；`distributed` 在 stage-load 点
创建 packet load，而在原 commit 点创建同一批 store。

## Same-Source 证明

两臂均从完全相同的 pre-branch MLIR 开始：

```text
SHA256(pre_k64_pipeline_stage_lowering.mlir)
= 5a0299c09bd8f2a3d7ed26c0bb269cf9bf42db8f6b3495b1dac71f57eafffda3
```

该 MLIR 仍含相同的 `stage_load -> stage_commit` 高层语义。分叉只在
`LowerQwenK64PipelineStagePass` 读取环境选项后发生。`post_kfrag_load_lowering.mlir`
保留 `avelang.qwen.k64.pipeline.global_packet` 与 `lds_commit` marker，可直接看到
packet load 的位置差异；两臂的该 dump 均不再含 stage op 或
`builtin.unrealized_conversion_cast`。

每一臂都通过：

| 检查 | immediate | distributed |
|:--|:--:|:--:|
| current / next 输出 finite | pass | pass |
| next K LDS snapshot 对 `K[64:128, :64]^T` | byte-exact | byte-exact |
| immediate vs distributed current、next、snapshot | \- | 全部 byte-exact |
| MFMA 数 | 32 | 32 |

这项 repro 的正确性目标是 stage 语义与 C0 consumer 不变，不替代 nonzero-W full
recurrence correctness reference。

## ISA 顺序证据

两臂的 ISA 不再收敛。

| arm | next-K `global_load_dwordx4` | current C0 MFMA | next-K LDS commit |
|:--|:--|:--|:--|
| immediate | `0x222c..0x2298` | `0x20fc..0x21f0` | 随 load 后进入 next bank staging |
| distributed | `0x1ff8..0x2014` | `0x2150..0x2244` | `ds_write_b16` 从 `0x236c` 开始 |

distributed 的 next-K vector loads 在 current C0 first MFMA 前约 `0x158` bytes
发射；current MFMA 完成后有两个 `s_barrier`（`0x2360/0x2364`），才开始对同一
LDS bank overwrite。正是 S0 所要求的 load/compute/late-commit 拓扑。

静态 ISA 指令数完全相同：

| 指令 | immediate | distributed |
|:--|--:|--:|
| `global_load_dwordx4` | 8 | 8 |
| `global_load_ushort` | 16 | 16 |
| `ds_write_b16`（含 `_d16_hi` 的基名计数） | 80 | 80 |
| `ds_read_b128` | 24 | 24 |
| `v_mfma_f32_32x32x8_bf16` | 32 | 32 |
| `s_barrier` | 8 | 8 |

所以 S0 是搬移同一工作，而不是复制 global/LDS/MFMA 工作；它也没有出现 B2 的
scalar `global_load_ushort` 爆炸。

## 动态工作与资源

T=128、grid=128、WG=128 的 rocprof 中，26 次匹配 dispatch 的动态指令数相同：

| 指标 | immediate | distributed |
|:--|--:|--:|
| MFMA | 32 | 32 |
| VALU | 920 | 920 |
| SALU | 42 | 42 |
| VMEM | 240 | 240 |
| LDS instructions | 248 | 248 |
| profiler VGPR / AccVGPR / SGPR | `92 / 84 / 112` | `92 / 84 / 112` |
| profiler LDS / scratch | `20480 B / 0` | `20480 B / 0` |

两份 HSACO metadata 也完全一致：`.vgpr_count=156`、`.agpr_count=64`、
`.group_segment_fixed_size=20480`、`.private_segment_fixed_size=0`、
`.vgpr_spill_count=0`、`.sgpr_spill_count=0`。ISA 中 `spill`/`scratch` token 计数为
零。这里明确区分 code-object 的 VGPR/AGPR metadata 与 rocprof 的
`VGPR_Count/Accum_VGPR_Count`，二者不可互换。

## 性能复核

第一次 5-session、warmup=10、repeat=50 的中位数为 immediate `0.025117 ms`、
distributed `0.025878 ms`，后者慢约 `3.0%`。由于差异小于 1 us，进行了预先额外的
15-session confirmation：每臂 fresh process，warmup=20、repeat=100，session 内
rotating order。

| 统计 | immediate | distributed |
|:--|--:|--:|
| 15 session median of medians | `0.024917 ms` | `0.024917 ms` |
| distributed / immediate | \- | `1.000000x` |
| paired mean (distributed - immediate) | \- | `-0.057 us` |
| paired bootstrap 95% CI | \- | `[-0.328, +0.235] us` |

不存在可重复的正收益。前一轮的 `+0.761 us` median delta 和后一轮的近零 delta
共同说明，这个仅有一个 K64 tile 的窗口太窄，测到的是正常短 kernel 波动而非稳定的
latency hiding。

## 与 B2 的关系

| 路线 | next tile 表示 | scratch / spill | T=2048 结果 |
|:--|:--|:--|:--|
| B2 source lookahead | 每线程 `w_next[64]` / `k_next[64]` 等普通 local array | `840 B` / 209 VGPR spill words | 比 B0 慢约 60% |
| S0 compiler stage | 每 lane 4 个 BF16x8 packet，late commit | `0 B` / 0 | isolated 性能中性 |

因此 B2 仅否定 register-materialized lookahead；S0 修复了其表示和资源问题，但没有
证明单 K64 lookahead 本身足以改善吞吐。

## 决策

S0 的结构、same-source、正确性、vector-width、scratch/spill 和 dynamic-work 门均
通过；性能门失败。因此：

1. 不把该 primitive 接入 B0 或 production。
2. 不实施 S1 `next K0 + K1`，不继续加入 W/U/g，也不进行 B3 full recurrence。
3. 保留这次 experimental-only compiler primitive 与 artifacts，作为“late placement
   能不引入 B2 spill”的证据。
4. 若未来重启这条线，必须先用 Triton trace 证明一个更宽的 load/compute window 以及
   target-specific waitcnt/packet permutation/typed fragment 计划；不能只放大当前 S0。

## 复现

```bash
cmake --build /tmp/avelang-build-kfrag-qwen-rocm722 --target _avelang_bindings -j 16

PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:python:test/examples/linear_attention/vllm_compare \
PYTHONDONTWRITEBYTECODE=1 \
python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_direct_k64_pipeline_stage_s0.py \
  --seed 20260728 --warmup 20 --repeat 100 --sessions 15 --json
```

完整 MLIR、HSACO、ISA 和 rocprof CSV 位于：
`test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_pipeline_stage_s0/`。
