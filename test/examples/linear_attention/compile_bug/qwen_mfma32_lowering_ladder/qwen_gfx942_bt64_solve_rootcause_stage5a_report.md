# Qwen gfx942 BT64 FP32 Solve Stage 5A Root-Cause Audit

## 结论

Stage 5A 完成，且严格保持 audit-only：没有新增 solve、没有改 v18、没有改
Stage 4、asm-v0、compiler、LLVM 或 AMDGCN assembly。

当前 v18 solve 与 vLLM solve 计算的是同一个 FP32 契约：对每个 strict-lower
64x64 `A` 输出 `X=(I+A)^-1`，布局均为连续的 `[1,T,8,64]`。54 个同输入
case 均通过 authority 检查。造成差距的主因不是 input、kernel 数、scratch、
spill，也不是单纯的 global store；是 **v18 的 63 行串行 LDS/标量 recurrence
结构**，相对 vLLM 的 **4x16 hierarchical block inverse + FP32 MFMA block dot**
拥有更长的依赖关键路径、更高 SALU/LDS/VALU，以及更差的 wave/resource 形态。

唯一建议的 Stage 5B 是：先独立实现一个 FP32 BT64 4x16 hierarchical block
inverse，使用 256-thread CTA 和 MFMA16 block product；不接入 full pipeline
直到 standalone correctness/resource/performance gate 全部通过。

## 1. 真实调用路径

### Avelang v18

Stage 4 的
`qwen_gdn_full_bt64_stage4_all_s0_stages` 在
[qwen_gdn_bt64_nonrecurrence_mfma_v2.py](../../vllm_compare/qwen_gdn_bt64_nonrecurrence_mfma_v2.py:773)
调用 `qwen_gdn_solve_avelang_v18_bt64_layout`，该 wrapper 再发射
`_qwen_gdn_solve_kernel_v18_parallel`。源码和 launch 在
[qwen_gdn_chunked_avelang_v18_bt64_layout_fixed.py](../../vllm_compare/qwen_gdn_chunked_avelang_v18_bt64_layout_fixed.py:36)：

- 输入/输出：FP32、contiguous `[1,T,8,64]`；
- 一 CTA 对应一个 `(chunk, value_head)`；
- grid：`T/64 * 8`；workgroup：128 threads，即 2 waves；
- wrapper 的 `torch.empty_like` 已在 body-only benchmark 中排除；
- solve 后直接由 Stage 4 的 native W/U 读取 `a_solved`。

### vLLM/FLA

安装版的 `solve_tril` 位于审计副本
[`solve_tril_vllm_installed.py`](codex_qwen_bt64_solve_rootcause_stage5a/ir/vllm/solve_tril_vllm_installed.py:506)。
BT64 选择 `merge_16x16_to_64x64_inverse_kernel`（源码第 238 行），其 wrapper
分配 `torch.zeros_like(A)`，再以 grid `(T/64, B*H)` 发射一个 Triton kernel。

正常调用的实际 autotune 选择已捕获，而非猜测：`num_warps=4`、
`num_stages=5`、`num_ctas=1`、`precision=ieee`、`USE_TMA=false`。body-only
harness 用这个 exact config 直接发射；必要的输出清零在每个 event 之前完成，
不混入 body 时间。

完整 source map 见
[`source_call_graph.md`](codex_qwen_bt64_solve_rootcause_stage5a/source_call_graph.md)、
[`avelang_solve_source_map.json`](codex_qwen_bt64_solve_rootcause_stage5a/avelang_solve_source_map.json)、
[`vllm_solve_source_map.json`](codex_qwen_bt64_solve_rootcause_stage5a/vllm_solve_source_map.json)。

## 2. 数学、数值与依赖

v18 先把 `-A` stage 到 `mat[64,64]`，再按 `r=1..63` 做前向递推；每一行按列
并行，但下一行依赖上一行。每个 row 有三次 workgroup barrier：复制 row、完成
group partial、写回 final result。即 63 个串行 row stage，189 个 recurrence
barrier（加初始 staging barrier 共 190）。

vLLM 将 64x64 分成 4x4 个 16x16 block。四个 diagonal block 先做 local inverse，
然后计算：

```text
X21 = -X22 A21 X11      X32 = -X33 A32 X22      X43 = -X44 A43 X33
X31 = -X33 (A31 X11 + A32 X21)
X42 = -X44 (A42 X22 + A43 X32)
X41 = -X44 (A41 X11 + A42 X21 + A43 X31)
```

因此它不是不同数学，而是等价的 block 重排。它有四条局部的 14-step 16x16
inverse 链，以及 `{21,32,43}` -> `{31,42}` -> `{41}` 的三层 block-DAG；其
off-diagonal 工作变成 `tl.dot`。最终 HSACO 实际包含
`v_mfma_f32_16x16x4_f32`，不是只从 Triton `tl.dot` 文本推断。

两条路径的 FP32 结合顺序不同，所以不要求 bitwise identical。54 cases 都使用
同一个 `A` 对象，authority 是 `torch.linalg.solve_triangular(I+A,I)`：

| 全部 case 最大值 | v18 | vLLM |
|:--|--:|--:|
| max abs vs authority | `4.5776367e-05` | `3.0517578e-05` |
| max residual inf | `4.8995018e-05` | `1.9311905e-05` |
| 两实现 max abs | `6.1035156e-05` | - |

最大误差来自刻意构造的 high-dynamic case；所有 case 均 `passed=true`。详见
[`correctness.csv`](codex_qwen_bt64_solve_rootcause_stage5a/correctness.csv) 与
[`numerical_order_comparison.md`](codex_qwen_bt64_solve_rootcause_stage5a/numerical_order_comparison.md)。

## 3. 同口径性能

Body-only 表排除了 allocation、compile 和 vLLM `zeros_like` 准备；每点为 3 个
独立 session 的中位数，每 session warmup=20/repeat=100：

| T | chunks | v18 body ms | vLLM body ms | v18/vLLM |
|--:|--:|--:|--:|--:|
| 512 | 8 | 0.116694 | 0.035372 | 3.30x |
| 2048 | 32 | 0.122422 | 0.035613 | 3.44x |
| 8192 | 128 | 0.229501 | 0.043484 | 5.28x |

为和历史 Stage 4 对齐，public wrapper 的 T=2048 是 `0.125386 ms` vs
`0.053179 ms`，比例 `2.36x`，复现了先前约 2.36x 的观测。body-only 更严格，
因此揭示了被 vLLM 输出初始化掩盖的 compute 差距。

chunk sweep 拟合：

| implementation | intercept | slope/chunk | R2 |
|:--|--:|--:|--:|
| v18 | 103.069 us | 0.895079 us | 0.9586 |
| vLLM | 33.866 us | 0.087651 us | 0.9728 |

所以不只是 fixed overhead：v18 的 per-chunk slope 为 vLLM 的 10.21x。T=512
和 2048 看起来接近，是因为两者在这些 grid 下都能并行调度大量 CTA；到 T=8192
后 gap 明显扩大。原始 sample 与方法在
[`benchmark_methodology.md`](codex_qwen_bt64_solve_rootcause_stage5a/benchmark_methodology.md)、
[`standalone_benchmark.csv`](codex_qwen_bt64_solve_rootcause_stage5a/standalone_benchmark.csv)、
[`latency_fit.json`](codex_qwen_bt64_solve_rootcause_stage5a/latency_fit.json)。

## 4. rocprof 和 ISA

T=2048 的 selected dispatch 比较：

| metric | v18 | vLLM |
|:--|--:|--:|
| trace median | 110.224 us | 23.875 us |
| workgroup / waves | 128 / 2 | 256 / 4 |
| grid global workitems | 32768 x 1 | 8192 x 8 |
| CTA 数 | 256 | 256 |
| VGPR / AccVGPR / SGPR | 96 / 128 / 112 | 56 / 16 / 48 |
| LDS block / scratch | 17920 B / 0 | 0 B / 0 |
| OccupancyPercent | 4.733 | 6.991 |
| MFMA | 0 | 65536 |
| VALU | 2695424 | 1419264 |
| SALU | 2535680 | 570368 |
| VMEM | 32768 | 135168 |
| LDS instructions | 1177856 | 397312 |

两边都只有一个 solve CTA dispatch，均为零 scratch；HSACO metadata 也显示零
private segment、零 VGPR spill。没有证据表明需要先修 compiler/RA 或写 assembly。
vLLM 的 VMEM 更高却更快，也反证 global traffic 不是主解释；v18 的 SALU 是
4.45x，LDS instruction 是 2.96x，且没有 MFMA block update。

barrier 的精确动态 counter 不可得，故未伪造。v18 的 189 次 recurrence barrier
来自真实源码；两边 static ISA 的 `s_barrier` text count 受 unrolling/loop 影响，
不用于直接时延归因。完整资料在
[`resource_comparison.csv`](codex_qwen_bt64_solve_rootcause_stage5a/resource_comparison.csv)、
[`dynamic_counter_comparison.csv`](codex_qwen_bt64_solve_rootcause_stage5a/dynamic_counter_comparison.csv)、
[`static_isa_comparison.md`](codex_qwen_bt64_solve_rootcause_stage5a/static_isa_comparison.md)。

## 5. 受控消融

| T/input | real v18 | launch floor | load/store only | no-store checksum |
|:--|--:|--:|--:|--:|
| 64 random | 0.116053 | 0.018027 | 0.018908 | 0.105997 |
| 2048 random | 0.122302 | 0.017866 | 0.024397 | 0.106959 |
| 2048 diagonal-boundary | 0.122061 | 0.017266 | 0.022473 | 0.106419 |

`load/store-only` 仅保留原始地址映射和 full output 写回；它远小于 real solve。
`no-store` 保留 LDS staging/recurrence，并以最后一行 checksum 防止整个 kernel 被
DCE；它不是数学等价 variant，故只用于界定 final output store 不是主因。对角边界
输入几乎不改变 v18 时间，也支持 fixed control/barrier schedule 才是主要负担。

## 6. Stage 5B 决策

**继续，但只做一个方向：独立的 FP32 BT64 4x16 hierarchical block inverse。**

- 每 CTA 一个 chunk/head，256 threads/4 waves；
- 先四个 16x16 diagonal local inverse，再按三层 block-DAG 完成 six lower blocks；
- block product 目标使用 FP32 MFMA16；不改 public layout；
- 首先只测 standalone，先跑当前 54-case matrix 与 W/U consumer；
- gate：scratch=0、显式 LDS <=8 KiB、T=2048 body <=0.060 ms，强目标 <=0.050 ms；
- 若实现失败 gate，保留 v18，不开始 compiler 或 assembly 支线。

vLLM body `0.035613 ms` 是硬件/算法参考下界，不承诺 Avelang 可直接相同。若达到
0.06-0.05 ms，按 Stage 4 historical `solve=0.125687 ms` 和 full `0.475867 ms`
作简单替换估算，T=2048 full 可到约 `0.410-0.400 ms`；完整 pipeline 必须重测，
不得将该加减当结果。

实现蓝图与 gate 见
[`stage5b_implementation_blueprint.md`](codex_qwen_bt64_solve_rootcause_stage5a/stage5b_implementation_blueprint.md)。

## 7. 回归与产物

- standalone solve: 54/54 authority cases passed；
- Stage 4 KKT/W-U/chunk-o/solve regression: `29 passed in 23.42s`；
- asm-v0 full contract: `4 passed in 24.48s`；
- 新跑 Stage 4 T=2048 smoke: `stage4_all=0.482517 ms`，与历史
  `0.475867 ms` 同量级；此次 smoke 使用 warmup=2/repeat=5，因此只验证路径未被
  audit 改动，不替代正式 benchmark。

所有命令、原始 CSV、HSACO/ISA、rocprof trace 和最终 JSON 均在
[`codex_qwen_bt64_solve_rootcause_stage5a`](codex_qwen_bt64_solve_rootcause_stage5a) 下。
最终可机器读取结论是
[`final_decision.json`](codex_qwen_bt64_solve_rootcause_stage5a/final_decision.json)。
