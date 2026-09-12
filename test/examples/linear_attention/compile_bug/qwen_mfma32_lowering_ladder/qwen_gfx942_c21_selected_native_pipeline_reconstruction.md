# Qwen gfx942 C21 Selected-Native Pipeline Reconstruction

## 结论

**Case B: `STOP_C21_PIPELINE_NOT_MATERIALIZED`。** C21 的 source 和编译器 plan
确实把 Q@H 与两个 Q@K consumer 写进同一个 K32 superloop，且 T=2048 五类
正确性均与 Z5B BF16 byte-exact；然而 final ISA 没有形成所需的 next-stage
producer 与当前 MFMA window 的真实重叠。C21 仍是 `producer -> immediate
wait -> LDS publication -> barrier -> consumer` 的子阶段序列。因此，不应把它
晋级为新的 Stage6Z performance baseline，也不应自动进入 C22。

## Fresh Selected Native 身份

- 目标：gfx942，`T=2048`，`BT64/BV64/BK32`，WG256（4 waves/CTA），`num_stages=2`。
- selected metadata hash：`d5c5e6b6d5ee7abce52cb10ce5d3937161f4d0f195f594f096cf9b2b4f75e1a1`。
- HSACO SHA256：`cc74acf84104a0900ed327fc469826c228c94fdf07fd55d8d302c3e88c04e72d`。
- shared metadata：`12288 B`；launch grid：`(2, 32, 8)`。
- 这是从 fresh selector cache 固定出的 stage-2 HSACO；当前运行时 cache 若重新选择其他 stage，不作为本报告对照。

`num_stages=2` 的可见形式是 TTGIR 中一个带 slot/index 的循环携带状态和 `memdesc<1x...>` staging 表示。它支持 rotating staged schedule 的结论，但单独不能被夸大为“所有物理 buffer 必然完整双缓冲”。

## Native 与 C19

fresh native TTGIR 在同一个 K32 `scf.for` 中同时携带 Q@H 与 Q@K 的两个 MMA accumulator：loop 内先 global-load next Q/K/H，再 local-load current Q/K/H，随后连续执行 Q@H 与 Q@K dot，最后写回 next staged slot。最后一个 staged slot 在 loop epilogue 被消费。

C19 则是完整 Q@H 四个 K32 stage，接着完整 Q@K half-0 四个 stage，最后 Q@K half-1 四个 stage；这三个 source loop 之间有显式 phase 边界。C19/C21/native 的 final ISA lexical 结构不能替代动态 PMC，但足以说明同步图形状不同：

| arm | MFMA32 | barrier | waitcnt | global load | LDS read/write |
|:--|--:|--:|--:|--:|--:|
| C19 | 56 | 46 | 147 | 104 | 112/112 |
| C21 | 20 | 16 | 94 | 94 | 40/100 |
| fresh native | 40 | 11 | 48 | 31 | 80/40 |

## C21 实际结果

C21 的 `ChunkOPipelinePlan` 确实进入 lowering，并形成与 C19 不同的 LLVM/MIR/ISA/HSACO；C21 HSACO 与 C19、native 均为不同 hash。它也保持 scratch/private/spill 为零。但硬件发射顺序未达标：C21 final ISA 中 current QH MFMA 在约 378/383/387/391 行，下一 K producer issue 出现在约 394 行之后，紧跟 wait/publish/barrier，而 score MFMA 到约 422 行才开始。这不是 native 那种在循环内扩大 load/MFMA 调度窗口的 materialization。

因此，wide typed feeding 仅证明为机器图中存在相应宽 packet family，不能宣称它已构成 selected-native 级别的 producer/consumer pipeline。

## 正确性与资源

C21 在 `T=2048` 的 random、zero-V-new、caller-owned NaN-prefill、structured Q/K/H one-hot 与 token/value pattern 五项均 finite，且对 Z5B BF16 byte-exact。C21 code object 为 VGPR188、AGPR80、SGPR36、LDS24576 B、private0、reported spill0；native selected 为 metadata shared12288 B、private0/spill0，资源字段不可与 rocprof `Accum_VGPR_Count` 混为同一量。

## 动态 PMC（每 CTA，诊断）

| arm | MFMA | VMEM | LDS | VALU | SALU | profiler VGPR/AccumVGPR |
|:--|--:|--:|--:|--:|--:|:--|
| z5b | 160.00 | 672.00 | 672.00 | 7072.00 | 768.00 | 76/100 |
| c19_frozen | 160.00 | 304.00 | 688.00 | 6174.00 | 618.00 | 88/88 |
| c21_frozen | 160.00 | 304.00 | 688.00 | 8018.00 | 722.00 | 108/84 |
| native_selected | 160.00 | 140.00 | 480.00 | 3376.00 | 660.00 | 100/36 |

这些值来自最后五个匹配 dispatch 的 counter median，按实际 `Grid_Size / Workgroup_Size` 归一为 CTA；trace 时间未用作正式延迟结论。

## 正式 T=2048 Body 性能

口径：7 个独立 fresh Python process、caller-owned preallocated output、current HIP stream、无 Graph、warmup=10、repeat=50、平衡轮换顺序；数值为 HIP-event session median 的中位数。

| arm | ms | us | 相对 Z5B |
|:--|--:|--:|--:|
| Z5B | 0.066499 | 66.499 | 1.000x |
| C19 | 0.094140 | 94.140 | 0.706x |
| C21 | 0.091355 | 91.355 | 0.728x |
| fresh selected native | 0.060390 | 60.390 | 1.101x |

C21 比 C19 快约 `2.785 us`，但比 Z5B 慢约 `24.857 us`（约 `37.4%`）。相对 Z5B 的 speedup 是 `0.728x`；C21/native 是 `1.513x`。按指定公式，C21 捕获的 Z5B→native gap 为 `-406.9%`，为负值，说明它没有回收 Z5B 与 native 的间距。

## 停止决定

C21 的 correctness、独立 code object、MLIR/LLVM/MIR/ISA capture 和正式性能均已完成；但 overlap 证据未通过，且 C21 不快于 Z5B。按预注册规则必须停止在 `STOP_C21_PIPELINE_NOT_MATERIALIZED`，不继续 C22 barrier、packet、VALU 或多长度扩展。本轮没有修改 X2 immutable recurrence HSACO、R4 recurrence、allocator/RA 或 production selector。

## 工件

同目录 JSON 记录 selected identity、native timeline、C19 gap、C21 plan、overlap evidence、correctness、machine evidence、PMC、formal body 和 regression 状态。
