# Qwen gfx942 BT64 Stage 6Z Z2: Phase-Aware BV64 Chunk-O

## 结论

**Z2 不晋级，也没有接入 X2 full graph。**

Z2 在保持 BT64/BV64/BK32、WG256、每个 `[64,64]` 输出 tile 的 CTA ownership、
MFMA32 数学和所有 ABI 不变的条件下，移除了 Z1 的 lane-private `frag_words`
LDS round trip。它在完整 isolated chunk-o 上通过了 `T=64/512/2048/8192`
的 Z1 byte-exact 检查，且在 `T=2048` 与 `T=8192` 都稳定快于 Z1；scratch 和
MIR/code-object spill 都为零。

不过，最终 HSACO 仍有 **27 条静态 `s_barrier`**。虽然比 Z1 的 41 条少 14 条，
仍不满足本轮预注册硬门槛 `static barrier < 19`。因此本报告只保留 Z2 作为一个
有证据的未晋级候选，**不创建 X2 替换版、不运行 X2 public Eager API 计时、也不修改
production selector**。

## 范围与冻结项

Z2 只修改了 isolated native-style chunk-o：

| 项目 | 固定值 |
|:--|:--|
| source | `qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py` |
| kernel | `_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z2` |
| tile / ownership | `BT64, BV64, BK32`，一个 CTA 负责一个 `[64,64]` 输出 tile，两个 CTA/chunk-head |
| workgroup | `WG256`，四个 wave |
| dtype | `q/k/v_new/h/out=BF16`，`g=FP32`，accumulator=`FP32` |
| MFMA | `v_mfma_f32_32x32x8_bf16`，K32 reduction order 不变 |
| 未修改 | X2 immutable current-vLLM recurrence HSACO、R4 recurrence、allocator/RA、全图 dispatch、tile/WG、数学和 public ABI |

## WG shape hard guard（后续实验必读）

Z2 源码只有一个 launch shape：`WG256`。在
`qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py` 中，
`WORKGROUP=256`，host launch tuple 固定为 `(WORKGROUP, 1, 1)`，并新增了
`Z2_WORKGROUP_CONTRACT=256` 检查。Z2 没有 autotuner，也没有 WG128 fallback。

本路线中出现过的 WG128 有两个完全不同的来源，不能与 Z2 混写：

1. `qwen_gdn_bt64_native_chunko_stage6z_z3_wg128.py` 是独立的 Z3 source candidate，
   它明确使用 `WORKGROUP=128`；
2. native Triton 在 `T>=4096` 的 public selector 可能实际选择 WG128；这是 native
   的配置，不是 Z2 的配置。

此前错误的同进程三方 benchmark 在 `_native_launch()` 中直接调用
`chunk_fwd_kernel_o[...]`，没有先复用 public selector 的 chosen Config。Triton
因此可以在 benchmark 内自行选择另一个 candidate，造成“Z2 对 WG128 native”的
shape 混淆。该不安全 direct-call 路径已经从
`bench_qwen_gdn_bt64_stage6z_fixed_z2_z3_native.py` 删除，native 现在统一调用
`bench_qwen_gdn_bt64_stage6z_native_selected_wg256.py` 的
`select_and_pin()` + `launch()`。

后续 agent 的硬规则：

```text
Z2 vs native WG256：只能使用
bench_qwen_gdn_bt64_stage6z_fixed_z2_vs_native_selected.py
或 profile_qwen_gdn_bt64_stage6z_fixed_z2_vs_native.py。

任何直接 chunk_fwd_kernel_o[...] 的 native 调用都必须先 pin exact Config；
否则结果标记为 INVALID，不得写入 Z2 性能表。
```

当前同形状 T=2048 有效配置是：
`BK=32, BV=64, num_warps=4, num_stages=2, WG256`。旧的 WG128 或 autotune
污染数据只能作为历史错误记录，不得重新使用。

### Shape guard 修复后的 smoke 证据

在 gfx942 Docker 中重新执行了：

```bash
python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_native_chunko_stage6z.py \
  --T 64 --implementation z2 --warmup 1 --repeat 1
```

输出明确为：

```text
implementation=z2
kernel=_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z2
workgroup=256
finite=true
```

随后执行修复后的 mixed smoke：

```bash
python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_stage6z_fixed_z2_z3_native.py \
  --T 64 --sessions 1 --warmup 1 --repeat 1
```

它记录的 native pin 是：

```text
BK=32, BV=64, num_warps=4, num_stages=2, num_ctas=1, WG256
```

同一输出中 Z2 为 `0.082082 ms`、native 为 `0.061972 ms`、Z3 为
`0.100389 ms`；这里的单 session/单 repeat 只验证 shape，不用于性能排名。
这证明旧的 silent WG128 native fallback 已从 mixed benchmark 中移除；Z3
仍然是唯一明确的 WG128 source candidate。

Z1 使用 16 KiB `phase` LDS 加 12 KiB `frag_words[768,4]` i32 LDS。后者只是把
每个 lane 已从 `phase_vec` 读到的 i32 fragment 再写入 LDS、同步、随后读回。
Z2 保留 16 KiB `phase` 的跨 wave producer/consumer 边界，删除 `frag_words`，让
lane 直接把已读取的 `q_words/k_words/h_words/score_words/v_words` view 成 MFMA operand。
这是一条 phase-aware lifetime 缩短，不是批量删除 barrier。

## Z1 的 41 条 Barrier 归属

Stage 7A 已把 Z1 source、pre-link LLVM、pre-LTO assembly 与 final HSACO 按 ordinal
对齐；四层都得出 41 条。它们全部来自 `al.syncthreads()`，不是 AMDGPU backend
额外插入。下表的 ordinal range 展开后覆盖全部 41 条 `s_barrier`：

| Z1 ISA ordinal | source phase / site | 静态数 | buffer 与跨-wave hazard | Z2 处理 |
|:--|:--|--:|:--|:--|
| 1-4 | A `stage_qh` | 4 | `phase`: Q/H producer -> inter MFMA consumer，RAW | 保留；Z2 对应 source 87 |
| 5-12 | A `pack_frag` | 8 | `frag_words` lane write -> fragment LDS read | 删除中间 LDS；不保留此 barrier |
| 13-20 | A `reuse_frag` | 8 | fragment read/MFMA -> 下一 fragment overwrite | 用每 K32 tile 的 phase release barrier 替代 |
| 21-22 | B `stage_qk` | 2 | `phase`: Q/K producer -> score MFMA owner wave，RAW | 保留；Z2 对应 source 118 |
| 23-26 | B `pack_frag` | 4 | `frag_words` lane write -> LDS read | 删除中间 LDS；不保留此 barrier |
| 27-30 | B `reuse_frag` | 4 | fragment read/MFMA -> 下一 K32 producer overwrite | 每 K32 phase release barrier 保留；Z2 source 130 |
| 31-32 | B `serialize_score` | 2 | score half store -> 后续 score/V consumer | 保留；Z2 source 145 |
| 33 | C `stage_v` | 1 | score/V producer -> intra MFMA consumer，RAW | 保留；Z2 source 157 |
| 34-37 | C `pack_frag` | 4 | `frag_words` lane write -> LDS read | 删除中间 LDS；C 后 phase 不再复用 |
| 38-41 | C `reuse_frag` | 4 | fragment read/MFMA -> fragment buffer reuse | 删除中间 LDS；C 后 phase 不再复用 |

所以 Z2 的 source-level意图是：仅删除 `frag_words` 专用的同步，把每个仍然存在的
barrier 都绑定到 `phase` 的 CTA-wide 发布或覆盖前 release。它没有删除
`stage_qh`、`stage_qk`、score-half serialization 或 `stage_v` 的跨 wave RAW 边界。

Z2 source 的显式 phase boundary 为 A 的 publish/release、B 的 publish/release 以及
score serialization、C 的 score/V publication。循环展开和控制流复制后，final HSACO
含 27 条 lexical `s_barrier`。该数目而非简单 source 行数是 gate 使用的数值。

## 与 Native vLLM 的同 tile 机器工作对齐

native selected T2048 `chunk_fwd_kernel_o` 与 Z1/Z2 都是 `[64,64]` tile、BT64/BV64/BK32、
WG256、两个 CTA/chunk-head。下表的 native 数来自 selected HSACO 的静态 ISA；其动态 PMC
没有以这个窄 body ABI 单独捕获，因此标为 N/A，不能由静态指令数反推或伪造。

| 每 `[64,64]` tile，T2048 | current native vLLM | Z1 | Z2 |
|:--|--:|--:|--:|
| static `v_mfma_f32_32x32x8_bf16` | 40 | 32 | 56 |
| static buffer/global load | 14 | 6 | 6 |
| static buffer/global store | 4 | 6 | 6 |
| static `ds_read*` | 72 | 64 | 56 |
| static `ds_write*` | 40 | 176 | 240 |
| static `s_waitcnt` | N/A in Z0 summary | 230 | 244 |
| static `s_barrier` | 11 (direct selected-ISA recount) | 41 | 27 |
| dynamic MFMA / CTA | N/A | 160 | 160 |
| dynamic VMEM issues / CTA | N/A | 1,056 | 1,056 |
| dynamic LDS instructions / CTA | N/A | 1,504 | 1,056 |
| dynamic VALU / CTA | N/A | 12,942 | 11,504 |
| dynamic SALU / CTA | N/A | 1,192 | 1,072 |

Z1/Z2 的动态数由 T2048 PMC 总数除以 512 CTA 得到。Z2 static MFMA/DS-write/waitcnt
增加不表示计算数学改变：它反映 LLVM/LTO 生成的 lexical instruction paths；动态 MFMA
仍严格为 160/CTA，数学工作不变。最终 body 计时和 PMC 才用于性能结论。

三者的语义 ABI 相同，每 tile 的无缓存逻辑 I/O 下界也是相同的：Q 16 KiB、K 16 KiB、
H 16 KiB、V-new 8 KiB、g 256 B 读取，BF16 output 8 KiB 写入。合计读取 57,600 B、
写入 8,192 B。实际 VMEM 字节不能仅从 `SQ_INSTS_VMEM` 推出，因为该 counter 是 issue
计数而非字节计数，且 ISA 中混有不同 vector widths；本轮不把它误报为精确硬件字节数。

与 native 的阶段差异仍是结构性的：native Triton 的 TTGIR 管理 local operand buffer
allocation/deallocation，selected T2048 ISA 仅 11 barriers；Z2 仍把可见 `phase` 当作 CTA
级复用边界。因此 Z2 证明 lane-private fragment LDS 是可消除的冗余 materialization，
但没有证明当前 source schedule 可复现 native 的完整 local-operand pipeline。

## 正确性

测试：

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_native_chunko_stage6z_z2.py -k 64 -s
```

结果为 `5 passed in 11.41s`。该文件覆盖：

| 检查 | 结果 |
|:--|:--|
| `T=64/512/2048/8192` 对 Z1 | BF16 output byte-exact |
| frozen Stage6W tolerance | 通过，阈值 `1/128` |
| finite | 通过 |
| zero V-new + NaN-prefilled caller-owned output reuse | 通过且对 Z1 byte-exact |
| MFMA geometry / K32 order | 不变 |

## T2048 资源与 ISA

T2048、512 CTA、WG256 的 fresh rocprof capture：

| 指标 | Z1 | Z2 | 变化 |
|:--|--:|--:|--:|
| trace median | 52.198 us | 45.788 us | -12.28% |
| profiler Accum_VGPR | 172 | 36 | -136 |
| LDS block | 28,672 B | 16,384 B | -12,288 B |
| code-object VGPR | 164 | 156 | -8 |
| code-object AGPR | 32 | 32 | 0 |
| code-object SGPR | 46 | 42 | -4 |
| dynamic MFMA | 81,920 | 81,920 | 0 |
| dynamic VMEM | 540,672 | 540,672 | 0 |
| dynamic LDS | 770,048 | 540,672 | -29.79% |
| dynamic VALU | 6,626,304 | 5,890,048 | -11.11% |
| dynamic SALU | 610,304 | 548,864 | -10.07% |
| occupancy percent | 15.227565 | 14.796711 | -0.43 pt |
| scratch | 0 B | 0 B | 0 |
| code-object VGPR/SGPR spills | 0 / 0 | 0 / 0 | 0 / 0 |
| static barriers | 41 | 27 | -14, **仍失败** |

`rocprof` 的 `VGPR_Count` 与 HSACO static metadata 不一致，因此本表以 HSACO 为
静态 VGPR/AGPR/SGPR 的依据，以 rocprof `Accum_VGPR` 和 PMC 为采样资源/机器工作依据。

## Caller-Owned Isolated Body

这是 preallocated caller-owned body 诊断，不是 public Eager API 排名。每长度五个
fresh-process session，current HIP stream，warmup 10、repeat 50，session 间轮换
`Z1,Z2` / `Z2,Z1`，没有 Graph capture。

| T | Z1 median of session medians | Z2 median of session medians | Z1-Z2 paired median | Z2 加速 |
|--:|--:|--:|--:|--:|
| 2048 | 0.076213 ms | 0.070044 ms | 6.309 us | 8.09% |
| 8192 | 0.200658 ms | 0.168691 ms | 31.968 us | 15.93% |

五个 paired session 全部是正收益：

| T | paired Z1-Z2 range | paired mean |
|--:|--:|--:|
| 2048 | 5.368 to 6.569 us | 6.029 us |
| 8192 | 31.065 to 33.670 us | 32.156 us |

这与 PMC 的 LDS/VALU/SALU 下降一致：Z2 的收益来自删去 lane-private LDS materialize
及其相关 address/fragment 工作，而不是删 MFMA、改结果或引入 V-new reload。

## Gate 判定

| 门槛 | 结果 | 判定 |
|:--|:--|:--|
| T64/512/2048/8192 bit-exact | 通过 | pass |
| scratch = 0，MIR/code-object spill = 0 | 通过 | pass |
| MFMA 数、数学、WG256、ownership 不变 | 通过 | pass |
| AccVGPR 显著低于 172 | 36 | pass |
| isolated body T2048 与 T8192 都快于 Z1 | +8.09%，+15.93% | pass |
| static `s_barrier < 19` | 27 | **fail** |

**最终决定：No-Go。** 这个候选没有满足所有预注册 gate，不能用正的 isolated speedup
覆盖 barrier 健康度要求。保持 X2、immutable recurrence HSACO 与 R4 不变；也不执行
Z2 的 public Eager full graph benchmark，因为那会违反“未过门槛不得接入 X2”的规则。

## 产物与复现

原始机器、PMC 与 benchmark 数据：

`codex_qwen_bt64_stage6z_z2_phase_aware/`

其中包括 `machine/z1.isa`、`machine/z2.isa`、两个 T2048 rocprof JSON，以及
`benchmark/fresh_process_z1_z2.json`。相关 source/test/harness：

- `qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py`
- `test_qwen_gdn_bt64_native_chunko_stage6z_z2.py`
- `run_qwen_gdn_bt64_native_chunko_stage6z_z2_confirmation.py`

## 补充：逐长度 native selector fresh capture

为了决定是否值得做 WG128 长文本候选，本轮没有用 T2048/T8192 推断其他长度，
而是在相同 MI300/gfx942 Docker 中分别以 fresh process 调用 native vLLM
public `chunk_fwd_o`，保存了实际命中的 `chunk_fwd_kernel_o` source、TTIR、TTGIR、
LLVM IR、AMDGCN ISA、HSACO 和 readobj。原始工件位于：

```text
codex_qwen_bt64_stage6z_native_chunko/native_refresh/T{T}/
```

`native_capture.json` 中的 `runtime_selected_config` 是 selector 实际结果，
`selected_static_summary` 是对应 selected AMDGCN 的静态计数。结果如下：

| T | BK | BV | num_warps | WG | num_stages | metadata shared | static MFMA32 | static barrier | selected HSACO SHA256 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 512 | 32 | 32 | 4 | 256 | 2 | 10240 B | 40 | 11 | `c87795d557485ced7f7ea6d3a3cc86c55f6f2640c5e948ef44ca4a0dcb65f976` |
| 1024 | 32 | 64 | 4 | 256 | 2 | 12288 B | 40 | 11 | `cc74acf84104a0900ed327fc469826c228c94fdf07fd55d8d302c3e88c04e72d` |
| 2048 | 32 | 64 | 4 | 256 | 2 | 12288 B | 40 | 11 | `cc74acf84104a0900ed327fc469826c228c94fdf07fd55d8d302c3e88c04e72d` |
| 4096 | 32 | 64 | 2 | 128 | 2 | 12288 B | 80 | 11 | `e201dd58c83e64565f10754789ac294df53343b9c9bbbe764e8f667817066ee5` |
| 8192 | 32 | 64 | 2 | 128 | 2 | 12288 B | 80 | 11 | `e201dd58c83e64565f10754789ac294df53343b9c9bbbe764e8f667817066ee5` |
| 16384 | 32 | 64 | 2 | 128 | 2 | 12288 B | 80 | 11 | `e201dd58c83e64565f10754789ac294df53343b9c9bbbe764e8f667817066ee5` |

最后一行的 hash 为原始捕获中的实际 selected HSACO 文件 hash；其 Triton metadata
编译 hash 是 `82614ff8e4b0ad1b389b2236643259f107401102f2fab48d92c841ab4fc54903`。
本报告不把 metadata hash 和 HSACO 文件 SHA256 混写。T16384 的实际 HSACO 文件 SHA256
应以 `native_refresh/T16384/native_capture.json` 为准；如果复制表格时只保留
metadata hash，也不能把它当成 code-object identity。

selector 的结论是明确的：短文本 `T<=2048` 使用四 wave/WG256，长文本
`T>=4096` 使用两 wave/WG128，均为两 stage。WG128 不是人为指定，也不是从两个点
插值出来的，而是四个长度的 fresh selector/capture 直接观察到的结果。

### native static resource 记录

以下字段来自每个 selected HSACO 的 `code_object_readobj.txt`。它们是 code-object
静态资源，不是 rocprof 动态 `Accum_VGPR`：

| T | `.vgpr_count` | `.agpr_count` | `.sgpr_count` | `.private_segment_fixed_size` | spills |
|---:|---:|---:|---:|---:|:--|
| 512 | 128 | 32 | 87 | 0 B | 0 |
| 1024 | 132 | 32 | 89 | 0 B | 0 |
| 2048 | 132 | 32 | 89 | 0 B | 0 |
| 4096 | 220 | 64 | 76 | 0 B | 0 |
| 8192 | 220 | 64 | 76 | 0 B | 0 |
| 16384 | 220 | 64 | 76 | 0 B | 0 |

fresh native narrow body 没有在本轮用与 Z2 相同的 PMC 采集动态 `Accum_VGPR`；
因此该列为 N/A，不能用 `.agpr_count` 代替。静态 ISA 计数在刷新工件中可复核：
T512 为 `buffer_load/store=13/4`、`ds_read/ds_write=104/33`；T1024/T2048
为 `14/4`、`80/40`；T4096/T8192/T16384 为 `28/8`、`56/36`。六种长度均为
`MFMA16=0`、`s_barrier=11`，且对应源/TTIR/TTGIR/LLVM/ISA/HSACO 文件齐全。

## 补充：native body 速度口径

在 selector capture 完成后，使用 fresh process、current HIP stream、无 Graph、
warmup=5、repeat=20，预先分配 `o` 的 direct body 和正常 eager `chunk_fwd_o` 各自
测量。首次 public call 的编译/autotune 不计入计时，结果保存于
`native_refresh/T{T}/body_speed.json`。这里的 native 速度只是 chunk-o body 对照，
不是完整 Qwen public API 端到端时间。

| T | chunks | native direct preallocated HIP ms | native public `chunk_fwd_o` HIP ms | direct slope-style us/chunk |
|---:|---:|---:|---:|---:|
| 512 | 8 | 0.037696 | 0.045167 | 4.712 |
| 1024 | 16 | 0.039178 | 0.045467 | 2.449 |
| 2048 | 32 | 0.043826 | 0.050555 | 1.370 |
| 4096 | 64 | 0.056483 | 0.063634 | 0.883 |
| 8192 | 128 | 0.089814 | 0.098987 | 0.702 |
| 16384 | 256 | 0.151886 | 0.154569 | 0.593 |

跨六个点的线性拟合为：

```text
native direct body = 0.030376 ms + 0.469496 us/chunk * chunks
native public body = 0.038122 ms + 0.455653 us/chunk * chunks
```

T=2048 的 native direct body 是 `0.043826 ms`，T=8192 是 `0.089814 ms`，
T=16384 是 `0.151886 ms`。这些值与 Z2 的 caller-owned isolated body 口径相近，
但仍要注意：Z2 的 WG 是固定 256，而 native 在 T>=4096 已经切换到 WG128；不能
把它们误写成同一个 kernel 的 result。

## 补充：Z3 WG128 长文本候选的 hard-stop

fresh selector 证实长文本应尝试 WG128 后，新增了唯一 Z3 source：

```text
vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z3_wg128.py
```

Z3 保持 Z2 的 direct-fragment 路径、BT64/BV64/BK32、每 `[64,64]` tile ownership、
BF16 ABI、MFMA32 数学和 phase barrier 语义，只把四 wave 的 value-half ownership
改为两 wave 顺序处理两个 32-value half。它没有批量删除 barrier，也没有接入 X2。

第一次 T64 smoke 暴露了一个实现错误：仍使用 WG256 的 `rep=8/16` 覆盖次数，
WG128 下只覆盖了一半 shared staging。该覆盖错误已经在 source 中修正为 A/B
`rep=16`、C `rep=32`，修正后的 T64 finite smoke 通过。随后改用一臂一进程的
validator，避免多 kernel 同进程异步错误污染后续测试：

| T | Z2 reference vs Z3 candidate | 证据 |
|---:|:--|:--|
| 64 | byte-exact | one-arm validator pass |
| 512 | byte-exact | one-arm validator pass |
| 2048 | byte-exact | one-arm validator pass |
| 4096 | byte-exact | one-arm validator pass |
| 8192 | 未得到可接受的 candidate 结果 | 长文本 gate 未通过/未完成 |
| 16384 | fail | candidate 进程 GPU memory fault，RC=134 |

T16384 的实际错误为 GPU memory access fault，而不是数值误差；因此它不能被解释为
“WG128 只是稍慢”或“需要放宽 atol”。T8192 也没有一条完整、被接受的 candidate
correctness 记录。按照预注册规则，任意长文本 correctness gate 未通过，就停止
后续性能和 selector 实现。

### Z3 最终判定

| gate | 结果 |
|:--|:--|
| T64/512/2048/4096 byte-exact | pass |
| T8192 accepted correctness | **未通过/未完成** |
| T16384 accepted correctness | **fail，GPU memory fault** |
| scratch/spill/resource | 未进入可接受的正式性能阶段 |
| T8192/T16384 快于 Z2 | 未测，禁止测 |
| static barrier < 19 | 未形成可接受 candidate，不能作为晋级依据 |

因此 Z3 不是一个可以和 Z2 做速度排名的候选。不能因为它在短长度的一臂检查中
byte-exact，就宣称 WG128 已经正确；也不能建立按长度选择 WG256/WG128 的 selector。
X2 immutable recurrence HSACO、R4 recurrence、production dispatch 和 production
selector 均未修改。

## 最终 Stage 6Z 状态

现在的证据链应这样记录：

1. Z0 完成了 native ownership/IR/ISA 审计，并证明 native selector 随长度切换。
2. Z1 证明直接改 source phase/lifetime 可以比 Z1 自己快，但 41 barriers 过高。
3. Z2 删除 `frag_words` 中间 materialization，在 correctness、AccVGPR、LDS、VALU、
   SALU 和 isolated body 上取得正收益，但 27 barriers 仍超过 `<19`，所以 No-Go。
4. Z3 是针对 native 长文本 selector 的唯一 WG128 候选；其短文本可以通过，长文本
   correctness 没有通过，且 T16384 出现 GPU memory fault，因此不运行性能、不接 full。

结论保持为：**Z2 是当前 isolated Avelang 性能最好的、但未晋级的候选；native
vLLM 的 WG128 选择已经得到逐长度实测；Z3 WG128 当前是 correctness No-Go。** 下一步
不能继续无依据地做更多 WG/tile/barrier 变体；若要继续，应先定位 Z3 长文本 fault 的
具体索引/编译器机器图原因，并重新建立 correctness gate。

## Stage 6Z 最终调试更新：Z3 长文本 fault 已修复

本节覆盖前面关于 Z3 “T8192 未完成、T16384 fault、correctness No-Go”的旧状态；
旧数据保留用于实验历史，最终判定以本节为准。

### 根因不是 compiler lowering

Z3 Phase B 的旧 K staging 对每个 `source_half` 使用 16 repetitions，产生 `row=0..63`，
但 score consumer 只读取 `lane_col=0..31`。对 `source_half=1`，死的 row 32 访问在
最后 chunk 变成：`global_token=(T-64)+1*32+32=T`，所以 T8192/T16384 的第一个非法
访问分别是 `k[token=8192]` 和 `k[token=16384]`。LDS phase row 仍在 `0..255`，没有
LDS OOB。Z3 源修复为 Q 保留 `rep=16`、K 改为 `rep=8`；实验 Z2 reference 也修复
同一死读，WG256 下 K 为 `rep=4`。

完整公式、buggy/fixed CSV/JSON 和 source-level audit 见：

```text
codex_qwen_bt64_stage6z_z3_wg128_debug/
```

### 修复后 correctness

| T | Z3 vs fixed Z2 | max abs | finite | zero-V caller-owned output |
---:|:---:|---:|:---:|:---:|
| 64 | byte-exact | 0.0 | pass | pass |
| 512 | byte-exact | 0.0 | pass | - |
| 2048 | byte-exact | 0.0 | pass | - |
| 4096 | byte-exact | 0.0 | pass | - |
| 8192 | byte-exact | 0.0 | pass | pass |
| 16384 | byte-exact | 0.0 | pass | pass |

本轮没有运行 Z3 性能、PMC、selector 或 X2 full graph。旧 Z2 timing/PMC 对应修复前
的 Z2 source，不能被重新标注为修复后的新测量。

### compile-only machine audit

Z3 T8192 的 compile-only machine 工件已经补齐：

```text
codex_qwen_bt64_stage6z_z3_wg128_debug/machine/
```

其中包含 `lowered_llvm.ll`、`pre_lto_amdgcn.s`、`final_isa.s`、`z3_fixed.hsaco`、
`machine/exact_lto/` 和 `machine/machine_evidence.json`。`get_mlir()` 在当前 Docker
绑定中单独会 core dump，所以没有伪造 initial MLIR；source-level address audit、
LLVM、MIR、ISA 和 HSACO 是可用的正式工件。

固定 Z3 code object 的关键字段为：`.group_segment_fixed_size=16384 B`、
`.private_segment_fixed_size=0`、`.vgpr_spill_count=0`、`.sgpr_spill_count=0`、
`.agpr_count=32`、`.vgpr_count=228`、`.sgpr_count=38`；静态 ISA 为 56 MFMA32、
21 barrier、336 global load、32 global store、50 LDS read、288 LDS write，且无
`ds_bpermute`。这些是编译期静态数据，不是动态 PMC。

### Stage 6Z 状态修正

现在应这样理解 Stage 6Z：

1. Z0 完成 native ownership/IR/ISA 审计，确认 native selector 按长度切换。
2. Z1 证明 source phase/lifetime 调整可以带来 isolated body 收益，但 41 barriers
   超过门槛。
3. Z2 删除 `frag_words` materialization，历史上是 isolated 最快的 Avelang 候选，
   但 27 barriers 仍超过 `<19`，所以仍为 No-Go。
4. Z3 是唯一 WG128 长文本候选；其 source-level dead K overread 已精确定位并修复，
   六个长度 correctness 全部通过，compile-only 资源无 spill。
5. Z3 的性能和 selector 状态仍为 pending；未接 X2、未改 production、未改 immutable
   recurrence HSACO/R4。

因此最终结论不再是“Z3 correctness No-Go”，而是：

```text
Z2 = 修复前历史测量中的 isolated 最快候选，但 barrier No-Go；
Z3 = 修复后 correctness 通过、机器资源审计通过、性能尚未重测；
production/X2 = 未改变。
```

## 修复后正式重测：Z2/Z3 当前有效结果

本节覆盖前文的 `performance pending` 状态。前文 Z2 的 timing/PMC 发生在
Phase-B dead K overread 修复之前，只能作为历史记录；本节的结果来自修复后的
Z2 和 Z3 source，覆盖 correctness、compile-only machine audit、caller-owned
body timing 和真实 rocprof PMC。两版都保持 BT64/BV64/BK32、MFMA32、BF16
ABI、K32 累加顺序、数学和 X2 immutable recurrence HSACO 不变。

### Correctness gate

每个 arm、每个长度均由独立 Python 进程运行；reference 使用冻结的 Stage6W
BF16 body，Z3 另外与独立进程生成的 Z2 输出做 BF16 byte-exact 比较。旧 Z1
source 在 T=1024 单臂仍暴露修复前的 dead-read fault，因此不再把旧 Z1 当作
修复后的 reference，也没有使用它的 timing。

| T | Z2/Z3 finite | Z2 vs Stage6W max abs | Z3 vs Stage6W max abs | Z3 vs Z2 |
|---:|:--:|---:|---:|:--:|
| 64 | pass | `1.16e-10` | `1.16e-10` | byte-exact |
| 512 | pass | `1.53e-05` | `1.53e-05` | byte-exact |
| 1024 | pass | `1.53e-05` | `1.53e-05` | byte-exact |
| 2048 | pass | `1.53e-05` | `1.53e-05` | byte-exact |
| 4096 | pass | `1.53e-05` | `1.53e-05` | byte-exact |
| 8192 | pass | `1.53e-05` | `1.53e-05` | byte-exact |
| 16384 | pass | `3.05e-05` | `3.05e-05` | byte-exact |

门槛为 `1/128`，全部通过。T64、T8192、T16384 的 Z2/Z3 zero-V
caller-owned output 测试也全部通过：NaN 预填输出被完整覆盖，输出 finite，
并与同 arm 的 fresh output byte-exact。

### Recompiled machine audit

重新编译使用
`dump_qwen_gdn_bt64_stage6z_z3_machine_artifacts.py --variant {z2,z3}`，
T=8192，`launch_executed=false`、`rocprof_executed=false`。initial MLIR
仍因当前 Docker binding 的 MLIR printer core dump 而不可用，未伪造该文件；
lowered LLVM、pre-LTO AMDGCN、exact-LTO replay、pre/post-greedy MIR、
virtregrewriter MIR、ISA、HSACO 均已保存。

| 字段 | fixed Z2 WG256 | fixed Z3 WG128 |
|:--|--:|--:|
| HSACO SHA256 | `2afaa8867da656421a9c87988ed4dcad757796c1b1d3b93e27307fb401a1c09a` | `a3b05564781bbb191038b506345183a5219e7f6959fd0a1dc5fbfafbaa92206c` |
| code-object VGPR | 168 | 228 |
| code-object AGPR | 32 | 32 |
| code-object SGPR | 44 | 38 |
| LDS fixed | 16384 B | 16384 B |
| private segment | 0 B | 0 B |
| VGPR/SGPR spill | 0 / 0 | 0 / 0 |
| static MFMA32 | 20 | 56 |
| static `s_barrier` | **9** | **21** |
| static global load/store | 136 / 16 | 336 / 32 |
| static LDS read/write | 20 / 88 | 50 / 288 |
| static `ds_bpermute` | 0 | 0 |

Z2 工件目录为
`codex_qwen_bt64_stage6z_fixed_rerun/machine/z2_T8192/`，Z3 工件目录为
`codex_qwen_bt64_stage6z_fixed_rerun/machine/z3_T8192/`。两个 HSACO hash
不同，说明这不是同一 code object 的重复测量。exact-LTO 的 pre-greedy、
post-greedy 和 virtregrewriter section 没有 `SI_SPILL_AV32/AV64`。
静态字段和动态 rocprof 字段分开记录，不能互相替代。

### Fresh-process isolated body timing

口径固定为 caller-owned output、current HIP stream、no Graph、每 session
warmup=10/repeat=50、每长度 5 个 fresh-process session、旋转三臂顺序。
native 是实际 selected `chunk_fwd_kernel_o` 的 preallocated direct body，
不是 public full graph。

| T | fixed Z2 ms | fixed Z3 ms | native selected ms | Z2/native | Z3/native | Z3/Z2 |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.061431 | 0.074491 | 0.036194 | 1.697x | 2.058x | 1.213x |
| 1024 | 0.063634 | 0.078456 | 0.037756 | 1.685x | 2.078x | 1.233x |
| 2048 | 0.077475 | 0.094661 | 0.042763 | 1.812x | 2.214x | 1.222x |
| 4096 | 0.110724 | 0.126388 | 0.056224 | 1.969x | 2.248x | 1.142x |
| 8192 | 0.177203 | 0.220769 | 0.091917 | 1.928x | 2.402x | 1.246x |
| 16384 | 0.317692 | 0.393445 | 0.141591 | 2.244x | 2.779x | 1.238x |

全六点线性拟合为：

```text
fixed Z2 = 0.046632 ms + 1.048349 us/chunk * chunks
fixed Z3 = 0.054739 ms + 1.309077 us/chunk * chunks
native   = 0.030928 ms + 0.438248 us/chunk * chunks
```

因此 Z3 在 T4096、8192、16384 分别比 Z2 慢约 `14.15%`、`24.60%`、
`23.84%`，不是预注册的“稳定快于 Z2”。T16384 的 paired session median
差值为 `+74.95 us`（Z3-Z2）。

### Dynamic PMC and trace

下面是 rocprofv3 的实际 kernel trace/PMC，warmup=2、repeat=5；这些值不是
静态 ISA 推算。`Grid_Size` 是采集器记录的 global work-items。

| T | arm | trace us | MFMA | VMEM | LDS | VALU | SALU | occupancy | profiler VGPR | AccVGPR |
|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | Z2 | 55.322 | 81920 | 475136 | 475136 | 5836800 | 548864 | 15.738% | 88 | 32 |
| 2048 | Z3 | 73.229 | 81920 | 458752 | 462848 | 5555200 | 421376 | 8.510% | 68 | 164 |
| 8192 | Z2 | 158.797 | 327680 | 1900544 | 1900544 | 23347200 | 2195456 | 36.798% | 88 | 32 |
| 8192 | Z3 | 208.591 | 327680 | 1835008 | 1851392 | 22220800 | 1685504 | 18.134% | 68 | 164 |
| 16384 | Z2 | 299.926 | 655360 | 3801088 | 3801088 | 46694400 | 4390912 | 40.650% | 88 | 32 |
| 16384 | Z3 | 391.542 | 655360 | 3670016 | 3702784 | 44441600 | 3371008 | 20.403% | 68 | 164 |

Z3 的动态 MFMA 完全保持一致，VMEM/LDS/VALU/SALU 约有小幅下降；但它把
AccumVGPR 从 32 提高到 164，occupancy 大约降到 Z2 的一半，且 static
barrier 从 9 变成 21。减少的标量/内存指令没有抵消 wave-level 资源压力和
同步成本，这解释了 Z3 在长文本反而变慢。

### 晋级判定与当前有效排名

| gate | fixed Z2 | fixed Z3 |
|:--|:--:|:--:|
| correctness 全长度 | pass | pass |
| zero-V/NaN caller-owned | pass | pass |
| scratch/spill=0 | pass | pass |
| static barrier < 19 | pass, 9 | **fail, 21** |
| T4096/8192/16384 稳定快于 Z2 | N/A baseline | **fail，三点均更慢** |
| selector/X2 integration | 未执行 | 未执行 |

本轮结论：**修复后的 Z2 是当前 Avelang isolated chunk-o 的有效 baseline，
但仍不是 production 或 X2 晋级版本；Z3 correctness 通过、机器图有效，
但因 barrier 和性能双重失败正式 No-Go。** native selected body 仍明显更快，
但它是独立的 native vLLM 对照，不改变 Avelang selector。

所有新结果和原始 CSV/JSON 位于：
`codex_qwen_bt64_stage6z_fixed_rerun/`。前文 Z2 timing/PMC 必须继续标注为
修复前历史数据；今后 Stage 6Z 的性能排名只使用本节 fixed rerun。

---

## Stage 6Z Z7B：通用 block-dot V2 的 MFMA-B same-source A/B

完整报告：
`qwen_gfx942_stage6z_block_dot_v2_mfma_b_same_source_ab.md`。

### 实验状态

Z7B 从当前 Z5B direct-Q-cache-consumer 直接分叉。它只把 Z5B 中的 `Q@H.T`
和 `Q@K.T` MFMA-B 计算改为同一个高层
`al.amdgpu.block_dot_bf16_f32` logical block-dot contract；Q cache、Q producer
pass=1、phase-separated accumulator、g/V-new/output、BT64/BV64/BK32、WG256、
MFMA32、BF16 ABI 和 caller-owned output 全部冻结。

两条 arm 的 source function SHA256 相同：

```text
0a73d0ca01d97d4a49b9d7eed118d7bcaa775b1ac9ab5dbd7f80dadde605873c
```

generic 和 specialized 只通过
`AVELANG_BLOCK_DOT_LOWERING=generic|specialized` 分叉。当前 Docker binding 的
`get_mlir()` 在 initial MLIR 打印阶段 SIGSEGV（退出码 139），所以没有伪造
pre-branch MLIR hash；source identity、lowered LLVM、exact-LTO MIR、ISA 和 HSACO
均已保留，并在完整报告中标注证据等级。

### Correctness

generic/specialized 在 `T=64/512/1024/2048/4096/8192/16384` 均与 Z5B
BF16 byte-exact、finite；`T=64/8192/16384` 的 caller-owned output、zero-V-new、
NaN-prefill 检查也全部通过。没有修改容差或跳过错误。

### Compiler boundary evidence

lowered LLVM 已经出现真实分叉：

| LLVM 文本项 | generic | specialized |
|:--|--:|--:|
| `load bfloat` | 36 | 4 |
| `load <8 x bfloat>` | 4 | 8 |
| `extractelement <8 x bfloat>` | 48 | 80 |

但 exact-LTO 最终 machine graph 收敛：

| final machine 项 | generic | specialized |
|:--|--:|--:|
| HSACO SHA256 | `da2b8532146a1c74b1ff99a00596272543bfd68ce57ba4a7301c5eb922f028ac` | 同左 |
| code-object VGPR/AGPR/SGPR | 108/32/28 | 108/32/28 |
| LDS/private/spill | 32768 B / 0 B / 0 | 同左 |
| static MFMA32/barrier | 56/32 | 56/32 |
| static global load/store | 192/16 | 192/16 |
| static LDS read/write | 56/144 | 56/144 |
| `ds_bpermute` | 0 | 0 |

final ISA 文本的差异只有 objdump 文件路径头，指令主体相同。当前证据可以把
第一次收敛定位到 lowered LLVM 之后、最终 AMDGPU machine code/HSACO 之前；具体
LLVM/AMDGPU pass 尚未 bisect。

### T=2048 dynamic PMC（按 CTA）

| arm | MFMA | VMEM | LDS | VALU | SALU | profiler VGPR/AccumVGPR | occupancy |
|:--|--:|--:|--:|--:|--:|:--|--:|
| Z5B | 160 | 672 | 672 | 7072 | 768 | 76/100 | 14.778940% |
| Z7B generic | 160 | 672 | 688 | 7178 | 704 | 84/92 | 14.605566% |
| Z7B specialized | 160 | 672 | 688 | 7178 | 704 | 84/92 | 14.789358% |
| native WG256 diagnostic | 160 | 140 | 480 | 3376 | 660 | 100/36 | 9.357701% |

native 的 `LDS_Block_Size=0` 是 collector metadata limitation，不表示 native 没有
LDS；dynamic LDS=480/CTA 是有效数据。

### Formal body timing

口径为 current HIP stream、caller-owned preallocated output、no Graph、warmup=10、
repeat=50、7 个 fresh-process session、轮换顺序。native 为同形状 WG256 direct
body diagnostic，不是 public full API 排名。

| T | Z5B ms | Z7B generic ms | Z7B specialized ms | native ms | specialized/Z5B | Z5B/native |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 0.067360 | 0.069162 | 0.069283 | 0.042583 | 1.0285x | 1.5818x |
| 8192 | 0.158055 | 0.164845 | 0.165146 | 0.091196 | 1.0449x | 1.7331x |

endpoint slope：

```text
Z5B             0.944740 us/chunk
Z7B generic     0.996693 us/chunk
Z7B specialized 0.998568 us/chunk
native          0.506375 us/chunk
```

Z7B generic 和 specialized 在 T=2048、8192 都比 Z5B 慢，故没有按预注册条件
继续运行 T=16384 性能，也没有建立 selector、没有接入 X2 或 production。

### Stage 6Z 当前排名更新

```text
Z5B = 当前 isolated research baseline
Z6G-S/I = performance No-Go
Z7B-G/B = correctness PASS；LLVM A/B 分叉 PASS；final machine 收敛；No-Go
X2/production = 未改变
```

Z7B 证明了“通用 block-dot source 和 LLVM lowering 分叉可以建立”，但没有证明
“specialized B operand 已经实现了 native-style machine path”。下一步若继续，
应先做 LLVM/AMDGPU/LTO convergence bisect，定位 typed operand 第一次丢失的位置；
不要直接扩展 Q 或 V-new，也不要把 LLVM 文本中的 vector load 变化当成性能收益。

---

## Stage 6Z BDV2：通用 `block_dot_bf16_f32` Full-Scope Generalization

完整报告：
`qwen_gfx942_stage6z_block_dot_v2_full_scope_generalization.md`。

### 实验范围

BDV2 从 Z5B direct-Q-cache-consumer 分叉。它冻结了 BT64/BV64/BK32、WG256、
2 CTA/chunk-head、BF16 ABI、MFMA32、K32 reduction order、dedicated Q cache、
phase-separated accumulator、数学和 caller-owned output。只把 K/H 的 source
producer 与 MFMA-B consumer 改成同一个通用 full-scope logical block-dot contract。

新增的 `block_dot_bf16_f32_logical` 与
`block_dot_bf16_f32_logical_transposed` 都创建同一个既有
`AMDGPUBlockDotBF16F32Op`，没有增加 Qwen-specific intrinsic。K/H 都由
`emitFullScopeProducer` 和 `emitGenericOperandBPair` 管理；specialized 只在
late lowering 选择 BF16x8 typed producer，generic 保留 scalar fallback。

### Same-source 机器证据

两臂 source SHA256 相同：

```text
aee02e0309152ab34893836720c38266c6519c57499ef640db3e4ff48152047e
```

当前 binding 的 initial `get_mlir()` printer 会 SIGSEGV，所以 artifact 明确记录
`initial_mlir=skipped by command line`，没有伪造 pre-branch MLIR hash。source、
lowered LLVM、pre-LTO AMDGCN、exact-LTO MIR、ISA、HSACO 均已保存，分层 hash
显示 generic/specialized 真正不同：

| 层次 | generic | specialized |
|:--|:--|:--|
| lowered LLVM | `a8d17cb4...8b44de65` | `df3dbcce...8d5e8f2` |
| pre-LTO AMDGCN | `f7805897...174051c` | `ff7e5462...aa3407d6` |
| final ISA | `785ec664...3d8f523` | `a8deb1d3...c311938` |
| HSACO | `6b2b07fb...74cc2c35` | `d17483b9...c9091af5a` |

T=2048 exact HSACO metadata：

| arm | code VGPR | code AGPR | SGPR | LDS | private/spill |
|:--|--:|--:|--:|--:|:--|
| BDV2-G | 136 | 48 | 30 | 32768 B | 0 / 0 |
| BDV2-S | 132 | 48 | 30 | 32768 B | 0 / 0 |

两臂 static lexical MFMA32/barrier/global load/store/LDS read/write 分别都是
`56/44/140/16/56/92`。静态计数相同不代表 machine graph 相同，因为 HSACO
和 final ISA hash 已不同；两臂 exact-LTO MIR 也都没有 AV32/AV64 spill save。

### Correctness

generic/specialized 在 `T=64/512/1024/2048/4096/8192/16384` 全部相对 Z5B
BF16 byte-exact、finite。T=64/8192/16384 的 caller-owned output、zero-V-new、
NaN-prefill 也全部通过。

早期实现误用了 `tid < 128`，只覆盖 wave 0，T=64 产生
`max_abs=0.004642...`。审计后改为 `value_half = (thread_id // 64) % 2`，K
producer/consumer 只在 value-half 0，也就是 wave 0 和 wave 2 工作；修复后全矩阵
通过。这是 ownership 证据，不是放宽容差。

### T=2048 动态 PMC（per CTA）

rocprof 原始 `Grid_Size=131072`、WG256 对应 512 CTA；下表由真实 counter 除以
512 归一化，不能由 static ISA 推导：

| arm | MFMA | VMEM | LDS | VALU | SALU | profiler VGPR/AccumVGPR | occupancy |
|:--|--:|--:|--:|--:|--:|:--|--:|
| Z5B | 160 | 672 | 672 | 7072 | 768 | 76/100 | 14.498718% |
| BDV2 generic | 160 | 448 | 464 | 8410 | 780 | 88/88 | 14.634712% |
| BDV2 specialized | 160 | 448 | 464 | 8410 | 780 | 88/88 | 14.448873% |

BDV2 的 VMEM/LDS 确实下降，但 VALU 增加约 19%，因此 typed producer 的减少没
有变成 latency 收益。generic 与 specialized 的动态 PMC 完全相同，说明当前
specialized 仍未形成 native-style 低地址/低 VALU feeding。

### fresh-process body benchmark

口径为 current HIP stream、caller-owned output、no Graph、warmup=10、repeat=50、
7 fresh-process sessions、rotating order。native 是 same-shape selected chunk-o
diagnostic，不是 public Eager full-graph 排名。

| T | Z5B | BDV2-G | BDV2-S | native | G/Z5B | S/Z5B |
|--:|--:|--:|--:|--:|--:|--:|
| 2048 | 0.067641 ms | 0.069043 ms | 0.068962 ms | 0.042744 ms | 1.0207x | 1.0195x |
| 8192 | 0.157294 ms | 0.166847 ms | 0.167048 ms | 0.090875 ms | 1.0607x | 1.0620x |

endpoint slope：Z5B `0.933891 us/chunk`，BDV2-G `1.018802`，BDV2-S `1.021724`，
native `0.501364`。T=2048 和 T=8192 的 7 个 paired session 都没有正收益，故
BDV2-G/S 不晋级、不建 selector、不接 X2。

### Stage 6Z 排名

```text
Z5B = 当前 isolated research baseline
BDV2-G/S = correctness PASS；full-scope machine 分叉 PASS；performance No-Go
Z7B = 历史 logical block-dot A/B，最终 machine 收敛
X2/production = 未改变
```

本轮证明的是“通用 full-scope block-dot 表示和 late lowering 可以落地”，不是
“当前 specialized 已经等价于 Triton”。下一步若继续，只应先做 LLVM/AMDGPU/LTO
convergence 和 address/layout feeding 归因，不应添加新的 Qwen-specific intrinsic，
也不应同时修改 ownership、MFMA geometry、RA 或 full-v29。

---

## Stage 6Z BDV2-P1：通用 VALU/Layout Planner 后续审计

本节是对上面 BDV2 结论的后续补充。它不改写 BDV2 的历史数据，也不把 P1 的
LLVM 文本变化误记成 final machine 优化。

### P1 做了什么

P1 从同一份 BDV2 full-scope source 分叉，只增加通用
`LogicalBlockLayoutPlan`，由一个 planner 同时管理 K/H 的：

```text
tid/wave/lane ownership
-> packet row/column
-> k-stage/feature offset
-> global producer address
-> shared consumer address
-> MFMA B fragment
```

同时尝试让 MFMA consumer 直接使用 `<4xbf16>` LDS fragment，减少旧的
`<8xbf16> -> extractelement/insertelement -> <4xbf16>` 重建。P1 没有新建
Qwen/chunk-o 专用 op，没有改 WG256、MFMA32、K/H ownership、数学、ABI、Q cache
或 accumulator phase。

代码和设计说明：

```text
lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
  qwen_block_dot_bf16_f32_v2_design.md
```

P1 通过：

```text
AVELANG_BLOCK_DOT_LAYOUT_PLANNER=bdv2_p1_affine
```

### provenance 结果

T=2048 `lowered_llvm.ll` lexical count：

| pattern | Z5B | BDV2-S | P1-S |
|:--|--:|--:|--:|
| `udiv/urem` | `0/0` | `9/8` | `9/8` |
| `extractelement` | `96` | `130` | `66` |
| `insertelement` | `96` | `160` | `96` |
| `add` | `105` | `104` | `116` |

P1 确实删除了 LLVM 层 64 个 extract 和 64 个 insert，但 planner 仍让 affine
index 以动态 SSA 形式存在。P1 pre-LTO 也没有出现明确的 arithmetic reduction，
而是出现更多 `v_add/v_lshl_add/v_lshr/v_and/v_mov` lexical pattern。

最关键的 convergence 证据：

- BDV2-S/P1-S source SHA256 相同：
  `aee02e0309152ab34893836720c38266c6519c57499ef640db3e4ff48152047e`；
- lowered LLVM 不同；
- pre-LTO AMDGCN 不同；
- exact-LTO replay return code 均为 0；
- final HSACO SHA256 完全相同：
  `d17483b933a7abf68b34b0e193aa4e3e5d9f93f48a12cfa8d84b285c9091af5a`；
- 去掉 disassembler 绝对路径 header 后，final ISA body SHA256 完全相同：
  `7a02fc2af0542b4a2177681e1cd51663f98d9f0357ef664c3cc067988dafaf70`。

因此 P1 的变化在 LLVM/pre-LTO 表示层被 AMDGPU/LTO 收敛，不能声称它改变了
最终 machine graph。

### P1 资源与 PMC

P1-S 与 BDV2-S 的 code object 和真实 T=2048 PMC 完全相同：

| arm | MFMA/CTA | VMEM/CTA | LDS/CTA | VALU/CTA | SALU/CTA | VGPR | AGPR | LDS bytes | spill |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Z5B | 160 | 672 | 672 | 7072 | 768 | 76 | 32 | 32768 | 0 |
| BDV2-S | 160 | 448 | 464 | 8410 | 780 | 88 | 48 | 32768 | 0 |
| P1-S | 160 | 448 | 464 | 8410 | 780 | 88 | 48 | 32768 | 0 |
| native WG256 diagnostic | 160 | 140 | 480 | 3376 | 660 | native artifact | native artifact | native artifact | native artifact |

这里的 `VGPR/AGPR` 是 code-object metadata；P1 的 profiler row 同时为
`VGPR=88, AccumVGPR=88`。它们不与 static ISA lexical count 混用。

### P1 正确性与性能

P1-G/P1-S 在 `T=64/512/1024/2048/4096/8192/16384` 均相对 Z5B BF16
byte-exact、finite；caller-owned output、zero-V-new 和 NaN-prefill edge cases
也通过。旧 full-scope 与 direct-K64 block-dot regression 共 `7 passed`。

fresh-process caller-owned isolated body，current stream、no Graph、warmup=10、
repeat=50、7 sessions：

| T | Z5B ms | BDV2-S ms | P1-S ms | native ms | P1-S/Z5B | P1-S/native |
|--:|--:|--:|--:|--:|--:|--:|
| 2048 | `0.066899501` | `0.069463000` | `0.069863502` | `0.042483000` | `1.044305x` | `1.644505x` |
| 8192 | `0.157854497` | `0.167768501` | `0.167308502` | `0.090734500` | `1.059891x` | `1.843935x` |

端点 slope：

| arm | us/chunk |
|:--|--:|
| Z5B | `0.947448` |
| BDV2-S | `1.024016` |
| P1-S | `1.015052` |
| native | `0.502620` |

P1 在 T=2048 相对 Z5B 慢 `2.964 us`，T=8192 慢 `9.454 us`；没有跨长度正收益，
因此 P1 不晋级、不建 selector、不接 X2。T=16384 body 不运行，因为预注册
条件要求 T=2048 或 T=8192 先出现稳定正收益；T=16384 correctness 已通过。

### Stage 6Z 更新后的判断

P1 给 compiler 团队的窄结论是：

```text
同一通用 block_dot source 可以在 LLVM 层表达更紧凑的 typed fragment，
但当前 AMDGPU/LTO lowering 没有把这个 intent 保留到最终 ISA。
```

这支持继续研究通用 representation-preserving lowering，但不能把整个
Z5B/native gap 归结为单一 compiler pass，也不能把 P1 当成性能胜者。当前
Stage6Z 性能 baseline 仍然是 Z5B；BDV2/P1 保留为通用 block-dot compiler
infrastructure 和回归证据。

## BDV2-P2：First-Class MFMA Operand Preservation（当前结果）

P2 在同一份 `qwen_gdn_bt64_native_chunko_stage6z_block_dot_v2_full_scope.py`
上增加 compiler-only selector：

```text
AVELANG_BLOCK_DOT_OPERAND_PRESERVATION=p2_first_class
```

它没有新增 Qwen/chunk-o public op，也没有修改 BT64/BV64/BK32、WG256、MFMA32、
Q/K/H/V-new/g/output 数学、Q cache 或 accumulator phase。K/H 仍通过同一个
full-scope `block_dot_bf16_f32` planner；P2 只是把 A/B row、operand word、K32
顺序和 `gfx942_shared_b32_mfma32` identity 暂时保存为内部
`amdgpu_block_dot_mfma_operand`，在 GPU outlining 后用专门 late pass 物化。

### 机器边界证据

P2 的 `post_gpu_outlining.mlir` 仍包含：

```text
ave.gpu.amdgpu_block_dot_mfma_operand
  {operand_role, source_role, logical_shape,
   physical_encoding, fragment_mapping, target_mfma}
```

`post_block_dot_operand_materialization.mlir` 随后包含：

```text
llvm.load volatile i64, addrspace(3), align 8
llvm.bitcast i64 -> vector<4xbf16>
```

内部 op 在 late pass 后不残留，并继续调用既有 MFMA32 intrinsic。P2 的当前
machine artifact 为：

```text
codex_qwen_bt64_stage6z_bdv2_p2_machine_specialized/
```

它的 source SHA256 仍为
`aee02e0309152ab34893836720c38266c6519c57499ef640db3e4ff48152047e`，而 P2
HSACO SHA256 为
`f612f23968d1e9b9e205b1daf75a70cef2d9016d56f44df24fef6d3ae20c7947`，不同于
P1/BDV2-S 的 `d17483b933a7abf68b34b0e193aa4e3e5d9f93f48a12cfa8d84b285c9091af5a`。
这证明 P2 没有像第一版那样在最终 code object 收敛回 P1。

### 结果摘要

| arm | T2048 MFMA/CTA | VMEM/CTA | LDS/CTA | VALU/CTA | SALU/CTA | code VGPR/AGPR | LDS | scratch |
|:--|--:|--:|--:|--:|--:|:--|--:|--:|
| Z5B | 160 | 672 | 672 | 7072 | 768 | 76/32 | 32768 B | 0 |
| BDV2/P1-S | 160 | 448 | 464 | 8410 | 780 | 132/48 | 32768 B | 0 |
| P2-S | 160 | 448 | 592 | 8474 | 780 | 132/48 | 32768 B | 0 |

这些是 T=2048 fresh rocprof 结果按 512 CTA 归一化的 dynamic counters；不是
从 static ISA 推导。P2 的 static ISA 为 `MFMA32=56`、`global_load=140`、
`global_store=16`、`ds_write=92`、`ds_read=104`、`s_barrier=44`。P1 为
`ds_read=56`，并且主要是 `ds_read_b128=56`；P2 变为 `ds_read_b64=96` 加
`ds_read_b128=8`。因此 P2 确实保留了 first-class operand 到机器层，但当前
packed i64 物化没有降低机器工作，反而增加了 LDS 读取。

### fresh-process Eager body

口径仍为 caller-owned preallocated output、current HIP stream、no Graph、
warmup=10、repeat=50、7 fresh-process sessions、rotating order。这里是
isolated body diagnostic，不是 full public API 排名：

| T | Z5B ms | P1-S ms | P2-S ms | native ms | P2/Z5B | P2/native |
|--:|--:|--:|--:|--:|--:|--:|
| 2048 | 0.066862214 | 0.070825143 | 0.073120214 | 0.042594643 | 1.0936x | 1.717x |
| 8192 | 0.157708641 | 0.168587500 | 0.176261858 | 0.090960786 | 1.1177x | 1.938x |

P2 相对 Z5B 的 paired HIP-event median 差值为：T=2048 约 `+6.10 us`，
T=8192 约 `+18.55 us`。7 个 session 的方向一致，没有稳定正收益。由于
T=2048 和 T=8192 都没有正收益，本轮不运行条件性的 T=16384 性能测试；
T=16384 correctness 已通过。

### Stage6Z 决策

P2 选择 **Case B：保留为通用 compiler infrastructure，性能不晋级**：

- correctness：通过 T=64/512/1024/2048/4096/8192/16384、finite、
  caller-owned、zero-V-new、NaN-prefill；
- representation：内部 operand plan 从 post-GPU-outlining 保留到专用 late
  materialization，并且 P2 HSACO 与 P1 不同；
- performance：P2 比 Z5B 和 P1-S 都慢，不能成为 isolated baseline；
- baseline：Z5B 仍是当前 Stage6Z isolated performance baseline；
- scope：不接 X2、不接 production、不建 selector，不改 allocator/RA。

详细实验、命令、hash、correctness 原始结果和 benchmark JSON 见：
`qwen_gfx942_stage6z_block_dot_v2_p2_first_class_mfma_operand.md`。

## Stage 6Z BDV2-P3：Packed Operand Reuse / Wide LDS Consumer

### P3 结论

P3 是 P2 的唯一 packed-consumer 后续。它没有新增 Qwen/chunk-o public op，
没有改变 production/X2、WG256、BT64/BV64/BK32、MFMA32 数学、K32 order、
Q cache、K/H planner、accumulator phase 或 BF16 ABI。P3 只在同一份
`block_dot_bf16_f32` source 的 late lowering 中，把两个相邻的
`<4xbf16>` operand fragment 组成一个 aligned `vector<8xbf16>` LDS read，
再切成 low/high 两个 fragment 供两个原有 MFMA consumer 使用。

完整记录见：

```text
qwen_gfx942_stage6z_block_dot_v2_p3_packed_operand_reuse.md
```

### 机器证据

P3 的 post-materialization MLIR 含有：

```text
llvm.load addrspace(3), align 16 -> vector<8xbf16>
vector.extract_strided_slice [0] / [4]
two MFMA32 calls with consumer_group=packed_b128_pair
```

P3 的 machine artifact 为：

```text
codex_qwen_bt64_stage6z_bdv2_p3_machine_specialized/
```

P3 HSACO SHA256：

```text
d17483b933a7abf68b34b0e193aa4e3e5d9f93f48a12cfa8d84b285c9091af5a
```

P2 HSACO SHA256 为：

```text
f612f23968d1e9b9e205b1daf75a70cef2d9016d56f44df24fef6d3ae20c7947
```

因此 P3 与 P2 的 final code object 不同；P3 与 P1/BDV2-S 收敛到同一
packed LDS machine shape。当前 binding 的 initial `get_mlir()` 路径会
SIGSEGV，本轮 dump 使用 `--skip-initial-mlir`，不能把 source hash 写成
pre-branch MLIR hash。

### Static ISA 与动态 PMC

Static lexical count：

| arm | MFMA32 | global load | global store | ds_read | ds_write | s_barrier |
|:--|--:|--:|--:|--:|--:|--:|
| P2-S | 56 | 140 | 16 | 104 | 92 | 44 |
| P3-S | 56 | 140 | 16 | 56 | 92 | 44 |
| P1/BDV2-S | 56 | 140 | 16 | 56 | 92 | 44 |

T=2048 dynamic rocprof，按 CTA 归一化：

| arm | MFMA/CTA | VMEM/CTA | LDS/CTA | VALU/CTA | SALU/CTA | profiler VGPR | profiler AccVGPR | code LDS |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Z5B | 160 | 672 | 672 | 7072 | 768 | 76 | 100 | 32768 B |
| P1-S | 160 | 448 | 464 | 8410 | 780 | 88 | 88 | 32768 B |
| P2-S | 160 | 448 | 592 | 8474 | 780 | 88 | 88 | 32768 B |
| P3-S | 160 | 448 | 464 | 8410 | 780 | 88 | 88 | 32768 B |
| native WG256 | 160 | 140 | 480 | 3376 | 660 | diagnostic | diagnostic | diagnostic |

P3 让 P2 的额外 LDS 读取消失，但没有减少 P1 的 `8410 VALU/CTA`。因此
P3 的 machine representation 改善没有转化为 latency 改善。

### Correctness 与回归

P3 相对 Z5B 在 T=64/512/1024/2048/4096/8192/16384 全部 BF16 byte-exact、
finite；T=64/8192/16384 caller-owned output、zero-V-new、NaN-prefill 也全部
通过。旧 direct-K64/block-dot/Stage6S 回归为：

```text
23 passed in 89.49s
```

### Fresh-process body benchmark

口径为 caller-owned isolated body、current stream、no Graph、warmup=10、
repeat=50、7 fresh-process sessions、rotating order。下面使用 session medians
的 median，单位 ms：

| T | Z5B | P1-S | P2-S | P3-S | native | P3/Z5B |
|--:|--:|--:|--:|--:|--:|--:|
| 2048 | `0.067460500` | `0.070625000` | `0.072828002` | `0.070084002` | `0.042723501` | `1.0389x` |
| 8192 | `0.157334000` | `0.168270000` | `0.176522501` | `0.167669497` | `0.091196001` | `1.0657x` |

P3-Z5B 的七个 paired 差值在两个长度都全部为正。P3 的两点 slope 为
`1.016516 us/chunk`，Z5B 为 `0.936182 us/chunk`，native 为
`0.504922 us/chunk`。P3 因而没有通过性能晋级条件；条件性的 T=16384
body 不运行，T=16384 correctness 已通过。

### Stage 6Z 状态更新

正式状态：

```text
Z5B = isolated performance baseline
P3 = compiler representation PASS / performance No-Go
```

P3 保留为通用 block-dot infrastructure 和 regression evidence，不建立
selector，不接 X2，不接 production。下一条性能方向只能针对 P1/BDV2 共有的
affine address/layout/fragment feeding VALU，而不是继续做 P2/P3 load-width
枚举。

## BDV2-P4 Final-Machine VALU Provenance Closure

P4 是在 P3 之后唯一实施的 accumulator/fragment feeding 控制杆。它保持同一
份 full-scope `block_dot_bf16_f32` source、K/H common planner、BT64/BV64/BK32、
WG256、2 CTA/chunk-head、MFMA32、Q full cache、phase-separated accumulator、
BF16 ABI 和 caller-owned output。P4 通过内部
`AccumulatorForwardingMap` 在 accumulator reset boundary 内复用 SSA 结果，
并去掉后续不读取的 duplicated high tile；没有新建 Qwen/chunk-o op 或 source
fork。

本轮的机器结论：P3 相对 Z5B 的最强 provenance 是 MFMA result 到 accumulator
fragment 的 `v_accvgpr_read/write`、`COPY/REG_SEQUENCE` 和 select feeding，
不是 P3 多做的 global/LDS load。P4 的 post-materialization MLIR、LLVM、exact
LTO MIR、final ISA 和 HSACO 均与 P3 不同，code-object `VGPR/AGPR` 从 `132/48`
降为 `104/32`，但 final `v_accvgpr_read/write` 仍为 `160/224`，dynamic
`VALU` 仍为 `8410/CTA`。P4 只让 lexical `v_mov_b32` 减少 32 条，同时让
`v_cndmask_b32_e64` 增加 32 条，因此没有形成可观测的 dynamic VALU 删除。

T=2048 dynamic PMC（每 CTA）为：

| arm | MFMA | VMEM | LDS | VALU | SALU |
|:--|--:|--:|--:|--:|--:|
| Z5B | 160 | 672 | 672 | 7072 | 768 |
| P3 | 160 | 448 | 464 | 8410 | 780 |
| P4 | 160 | 448 | 464 | 8410 | 780 |
| native WG256 diagnostic | 160 | 140 | 480 | 3376 | 660 |

7-session fresh-process body benchmark也没有改善：

| T | Z5B ms | P3 ms | P4 ms | native ms |
|--:|--:|--:|--:|--:|
| 2048 | `0.066899501` | `0.069784001` | `0.070364498` | `0.042644000` |
| 8192 | `0.157253496` | `0.167788997` | `0.168190002` | `0.091215502` |

P4 的 paired difference 对 Z5B 在两个长度的 7 个 session 全部为正，故正式
状态保持：

```text
Z5B = isolated performance baseline
P3/P4 = generic block_dot compiler infrastructure, performance No-Go
```

完整 final-machine provenance、SHA、static/dynamic 区分和 paired raw samples
见 [`qwen_gfx942_stage6z_block_dot_v2_p4_final_machine_valu_provenance.md`](qwen_gfx942_stage6z_block_dot_v2_p4_final_machine_valu_provenance.md)
及 [`machine-readable valu_provenance_p3_vs_z5b_vs_native.json`](machine-readable%20valu_provenance_p3_vs_z5b_vs_native.json)。
本轮不接 X2，不启动 Q/V 扩展，也不继续衍生 P4.1/P4.2。

## C22 Z5B Schedule-Preserving Physical Lowering Closure

C22 将 C13-C16 的 static physical Q/H/K/V bridge 接入 Z5B，同时冻结 Q cache、
A -> B0 -> B1 -> C phase order、K32 order与 phase-separated accumulator。它没有
启用 C19/C21 的 full-region ownership、superloop 或 pipeline。

结果为 `STOP_C22_SCHEDULE_LAYOUT_INCOMPATIBLE`。在 T=64、zero-V-new、
caller-owned NaN-prefilled output 的真实 GPU gate 中，C22 finite 但不与 Z5B BF16
byte-exact：random Q/H/K 最大绝对误差 `0.0054473876953125`，structured Q/H/K 为
`52.0`。zero-V 将首错定位到 Q/H/K bridge；C13 layout 8/8 与 C14 codegen 1/1
仍通过。

这是 C16 selected-native fragment ownership 与 Z5B wave/value-half reuse 不可直接
组合的证据，而不是 physical-layout algebra 失败。按 correctness stop rule，C22
没有运行 PMC、T2048 或长文本 body/slope，也没有接入 X2；Z5B 保持唯一 isolated
performance baseline。完整 closure 见
[C22 report](qwen_gfx942_c22_z5b_schedule_preserving_physical_lowering.md)。
