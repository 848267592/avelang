# Direct-K64 BV32 C0: Persistent Typed Full-Block Operand Lowering

## 1. 结论

C0 通过。它在完全冻结的 direct-K64 BV32 update-suffix 实验中只改变
`block_dot_bf16_f32` 的 operand materialization granularity：从
`typed_immediate32` 的立即 32x32 tile stage/consume，改为
`persistent_typed_block` 的 CTA-local 全 V32xT64 和当前 K64xT64 block
stage/consume。高层 Qwen source、ABI、geometry、CTA ownership、数学和每
CTA MFMA 工作量都没有改变。

在 T=2048 的 fresh-process body benchmark 中，persistent 版本为
`0.216922 ms`，当前 typed-immediate32 为 `0.261739 ms`，降低 `17.12%`
（`1.2066x`）。T=2048 的 rocprof trace 从 `228.660 us` 降至
`183.273 us`（`-19.85%`），MFMA 动态数仍为 `32768`，scratch 和 MIR
spill 均为零。

这不是 full-v29 nonzero-W correctness 或 production promotion；它只是
current-vLLM BF16 ABI 下 direct-K64 recurrence-update suffix 的
experimental result。C0 已足以把 persistent typed block 保留为后续 C1
ping-pong prefetch 的唯一候选，不过本轮没有实现 C1。

## 2. 冻结的实验边界

两臂共享同一份 high-level source：

`test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_block_dot_bv32_coop.py`

| 项目 | 固定值 |
|:--|:--|
| 输入 K / V-new | BF16 |
| g / persistent state | FP32 |
| H 输出 | BF16 |
| final state 输出 | FP32 |
| token block / value block | BT64 / BV32 |
| CTA / workgroup | 32 CTA，WG128，two-wave cooperative |
| dot geometry | `v_mfma_f32_32x32x8_bf16` |
| K 顺序 | direct K[0:64] 后 direct K[64:128]，每 half 内 K32 accumulation order 不变 |
| persistent H1/H2 layout、global IO、launch | 不变 |
| 每 CTA MFMA | 1024 |
| 整网格 T=2048 MFMA | 32768 |

明确未修改：production dispatch、allocator/RA、旧 broad-K、旧 compact-K、
full-v29 pred/reference correctness、BV、CTA ownership、MFMA geometry 或
pipeline overlap。

## 3. 同源 A/B 与 lowering 分叉

运行时仅设置 operand-lowering 环境变量：

| arm | `AVELANG_BLOCK_DOT_OPERAND_LOWERING` | 语义 |
|:--|:--|:--|
| current typed-immediate32 | `typed_vector` | typed BF16x8 global load 后，按立即 32x32 operand tile stage、barrier、MFMA、再复用 LDS scratch |
| persistent-typed-block | `persistent_typed_block` | V32xT64 只在 `k_half==0` stage 一次；当前 K64xT64 每个 K-half stage 一次；随后四个 32x32x8 consumer 按原 K32 顺序读取 |

这两条路径在 `lower_qwen_block_dot_pass.cc` 中才分叉。两者的
`pre_kfrag_branch.mlir` SHA256 完全相同：

```text
8675f9e514cd6f07b981c5f978212fdc90965d7bb9b59fba5bf30d682be331f2
```

因此 A/B 没有通过修改 Qwen 高层代码、输入 shape 或 launch 来取得结果。
`post_block_dot_lowering.mlir` 才不同：persistent arm 出现 CTA-local
workgroup allocation：

```text
memref<1x32x64xbf16, #gpu.address_space<workgroup>>  # V block, 4 KiB
memref<64x64xbf16, #gpu.address_space<workgroup>>    # current K half, 8 KiB
```

V stage 被 `k_half == 0` guard 包住。K stage 在每个 K-half 执行一次。随后
一个 barrier 保护完整 block，四个 MFMA consumer 读取已存在的 operand，最后
一个 barrier 结束 buffer lifetime。这个差异继续保留至 LLVM、LTO MIR 和 HSACO
ISA；不是 frontend 里出现又在后端前消失的空分叉。

### 关键实现细节

global V 的 contiguous BF16x8 向量顺序与 MFMA consumer 所需的 token-major
LDS 访问顺序不同。因此 persistent path 保留 typed BF16x8 global load，但在
转置写入 LDS 时仍会使用标量 BF16 LDS store。C0 没有宣称已经实现完整的
vectorized LDS transpose/local-load；测到的收益来自扩大 operand lifetime、
合并 stage/consume phase，以及消除多余的重建和同步。

## 4. 正确性

参考是 direct-K64 FP32 update reference，不是尚未解决的 full-v29 nonzero-W
reference。T=64/512/2048 的两个 arm 均 finite，并满足冻结阈值：

| arm | T | h max abs | final-state max abs | 结果 |
|:--|--:|--:|--:|:--|
| typed-immediate32 | 64 / 512 / 2048 | <= `0.5` | <= `4.5776e-05` | pass |
| persistent-typed-block | 64 | `0` | `5.7220e-06` | pass |
| persistent-typed-block | 512 | `0.25` | `1.5259e-05` | pass |
| persistent-typed-block | 2048 | `0.5` | `4.5776e-05` | pass |

更强的 arm-to-arm T=2048 output digest 检查相同：

```text
42b98bf78a1b9f6e1ee4430b302a9657436fa5db1c87872d081e27c4da90edab
```

该 digest 包含 H 和 final state 的输出字节。故 persistent arm 不仅满足
reference tolerance，也与 typed-immediate32 bit-exact。

## 5. Fresh-Process Body 性能

所有数字使用同一 fresh-process harness，warmup=5、repeat=20。每个长度收集
两个三-session block，共六个 session median；第二个 block 更换 seed 并轮换 arm
的起始顺序。下面的数值是六个 session median 的中位数，单位为 ms。

| T | chunks | typed-immediate32 | persistent-typed-block | persistent gain | current Triton W=0 control | persistent / Triton |
|--:|--:|--:|--:|--:|--:|--:|
| 512 | 8 | `0.086689` | `0.078115` | `1.1098x` / `-9.89%` | `0.037556` | `2.080x` |
| 1024 | 16 | `0.133108` | `0.116232` | `1.1452x` / `-12.68%` | `0.068221` | `1.704x` |
| 2048 | 32 | `0.261739` | `0.216922` | `1.2066x` / `-17.12%` | `0.119778` | `1.811x` |

端点 T=512 到 T=2048 的线性斜率：

| arm | slope |
|:--|--:|
| typed-immediate32 | `7.294 us/chunk` |
| persistent-typed-block | `5.784 us/chunk` |
| current Triton W=0 control | `3.426 us/chunk` |

persistent 使 Avelang 端点 slope 降低 `1.510 us/chunk`（`-20.7%`），但仍比
当前 Triton control 多约 `2.358 us/chunk`。此前约 `2.16x` 的差距来自较早的
single-run/不同样本；本轮六 session 同口径下 immediate 为 `2.185x`、persistent
为 `1.811x`，故报告以这一轮重测值为准。

## 6. T=2048 rocprof 资源与动态工作

每臂使用 warmup=2、repeat=5，解析七个匹配 dispatch。trace 为其 dispatch
median；计数是同一 counter collection 中的动态总数。

| metric | typed-immediate32 | persistent-typed-block | persistent change |
|:--|--:|--:|--:|
| trace median | `228.660 us` | `183.273 us` | `-19.85%` |
| VGPR (rocprof) | 4 | 12 | +8 |
| AccVGPR | 164 | 196 | +32 |
| SGPR | 48 | 112 | +64 |
| LDS block | 4096 B | 12288 B | +8192 B |
| scratch | 0 B | 0 B | 0 |
| occupancy percent | `0.624036` | `0.618571` | `-0.005465` |
| SQ_INSTS_MFMA | 32768 | 32768 | 0 |
| SQ_INSTS_VALU | 1636032 | 1355840 | `-17.13%` |
| SQ_INSTS_SALU | 214272 | 189760 | `-11.44%` |
| SQ_INSTS_VMEM | 102912 | 94720 | `-7.96%` |
| SQ_INSTS_LDS | 229376 | 188416 | `-17.86%` |

由 HSACO metadata 读取的资源也确认没有 private segment 或 spills：

| code-object field | typed-immediate32 | persistent-typed-block |
|:--|--:|--:|
| private segment | 0 B | 0 B |
| VGPR spill count | 0 | 0 |
| SGPR spill count | 0 | 0 |
| group LDS | 4096 B | 12288 B |
| AGPR count | 32 | 64 |
| VGPR count | 152 | 184 |

因此这次是用更多 LDS 和 registers 换取更少的动态 operand/staging work；它没有
跨过 scratch、spill 或明显 occupancy cliff。rocprof 的 `VGPR` resource 列和
code-object VGPR count 来自不同 collector/metadata 字段，报告分别保留。

## 7. LLVM、MIR 与 ISA 证据

| static instruction family | typed-immediate32 | persistent-typed-block |
|:--|--:|--:|
| `v_mfma_f32_32x32x8_bf16` | 32 | 32 |
| 16x16 MFMA | 0 | 0 |
| `s_barrier` | 17 | 5 |
| `ds_read_b128` | 32 | 24 |
| `ds_write_b16` | 96 | 80 |
| `global_load_dwordx4` | 20 | 18 |
| all global/buffer loads | 30 | 26 |
| buffer stores | 102 | 102 |

post-lowering MLIR 的 `gpu.barrier` 数也从 9 到 3。ISA 中仍是
`v_mfma_f32_32x32x8_bf16`，且没有 16x16 MFMA；C0 没有偷换 MFMA geometry。

两臂均由同一 LTO replay 工具生成 20 个 machine sections。所有 section 搜索
`SI_SPILL_AV32_SAVE` 和 `SI_SPILL_AV64_SAVE` 均为零。representative
pre-greedy MIR line count 从 2392 降到 2136，prolog/epilog 后从 2311 降到
2067。这与 barrier、LDS、VALU 和 SALU 的减少方向一致，但 line count 本身不是
性能计数，不能单独作为性能因果结论。

## 8. 对问题的直接回答

1. **persistent full block 是否继续显著降低 VMEM 和 SALU？** 是。VMEM 降
   `7.96%`，SALU 降 `11.44%`，VALU 降 `17.13%`，trace 降 `19.85%`。VMEM
   降幅小于其他项，因为 immediate arm 已有 BF16x8 typed global load；C0 主要
   消除相邻 consumer 间的 staging/reconstruction phase，并非删除全部 input traffic。
2. **LDS 指令和 barrier 能否减少？** 是。动态 LDS 指令降 `17.86%`；静态
   barrier 从 17 到 5，MLIR barrier 从 9 到 3。代价是 LDS 多 8 KiB。
3. **typed global load 是否真正跨 MFMA consumer 复用？** 是。V32xT64 仅 first
   K-half stage 一次，供后续两个 K-half 的四个 32x32 consumer 使用；每个
   K64xT64 block stage 一次，供本 half 的四个 consumer 使用。global vector order
   与 LDS consumer order 不同，因此当前仍有 scalar LDS transpose store。
4. **相对 current Triton W=0 control 差距缩小多少？** T=2048 fresh-process
   ratio 从 `2.185x` 缩至 `1.811x`；绝对 gap 从 `141.961 us` 缩至 `97.144 us`。
   这是 W=0 recurrence control，不能等同 full public forward。
5. **C1 是否值得做？** 值得，但必须是单独实验：冻结 C0 的
   BV32/32CTA/WG128/MFMA32、typed block、state layout 和 K32 order，仅增加
   ping-pong K-half operand buffer。必须重测 barrier dependency、LDS footprint、
   scratch/spill 与 same-source correctness；不能同时改 BV、ownership、geometry 或 RA。

## 9. 剩余差距和限制

C0 证明 Avelang lowering 仍有可恢复的 operand-materialization overhead；但它不
证明所有剩余 `1.811x` 都来自 block-dot lowering。剩余差距仍可能含有 native fused
pipeline 的 producer/consumer scheduling、local operand transpose、decay/address
formation、persistent-state scheduling 或 control 的工作范围差异。native Triton
control 的工作流不能被此 isolated control 完整等同为 Avelang source 的 backend-only
差异。

full-v29 nonzero-W correctness 仍独立未解决，也不因 C0 而改变。

## 10. 文件与复现

实现和测试：

- `lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc`
- `test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_block_dot_bv32_coop.py`
- `test/examples/linear_attention/vllm_compare/test_qwen_gdn_direct_k64_block_dot_bv32_persistent_operand_c0.py`
- `test/examples/linear_attention/vllm_compare/bench_qwen_gdn_direct_k64_block_dot_bv32_persistent_operand_c0.py`
- `test/examples/linear_attention/vllm_compare/profile_qwen_gdn_direct_k64_block_dot_bv32_persistent_operand_c0.py`

所有 IR、LTO link、MIR、ISA 和 rocprof CSV 位于：

`test/examples/linear_attention/rocprof_outputs/qwen_block_dot_bv32_persistent_operand_c0/`

核心命令：

```bash
cmake --build /tmp/avelang-build-kfrag-qwen-rocm722 --target _avelang_bindings -j 16

PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:. \
  PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_direct_k64_block_dot_bv32_persistent_operand_c0.py -s

PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:. \
  PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_direct_k64_block_dot_bv32_persistent_operand_c0.py \
  --T 512 1024 2048 --warmup 5 --repeat 20 --sessions 3 --seed 20260818 --order-offset 0

PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:. \
  PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_direct_k64_block_dot_bv32_persistent_operand_c0.py \
  --T 512 1024 2048 --warmup 5 --repeat 20 --sessions 3 --seed 20260918 --order-offset 1
```
