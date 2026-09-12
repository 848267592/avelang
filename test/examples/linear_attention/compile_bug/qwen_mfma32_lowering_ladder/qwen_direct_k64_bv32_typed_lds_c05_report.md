# Direct-K64 BV32 C0.5: Typed/Swizzled LDS Operand-Layout Lowering

## 1. 结论

C0.5 完成了严格的同源 A/B。它只替换
`al.amdgpu.block_dot_bf16_f32` 的 **LDS operand-layout lowering**，将 C0 的
`persistent_typed_block` 标量转置写入，与 `persistent_typed_lds_layout`
的 packed/token-major 写入比较。Qwen 高层 source、BF16 ABI、BT64、BV32、
WG128、32 CTA、two-wave cooperative ownership、persistent state、K32
accumulation order、barrier phase、MFMA32 geometry、每 CTA 1024 MFMA 和输出
数学均保持不变。

packed arm 确实生成了 `ds_write_b128`，消除了静态 `ds_write_b16`。不过它
**没有**生成与之对称的 packed consumer local-load：在冻结的 MFMA32 operand
lane mapping 下，consumer 必须跨 token 聚集 eight BF16 elements，因此 ISA 变成
大量 `ds_read_u16`。这把 producer 的 scalar scatter 成本转移为 consumer 的 scalar
gather，不能宣称已经得到完整的 typed/vector LDS load path。

尽管如此，fresh-process body benchmark 的 packed arm 在三个长度均小幅更快：

| T | C0 scalar ms | C0.5 packed ms | packed gain |
|--:|--:|--:|--:|
| 512 | `0.078346` | `0.077616` | `1.0094x` / `-0.93%` |
| 1024 | `0.116623` | `0.114620` | `1.0175x` / `-1.72%` |
| 2048 | `0.217103` | `0.210513` | `1.0313x` / `-3.04%` |

T=2048 rocprof trace 也从 `183.072 us` 降至 `174.900 us`（`-4.46%`）。但 packed
arm 的动态 LDS 指令增加 `15.22%`，VALU 增加 `9.36%`，VGPR resource 从 `12`
升至 `68`。因此 C0.5 是一个 **正确、无 spill 的小幅 net improvement**，不是
“单 buffer LDS 已经足够紧凑”的确认。按预注册 stop rule，本轮 **不实施 C1
ping-pong/double buffering**；先保留 C0 scalar 作为结构更干净的 C0 baseline，
并将 packed C0.5 记录为说明 transposed LDS layout 需要 consumer-side fragment
layout 支持的反例/控制实验。

这不是 production promotion，也不涉及 full-v29 nonzero-W pred correctness。

## 2. 冻结边界

实验 kernel source 未改：

`test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_block_dot_bv32_coop.py`

| 属性 | 冻结值 |
|:--|:--|
| K / V-new | BF16 |
| g / persistent state | FP32 |
| H / final state | BF16 / FP32 |
| token/value block | BT64 / BV32 |
| grid / workgroup | 32 CTA / WG128，two-wave cooperative |
| dot operation | direct K[0:64]，再 direct K[64:128] |
| MFMA | `v_mfma_f32_32x32x8_bf16` |
| K accumulation | 原 K32 order 不变 |
| persistent blocks | V32xT64 一次；每 K-half K64xT64 一次 |
| barriers | C0 的完整-block stage/consume phase 不变 |
| dynamic MFMA | T=2048 全 grid `32768`；每 CTA `1024` |

明确未动：production dispatch、allocator/RA、旧 broad-K/compact-K、MFMA
geometry、BV32 ownership、full-v29 pred correctness、prefetch、double buffer 和
新的 overlap pipeline。

## 3. 两条 lowering arm

同一个 high-level `block_dot_bf16_f32` 在
`lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc` 才按
`AVELANG_BLOCK_DOT_OPERAND_LOWERING` 分叉。

| arm | option | shared layout 与读取方式 |
|:--|:--|:--|
| C0 scalar | `persistent_typed_block` | K `[64,64]` 与 V `[1,32,64]`，consumer 所需 row/token 顺序；BF16x8 global load 被拆成 eight scalar LDS stores，consumer 可作 packed row local-load。 |
| C0.5 packed | `persistent_typed_lds_layout` | K `[64,64]` 与 V `[64,32]` token-major；BF16x8 global load 直接以 packed LDS store 写入，consumer 为恢复 MFMA fragment 的 token sequence 做 eight scalar LDS reads/gather。 |

核心交换是：global memory 的 contiguous vector 方向是固定 token 内的 K/V
elements，而当前 MFMA32 fragment consumer 的 contiguous 方向是固定 K/V row 上的
eight token elements。若不改变冻结的 ownership、MFMA lane mapping 或 accumulation
order，两者就是一个转置关系。C0 用 scalar scatter 在 producer 完成这个转置；C0.5
延后转置，导致 consumer gather。

## 4. Same-Source Lowering 证据

两臂的 pre-branch MLIR 完全相同：

```text
SHA256 8675f9e514cd6f07b981c5f978212fdc90965d7bb9b59fba5bf30d682be331f2
```

它包含一个 high-level `block_dot_bf16_f32`，并有相同的 launch、barrier 和
`mfma_32x32x8` source call。只有 block-dot specialized lowering 之后出现差异：

| artifact | C0 scalar SHA256 | C0.5 packed SHA256 | 说明 |
|:--|:--|:--|:--|
| `post_block_dot_lowering.mlir` | `71c5f7c5dd78605980241cd087eb088497317ed28d99eb01920d29483f00ac78` | `791ee24e1e97b9d77b99b3f8a9504e834598a224e1e0c39b8db672c7e1ffd3fc` | 分叉已发生 |
| `preopt_llvm.ll` | `9dbdfd1e067aa7d5ed3fe70dbddb9d9ebf81b66a6b52bbfee8b7ec7607afad20` | `274bee63ada8701b82e7f317774ab831b3a78b055f3e6fe62453a711300bb9ad` | 差异保留至 LLVM |
| `postopt_llvm.ll` | `20e18a73cbc0fc4297f73822bd00e7de9ec8b59486df7ef28d58443bc4f198a9` | `98d15c04f430b17c0c1d8ca7a08cd830b8d268ef47d51ea96b039d4e94040068` | 未在 LLVM 优化中收敛 |

两臂各自捕获了 pre-link bitcode、LTO argv、post-LTO bitcode、pre/post-greedy
MIR 和 HSACO/ISA。故此 A/B 不是 frontend 中已经消失的无效开关。

## 5. 正确性

参考是 direct-K64 FP32 update reference，而非尚未解决的 full-v29 nonzero-W
reference。两臂的 6 个 reference cases 均通过：

| T | h max abs | final-state max abs | finite |
|--:|--:|--:|:--|
| 64 | `0` | `5.7220459e-06` | yes |
| 512 | `0.25` | `1.5258789e-05` | yes |
| 2048 | `0.5` | `4.5776367e-05` | yes |

最强的 C0-to-C0.5 output check 直接 hash H 和 final-state 原始 device bytes。三个
长度都逐字节一致：

| T | C0 / C0.5 SHA256 |
|--:|:--|
| 64 | `cd6ecac55fb7bab138bbbfa22694659f3d476a06ee321cb89eba94499ecb97f0` |
| 512 | `6d1055e25038a37dedaaef5f4546cab1dc08afe0ee3179b86261d024a96748bd` |
| 2048 | `914debb5131e42e5c980e2144aafc85966ed52d3c426b832dfbd0e0825fdeb8a` |

最初 byte-digest 测试错误地调用了 NumPy 的 BF16 conversion，环境不支持该 dtype。
已将 harness 改为 `tensor.view(torch.uint8)` 后再 hash 原始 bytes；这只修复测试
读取方式，不改 kernel 或 lowering。

## 6. Body Benchmark

同一 fresh-process harness：warmup=5、repeat=20、每长度六个 session median；两个
三-session block 使用不同 seed 和 arm-order offset。Triton W=0 是同一 isolated
recurrence-update control，非 full eager public API。

| T | chunks | C0 scalar ms | C0.5 packed ms | packed/C0 | Triton W=0 ms | packed/Triton |
|--:|--:|--:|--:|--:|--:|--:|
| 512 | 8 | `0.078346` | `0.077616` | `0.9907x` | `0.037556` | `2.0667x` |
| 1024 | 16 | `0.116623` | `0.114620` | `0.9828x` | `0.068121` | `1.6826x` |
| 2048 | 32 | `0.217103` | `0.210513` | `0.9696x` | `0.120118` | `1.7525x` |

端点 T=512 到 T=2048 的 chunk slope：

| arm | endpoint slope |
|:--|--:|
| C0 scalar | `5.7815 us/chunk` |
| C0.5 packed | `5.5374 us/chunk` |
| Triton W=0 control | `3.4401 us/chunk` |

packed 使 C0 slope 降 `0.2442 us/chunk`（`-4.22%`），但仍比 Triton control 多
`2.0973 us/chunk`。这说明 writer packing 不是剩余约 `1.75x` 差距的主导可恢复项。

## 7. T=2048 ROCprof 与资源

每臂 rocprof 使用 warmup=2/repeat=5。trace 是最后五个 matching dispatch 的 median；
PMC 动态计数对每个 matching dispatch 完全恒定。

| metric | C0 scalar | C0.5 packed | change |
|:--|--:|--:|--:|
| trace median | `183.072 us` | `174.900 us` | `-4.46%` |
| VGPR (rocprof) | 12 | 68 | +56 |
| AccVGPR | 196 | 196 | 0 |
| SGPR (rocprof) | 112 | 112 | 0 |
| LDS block | 12288 B | 12288 B | 0 |
| scratch | `0 B` | `0 B` | 0 |
| occupancy | `0.61937` | `0.61018` | `-0.00919` |
| SQ_INSTS_MFMA | 32768 | 32768 | 0 |
| SQ_INSTS_VMEM | 94720 | 94720 | 0 |
| SQ_INSTS_SALU | 189760 | 189824 | `+0.03%` |
| SQ_INSTS_VALU | 1355840 | 1482688 | `+9.36%` |
| SQ_INSTS_LDS | 188416 | 217088 | `+15.22%` |

ROCprof trace 与 body timing 的方向相同，但它带有 counter/trace instrumentation，
不把 `-4.46%` 作为比 fresh-process `-3.04%` 更权威的精确性能数字。二者共同支持
的结论仅是：packed path 的净延迟收益很小，且没有改变 MFMA 或 VMEM 工作量。

## 8. ISA、MIR 和低层差异

| static ISA family | C0 scalar | C0.5 packed |
|:--|--:|--:|
| `v_mfma_f32_32x32x8_bf16` | 32 | 32 |
| 16x16 MFMA | 0 | 0 |
| `s_barrier` | 5 | 5 |
| all `ds_write` | 80 | 10 |
| `ds_write_b16` | 80 | 0 |
| `ds_write_b128` | 0 | 10 |
| all `ds_read` | 24 | 192 |
| `ds_read_b128` | 24 | 0 |
| `ds_read_u16` | 0 | 192 |
| `global_load_dwordx4` | 18 | 18 |
| all buffer/global loads | 26 | 26 |
| buffer stores | 102 | 102 |

所以 C0.5 的 requested producer-side packed lowering 是真实发生的，例如 ISA 中有：

```text
ds_write_b128 v133, v[54:57]
ds_write_b128 v133, v[58:61] offset:64
```

但 consumer 紧接着是大量非连续地址的 `ds_read_u16`，并非 `ds_read_b128`。这正是
冻结 MFMA fragment mapping 与 global BF16x8 order 不同的机器级证据。

两臂都在 LTO replay 的 20 个 MIR sections 中检查了
`SI_SPILL_AV32_SAVE` 与 `SI_SPILL_AV64_SAVE`：计数均为零。最终 HSACO metadata
也显示 private segment、VGPR spill 和 SGPR spill 均为零。packed 路径没有产生
scratch/spill cliff，但 pre-greedy section 从 `2136` 行增加到 `2466` 行，与额外
gather/address/register work 一致。行数不是性能计数，只用作结构性佐证。

## 9. 回答本轮问题

1. **static `ds_write_b16` 是否显著下降？** 是，`80 -> 0`；`ds_write_b128` 为 `10`。
2. **是否生成 packed LDS store？** 是，真实 ISA 有 `ds_write_b128`，不是仅停留在
   MLIR `vector.store`。
3. **是否生成对应 typed/vector local-load？** 否。consumer 是 `192` 个静态
   `ds_read_u16`，而 C0 有 `24` 个 `ds_read_b128`。因此“swizzled”在本轮仅解决
   global-to-LDS producer 写入连续性，未同时解决 MFMA consumer fragment 连续性。
4. **LDS/VALU/SALU 与 latency 怎样变化？** LDS `+15.22%`、VALU `+9.36%`、SALU
   基本不变；body latency 仍小幅改善 `0.93%/1.72%/3.04%`，说明 packed write
   省下的 producer work 在当前机器上略大于 gather 增量，但空间很小。
5. **bank conflict 或 permute 是否抵消收益？** 未收集可直接归因 bank-conflict 的
   counter，不能把结果称为已证实的 bank conflict。已经可证实的是更多 LDS scalar
   reads、更多 VALU 和 VGPR；这些足以解释为何收益远小于 C0。
6. **相对 Triton W=0 gap 是否明显缩小？** T=2048 ratio 从 C0 的约 `1.807x`
   （同一 fresh C0.5 run）到 `1.753x`，只有 `~3%` 的缩小，未改变主结论。
7. **是否进入 C1 ping-pong prefetch？** 否。本轮没有实现 C1。C0.5 未确认单 buffer
   LDS layout 已经在 producer 和 consumer 两侧同时紧凑；现在直接加入 ping-pong 会把
   operand-layout gather、barrier、LDS footprint 和 overlap 四个变量混在一起。

## 10. 推荐与下一步边界

保留 C0.5 为 experimental lowering evidence，不升级为后续 pipeline baseline。下一步
不能把 C0.5 的 token-major packed layout直接扩展为 C1。若未来继续此方向，新的独立
实验必须首先改变或专门表达 **consumer MFMA fragment layout**，使 producer BF16x8
global vector 和 consumer local fragment 的方向一致，再比较 same-source A/B；否则
只是在 LDS 两端来回移动 scalarization。

在 user 已冻结的路线下，本实验的正式决策是：停止 C0.5 后续的 ping-pong 变体，保留
C0 persistent scalar-transpose path 作为较干净的 reference，并把剩余差距继续归因到
consumer fragment construction、decay/address dataflow 或更大 native fused pipeline，
而不是仅仅缺少 `ds_write_b128`。

## 11. 文件和复现

实现与 harness：

- `lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc`
- `test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_block_dot_bv32_coop.py`
- `test/examples/linear_attention/vllm_compare/test_qwen_gdn_direct_k64_block_dot_bv32_typed_lds_c05.py`
- `test/examples/linear_attention/vllm_compare/bench_qwen_gdn_direct_k64_block_dot_bv32_typed_lds_c05.py`
- `test/examples/linear_attention/vllm_compare/profile_qwen_gdn_direct_k64_block_dot_bv32_typed_lds_c05.py`

IR/LLVM/LTO/MIR/ISA/rocprof artifacts：

`test/examples/linear_attention/rocprof_outputs/qwen_block_dot_bv32_typed_lds_c05/`

关键命令：

```bash
cd /workspace/project/avelang
cmake --build /tmp/avelang-build-kfrag-qwen-rocm722 --target _avelang_bindings -j 16

export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:.
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_direct_k64_block_dot_bv32_typed_lds_c05.py -s

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_direct_k64_block_dot_bv32_typed_lds_c05.py \
  --T 512 1024 2048 --warmup 5 --repeat 20 --sessions 3 --seed 20260828 --order-offset 0

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_direct_k64_block_dot_bv32_typed_lds_c05.py \
  --T 512 1024 2048 --warmup 5 --repeat 20 --sessions 3 --seed 20260928 --order-offset 1

AVELANG_BLOCK_DOT_LOWERING=specialized \
AVELANG_BLOCK_DOT_BV32_MODE=1 \
AVELANG_BLOCK_DOT_OPERAND_LOWERING=persistent_typed_lds_layout \
/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex direct_k64_block_dot_bv32_coop \
  -d test/examples/linear_attention/rocprof_outputs/qwen_block_dot_bv32_typed_lds_c05/packed/rocprof_t2048 \
  -o counters -f csv -- python3 \
  test/examples/linear_attention/vllm_compare/profile_qwen_gdn_direct_k64_block_dot_bv32_typed_lds_c05.py \
  --operand persistent_typed_lds_layout --T 2048 --warmup 2 --repeat 5
```
