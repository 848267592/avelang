# Qwen Direct-K64 S0-C: Triton Packetized LDS Commit Feasibility Audit

## 结论

**S0-C 在 compiler lowering 实现之前以 NO-GO 结束。** 这不是因为无法恢复
current-vLLM Triton 的 K64 映射，恰好相反：本轮从同一 Stage-6R 捕获实例的
TTGIR 与 LLVM 中恢复了完整、可逆、逐元素验证的 `64 x 64` K tile 映射。

映射证明了 S0-C 的两个冻结要求彼此不兼容：

1. 保留当前 S0 的 early `vector<8xbf16>` global packet；
2. 只做 lane-local `in_thread` transpose/repack，且保持 C0 的逻辑 K LDS
   consumer 不变。

当前 S0 的一个 BF16x8 packet 是**一个 token 的连续八个 K 值**；native Triton
packet 是**一个 K 值的连续八个 token**。每个 native BF16x8 packet 的八个元素来自
**八个不同的 current-S0 lanes**。因此其构造至少需要 cross-lane exchange，或者改变
LDS physical/consumer layout。前者（`ds_bpermute`、猜测性的 lane transpose）和后者
（新 shared layout/swizzle、C0 consumer 变化）都被 S0-C 明确禁止。

所以不能诚实地新增一个名为 `triton_packet_commit` 的 branch：若它只含 lane-local
repack，就不可能复现 recovered native packet；若让它通过 cross-lane 操作完成，则不再是
本轮允许的唯一变量。没有进行编译、GPU correctness、PMC 或性能测试，因为那会测试一个
已经偏离 S0-C contract 的 kernel，而不是 same-source commit A/B。

这给 S0-C 一个明确的、结构性的 NO-GO；它不是“性能还没测”。结合 S0 已经得到的
性能中性结论，冻结约束下的 K64 pipeline-stage 路线关闭：不进入 S1，不接回 B0/full
recurrence，也不继续枚举 store width、barrier 或 RA。

## 范围与未变项

本轮没有修改：

- `amdgpu_qwen_k64_pipeline_stage_load/commit`；
- `lower_qwen_k64_pipeline_stage_pass.cc`；
- C0 preloaded-K Direct-K64 MFMA32 consumer；
- MFMA geometry、BV32/WG128 ownership、LDS bank 数、allocator/RA；
- production selector、external HSACO 或 full recurrence。

唯一新增的代码是 audit-only verifier：

- `audit_qwen_k64_triton_packet_commit_mapping_s0c.py`

它不 JIT、不会 launch GPU，也不会编写 compiler IR。生成产物位于：

- `rocprof_outputs/qwen_direct_k64_pipeline_stage_s0c/audit/`
  - `qwen_k64_triton_packet_commit_mapping.json`
  - `qwen_k64_triton_packet_commit_mapping.md`

## 读取的证据

审计使用 Stage 6R 保留的同一 current-vLLM recurrence 工件：

```text
codex_qwen_bt64_recurrence_reconciliation_stage6r/
  current_kernels/vllm/kernel.ttgir
  current_kernels/vllm/kernel.llir
  current_kernels/vllm/kernel.amdgcn
  current_kernels/vllm/disassembly.txt
```

TTGIR 明确给出：

```mlir
#blocked = #ttg.blocked<{
  sizePerThread = [8, 4], threadsPerWarp = [8, 8],
  warpsPerCTA = [1, 2], order = [0, 1]}>

#linear1 = #ttg.linear<{
  register = [[0,1], [0,2], [1,0], [2,0], [4,0]],
  lane     = [[8,0], [16,0], [32,0], [0,4], [0,8], [0,16]],
  warp     = [[0,32]]}>

#shared1 = #ttg.amd_rotating_shared<{
  vec = 4, perPhase = 1, maxPhase = 16, order = [1,0]}>
```

K 的链是：

```text
#blocked global K load
  -> amdg.in_thread_transpose
  -> #linear1
  -> ttg.local_store into #shared1
  -> ttg.local_load #ttg.dot_op(opIdx=0)
  -> v_mfma_f32_32x32x8_bf16
```

LLVM 将 K0 store 展开在 `kernel.llir:608..673`，将 K0 MFMA B operand load 展开在
`kernel.llir:3237..3266`。这两个位置提供了所需的具体 LDS 地址表达式；不依赖 ISA
变量名猜测。

## 完整 4096 元素映射

对任意逻辑 `source[token, k]`，其中 `token,k in [0,63]`，从 TTGIR `#linear1`
逆解得到：

```text
native_lane = 64 * (k // 32) + 8 * ((k % 32) // 4) + token // 8
load_packet = k % 4
packet_bf16_index = token % 8
post_in_thread_transpose_register = load_packet + 4 * packet_bf16_index
```

也就是说，native 的每个 lane 有四个 BF16x8 packet；每个 packet 固定一个 `k % 4`，
包含八个连续 token。LLVM K0 store 的 rotating-LDS byte 地址为：

```text
base = ((0 if lane bit0 == 0 else 1088) | ((lane & 6) << 2))
       xor (lane & 120)
       | ((lane & 6) << 10)
lds_byte = (base xor (136 * (token % 8))) + 2 * (k % 4)
```

LLVM K0 consumer 的读地址为：

```text
consumer_base = (2056 if lane & 16 else 0)
                xor ((4112 if lane & 64 else 0) | ((lane & 32) >> 2))
                xor ((lane & 15) << 3)
                | ((lane & 15) << 7)
lds_byte = consumer_base xor (16 * mfma_k8) + 2 * bf16_word
```

verifier 的 mechanical checks：

| 检查 | 结果 |
|:--|:--|
| producer source elements | `4096` |
| distinct rotating-LDS BF16 positions | `4096` |
| LDS byte range | `0..8190`，无 hole、无 duplicate |
| consumer BF16 words | `4096` |
| consumer 对应的 unique source elements | `4096` |
| producer/consumer inverse map | exact |
| out-of-range address | `0` |

因此 native mapping 不是抽象 layout 说明，而是已经写入 JSON 的 4096 行
`source -> lane -> packet -> reg -> LDS byte -> MFMA consumer` 证据。

## 为什么当前 S0 不能仅靠 lane-local commit 变成 native

S0 的 existing `emitPacketLoads` 固定了：

```text
linear = s0_lane + 128 * s0_packet
token = linear / 8
k     = 8 * (linear % 8) + bf16_index
```

反解为：

```text
s0_lane    = 8 * (token % 16) + k / 8
s0_packet  = token / 16
s0_bf16idx = k % 8
```

这意味着 S0 packet 的形状是：

```text
fixed token x eight consecutive K values
```

而 native packet 的形状是：

```text
fixed K value x eight consecutive token values
```

verifier 对全部 `128 lanes x 4 native packets = 512` 个 native BF16x8 packet
分别收集其 eight source elements 的 current-S0 owner，得到：

| 一个 native BF16x8 packet 所需的 distinct current-S0 lanes | packet 数 |
|---:|---:|
| `8` | `512` |

没有一个 native packet 能由一个 current-S0 lane 自己持有的四个 BF16x8 vectors 通过
register reorder 得到。这不是某个地址常量、store width 或 backend combine 的问题，而是
两个 packet 化方向正交。

## 对拟议 A/B 的影响

拟议的 scalar arm 可以保持：

```text
current-S0 BF16x8 K-contiguous packet
  -> extract BF16
  -> K_bank[k, token] scalar stores
```

但是 `triton_packet_commit` 要同时满足：

```text
same current-S0 early load packets
+ lane-local-only repack
+ unchanged C0 logical K consumer
+ native token-contiguous BF16x8 LDS packet
```

此方程没有解。可行的两个操作都违反预注册限制：

| 需要的操作 | 为什么不允许 |
|:--|:--|
| cross-lane retile / permutation | 每个 native packet 需要八个 S0 lanes；S0-C 禁止 `ds_bpermute` 与猜测性 cross-lane transpose |
| 改为 native rotating LDS physical layout 并同步改 consumer load | 这会改变 C0 consumer / 引入新的 shared layout 或 swizzle |
| 改变 early global packet ownership，使每 vector 沿 token 连续 | 不再保持 current-S0 early vector load recipe；对 T-major K 输入也不能由当前一次 `vector<8xbf16>` contiguous load 实现 |

所以新环境变量 `AVELANG_QWEN_K64_PIPELINE_COMMIT_LOWERING` 没有加入。加入一个实际上
无法合法实现的 `triton_packet_commit`，或让它悄悄 fallback 成 32 个 scalar store，只会
把结构失败伪装成性能实验。

## 为什么 S0 early-load 成功仍然性能中性

S0 原本已经证明：同一高层 opaque stage token 可在 AveLang compiler 中形成
`next global load -> current MFMA -> late commit`，且不引入 B2 的 local array、scratch
或 spill。它的 15-session 性能 CI 仍跨零，原因是单一 K64 的 window 太窄，并且 commit
本身仍然是 scalar BF16 stores。

S0-C 的目标本来是仅将后者替换为 native-style packet commit。然而 recovered mapping
说明 native packetization 并不只是 commit-time store packing；它从 global-load ownership、
`in_thread_transpose`、rotating shared layout 到 dot operand consumer 是一个整体。只替换
最后一个 commit 而冻结其他三端，无法保持语义。

## 决策

```text
S0-C structural feasibility: NO-GO
same-source scalar-vs-packet commit A/B: not legal under frozen contract
GPU correctness / PMC / benchmark: intentionally not run
S1 / B0 integration / more K64 stage tuning: closed
```

这不代表 gfx942 或 Triton packetized K64 staging 不可行，也不构成“AveLang backend 不支持
宽 LDS store”的结论。它只说明：**当前 S0 source packet ownership 与 C0 consumer ABI
不具备把 Triton mapping 投影成唯一 commit lowering 的条件。**

若未来需要重新开启该方向，必须作为一个新的、有不同 contract 的研究题：同时定义
global-load ownership、cross-lane transpose/typed fragment 和 rotating LDS consumer，而不是
继续在 current S0 commit 上调整 `ds_write_b16/b64`。

## 复现

```bash
cd /home/jiandongliu/project/avelang
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/\
  audit_qwen_k64_triton_packet_commit_mapping_s0c.py

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/\
  audit_qwen_k64_triton_packet_commit_mapping_s0c.py \
  --out-dir test/examples/linear_attention/rocprof_outputs/\
qwen_direct_k64_pipeline_stage_s0c/audit
```
