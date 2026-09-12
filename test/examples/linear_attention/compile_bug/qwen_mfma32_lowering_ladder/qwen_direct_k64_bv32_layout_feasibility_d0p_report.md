# Direct-K64 BV32 D0-P: LDS Layout Feasibility

## 1. 结论

**D0-P 没有通过，按预注册停止条件关闭局部 LDS-layout 路线。**

本轮没有重复 C0.5S 已做过的“普通标量 store 对显式 `u32` packed
store”比较。它验证的是更窄也更关键的问题：能否让 producer 和
MFMA32 consumer 同时保持紧凑，而不是一端 packed、另一端退化成 LDS
gather。

结果分为三层：

1. 当前 Triton/current-vLLM 的 TTGIR 确实拥有 AveLang C0.5 所没有的
   `amd_rotating_shared` / `swizzled_shared` / `ttg.dot_op` typed operand
   表示，并对 K 使用 `amdg.in_thread_transpose`。
2. AveLang 公共 source 已能表达这个方向的几个硬件原语：BF16x8 raw
   global load、跨 lane `al.shuffle -> ds_bpermute_b32`、`v_perm_b32`、
   `ds_write_b128`、`ds_read_b128` 和 `v_mfma_f32_32x32x8_bf16`。
3. 但是完整的 BV32 direct-K64 register-transpose update arm 在 `T=64`
   已经出现 15 个 `H` NaN，final-state 最大绝对误差为 `61.0416`。它不满足
   最基本的 semantic gate，故没有运行 T=512/1024/2048 benchmark、rocprof
   或任何“性能改善”结论。

因此，这个结果不能说“硬件布局不可行”，也不能说“AveLang 不能发出
`ds_bpermute` 或 packed LDS 指令”。它只能严格地说：**当前公开 source
操作的组合尚未构成一个 correctness-preserving 的完整 Qwen direct update
implementation；继续枚举更多 swizzle 没有数据支持。**

下一步应转向 matched fused recurrence pipeline，而不是继续 D0-P local
LDS-layout sweep、ping-pong 或更多 swizzle 枚举。

## 2. 冻结边界

本轮保持不变：

| 项目 | 值 |
|:--|:--|
| ABI | BF16 K/V-new，FP32 g/state，BF16 H，FP32 final state |
| token/value block | BT64 / BV32 |
| workgroup / ownership | WG128，32 CTA，two-wave cooperative |
| dot | `v_mfma_f32_32x32x8_bf16` |
| K order | 每 half 内 K32 accumulation order 不变 |
| state/global IO | C0 的 persistent state 和 public output contract |
| 禁止项 | production、allocator/RA、旧 broad/compact-K、full-v29 nonzero-W、BV、MFMA geometry、`lower_qwen_block_dot_pass.cc`、ping-pong |

本轮没有修改 `lower_qwen_block_dot_pass.cc`，没有改变 production selector，也
没有把任何 D0-P 代码接到 full Qwen forward。

## 3. Triton/current-vLLM Mapping Audit

审计固定使用 Stage 6R 已保存的 current-vLLM recurrence 工件：

```text
codex_qwen_bt64_recurrence_reconciliation_stage6r/
  current_kernels/vllm/kernel.{ttir,ttgir,llir,amdgcn,hsaco}
```

机器可检查输出：

```text
rocprof_outputs/qwen_direct_k64_bv32_layout_d0p/audit/
  triton_k64_mapping.json
  triton_k64_lane_global_mapping.csv
```

输入工件 SHA256、所有布局属性、local alloc/store/load、dot site 和 ISA 静态计数
均写入 JSON；CSV 枚举 4096 个 `K[64,64]` global-load 元素的 lane ownership。

### 3.1 TTIR / TTGIR 的直接证据

TTIR 中的 update dot 是：

```mlir
tt.dot : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
```

TTGIR 的关键 layout 不是猜测：

```mlir
#blocked = #ttg.blocked<{
  sizePerThread = [8, 4], threadsPerWarp = [8, 8],
  warpsPerCTA = [1, 2], order = [0, 1]}>
#mma = #ttg.amd_mfma<{
  version = 3, warpsPerCTA = [2, 1], instrShape = [32, 32, 8],
  isTransposed = true}>
#shared1 = #ttg.amd_rotating_shared<{
  vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared2 = #ttg.swizzled_shared<{
  vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
```

K operand 的实际链为：

```mlir
amdg.buffer_load K[64,64], #blocked
  -> amdg.in_thread_transpose
  -> ttg.local_store into #shared1
  -> ttg.local_load -> #ttg.dot_op{opIdx=0, parent=#mma, kWidth=4}
  -> tt.dot
```

V-decay operand 的链为：

```mlir
BF16 V-decay tensor
  -> ttg.local_alloc into #shared2
  -> ttg.local_load -> #ttg.dot_op{opIdx=1, parent=#mma, kWidth=4}
  -> tt.dot
```

这就是 Triton 已经拥有 first-class dot operand layout，而 AveLang C0/C0.5
只有 shared tensor/view 的实物证据。

### 3.2 可严格导出的 lane -> global K mapping

`#blocked` 的 `64x64` tensor map 可由 encoding 精确展开。对 `warp in {0,1}`、
`lane in [0,63]`、`row_item in [0,7]`、`col_item in [0,3]`：

```text
logical_token          = (lane % 8) * 8 + row_item
logical_feature_in_k64 = warp * 32 + (lane // 8) * 4 + col_item
byte_offset_in_K64     = 2 * (logical_token * 64 + logical_feature_in_k64)
```

这正是 CSV 的 4096 行数据。它说明每 wave 占 K64 的一个 32-feature half，
每 lane 初始拥有一个 `8 x 4` register tile；这与当前 BV32 two-wave ownership
一致。

### 3.3 不应伪造的 mapping 部分

当前保存的 TTGIR 在 `ttg.local_load -> #ttg.dot_op` 已抽象掉最终 lane-address
展开。AMDGPU ISA 只保留寄存器和动态地址寄存器，而不会带原始 tensor element
provenance。因此，本轮**没有猜测**以下字段：

- 每个 lane 的精确 LDS byte offset；
- 每个 `ds_read_b64` 字中每个 BF16 的逻辑 token/feature；
- 每个 MFMA operand word 的完整 source-element 集合。

JSON 将这些字段明确标为 `not materialized by TTGIR element map`，而不是用
Triton 变量名填充。若将来需要“每个 operand word”的完全映射，正确的工具是
Triton layout interpreter 或在 `ttg.local_load` lowering 前输出 element map；仅靠
此 HSACO/TTGIR dump 不能诚实地恢复。

### 3.4 真实 ISA 形态

current-vLLM artifact 的静态计数：

| family | count |
|:--|--:|
| `buffer_load_dwordx4` | 36 |
| `ds_write_b16` | 88 |
| `ds_write_b64` | 30 |
| `ds_write2st64_b64` | 25 |
| `ds_read_b64` | 112 |
| `ds_read2_b64` | 8 |
| `ds_read2st64_b32` | 18 |
| `v_perm_b32` | 144 |
| `ds_bpermute_b32` | 0 |
| `v_mov_b32_dpp` | 0 |
| `v_mfma_f32_32x32x8_bf16` | 64 |
| `s_barrier` | 32 |

这份**特定 current-vLLM recurrence artifact**使用 `v_perm_b32`，但没有
`ds_bpermute` 和 DPP。仓库里其他 Triton kernels 出现过它们，不能移花接木到
这个 recurrence update。D0-P 的 source transpose 使用 `ds_bpermute` 是一种
AveLang feasibility construction，不是声称复制了这份 ISA 的全部调度。

## 4. AveLang Source Expressibility Gate

新增 gate：

```text
vllm_compare/repro_qwen_gdn_direct_k64_bv32_layout_expressibility_d0p.py
```

通过工件已 dump 到：

```text
rocprof_outputs/qwen_direct_k64_bv32_layout_d0p/expressibility_hsaco/
```

### 4.1 结果

| case | source capability | MLIR -> LLVM -> LTO -> HSACO | 结论 |
|:--|:--|:--|:--|
| `packed_local_load` | BF16x8 raw load -> packed LDS -> fragment | pass | 可表达 |
| `lane_shuffle` | `al.shuffle(u32, lane, 64)` | pass | 可表达 |
| `register_transpose_fragment` | 8x8 transpose -> packed LDS row -> fragment | pass | 可表达 |
| `scaled_register_transpose_fragment` | BF16 -> FP32 decay -> lane shuffle -> BF16 store -> fragment | pass | 可表达 |
| initial scaled-local-repack | local BF16x8 -> `view(u32)` | fail | 不能表达 |
| `noncontiguous_fragment` | token-major LDS gather -> local fragment | fail | 不能表达 |

两项失败都停在 LLVM translation，留下 `builtin.unrealized_conversion_cast`；不是
runtime error，也不是 benchmark 噪声。

### 4.2 通过 gate 的 ISA

最小 register-transpose gate 发出：

```text
buffer_load_dwordx4
ds_bpermute_b32  (8 fixed exchanges)
v_perm_b32
ds_write_b128
s_barrier
ds_read_b128
v_mfma_f32_32x32x8_bf16
```

scaled V-decay 版本同样发出 `v_exp_f32_e32`、8 个 `ds_bpermute_b32`、
`ds_write_b128`、`ds_read_b128` 和 MFMA32。它证明：AveLang source 能把
FP32 decay 结果经过 lane exchange 放进 consumer-contiguous LDS layout，且不必
先把 local BF16 vector bitcast 回 `u32`。

相反，C0.5 的 token-major packed 方案需要从 LDS 取非连续元素再把它们重新装成
typed MFMA fragment；这仍在 LLVM lowering 前留下 unrealized cast。这确认真正的
缺口是 **typed non-contiguous fragment gather/repack**，不是一般 vector store API。

## 5. D0-P Feasibility Arms

新增 full-suffix source：

```text
vllm_compare/repro_qwen_gdn_direct_k64_bv32_layout_feasibility_d0p.py
```

| arm | 状态 | 说明 |
|:--|:--|:--|
| `c0_reference` | existing reference | 复用 C0 persistent typed block，未重复 benchmark |
| `blocked_8x8` | N/A | 需要 token-major LDS non-contiguous gather；gate failed，明确抛错 |
| `register_transpose` | compile/run but fail | source 做 K/V 的 producer-side 8x8 transpose，consumer 直接从 row-contiguous LDS `u32` view 取 MFMA fragment |

`register_transpose` 的设计保持 32 CTA、BV32、WG128、K64 half、K32 reduction
order和 two-wave state ownership：

```text
source lane: one token x contiguous BF16x8
  -> four u32 DS-bpermute fetches / token for destination feature selection
  -> V: BF16 -> FP32 decay -> BF16 LDS store
  -> K: direct BF16 LDS store
  -> row-contiguous V32xT64 / K64xT64 LDS
  -> ds_read_b128 fragment -> MFMA32
```

这不是 hidden block-dot schedule：lane/subgroup ownership、token/feature index、
source lane order和 LDS write 都直接写在 experimental Python source 中。

## 6. Correctness Gate

T=64 direct-K64 FP32 update reference smoke：

| arm | finite | H max/mean abs | final max/mean abs | 结果 |
|:--|:--|:--|:--|:--|
| `register_transpose` | no | `NaN / NaN` | `61.041603 / 4.832937` | fail |

额外 output inspection 显示 H 共 `131072` 个元素中有 `15` 个非有限值；final
state 本身是 finite，但不能掩盖 H contract 和 numerical mismatch 已失败。

前两个 H snapshot rows 与 input state 大体相符，说明不是整块 output 没写；NaN
分散在多个 head/value/K 坐标。基于这一次失败，不能严谨地将问题归因于某一条
`ds_bpermute`、bank conflict 或 register allocator：它可能是 full source
composition、state/output mapping、barrier/liveness 或当前 conversion lowering 的任一
组合。没有通过 semantic gate 前，继续把它当性能候选是错误的。

## 7. 无效 arm 的静态诊断

尽管其数值无效，HSACO 仍记录为失败证据，不能用于性能宣称：

| static family | register-transpose full arm |
|:--|--:|
| `v_mfma_f32_32x32x8_bf16` | 16 |
| `buffer_load_dwordx4` | 6 |
| `ds_write_b128` | 6 |
| `ds_write_b16` | 0 |
| `ds_read_b128` | 16 |
| `ds_read_u16` | 0 |
| `ds_bpermute_b32` | 208 |
| `v_perm_b32` | 24 |
| `s_barrier` | 3 |

它确实达成了“producer/consumer 没有 `ds_read_u16` gather”的静态形态，但代价是
208 个 static `ds_bpermute`，并且 correctness 已失败。这两个事实共同禁止把它同
C0 的 `0.216922 ms` 或 Triton W=0 `0.119778 ms` 作 latency 比较。

## 8. 为什么不跑 benchmark / rocprof

预注册 gate 要求 T=64/512/2048 的 update reference、finite、C0 output check、
MFMA/K32 order与 zero scratch/spill 都先通过。第一个 T=64 已失败，所以：

- 没有 T=512/1024/2048 body benchmark；
- 没有 multi-session rotating-order slope；
- 没有 rocprof PMC、occupancy或 spill 宣称；
- 没有 full-path 接入；
- 没有把 invalid arm 与 Triton W=0 control 比倍数。

这不是遗漏数据，而是为了避免错误 mapping 在 profiler 下看起来“快”而污染后续
结论。

## 9. 最终决策

**停止 D0-P local LDS-layout route。**

理由不是“没有任何 AveLang source 能力”：最小 gate 已证实 raw x4、DS bpermute、
V decay 后的 lane transpose、packed LDS store/load 和 MFMA32 均可从公开 source
发出。停止理由是：

1. producer-only token-major packed 已由 C0.5/C0.5S 证明不是答案；
2. gather-to-fragment path 仍缺 first-class typed fragment lowering；
3. source register-transpose 虽可发出所需指令，但完整 update 在最小 correctness
   case 已失败，且静态 lane-exchange 成本很高；
4. 再试其它 swizzle 会同时改变 exchange、address、barrier和 liveness，无法得到
   清晰因果。

若继续追求 end-to-end performance，应转向 **matched fused recurrence pipeline**。
如果未来重启该局部路线，前提不是再换一个 swizzle，而是提供可验证的 first-class
typed MFMA fragment/layout IR，并在 lowering 前输出每个 lane 的 LDS/fragment element
map；届时才能把 Triton `#shared/#dot_op` 的语义逐元素对齐。

## 10. 复现

```bash
cd /workspace/project/avelang
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:.

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/audit_qwen_gdn_direct_k64_bv32_layout_d0p.py

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_layout_expressibility_d0p.py \
  --case all --dump-hsaco-dir \
  test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_layout_d0p/expressibility_hsaco --json

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_layout_feasibility_d0p.py \
  --arm register_transpose --T 64 --warmup 1 --repeat 2 --json
```

The last command is expected to reproduce the current correctness failure;
it is not a benchmark command.
