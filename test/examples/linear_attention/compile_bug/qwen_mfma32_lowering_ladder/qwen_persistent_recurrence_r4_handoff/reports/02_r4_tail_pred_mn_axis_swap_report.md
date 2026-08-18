# R4-tail pred MFMA 输出 M/N 轴交换：机制验证

## 结论

历史中**没有**完成过与本轮等价的 R4-tail pred-producer M/N 轴交换。本轮已在同一条
`gfx942_bt64_bv32_joint_v4_tail_issue` persistent recurrence 中接入唯一候选，并由
MLIR、LLVM、exact-LTO final-isel MIR 和 ISA 证明它真实交换了 pred MFMA 的两个
机器 operand：

```text
R4：       C[M=V, N=T] = state[V,K] × W[T,K]^T
candidate：C[M=T, N=V] = W[T,K] × state[V,K]^T
```

候选得到的是 **T1×V4**，不是单 lane 的 T1×V8：每个 lane 有四个彼此相隔 8 个
V 元素的连续 V4 fragment。因此本轮不是 No-Go；它允许下一轮只做静态
`persistent_io_packet<4xbf16>` 的 **b64** U/v_new 实验。它本身不足以宣称可以直接
发出单-lane b128；那需要两个 lane 的配对或另一种 accumulator layout，均不在本轮范围。

本轮按要求没有改 U/v_new、update、H/state ownership、specialized block-dot、tail
issue 或 LDS 容量；也没有运行 correctness、PMC 或 benchmark。这是纯 pred producer
机制和所有权验证。

## 历史排除

以下不算本实验：

- `qwen_gdn_v29_pred_only_mfma32_report.md` 的 MFMA 仍为
  `mfma(b_frag=state, a_frag=W, acc)`，即和 R4 相同的物理 operand 方向；它改变的是
  pred-only 调度/accumulator 解包，并非 R4-tail 的 M/N 交换。
- `repro_qwen_gdn_persistent_recurrence_r4_tail_issue_state_kv.py` 改的是 loop-carried
  state 和 update dual-dot 的 K×V 方向；其 pred 仍为 `mfma(state_frag, w_frag, ...)`。
- iopacket 与 LDS bridge 候选只在 pred 后重排或 packetize U/v_new；没有改变 pred
  MFMA producer 的物理 C 轴。

AveLang intrinsic 实现将第一个实参作为 `a`、第二个实参作为 `b`，并为此 op 配置
`m=32,n=32,k=8`（`lib/IR/Intrinsics/amdgpu_module.cc:653-699`）。所以把 first operand
从 state 改成 W 是真实 M/N 交换，不能与 A/B 名字交换、state-KV 或下游 transpose 混为一谈。

## 实现

基础 kernel 新增一个默认关闭的 constexpr `pred_mn_axis_swap`：

```python
if pred_mn_axis_swap:
    pred_acc = mfma_32x32x8_bf16_f32(w_frag, state_frag, pred_acc)
else:
    pred_acc = mfma_32x32x8_bf16_f32(state_frag, w_frag, pred_acc)
```

见 [R4 kernel](repro_qwen_gdn_persistent_recurrence_r4_joint_v4.py:283)；所有既有调用者仍取默认
`False`。新入口
[pred M/N candidate](repro_qwen_gdn_persistent_recurrence_r4_tail_issue_pred_mn_axis_swap.py:45)
通过同一 public-shaped R4-tail launch 传入：

```text
emit_audit=False, persistent_semantic=True,
io_packet_ownership=False, pred_mn_axis_swap=True
```

这保留原有的两-wave pred partial reduction，但没有新建 post-pred LDS bridge、packet gather 或
V×T 恢复 view。

## 精确 lane → (t,v) 公式

令当前 32×32 pred tile 的 wave-local lane 为 `l ∈ [0,63]`，accumulator index 为
`i ∈ [0,15]`，则：

```text
r = l mod 32                 # pred M axis, now token t
g = floor(l / 32)            # 0 或 1
i = 4q + p, q,p ∈ [0,3]

t = token_base + r
v = value_base + 8q + 4g + p
```

也就是每 lane 的 16 个 FP32 accumulator 对应：

```text
(t, 4g+0..3), (t, 8+4g+0..3),
(t,16+4g+0..3), (t,24+4g+0..3)
```

因此：

- 单 lane 每个 `q` 自然拥有一个连续 `V4`；可直接定义 b64 packet；
- `l=r` 与 `l=r+32` 的同 token fragments 分别是 `V[0:4]` 与 `V[4:8]`（以及
  `V[8:12]`/`V[12:16]` 等），故 V8 跨两个 lane；
- 这不再是原 R4 的 `V1×T8` producer，亦未通过执行 mask 循环模拟 packet。

该公式正是候选的直接 store map
`out_row=lane_row`、`out_col=(i>>2)*8+(lane>>5)*4+(i&3)`；见
[R4 kernel](repro_qwen_gdn_persistent_recurrence_r4_joint_v4.py:306)。

## 同源 production-shaped capture（T=64）

环境为 `ljd_qwen_vllm_avelang_rocm722` 的 gfx942/MI300X ROCm 容器，入口的 logical grid
为 32、workgroup 为 128，`emit_audit=False`。候选 HSACO SHA256：

```text
00ae277a950b5a26c3ef75b399fad8049e50e549132268405ce4cb1478e4802e
```

同配置 baseline HSACO SHA256：

```text
df21d6e2bb2aa16b745e1694e9554b94075fd9a53cfecf25b77367563da2abd5
```

制品目录：

- [MLIR snapshots](r4_tail_pred_mn_axis_swap_artifacts_t64/mlir/)
- [pre/post LLVM](r4_tail_pred_mn_axis_swap_artifacts_t64/mlir/preopt_llvm.ll)
  与 [postopt LLVM](r4_tail_pred_mn_axis_swap_artifacts_t64/mlir/postopt_llvm.ll)
- [exact-LTO final MIR](r4_tail_pred_mn_axis_swap_artifacts_t64/exact_lto/kernel_section_19.mir)
  （完整 replay 和所有 phase sections 位于同目录）
- [ISA](r4_tail_pred_mn_axis_swap_artifacts_t64/pred_mn_axis_swap_t64.isa.s)
  和 [HSA metadata](r4_tail_pred_mn_axis_swap_artifacts_t64/hsa_metadata.txt)
- [capture manifest](r4_tail_pred_mn_axis_swap_artifacts_t64/manifest.json)

### 轴交换不是元数据

post-opt LLVM 的 candidate 在 `postopt_llvm.ll:1017-1024` 先从 W 的动态 token LDS address
取 `%648`，从 state LDS address 取 `%649`，随后发出：

```llvm
@llvm.amdgcn.mfma.f32.32x32x8bf16.1k(%651 /* W */, %653 /* state */, ...)
```

同行号的 baseline 反过来是 `%651 /* state */`、`%653 /* W */`。final-isel MIR 同样保留
candidate 的 operand 顺序：

```text
V_MFMA_F32_32X32X8BF16_1K ... $vgpr100_vgpr101, $vgpr92_vgpr93
```

而 baseline 是 `$vgpr92_vgpr93, $vgpr100_vgpr101`。最终 ISA 的首个 pred MFMA（PC
`0x2ca0`）也逐字交换：

```text
baseline : v_mfma ... a[0:15], v[92:93],  v[100:101], 0
candidate: v_mfma ... a[0:15], v[100:101], v[92:93],  0
```

只有 pred 的前 8 条 MFMA 与其 producer scheduling 改变；后续 update MFMA 保持相同顺序。
总 MFMA 数两侧皆为 40（T=64）。

### 约束检查

| 指标 | baseline | candidate |
|---|---:|---:|
| MFMA32 static count | 40 | 40 |
| LDS | 53,248 B | 53,248 B |
| private segment | 0 B | 0 B |
| VGPR | 212 | 212 |
| SGPR | 29 | 29 |
| VGPR/SGPR spill | 0 / 0 | 0 / 0 |
| `ds_bpermute` / `v_bpermute` | 0 / 0 | 0 / 0 |
| `v_perm_b32` | 64 | 64 |

`v_perm_b32` 是原有 BF16 operand packing，数量没有增加；没有新出现的 cross-lane gather。
candidate 的 U/v_new 路径仍为 8 条 `global_load_ushort` 和 8 条
`global_store_short_d16_hi`（T=64 production body），符合“本轮不修改 U/v_new”的约束。
exact-LTO final MIR 的 `SI_SPILL_AV32_SAVE` 与 `SI_SPILL_AV64_SAVE` 均为 0。

## 下一轮边界

可做且只应做一项：在这个 pred M=T/N=V producer 上，为每个 `q` 保留 first-class
`T1×V4` / `4xbf16` ownership，令 U load 与 v_new store 直接变为 b64，禁止 packet loop、
exec-mask 回边、LDS bridge/gather 和 update 改动。先检查 ISA 的实际 b64 与无新增动态
transpose，再决定是否进入 correctness/性能测量。

不应做：宣称单 lane b128、重开 software pipeline/distance、修改 update 或新增 LDS bank。
