# R4-tail State-KV pred M/N-swap：直接 K×V B fragment 机制验证

## 结论

历史中没有与本轮完全等价的候选。已有 `state_kv_dual_dot` 虽然将 loop-carried
feedback/update 保留为 K×V，却在 pred 前写入 `state_pred[wave,V,K]`，并以
`mfma(state_frag, w_frag)` 计算 pred；它不是本实验。

本轮新候选 `gfx942_bt64_bv32_joint_v4_tail_issue_state_kv_pred_mn_swap`
完成了以下三项，而且仅改变 pred producer：

```text
state feedback LDS: [wave, K, V] = [2,64,32]
pred:               W[T,K] @ state[K,V]
MFMA physical C:    M=T, N=V
```

它得到 **T1×V4** ownership，而不是 V1×T8，也不是单 lane T1×V8。因此本轮
不是 No-Go：下一轮可以只尝试保留这一 ownership 的
`persistent_io_packet<4xbf16>`/b64 U-load 与 v_new-store；不能声称已经有单-lane
b128，V8 仍跨两个 lane。本轮没有修改 U/v_new、update、H ABI、tail issue 或 LDS
bank，也没有以此候选运行 correctness、PMC 或 benchmark。

## 为什么历史候选不等价

| 路径 | loop-carried state | pred 前 operand | pred MFMA | 是否本实验 |
|---|---|---|---|---|
| R4-tail | legacy V×K fragment | state V×K | `mfma(state,W)` | 否 |
| `state_kv_dual_dot` | K×V | `state_pred[V,K]` materialization | `mfma(state,W)` | 否 |
| 本候选 | K×V | direct `state_kv[wave,K,V]` | `mfma(W,state)` | 是 |

旧候选的 V×K materialization 位于
[`repro_qwen_gdn_persistent_recurrence_r4_tail_issue_state_kv.py`](repro_qwen_gdn_persistent_recurrence_r4_tail_issue_state_kv.py)
的 constexpr false 分支。AveLang 在 AST construction 前会解析两个 constexpr 分支，
故源码仍声明该 control buffer；但 `pred_mn_axis_swap=True` 的 post-block MLIR、post-GPU
MLIR 都没有 `memref<2x32x64xbf16>`，最终 53,248-B LDS allocation 也只保留一个
8-KiB K×V snapshot。这是生成物而非仅源码的消除证据。

## 接入的 first-class lowering

普通 `al.make_local((8,), bf16)` gather 接到 MFMA 时在 LLVM conversion 留下
`builtin.unrealized_conversion_cast`，所以不能拿“能生成 MLIR”误当成功。为保持 K×V
ownership 到 late lowering，本轮新增受固定 Qwen 几何约束的 op：

```text
ave.amdgpu_qwen_pred_state_kv_frag_load(
    state_kv[2,64,32], wave, k_vector, v_lane, fragment_offset
) -> vector<4xbf16>
```

相关实现：

- [op definition](../../../../lib/Dialect/AveLang/IR/AveLangOps.td)
- [AMDGPU intrinsic creation/check](../../../../lib/IR/Intrinsics/amdgpu_module.cc)
- [late vector-SSA lowering](../../../../lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc)
- [State-KV candidate source](repro_qwen_gdn_persistent_recurrence_r4_tail_issue_state_kv.py)
- [production-shaped export entry](run_qwen_gdn_persistent_recurrence_r4_tail_issue_state_kv_pred_mn_swap.py)

late lowering emits, for `e=0..3`:

```text
B_fragment[e] = state_kv[wave,
                          8*k_vector + 4*fragment_offset + e,
                          v_lane]
MFMA(A=W_fragment, B=B_fragment, acc)
```

所以它没有 V×K shared view、private gather memref、`ds_bpermute` 或 exec-mask packet
loop。post-block MLIR 的 direct fragment carries
`avelang.qwen.pred_state_kv_mn_swap.direct_b_fragment`，并紧接着成为
`_mfma_f32_32x32x8bf16_1k(W, state, acc)` 的第二个 operand。

## 精确 lane → (t,v) 公式

令当前 pred 32×32 tile 的 lane 为 `l in [0,63]`，MFMA accumulator index 为
`i in [0,15]`：

```text
r = l mod 32
g = floor(l / 32)
i = 4q + p,  q,p in [0,3]

t = token_base + r
v = value_base + 8q + 4g + p
```

含义分两层：

```text
MFMA B input owner:
  lane l supplies state[K=wave*64 + 8*k_vector + 4*f + e,
                        V=value_base + r]
  (f=fragment_offset, e=0..3)

MFMA C accumulator owner:
  lane l owns (t, V[4g:4g+4)), (t, V[8+4g:8+4g+4)),
               (t, V[16+4g:16+4g+4)), (t, V[24+4g:24+4g+4))
```

每个 `q` 是一个连续 V4；`g=0` 与 `g=1` 分别拥有相邻的低/高 V4，故 V8 必须跨 lane
`r` 和 `r+32`。这正是此候选可以研究 b64、但不能直接承诺 b128 的原因。

## 同源 production-shaped capture（T=64）

运行环境是 `ljd_qwen_vllm_avelang_rocm722` 的 gfx942 容器。入口采用
`emit_audit=False`、logical grid=32、workgroup=128；它只用于 codegen/ownership
机制检查。

| 项目 | 结果 |
|---|---|
| kernel symbol | `_state_kv_kernel` |
| HSACO SHA256 | `18f6687a17e26badfb2c194653fc81954d5401f46818ee5c931fcae09f30b820` |
| LDS | 53,248 B |
| private segment | 0 B |
| code-object `num_vgpr` | 164 |
| HSA `.vgpr_count` / `.agpr_count` | 228 / 64 |
| HSA `.sgpr_count` | 31 |
| HSA VGPR/SGPR spills | 0 / 0 |
| exact-LTO `SI_SPILL_AV32/AV64_SAVE` | 0 / 0 |
| static MFMA32 | 48 |
| `ds_bpermute` | 0 |

`num_vgpr` symbol and HSA metadata expose不同 resource fields，因此原样同时报告，
不把 164、228 与 64 混成一个寄存器数。

制品均来自上述唯一 HSACO：

- [post-block MLIR](r4_tail_state_kv_pred_mn_swap_artifacts_t64_v2/mlir/post_block_dot_lowering.mlir)
- [pre/post LLVM](r4_tail_state_kv_pred_mn_swap_artifacts_t64_v2/mlir/preopt_llvm.ll)
  / [postopt LLVM](r4_tail_state_kv_pred_mn_swap_artifacts_t64_v2/mlir/postopt_llvm.ll)
- [exact-LTO final-isel MIR](r4_tail_state_kv_pred_mn_swap_artifacts_t64_v2/exact_lto/kernel_section_09.mir)
- [ISA](r4_tail_state_kv_pred_mn_swap_artifacts_t64_v2/state_kv_pred_mn_swap_t64.isa.s)
- [HSA metadata](r4_tail_state_kv_pred_mn_swap_artifacts_t64_v2/hsa_metadata.txt)

ISA 的 pred region（约 PC `0x3530..0x3774`）在 `v_mfma` 前交错
`ds_read_u16` 与 `v_perm_b32`，但没有 `ds_bpermute`。这是 K×V memory 在固定 V 上
取连续 K 的真实成本；本结论只说明 output accumulator ownership 已交换，并不把这项
operand-preparation 成本包装成性能收益。

## 决策

允许的下一项且仅一项是：将 pred partial 的上述 T1×V4 label 保留到 corrected 边界，
重新指派 U/v_new 的同一 packet owner，并以 ISA 验收真正的 b64 与无 exec-loop。若该
label 在 two-wave pred reduction 或 public I/O ABI 前被迫丢失，则关闭 I/O 路线，转向已
量化的 update operand preparation 主矛盾。
