# R4-tail State-KV pred-M/N T1xV4 b64 I/O：MFMA 编码修复与正确性 GO

日期：2026-08-05

## 范围

本轮以 `gfx942_bt64_bv32_joint_v4_tail_issue_state_kv_pred_mn_swap` 为唯一
基础，新增 constexpr `pred_mn_v4_io`。没有改 H、W/K、tail issue、LDS 大小、
MFMA 数量、BF16 边界、state feedback 或现有 specialized update block-dot。

每个 pred M=T/N=V producer lane 的静态所有权为：令 wave-0 lane 为 `l`，
`r=l mod 32`、`g=floor(l/32)`、`q in [0,4)`、`p in [0,4)`，则

```text
token = token_base + r
V     = value_base + 8*q + 4*g + p
```

因此一个 lane 的每个 `q` 是一个连续 `T1 x V4` BF16 packet；64 个 wave-0
lane 的四个 packet 正好覆盖一个 `T32 x V32` tile。wave 1 仍只产生另一个 K64
pred partial。wave 0 从 `pred_partial[0]` 和 `pred_partial[1]` 读取这四个 FP32
值，做 vector-4 corrected/BF16 rounding，随后把四个 BF16 pack 为两个 `u32`。
`v_decay_stage[0, V, T]` 的写入仍是既有 update consumer mapping。

## 实现与第一次 ISA 问题

新增文件：

- `repro_qwen_gdn_persistent_recurrence_r4_tail_issue_state_kv.py`
- `run_qwen_gdn_persistent_recurrence_r4_tail_issue_state_kv_pred_mn_swap_v4_io.py`

初始 raw-buffer 调用将 varying `io_offset` 放在 `soffset`。最终 ISA 虽有
`buffer_load/store_dwordx2`，但每次访问是
`v_readfirstlane -> v_cmp_eq -> s_and_saveexec -> buffer_* -> s_xor_b64 exec`
的地址分组循环，不符合候选要求。

修正后把 `io_offset` 作为 raw-buffer `vindex`、`soffset=0`。这没有改变 packet
ownership，只修正 AMDGPU raw-buffer 的地址操作数位置。

## production `emit_audit=False` 制品（修正后）

- T=64，public-shaped body，grid `(32,1,1)`，WG `(128,1,1)`。
- HSACO：`r4_tail_state_kv_pred_mn_swap_v4_io_artifacts_t64_v3/hsaco/_state_kv_kernel.hsaco`
- SHA256：`32e55a53719c5e9ed90e11fc1d52febe4934e74b4b8ee4ee4e909c1a7aa1275a`
- post-opt LLVM 直接含 `llvm.amdgcn.raw.buffer.load/store.v2i32`，其中 varying
  byte address 是第二个（`vindex`）实参，第三个 `soffset` 为零。
- 最终 ISA 有 8 个静态 `buffer_load_dwordx2` 和 8 个静态
  `buffer_store_dwordx2`：两个 T32 token tile 各四个 V4 packet。第一个 token
  tile 的 load/store PC 对为
  `0x37c4/0x393c`、`0x3944/0x3a58`、`0x3a60/0x3c38`、`0x3c40/0x40a8`。
  这些指令形式均为 `... v<index>, s[rsrc], 0 offen`，没有紧邻的
  `v_readfirstlane` 或按不同地址反复 `saveexec/xor` 回边。
- `ds_bpermute`/`ds_permute`：0。
- `global_store_short` 仍存在，但位于本轮明确未改的 H/state ABI write 区间，
  而不是 U/v_new b64 区间。
- 资源：LDS 53,248 B；private 0 B；VGPR 228；AccVGPR 64；SGPR 31；VGPR/SGPR
  spill 均为 0。

制品中的 MLIR、LLVM、link input 和 ISA 位于同一 artifact 目录；编译日志继续
确认命中 `gfx942_bt64_bv32_joint_v4_tail_issue` complete recurrence lowering 和
`specialized operand=persistent_typed_block` block-dot 路径。

## 单 MFMA32 microscope 与修复

新增 `repro_qwen_mfma32_pred_mn_swap_microscope.py`。它以一个 wave、一个
`v_mfma_f32_32x32x8_bf16`、可识别 BF16 source tag，回收每 lane 的两个 BF16x4
fragment 和所有 16 个 FP32 accumulator。256 个 arg0 source slot 全部通过。

实测 primitive 的物理参数顺序是 **B,A**：令 `l` 为 lane、`r=l mod 32`、
`g=floor(l/32)`、`p in [0,4)`、`i in [0,16)`，则

```text
arg0[l,p] = B[K=4g+p, N=r]       # state-KV
arg1[l,p] = A[M=r, K=4g+p]       # W
acc[l,i]  = C[M=r, N=8*(i//4)+4g+(i%4)]
```

旧 direct pred 错把 `W` 放在 arg0、`state-KV` 放在 arg1。修复仅改为
`mfma(state_B_frag, w_A_frag, pred_acc)`，同时保留原有 typed-W
`k_vec=2*kpack+lane_group` 和两个固定 BF16x4 half 的 K traversal；没有改变
MFMA 数量、形状、LDS 或 recurrence schedule。修复后 production ISA 的 MFMA32
静态数仍为 48（修复前后相同）。

## 正确性：通过

全 nonzero-W P2 gate：

| 配置 | T=64 | T=128 |
| --- | --- | --- |
| pred-M/N swap，`pred_mn_v4_io=False` | 通过；host pred_f32 byte-equal | 通过；host pred_f32 byte-equal |
| pred-M/N swap，`pred_mn_v4_io=True` | 全阶段通过 | 全阶段通过 |

两种长度的 host microscope 对 `h/pred_f32/pred_bf16/v_new/v_decay/state_after/final_state`
均 byte-equal。device contract 的 `pred_f32` 最大差为 `9.313225746e-10`，T=128
final-state 最大差为 `5.587935448e-09`，均在公开 P2 容差内；BF16 `v_new` 与
`v_decay` 保持 byte-equal。

## 决策

该候选现在是正确性 **GO**：pred M/N producer、T1xV4 b64 U/v_new 和未改的 update
在同一 full recurrence 中已通过。按本轮范围未采集 PMC 或 Eager benchmark，因此本报告
不做性能收益结论；后续性能工作必须以 v3 同源 HSACO 为基准。
