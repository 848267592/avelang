# R4-tail U/v_new LDS-mediated bridge 候选

## 结论

**No-Go。** 候选数学正确，并且最终 production ISA 确实将 U/v_new 的
`global_load_ushort` / `global_store_short_d16_hi` 变为 `buffer_*_dwordx4`。
但 AveLang 将 T1xV8 packet materialization 变成带 exec-mask 回边的动态
packet body：VMEM、VALU、SALU 都显著增加。这违反了本轮“不得有动态
packet loop”的硬约束；不继续扩展本候选，也不引入 D0-P/R3
`ds_bpermute` transpose。

## 范围与实现

基线是 `gfx942_bt64_bv32_joint_v4_tail_issue`，形状为
`B=1,Hk=4,Hv=8,K=128,V=128,BT=64,BV=32,WG=128,grid=32`。候选名为
`gfx942_bt64_bv32_joint_v4_tail_issue_u_vnew_lds_bridge`。

改动仅在每个 `token_tile` 的 pred-finalize 到 update 输入之间：

1. 原有 `pred_partial[2,32,32]` 两个 wave plane 已在 pred reduction 后写完。
   当前 V1xT8 owner 读两个 plane 并把和写回 plane 0；plane 1 此时已死。
2. barrier 后将既有 plane 0 用 `u32[32,4,8]` view 看作 token-major
   `[token,V8-packet,dword]` LDS bridge。这是原位复用，不增加 LDS 分配。
3. 128 个静态 `packet=tid` owner 各负责一个 T1xV8：从 bridge 用两个
   `ds_read_b128` 读 FP32 pred、从 U 用 `raw_buffer_load_x4` 读 BF16x8，
   保留 BF16 rounding boundary，以 `raw_buffer_store_x4` 写 V-new。
4. V-decay 仍写回原有 `vdecay_stage[0,V,T]` physical mapping；后续仍由
   既有 `persistent_typed_block` specialized block-dot 消费。

H、W/K typed producer、single-bank tail issue/commit、FP32 state feedback、
MFMA 几何和数量均未改动。新增 mode `2` 只被 bridge wrapper 传入；原有
mode `0`（R4）和 mode `1`（历史 iopacket）不变。

涉及文件：

- `repro_qwen_gdn_persistent_recurrence_r4_joint_v4.py`
- `repro_qwen_gdn_persistent_recurrence_r4_tail_issue_u_vnew_lds_bridge.py`
- `run_qwen_gdn_persistent_recurrence_r4_tail_issue_u_vnew_lds_bridge_body.py`

## 正确性

在 `ljd_qwen_vllm_avelang_rocm722` 内，使用 current project binding
`build-software-pipeline/python` 与公开 R4 body harness 执行：

| T | P2 host microscope | device contract | h/pred/v_new/v_decay/state/final |
|---:|:---:|:---:|:---:|
| 64 | pass | pass | 对 microscope 全部 byte-equal |
| 128 | pass | pass | 对 microscope 全部 byte-equal |

与 device contract 的 FP32 pred/state 仅有既有的 FP32 小舍入差；
`v_new`、`v_decay` 均 byte-equal，两个长度均 finite。日志同时确认：
`mode=gfx942_bt64_bv32_joint_v4_tail_issue` 和
`mode=specialized operand=persistent_typed_block`。

## Production ISA 与资源

production body 使用 `emit_audit=False`、T=2048，导出的 HSACO SHA256 为：

```text
f90951deebb4a12d2bf373e6ce38b2699864d46201ec6d55aa4adadae544c124
```

制品位于
`test/examples/linear_attention/rocprof_outputs/qwen_r4_tail_u_vnew_lds_bridge_t2048/hsaco/`。

| static ISA opcode | R4-tail | bridge | 含义 |
|:--|--:|--:|:--|
| `global_load_ushort` | 16 | 0 | U 窄读已删除 |
| `global_store_short_d16_hi` | 16 | 0 | v_new 窄写已删除 |
| `buffer_load_dwordx4` | 0 | 2 | 两个 V32 token tile 的 U b128 |
| `buffer_store_dwordx4` | 0 | 2 | 两个 V32 token tile 的 v_new b128 |
| `global_store_short` | 64 | 64 | H，按要求未修改 |
| `ds_read_b128` | 24 | 28 | bridge 额外两个 b128 read / token tile |
| `s_barrier` | 12 | 14 | partial-reduction 与 bridge-consumer 边界 |
| `v_mfma_f32_32x32x8_bf16` | 48 | 48 | MFMA 未增加 |
| `ds_bpermute` / `ds_permute` | 0 / 0 | 0 / 0 | 没有 register transpose |

HSA metadata：LDS `53,248 B`、private segment `0 B`。生产 PMC 资源为
VGPR 76、AccVGPR 188、SGPR 112、Scratch 0、`OccupancyPercent=0.643491`。

但是宽访问周围不是一次直线 packet。ISA 在第一个 token tile 的
`0x36f0..0x3b98` 中有：

```text
0x36f4  ds_read_b128
0x36fc  ds_read_b128
0x371c  buffer_load_dwordx4
0x3728  s_cbranch_execnz ... <backedge>
...
0x3b80  buffer_store_dwordx4
0x3b8c  s_cbranch_execnz ... <backedge>
```

第二个 token tile 有同类回边。即使 source 的 `packet_i` 长度是常数 8，
当前 lowering 仍使用 exec 驱动循环，而非一个无回边的静态 packet
straight-line body；因此不能把它认定为“无动态 packet loop”。

## T=2048 PMC

两者都是 `Grid_Size=4096` global work-items、WG128，即 32 CTA、每 CTA
32 recurrence chunks，总归一化分母为 1024 physical V32/chunk。R4 数值来自
同一 MI300X/gfx942 的既有 production PMC capture；bridge 为本轮
`emit_audit=False` production-body 采集。

| metric | R4 / dispatch | bridge / dispatch | R4 / V32 | bridge / V32 | delta / V32 |
|:--|--:|--:|--:|--:|--:|
| MFMA | 65,536 | 65,536 | 64.000 | 64.000 | 0 |
| VMEM | 202,240 | 632,320 | 197.500 | 617.500 | +420.000 |
| VALU | 2,239,872 | 2,774,720 | 2,187.375 | 2,709.688 | +522.312 |
| SALU | 163,072 | 1,228,480 | 159.250 | 1,199.688 | +1,040.438 |
| LDS | 381,952 | 422,912 | 373.000 | 413.000 | +40.000 |

因此，静态窄 U/v_new 访问减少并没有转化为动态 VMEM 减少；VMEM 是 R4 的
3.13 倍，且 SALU 增加约 7.53 倍。它与 ISA 的 exec backedge 一致，说明主要
成本是 packet materialization/lowering，而不是 LDS 容量、MFMA 或
`ds_bpermute`。

## 决策

关闭 U/v_new LDS-mediated bridge 路线。下一轮不应再改其 lane mapping、宽
access、D0-P/R3 shuffle 或 packet distance；应转向已经量化的
**update operand preparation** 机器工作差额。
