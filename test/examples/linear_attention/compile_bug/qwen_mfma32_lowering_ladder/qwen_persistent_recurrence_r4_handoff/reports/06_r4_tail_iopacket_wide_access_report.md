# Qwen R4-tail：I/O packet ownership / wide global access

## 结论

**No-Go。** `gfx942_bt64_bv32_joint_v4_tail_issue_iopacket` 真实生成了要求的
`b64`/`b128` I/O 指令，也消除了所有 `global_store_short`；但它把每个 physical
V32/chunk 的 VMEM 从 197.5 增至 1577.5，long-sequence slope 从 10.60 增至
35.03 us/chunk。因此不能合入或替代 R4-tail。

这不是 “ISA 没有变化” 的 No-Go，也不是把工作转移到 LDS：最终 ISA、final-isel
MIR、PMC 和完整 ATT 都显示宽访问在运行；LDS 指令反而减少 100/V32。真正的问题是
该 raw-packet ownership 在完整 recurrence 的 H snapshot 与 U/v_new 路径上产生了
大量动态宽访问、BF16-to-u32 打包和 exec 控制。

## 同源 production 制品

| 项目 | 值 |
|---|---|
| 基线 plan | `gfx942_bt64_bv32_joint_v4_tail_issue` |
| 候选 plan | `gfx942_bt64_bv32_joint_v4_tail_issue_iopacket` |
| 形状 | B=1, Hk=4, Hv=8, K=128, V=128, BF16, BT=64, BV=32, nonzero-W |
| production 入口 | public Eager recurrence API，`emit_audit=False` |
| T=2048 launch | logical grid=32；HSA global work-items=4096；WG=128；32 workgroups |
| 候选 HSACO SHA256 | `14c4745cd5b48ee637e6443dbc17b91dedb0c91942481b16e0b0c491e5376409` |
| 基线 HSACO SHA256 | `5ebff98fd16fe1fa47501fbeb7dd4bd738f716d6445c0d0fd52d71e0621740c7` |

ATT dispatch 14 的 code object SHA256 与候选 production HSACO 完全相同。它包含两个
完整 wave 的 decoder JSON；将一 workgroup 的 hitcount 乘 32 后，与同一 public-Eager
production dispatch 的 PMC 在 VMEM、LDS、MFMA 和 VALU 上逐项完全闭合。

`post_block_dot_lowering.mlir` 仍带有
`avelang.block_dot.lowering = "specialized"` 和
`operand_lowering = "persistent_typed_block"`，因此没有 generic block-dot fallback。

制品目录：
`test/examples/linear_attention/rocprof_outputs/qwen_r4_tail_iopacket_t2048/`。

## 实现范围

只为同一个 full recurrence op 增加 constexpr 控制的 I/O ownership 分支：

- H pre-update snapshot：四个相邻 BF16 accumulator 值组装为两个 `u32`，并以
  `raw_buffer_store_x2`（64 bit）写回；保留给 pred 的同一 `state_bf16` 写入。
- U/v_new：`tid -> (token_off, local_v)` 拥有连续 V8 packet，以
  `raw_buffer_load_x4` / `raw_buffer_store_x4`（128 bit）访问；`g` 的一个标量读取在
  V8 内复用，而不伪造跨 head 的宽 g load。
- W/K producer、single-bank K LDS retile、BT64/BV32/WG128、tail issue、MFMA 几何、
  BF16 boundary、FP32 feedback 都没有改动。

实现位置是
`repro_qwen_gdn_persistent_recurrence_r4_joint_v4.py` 的 H 分支（约 197--244 行）和
U/v_new 分支（约 293--333 行）。候选入口为
`repro_qwen_gdn_persistent_recurrence_r4_tail_issue_iopacket.py`。

## 最终代码生成

final-isel MIR 明确包含 `BUFFER_STORE_DWORDX2_OFFSET_exact`（s64）以及
`BUFFER_LOAD_DWORDX4_OFFSET` / `BUFFER_STORE_DWORDX4_OFFSET_exact`（s128）。最终 ISA
的实际 PC 是：

| 路径 | ISA PC | 实际指令 |
|---|---:|---|
| H snapshot | `0x29b8..0x306c` | 8 × `buffer_store_dwordx2` |
| U packet | `0x3168`, `0x3714` | 2 × `buffer_load_dwordx4` |
| v_new packet | `0x3618`, `0x3bbc` | 2 × `buffer_store_dwordx4` |

| 静态 ISA 项 | R4-tail | I/O candidate |
|---|---:|---:|
| `global_store_short` | 80 | 0 |
| `buffer_store_dwordx2` | 0 | 8 |
| `buffer_load_dwordx4` | 0 | 2 |
| `buffer_store_dwordx4` | 0 | 2 |
| `s_and_saveexec_b64` | 38 | 35 |
| `s_xor_b64` | 52 | 50 |
| `s_andn2_saveexec_b64` | 16 | 0 |
| `s_or_b64` | 34 | 19 |
| 静态 `v_mfma` sites | 48 | 48 |

所以“宽访问是否真的产生”和“窄 H store / 部分 mask 是否删除”两项都通过；失败的是
动态工作量，而不是静态选择失败。

## Correctness、资源和动态机器工作

T=64/128/512/2048 full nonzero-W P2 correctness 全部通过。`h`、pred FP32/BF16、
`v_new`、`v_decay`、每 chunk state 和 final state 都 byte-equal。production HSA
metadata 的 private segment 为 0；final-isel MIR 无 `spill` / `reload`；PMC 的
Scratch_Size 也为 0。

PMC 的归一化分母是 `32 workgroups × 32 chunks = 1024 physical V32/chunk`：

| 指标 / V32/chunk | R4-tail | I/O candidate | 候选 - 基线 |
|---|---:|---:|---:|
| MFMA | 64.0 | 64.0 | 0 |
| VMEM | 197.5 | 1577.5 | +1380.0 |
| VALU（含 MFMA） | 2187.375 | 4821.688 | +2634.312 |
| SALU | 159.25 | 3166.25 | +3007.0 |
| LDS | 373.0 | 273.0 | -100.0 |
| VGPR（rocprof record） | 128 | 64 | -64 |
| AccVGPR（rocprof record） | 160 | 200 | +40 |
| SGPR | 112 | 112 | 0 |
| Scratch / LDS bytes | 0 / 53248 | 0 / 53248 | 0 / 0 |
| 候选 occupancy | — | 0.650 (five dispatches stable) | 无观察到 occupancy cliff |

动态 MFMA 仍为 64/V32/chunk，满足硬约束；不过候选并没有减少机器工作。LDS 下降也证明
这不是额外 transpose 或 LDS round-trip 所致。

### ATT 与 PMC 闭合

ATT 对同一 HSACO 的完整 workgroup trace 按 32 CTA 缩放后：

| 指标 | ATT 缩放值 / dispatch | PMC / dispatch |
|---|---:|---:|
| VMEM | 1,615,360 | 1,615,360 |
| LDS | 279,552 | 279,552 |
| MFMA | 65,536 | 65,536 |
| VALU | 4,871,872 + 65,536 MFMA = 4,937,408 | 4,937,408 |
| SALU | 3,242,240 | 3,242,240 |

ATT 的原始所有 `s_*` PC visit 为 8,228,544；`SQ_INSTS_SALU` 的定义不计入
`s_nop`、branch、`s_waitcnt`、`s_barrier`、SMEM load 和 `s_endpgm`。剔除这些后正好为
3,242,240，故 SALU 也严格闭合，而不是把 trace 的控制流 visit 误当作 PMC SALU。

### 宽 I/O 的动态归因

ATT 中的三个 raw-packet bucket 合计 1,572,864 VMEM/dispatch，即候选全部
1,615,360 VMEM 的 **97.37%**：

| bucket | PC | 动态 VMEM / dispatch | VMEM / V32/chunk | 占候选 VMEM |
|---|---|---:|---:|---:|
| H b64 store | `0x29b8..0x306c` | 1,048,576 | 1024 | 64.91% |
| U b128 load | `0x3168`, `0x3714` | 262,144 | 256 | 16.23% |
| v_new b128 store | `0x3618`, `0x3bbc` | 262,144 | 256 | 16.23% |
| 其他所有 VMEM | — | 42,496 | 41.5 | 2.63% |

H packet PC 区间还执行了大量 packing 与 exec 控制（ATT 缩放：
`s_and_saveexec_b64` 1,181,696、`s_xor_b64` 1,052,672、`s_cbranch_execnz`
1,048,576；原始 trace `s_nop` 2,392,064）。U/v_new 区间又有
`s_and_saveexec_b64` 393,216、`s_xor_b64` 524,288 和
`s_cbranch_execnz` 524,288。静态 mask site 虽减少，但 packetized loop 的动态
谓词/pack 开销远大于被删除的 short store path。

## Public-Eager fresh-process body benchmark

两个独立 session、每 implementation/长度均 fresh process、warmup=5、repeat=20；不使用
graph capture 或 private HSACO launch：

| T | current vLLM (ms) | R4-tail (ms) | I/O candidate (ms) | candidate / R4-tail |
|---:|---:|---:|---:|---:|
| 1024 | 0.105007 | 0.221981 | 0.590229 | 2.659x |
| 2048 | 0.154640 | 0.380547 | 1.141770 | 3.000x |
| 8192 | 0.454106 | 1.409179 | 4.513572 | 3.203x |

1024→8192 body slope：vLLM 3.117 us/chunk、R4-tail 10.600 us/chunk、候选
35.030 us/chunk。候选的回退随序列增长而恶化，排除了固定启动成本解释。

## 决策

保留候选和制品作为反例与可复现证据，但不启用该 plan。下一轮不得在这条 raw I/O packet
ownership 路径上继续微调宽度、mask 或地址；在不改变 full-op ownership 语义前，它已经违反
“机器工作必须实际下降”的门槛。
