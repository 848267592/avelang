# Qwen R4-tail vs current-vLLM：同源完整 ATT 的逐 bucket 动态差额

日期：2026-08-04
范围：只做 production 归因。**没有修改** layout、lowering、planner、kernel 或数学；没有创建性能候选。

## 结论

current-vLLM 的 production wave trace 已完整：decoder `status=0`、`INFO=[]`，选中的 code object 有 8 个完整 wave，即 WG128 下的 4 个完整 workgroup。将其乘以 `32 / 4 = 8` 后，VMEM、VALU、SALU、LDS、MFMA 与 32-CTA public-Eager PMC **逐项精确闭合**。

R4-tail 相比 current-vLLM 的 T=2048、每 physical V32/chunk 差额是：

- VMEM `+140.5`：`U/g load` 为 `+58.125`（41.4%），`H/v_new store` 为 `+82.0`（58.4%）；二者合计 `+140.125`，即 **99.7%**。
- VALU `+824.5`：最大单一正差额 bucket 是 `update operand preparation` 的 `+490.0`（59.4%）；其次是 `tail/loop W-K` 的 `+373.5`。R4 较小的 prologue 以及略少的 BF16/pred 工作抵消了其中一部分，所以这些正项之和可以大于净差额。
- SALU `+95.9375`：`loop/control` 自身为 `+131.0625`。其中 `128.0` 条来自 R4 每个 V32/chunk 反复执行 16 组输出-mask `saveexec/xor/andn2/or` 序列；这是该 bucket 差额的 97.7%。
- MFMA 总数同为 `64`，没有把差距归因于额外 MFMA。

因此下一轮只应实现一个机器路径：**full-op I/O packet ownership / wide global access，同时覆盖 U/g 输入和 H/v_new 输出。**不能误做 K-LDS layout、重新开启 core-lastuse，或只改某一条 `ds_read`。update-fragment 的 `+490 VALU` 是已证实的第二大机器工作差额，但不是本轮由 VMEM 压倒性主导时应优先改的路径。

## 同源性与完整 trace

两端均由公开 Eager API 驱动，T=2048、B=1、Hk=4、Hv=8、K=V=128、BF16、BT64、BV32、WG128、32 CTA：

| 实现 | public-Eager symbol | production HSACO SHA-256 | grid / workgroup |
| --- | --- | --- | --- |
| R4-tail | `_qwen_gdn_persistent_recurrence_r4_joint_v4_kernel` | `5ebff98fd16fe1fa47501fbeb7dd4bd738f716d6445c0d0fd52d71e0621740c7` | 4096 work-items / 128 = 32 CTA |
| current-vLLM | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | `632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e` | 4096 work-items / 128 = 32 CTA |

current-vLLM HSACO 的 hash 与现有 `qwen_gdn_bt64_bf16_recurrence_full_stage6s.py` contract 固定值一致。ATT 从 public-Eager `current_vllm_eager` worker 采集；autotune 期间 trace 文件中还出现其他 code object，但只把 ID 5（上述 SHA）的执行 PC 归入本报告。唯一未映射的 ID 0 记录是 host address，未进入选中 kernel 的计数；decoder 仍返回成功且无 INFO。

完整 current-vLLM raw trace：

```text
test/examples/linear_attention/rocprof_outputs/
qwen_current_vllm_t2048_att_20260804/c100/trace/0364d3a007f9/
758872_17843_shader_engine_0_14.att
```

选中 wave 是 `(CU=1, SIMD=1/2/3, wave=0)` 的 8 个完整 record，各含 31,060 条 instruction record；对应 4 个完整 WG。没有 `Wave incomplete`。

## ATT 与 PMC 闭合

ATT 的 executed-PC category 口径为：`FLAT+VMEM -> VMEM`、`VALU`、`SALU`、`LDS`；`v_mfma` 另列为 MFMA，且仍包含在 VALU 内。缩放是 `8`，没有用静态 ISA site 乘循环 trip count。

| metric | ATT：4 WG 原始 | ATT：缩放 32 CTA | current-vLLM PMC | delta |
| --- | ---: | ---: | ---: | ---: |
| VMEM | 7,296 | 58,368 | 58,368 | 0 |
| VALU | 174,448 | 1,395,584 | 1,395,584 | 0 |
| SALU | 8,104 | 64,832 | 64,832 | 0 |
| LDS | 38,184 | 305,472 | 305,472 | 0 |
| MFMA | 8,192 | 65,536 | 65,536 | 0 |

PMC 的 current-vLLM resource record 是 VGPR=104、AccVGPR=160、SGPR=96、LDS=0、scratch=0；这与 trace 的 kernel symbol、4096 global work-items 和 WG128 一致。R4 的同源完整 ATT/PMC 闭合见已有 [R4 production ATT 报告](qwen_persistent_recurrence_r4_tail_production_att_attribution_report.md)。

## 共同 recurrence bucket 的 PC 归属

current-vLLM 的 TTGIR 在 `kernel.ttgir:328` 开始单一 `scf.for` recurrence。下面的 PC 划分采用该 loop 的真实次序，而非静态 ISA 密度：

- `H` stores：TTGIR 331–343，对应 loop store PCs `0x2be0/0x2c24/0x2eb0/0x2ef0`；`v_new` store 是 TTGIR 406–414，对应 `0x35b0/0x3644/0x36e0/0x377c`。
- next-W：TTGIR 345–387；global `buffer_load_dwordx4` PCs `0x2fdc–0x2ffc` 与 `0x31e0–0x3200`，以及它们的 `0x2f04–0x30a0` address/LDS-commit block。
- pred：TTGIR 376–393；MFMA cluster `0x30a0–0x3374`。
- U/g/BF16：U load TTGIR 394–406、PC `0x33a4/0x33b0`；g loads TTGIR 422–435、PC `0x3788/0x37b8`；它们之间的 non-memory code 是 `0x33b4–0x3b2c` BF16 bridge。
- next-K/update：TTGIR 448–487；K global `buffer_load_dwordx4` PCs `0x3b2c/0x3b64/0x3b74/0x3b84/0x3cd8/0x3cec/0x3d18/0x3e3c`，update MFMA cluster `0x3ca0–0x406c`。
- state feedback：`v_accvgpr_read_b32` + `v_pk_fma_f32` PC ranges `0x3d6c–0x3e24` and `0x40b8–0x4174`; terminal state stores `0x5ce0–0x5e6c` are assigned here rather than mislabeled as H/v_new.

这使 W/K 的 next-packet memory issue 与 pred/update consumer prep 分开，即使 K commit 和 update MFMAs 在后端做了交错。R4 采用前一份报告已经固定的同名 phase selection；两端的 per-V32/chunk 都以 `32 chunks × 32 physical V32 tiles = 1,024` 归一化。

## 逐 bucket 的动态工作与 R4-current-vLLM 差额

每个单元格顺序为 `VMEM / VALU / SALU / LDS / MFMA`，均为每 physical V32/chunk。`delta = R4-tail - current-vLLM`。

| bucket | R4-tail | current-vLLM | delta |
| --- | ---: | ---: | ---: |
| W/K prologue（含 initial state） | 1.5 / 14.625 / 2.125 / 1 / 0 | 1.625 / 229.938 / 10.75 / 17.938 / 0 | -0.125 / -215.313 / -8.625 / -16.938 / 0 |
| tail/loop W-K | 32 / 420 / 18 / 32 / 0 | 31 / 46.5 / 0 / 48.438 / 0 | +1 / +373.5 / +18 / -16.438 / 0 |
| U/g load | 66 / 0 / 0 / 0 / 0 | 7.875 / 0 / 0 / 0 / 0 | **+58.125 / 0 / 0 / 0 / 0** |
| H/v_new store | 98 / 0 / 0 / 0 / 0 | 16 / 0 / 0 / 0 / 0 | **+82 / 0 / 0 / 0 / 0** |
| pred operand preparation | 0 / 88 / 0 / 24 / 16 | 0 / 96.875 / 11.625 / 75.562 / 31 | 0 / -8.875 / -11.625 / -51.562 / -15 |
| BF16 bridge | 0 / 486 / 0 / 46 / 0 | 0 / 503.75 / 23.25 / 7.75 / 0 | 0 / -17.75 / -23.25 / +38.25 / 0 |
| update operand preparation | 0 / 614 / 2 / 138 / 32 | 0 / 124 / 15.5 / 124 / 31 | 0 / **+490** / -13.5 / +14 / +1 |
| state feedback / terminal state | 0 / 176 / 4 / 68 / 16 | 0.5 / 98 / 0.125 / 2.938 / 2 | -0.5 / +78 / +3.875 / +65.062 / +14 |
| loop/control | 0 / 388.75 / 133.125 / 64 / 0 | 0 / 263.813 / 2.063 / 21.688 / 0 | 0 / +124.938 / **+131.063** / +42.313 / 0 |
| **sum** | **197.5 / 2,187.375 / 159.25 / 373 / 64** | **57 / 1,362.875 / 63.313 / 298.313 / 64** | **+140.5 / +824.5 / +95.938 / +74.688 / 0** |

MFMA 的 pred/update 分配不同，来自 vLLM 将 31 个 loop iteration 与 final epilogue replay 分开：loop pred 和 update 各为 31，终态 feedback/epilogue 合计 2；总数仍严格是 64。因此不能把单个 MFMA subrange 的不同布局误写成总 MFMA 差额。

## VMEM：U/g 与 H/v_new 是几乎全部的净差额

current-vLLM 的 U/g 动态 VMEM 为 7.875：loop 中两个 `buffer_load_dwordx4` U site（`0x33a4, 0x33b0`）和两个 `global_load_dword` g site（`0x3788, 0x37b8`）各为 1.9375/V32-chunk，另有 two one-time g setup loads（`0x2018, 0x2070`）合计 0.125。R4 的同一 bucket 是 66 个窄 global accesses，故差额为 58.125。

current-vLLM 的 H/v_new 是 16 个 VMEM/V32-chunk：四个 loop `buffer_store_dwordx4` H sites、四个 `buffer_store_dwordx2` v_new sites 以及 0.5 的终态 H/v_new store。R4 的 98 则是 `global_store_short`、`global_store_short_d16_hi` 和少量 dword terminal store 的组合，故差额为 82。final-state 的另 0.5 个 vLLM VMEM 已放在 state-feedback 行，避免夸大 H/v_new。

| VMEM net delta source | R4 | current-vLLM | delta | total +140.5 中占比 |
| --- | ---: | ---: | ---: | ---: |
| U/g | 66 | 7.875 | +58.125 | 41.37% |
| H/v_new | 98 | 16 | +82 | 58.36% |
| **上述 I/O 合计** | **164** | **23.875** | **+140.125** | **99.73%** |

余下 0.375 是 W/K prologue、tail 和 terminal state 的小幅互相抵消；它不会改变结论。

## VALU 与 SALU 的实际来源

VALU 的最大正差额不是猜测：R4 的 `update operand preparation` 为 614，而 current-vLLM 为 124，净 `+490`。该 bucket 覆盖 R4 specialized block-dot update fragment/operand path；current-vLLM 对应 TTGIR 448–487、ISA `0x3ca0–0x4178`。R4 `tail/loop W-K` 另有 `+373.5`，而 current-vLLM 以 46.5 VALU 处理其 fused next-packet issue。二者都是真实候选问题，但在本轮优先级中应排在 U/g + H/v_new I/O 后面。

SALU 则有具体、可复现的 R4 根因。R4 loop/control 的 133.125 中有 **128.0** 条/V32-chunk 来自 16 个重复的 mask/reconvergence group；每组四条 category-2 SALU、每条动态执行两次：

```text
0x2a40 ... 0x36f8（16 个同构 group）
s_and_saveexec_b64 s[8:9], vcc
s_xor_b64          s[10:11], exec, s[8:9]
s_andn2_saveexec_b64 s[10:11], s[10:11]
s_or_b64           exec, exec, s[10:11]
```

current-vLLM 的 loop/control SALU 仅 2.0625，所以该组本身解释 `128 / 131.0625 = 97.7%` 的 loop/control SALU 差额。附近的 `s_cbranch_execz` 是 JUMP category，未被错误计入 SALU；其余 5.125 是较少的终态 scalar/mask work。

## 决策

本轮不编码。下一次只修改 **U/g + H/v_new 的同一个 full-op I/O packet path**：建立一致的 lane ownership、地址计算和宽 global load/store，以删除 R4 的窄/fractured global I/O；不得把它拆成 K-LDS 小改动或重新试验 issue distance。验收应继续要求 public-Eager、同 HSACO manifest、scratch/spill=0，以及以 PMC/ATT 证明实际下降的是上述 140.125 VMEM/V32-chunk I/O 差额。
