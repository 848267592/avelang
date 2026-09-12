# Qwen v29 Broad/Compact MIR 压力时间线审计

## 范围与结论边界

这是 Experiment 0：没有改 kernel 源码、compiler pass、launch 或数值路径。它比较两份 exact full-v29 的 `IR Dump After Greedy Register Allocator`：原始 broad-K 路径与 compact producer/consumer rewrite。

已知的最终资源事实不由本脚本重新估算：broad 为 `AccVGPR=264`、`scratch=0`；compact 为 `AccVGPR=384`、`scratch=736 B`。compact 的 post-greedy MIR 中有 190 个 AV spill words （70 个 AV32 save + 60 个 AV64 save）。

MIR 没有保留 LiveIntervals 或 source debug location。因此本文的逐指令压力是“虚寄存器在 MIR 文本中首次定义到最后出现”的静态代理，而不是硬件物理寄存器压力的直接读数。`areg_*` 计作 AGPR-class；`av_*` 是可分配的 flexible class，单独显示，绝不冒充确定的物理 AGPR。

phase 标签完全由机器锚点得到：第一/最后一条 MFMA32、两类 MFMA 之间的 barrier、第一/最后一条 MFMA16。它们是可复现的 disassembly 区间，不是因 LTO 缺失 debug location 而无法证明的精确 AveLang 源代码行映射。

## 分阶段峰值

| phase | broad VGPR-class peak | compact VGPR-class peak | broad AGPR-class peak | compact AGPR-class peak | broad flexible-AV peak | compact flexible-AV peak | broad address peak | compact address peak | broad fragment peak | compact fragment peak | broad pred-acc peak | compact pred-acc peak | broad update-acc peak | compact update-acc peak |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| prelude | 306 | 286 | 0 | 0 | 116 | 232 | 356 | 365 | 0 | 0 | 0 | 0 | 0 | 0 |
| pred_mfma32 | 255 | 231 | 16 | 16 | 8 | 178 | 198 | 293 | 0 | 0 | 16 | 16 | 0 | 0 |
| pred_epilogue | 88 | 64 | 16 | 16 | 0 | 128 | 62 | 179 | 0 | 0 | 16 | 16 | 0 | 0 |
| k_producer | 89 | 65 | 0 | 0 | 0 | 128 | 63 | 180 | 0 | 0 | 0 | 0 | 0 | 0 |
| update_prologue | 11 | 11 | 4 | 4 | 0 | 0 | 5 | 5 | 2 | 2 | 0 | 0 | 4 | 4 |
| update_mfma16 | 515 | 515 | 4 | 4 | 0 | 0 | 8 | 8 | 256 | 256 | 0 | 0 | 4 | 4 |
| state_writeback | 522 | 522 | 4 | 4 | 0 | 0 | 1 | 1 | 256 | 256 | 0 | 0 | 4 | 4 |

最强的差异出现在 update 之前：compact 的 flexible-AV 词数在 `pred_mfma32` 为 178，而 broad 只有 8；`pred_epilogue` 与 `k_producer` 都是 128 对 0。同一张表还显示 pred accumulator（16 对 16）、update accumulator（4 对 4）和 update 阶段 fragment 代理（256 对 256）没有差异。这不支持“MFMA16 geometry 或最终 B fragment 本身变大”作为该 broad/compact 差异的解释。完整逐行数据在 `phase_comparison.csv`、`broad_timeline.csv`、`compact_timeline.csv`。

## 首个可见的 RA 危险点

compact 的第一条 AV spill 在 MIR line `247` / byte `2952`：`%6842`（`vreg_64_align2`，2 words），属于 `prelude`。

它的文本区间为 `240` 到 `248`，跨度仅 `8` 行；定义是 `undef %6842.sub0:vreg_64_align2 = nsw V_SUB_U32_e32 %907:vgpr_32, %911:vgpr_32, implicit $exec`。

它位于第一条 pred MFMA32 之前，且自身是短寿命地址分量。因此不能把它误解成“该 vreg 跨过 pred/update”。这份 post-greedy dump 只能严格证明：Greedy 在 compact prelude 已经开始插 AV spill；它不能单独给出物理峰值的因果归属。

## compact 的长寿命地址/fragment 候选

| vreg | class | spill words | first line | last line | span | def opcode | address | fragment | crosses pred/update |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| %164 | vgpr_32 | 0 | 1391 | 6573 | 5182 | V_LSHL_OR_B32 | True | False | True |
| %167 | vgpr_32 | 0 | 1441 | 6574 | 5133 | V_OR_B32 | True | False | True |
| %166 | vgpr_32 | 0 | 1440 | 6250 | 4810 | V_OR_B32 | True | False | True |
| %165 | vgpr_32 | 0 | 1439 | 5926 | 4487 | V_OR_B32 | True | False | True |
| %4382 | vgpr_32 | 0 | 1382 | 5622 | 4240 | V_ADD_U32 | True | False | True |
| %162 | vgpr_32 | 0 | 1436 | 3999 | 2563 | V_ADD_U32 | True | False | False |
| %175 | vgpr_32 | 0 | 1451 | 3740 | 2289 | V_LSHL_OR_B32 | True | False | False |
| %7449 | av_64_align2 | 0 | 1720 | 3996 | 2276 | COPY | True | False | False |
| %174 | vgpr_32 | 0 | 1450 | 3724 | 2274 | V_LSHL_OR_B32 | True | False | False |
| %7452 | av_64_align2 | 0 | 1723 | 3992 | 2269 | COPY | True | False | False |
| %7455 | av_64_align2 | 0 | 1726 | 3988 | 2262 | COPY | True | False | False |
| %173 | vgpr_32 | 0 | 1449 | 3708 | 2259 | V_LSHL_OR_B32 | True | False | False |
| %7460 | av_64_align2 | 0 | 1731 | 3984 | 2253 | COPY | True | False | False |
| %7464 | av_64_align2 | 0 | 1734 | 3980 | 2246 | COPY | True | False | False |
| %172 | vgpr_32 | 0 | 1448 | 3692 | 2244 | V_LSHL_OR_B32 | True | False | False |
| %7471 | av_64_align2 | 0 | 1737 | 3976 | 2239 | COPY | True | False | False |
| %7475 | av_64_align2 | 0 | 1740 | 3972 | 2232 | COPY | True | False | False |
| %171 | vgpr_32 | 0 | 1447 | 3676 | 2229 | V_LSHL_OR_B32 | True | False | False |
| %7479 | av_64_align2 | 0 | 1743 | 3968 | 2225 | COPY | True | False | False |
| %7487 | av_64_align2 | 0 | 1746 | 3964 | 2218 | COPY | True | False | False |
| %170 | vgpr_32 | 0 | 1446 | 3660 | 2214 | V_LSHL_OR_B32 | True | False | False |
| %7492 | av_64_align2 | 0 | 1749 | 3960 | 2211 | COPY | True | False | False |
| %7501 | av_64_align2 | 0 | 1752 | 3956 | 2204 | COPY | True | False | False |
| %7506 | av_64_align2 | 0 | 1757 | 3952 | 2195 | COPY | True | False | False |

## pred 阶段 flexible-AV 峰值

compact 的 pred-MFMA32 flexible-AV 文本峰值在 MIR line `3047` / byte `101904`，为 `178` words。其中 `144` words 是地址标记链，而短寿命 `DS_READ` pred 输入只有 `32` words。活跃对象完整清单在 `compact_pred_flexible_av_peak_live_vregs.csv`；下面列出跨度最长的对象。

| vreg | class | words | first | last | span | def opcode | address | fragment |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| %3352 | areg_512_align2 | 16 | 3047 | 3070 | 23 | V_MFMA | False | False |
| %3316 | av_128_align2 | 4 | 3042 | 3054 | 12 | DS_READ | False | False |
| %3315 | av_128_align2 | 4 | 3041 | 3052 | 11 | DS_READ | False | False |
| %3314 | av_128_align2 | 4 | 3040 | 3050 | 10 | DS_READ | False | False |
| %3313 | av_128_align2 | 4 | 3039 | 3048 | 9 | DS_READ | False | False |
| %3325 | av_128_align2 | 4 | 3046 | 3054 | 8 | DS_READ | False | False |
| %3324 | av_128_align2 | 4 | 3045 | 3052 | 7 | DS_READ | False | False |
| %3323 | av_128_align2 | 4 | 3044 | 3050 | 6 | DS_READ | False | False |
| %3322 | av_128_align2 | 4 | 3043 | 3048 | 5 | DS_READ | False | False |
| %5792 | vreg_64_align2 | 2 | 1155 | 3559 | 2404 | AV_MOV_B32_IMM_PSEUDO | True | False |
| %7449 | av_64_align2 | 2 | 1720 | 3996 | 2276 | COPY | True | False |
| %7452 | av_64_align2 | 2 | 1723 | 3992 | 2269 | COPY | True | False |
| %7455 | av_64_align2 | 2 | 1726 | 3988 | 2262 | COPY | True | False |
| %7460 | av_64_align2 | 2 | 1731 | 3984 | 2253 | COPY | True | False |
| %7464 | av_64_align2 | 2 | 1734 | 3980 | 2246 | COPY | True | False |
| %7471 | av_64_align2 | 2 | 1737 | 3976 | 2239 | COPY | True | False |
| %7475 | av_64_align2 | 2 | 1740 | 3972 | 2232 | COPY | True | False |
| %7479 | av_64_align2 | 2 | 1743 | 3968 | 2225 | COPY | True | False |
| %7487 | av_64_align2 | 2 | 1746 | 3964 | 2218 | COPY | True | False |
| %7492 | av_64_align2 | 2 | 1749 | 3960 | 2211 | COPY | True | False |

## 对最终 B fragment 的负控制

broad 的第一条 update MFMA16 B operand 是 `%3690`，在 MIR line `2907` 定义、`2908` 使用；compact 对应为 `%4300`，在 `4018` 定义、`4019` 使用。两者都是一行 def-use 的 `DS_READ_B64` handoff。这与既有 strict terminal-load A/B 的“post-opt LLVM 和 ISA 完全相同、资源不变”结果一致：只替换最终 generic/direct LDS load 不能修复 full-v29 cliff。

## 可行动的中间结论

机器文本已直接显示 compact 独有的一批 `V_LSHL_ADD_U64 -> COPY av_64` 链：例如 `%7449` 在 line 1720 创建，跨越整个 pred MFMA32 区间，直到 line 3996 才重新被 COPY 后进入后续 `V_LSHL_ADD_U64` / global load / LDS write 链。这类对象使 compact 在 pred 阶段多出 170 flexible-AV words，并在 pred epilogue/K producer 仍多出 128 words。所以 Experiment 1 应只测试“将这些 address tuple 延后到紧邻其 global/LDS consumer 创建”，并保留 MFMA、LDS write 数、tile ownership、barrier 和数学不变。它必须预注册 post-opt LLVM、pre-RA MIR、def-use 距离与峰值 flexible-AV 的下降 gate。

这仍不是“已证明 Avelang backend 必然错误”的铁证：broad/compact 的上层 producer graph 本来不同，且本审计使用静态代理。它是对下一条严格 single-variable late-address 实验的具体、可复查定位。

## 复现

```bash
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/analyze_qwen_v29_broad_compact_pressure_timeline.py \
  --report test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_v29_broad_compact_pressure_timeline_report.md
```

派生产物写入 `compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_v29_broad_compact_pressure_timeline/`：逐指令 CSV、phase 比较、首个 spill 上下文、pred flexible-AV peak live-set 与完整 JSON 都保留在该目录。
