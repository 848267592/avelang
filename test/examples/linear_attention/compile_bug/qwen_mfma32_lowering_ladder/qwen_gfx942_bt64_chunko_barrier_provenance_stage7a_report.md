# Qwen gfx942 BT64 Stage 7A: Chunk-O Barrier Provenance And Phase Audit

## 结论

**Case C: 关闭当前 Stage 6Z source-native chunk-o 路线。**

Stage 7A 证明了两件同时成立的事：

1. Z1 的 `41` 个静态 `s_barrier` 不是 AMDGPU backend 额外保守插入的。
   它们在 source 显式 `al.syncthreads()` 的循环展开后已经存在于 pre-link
   LLVM，并且 pre-LTO assembly 与最终 HSACO ISA 都是同一个 `41`。
2. 某些局部 barrier 在最小 repro 中确实可删且 bit-exact；但按规则只做的
   一次完整 Z1 phase-compaction 在 `T=8192` 立刻失去 bit exactness。

因此，不能把最小 repro 的“same-lane fragment”结论直接推广到 full CTA MFMA
pipeline。对**当前** Avelang source schedule 来说，这些 phase boundary 是实际
需要的同步形状。不能继续删 barrier、不能改 tile/WG 扫描、不能接入 full graph。

Stage 6Z 保持 No-Go，Stage6W/X2/v24/default selector 均未改变。

## 冻结边界

本轮没有修改：

- Stage6Z Z1 kernel；
- Stage6W、Stage6X X2、v24、default selector；
- BT64/BV64/BK32、WG256、CTA mapping、MFMA32、dtype 或数学；
- recurrence ABI、full graph、compiler、LLVM/AMDGPU RA、assembly、vLLM source。

唯一新增 full-kernel source 是一个**未晋级的失败实验**：

[`qwen_gdn_bt64_native_chunko_stage7a_phase_compact.py`](../../vllm_compare/qwen_gdn_bt64_native_chunko_stage7a_phase_compact.py)

它只删除 lane-private `frag_words` 同步并把 score-half sync 延迟到已有的
V-new producer-to-consumer barrier。它未通过 correctness gate，不能使用。

## 1. 精确 Barrier 链

审计对象是冻结 Z1：

```text
_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z1
```

生成的四层 artifact：

| 层 | artifact | barrier 数 | 结论 |
|:--|:--|--:|:--|
| AveLang source | `qwen_gdn_bt64_native_chunko_stage6z.py` | 10 个语法 site | 全部是显式 `al.syncthreads()` |
| AveLang IR 语义 | `lib/IR/builtin_module.cc` | 1:1 | `syncthreads()` 直接构造 `gpu::BarrierOp` |
| pre-link LLVM | `z1_prelink.ll` | 41 | `fence release -> llvm.amdgcn.s.barrier -> fence acquire` |
| pre-LTO assembly | `z1_prelink.s` | 41 | 与 LLVM barrier ordinal 相同 |
| final HSACO ISA | `z1_final.isa` | 41 | 逐 ordinal 与 pre-LTO 链对齐 |

因此本轮能严谨地说：**没有看到 compiler/backend 额外创建 barrier。** 它做的是
保留 source barrier 并对 `al.range` 的部分循环展开。

初始 MLIR 也尝试在独立子进程导出，但当前 Docker binding 的 `get_mlir()` 触发
segmentation fault（return code `-11`）。这个调试 API 限制不会影响 LLVM/HSACO 的
确证：同一 AST source 的 LLVM、pre-LTO assembly 与 freshly captured final HSACO 都
完成且数目一致。它意味着 source-line DebugLoc 没有可用的 MLIR 打印 artifact，不能
声称拥有 MLIR source location 到 ISA PC 的逐条 debug metadata 映射。

完整 41 条 PC ledger 在：

- [`z1_barrier_ledger.md`](codex_qwen_bt64_chunko_barrier_stage7a/z1_barrier_ledger.md)
- [`z1_barrier_ledger.json`](codex_qwen_bt64_chunko_barrier_stage7a/z1_barrier_ledger.json)
- [`audit_summary.json`](codex_qwen_bt64_chunko_barrier_stage7a/audit_summary.json)

ledger 使用已验证的同序 `source schedule -> LLVM call -> assembly -> final ISA`
ordinal 对齐，而不是伪造不存在的 DebugLoc。

## 2. 41 个 Barrier 的 source provenance

Z1 source 有 10 个同步 site。对 `T=2048, WG256` 的 exact specialization，静态
展开/保留结果如下：

| source line | site | static count | dynamic context | hazard | 初始分类 |
|--:|:--|--:|:--|:--|:--|
| 90 | `A.stage_qh` | 4 | 4 个 inter K32 stage | CTA Q/H producer -> MFMA consumer RAW | 必要 |
| 96 | `A.pack_frag` | 8 | 4 stage x 2 kt | `frag_words` pack -> load | 待验证 |
| 103 | `A.reuse_frag` | 8 | 4 stage x 2 kt | MFMA 后下一 fragment reuse | 待验证 |
| 125 | `B.stage_qk` | 2 | 2 个 source half 的 K-stage loop body，各动态 x4 | CTA Q/K producer -> owner MFMA RAW | 必要 |
| 131 | `B.pack_frag` | 4 | 2 half x 2 kt，K-stage body 动态 x4 | `frag_words` pack -> load | 待验证 |
| 139 | `B.reuse_frag` | 4 | 2 half x 2 kt，K-stage body 动态 x4 | MFMA 后 fragment reuse | 待验证 |
| 153 | `B.serialize_score` | 2 | 每个 score half 一次 | score write -> later score/V consume | 待验证 |
| 164 | `C.stage_v` | 1 | V-new transpose | score/V producer -> intra MFMA RAW | 必要 |
| 172 | `C.pack_frag` | 4 | 2 half x 2 kt | score/V fragment pack -> load | 待验证 |
| 179 | `C.reuse_frag` | 4 | 2 half x 2 kt | MFMA 后 fragment reuse | 待验证 |

总数为：`4 + 8 + 8 + 2 + 4 + 4 + 2 + 1 + 4 + 4 = 41`。

这也解释了表面矛盾：source 只有 10 行 barrier，但它不是 10 个静态 ISA barrier。
Phase A 的 `k_stage=4` 被展开；Phase B 保留了动态 K loop body；Phase C 的两个
score half/两个 fragment 被展开。

## 3. 与 Native vLLM 的阶段对齐

native selected `chunk_fwd_kernel_o` 在这次重新读取的 T2048 selected AMDGCN artifact
有 `11` 个 lexical `s_barrier`（此前 Z0 汇总的 `10` 是旧统计口径；Stage 7A 使用同一
selected file直接计数）。它的 Python source 没有 `tl.barrier()`；这些 barrier 是 Triton
local-memory/dot pipeline lowering 的结果，不能对 Python 行号做虚假的一对一归因。

| native phase | native source | Z1 对应 phase | 核心差异 |
|:--|:--|:--|:--|
| Q/K/H K32 load + two dots | `chunk_o.py:93-113` | A + B 的 source stage/fragment sequence | native 用 compiler-managed local operand pipeline；Z1 手动把 fragment 反复存入/读出 CTA LDS |
| decay/mask/score BF16 operand | `115-125` | B score serialize | native score 保持 local dot operand；Z1 将两个 score half 显式序列化到 `phase` |
| V load + score-times-V + BF16 store | `127-138` | C V transpose/intra | native TTGIR 释放 Q/K/H local buffers 后再分配 score/V local buffers；Z1 以 CTA-wide phase boundaries 保护共享复用 |

native T2048 TTGIR 有 5 个 local allocation、3 个 local deallocation；这就是它能以
11 个 barrier 完成多阶段局部 pipeline 的直接 evidence。Z1 的 41 个 barrier 不能仅用
“所有权变成 BV64”消掉。

## 4. 最小 Repro

新增 Qwen-free repro：

- [`repro_qwen_bt64_chunko_barrier_stage7a.py`](repro_qwen_bt64_chunko_barrier_stage7a.py)
- [`profile_qwen_bt64_chunko_barrier_stage7a.py`](profile_qwen_bt64_chunko_barrier_stage7a.py)

所有模式都是 WG256、shared BF16、MFMA32（A/C）且 zero scratch/spill。

| experiment | comparison | barrier | output | 解释 |
|:--|:--|--:|:--|:--|
| A | per-lane fragment pack without/with extra barrier | 1 / 2 | bit-exact | 单独的 lane-private write 后 barrier 可去掉 |
| B | score lower half write, then disjoint upper half write | 2 / 1 | bit-exact | 在没有中间 consumer 时，可合并为最终 consumer 前一条 barrier |
| C | all CTA stage, one owner wave MFMA | 1 | finite | owner wave 不意味着 source stage barrier 可以去掉 |

原始 JSON/ISA/readobj：

- [`minimal_repros.md`](codex_qwen_bt64_chunko_barrier_stage7a/minimal_repros.md)
- [`minimal_repros.json`](codex_qwen_bt64_chunko_barrier_stage7a/minimal_repros.json)

这些 repro 的价值是限定因果：它们证明 A/B 的 barrier 在**最小独立内存关系**中不是
必需的；它们没有证明完整 MFMA pipeline 在不同 wave 进度、反复 MFMA issue 与 LDS reuse
下也可安全删除。

## 5. 唯一允许的 Local Fix 与失败

依据 A/B，实施了唯一一个 phase-scheduling experiment：

```text
删除 A/B/C 的 frag_words pack/reuse barriers
删除两个 B.serialize_score barriers
保留 A.stage_qh、B.stage_qk、C.stage_v
```

理论上它会让静态 barrier 从 `41` 降至 `7`，且没有改动 tile、WG、MFMA、layout、dtype 或
math。该候选第一个完整 Z1 correctness case（T8192）即失败：

| comparison | result |
|:--|:--|
| phase-compact vs frozen Z1 | `bit_exact = false` |
| max abs | `0.00206613541` |
| gate | 失败，要求 bit-exact |

在该失配 kernel 后同一 pytest process 继续编译下一 specialization 时，HIP report 了
memory access fault 并 abort。该 abort 不用于归因；唯一可靠的 stop fact 是更早出现的
T8192 numerical mismatch。没有继续收集该无效候选的 body、rocprof 或 full graph 数据。

为什么最小 repro 不足以放行？最可能是完整 kernel 中 `frag_words` 的 reuse 不只是普通
“同一 lane store 后同一 lane load”：它夹在跨 wave 的 LDS source read、MFMA issue、下一
round LDS overwrite 和非锁步 wave progress 中。一个 barrier 可能同时充当 schedule-wide
phase boundary。当前一次修复同时移除了多类同步，Stage 7A 规则禁止再逐个恢复/扫组合，
所以不能把责任精确归给某一条 barrier。

## 决策

这不是 Case B：LLVM/pre-LTO/final ISA 都未显示额外 compiler-inserted barrier；问题不是
generic backend hazard analysis 平白增加了同步。

也不是可以继续的 Case A：唯一允许的 source compaction 未通过 full Z1 exactness。

所以是 **Case C for the current Avelang source schedule**。停止 Stage 6Z pure-source
native chunk-o 路线，不做 barrier subset sweep、tile/WG sweep、Z2 或 full integration。

后续如果追求端到端性能，下一条独立路线可以是已捕获 native `chunk_fwd_kernel_o` HSACO 的
external-kernel bridge，并明确标记 external integration；它不是 Avelang source kernel
优化。若坚持纯 Avelang source，按此前 Stage 6Z 排序转向 W/U runner-up gap，预期收益较小。

## 复现

```bash
cd /workspace/project/avelang

PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/audit_qwen_bt64_chunko_barrier_stage7a.py \
  --T 2048 \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_chunko_barrier_stage7a

PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/profile_qwen_bt64_chunko_barrier_stage7a.py \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_chunko_barrier_stage7a \
  --warmup 5 --repeat 20
```

The phase-compact test is intentionally skipped in ordinary collection because
its first full-kernel exactness gate already failed; the source is retained as
a documented failed experiment rather than a candidate baseline.
