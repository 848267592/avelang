# Qwen R4-tail：production ATT 完整 wave 与 PC-range 动态归因

日期：2026-08-04
范围：**只做 production `emit_audit=False` 的动态归因；未修改 layout、lowering、planner、数学或 kernel。**

## 结论

已修复 ATT 的 `Wave incomplete`：同一条 public-Eager R4-tail production kernel 现在有 4 个完整 wave record（2 个完整 workgroup），decoder 返回 `status=0`、`INFO=[]`，没有 `INFO=3 (Wave incomplete)`。以这 2 个 workgroup 的实际 PC 执行次数乘以 `32 / 2 = 16` 后，ATT 与同一 production HSACO 的 PMC 在 VMEM、VALU、SALU、LDS、MFMA 五项上**逐项精确相等**。

固定制品：

- plan：`gfx942_bt64_bv32_joint_v4_tail_issue`；T=2048；B=1、Hk=4、Hv=8、K=V=128、BT=64、BV=32、WG=128、logical grid=32 CTA。
- public Eager entry：`bench_qwen_gdn_r4_tail_vs_current_vllm_eager.py --worker --implementation r4_tail`，没有 private kernel launch。
- symbol：`_qwen_gdn_persistent_recurrence_r4_joint_v4_kernel`。
- production HSACO SHA-256：`5ebff98fd16fe1fa47501fbeb7dd4bd738f716d6445c0d0fd52d71e0621740c7`。
- production ISA 中没有 `pred_f32`、`pred_bf16`、`v_decay` 或 `state_after` audit sink；因此本报告没有混用 `emit_audit=True` correctness ISA。

最大的已证实 R4 VMEM 工作不是 tail W/K issue：每 physical V32/chunk 中，`H/v_new/terminal store` 为 98 条 VMEM，`U/g` load 为 66 条，tail W/K issue 为 32 条。R4 的 H/U/g 组合是 164 条；current-vLLM 的**全 kernel** VMEM 总数只有 57 条/V32/chunk。因此即使把 Triton 的全部 VMEM 都有利地分配给该组合，R4 在该组合的 VMEM 超额仍至少为 107/140.5 = **76.2%**（T=2048）。这把下一次机器路径优先级指向 full-op I/O packet ownership / wide global load-store，而不是 K LDS layout 或重新开启 core-lastuse issue。

对 VALU，R4 的最大绝对 PC bucket 是 `update operand preparation` 的 614 条/V32/chunk，其后是 BF16 bridge 的 486、tail W/K issue 的 420、loop/control 的 388.75。现有 current-vLLM 没有同源的完整 PC trace，所以不能诚实地把 `+824.5 VALU/V32/chunk` 再拆到上述单一 R4 bucket；本报告不把 R4 的绝对热点伪称为跨实现的因果差额。SALU 则有一个可靠下界：R4 的 loop/control bucket 为 133.125，而 Triton 的全 kernel SALU 仅 63.313，因此该 bucket 单独至少覆盖 `69.812/95.938 = 72.8%` 的 T=2048 SALU 差额。

## ATT 截断根因与修复

最初的单 dispatch trace 只有约 193 KB，decoder 给出 `INFO=3`。decoder 的枚举说明该值表示 trace 在 wave 全部结束前被 cutoff。下列检查排除了其余原因：

| 项目 | 检查结果 | 结论 |
| --- | --- | --- |
| trace buffer | `rocprofv3` 默认 ATT buffer 为 256 MB；未完成 raw trace 仅约 193 KB，完整 raw trace 为 2,593,096 B | 不是 buffer exhaustion。尝试把 `--att-buffer-size` 误传为 `512/1024` 被 rocprofv3 拒绝，未作为有效实验。 |
| capture duration | `--att-consecutive-kernels=2` 在只有一次匹配 dispatch 时出现 `Thread tracer being destroyed with thread trace active` | 这是 wave 在 profiler teardown 时仍活跃的直接原因。 |
| CU/SIMD selection | 目标 CU=1、SIMD mask=`0xF`；完整 raw 文件为 `shader_engine_0_14.att`，实际得到 SIMD 0/1/2/3 的 4 个 wave | 选择覆盖了一个实际执行的 CU 和全部 SIMD；没有不必要的 SE filter。 |
| decoder/tool 版本 | ROCm 7.2.2；rocprofv3 1.1.0（git `671d39a71e33c49fba50b12e30b1aea45c5ed366`）；`rocprof-trace-decoder` 0.1.6 | decoder 是适用于 ROCm < 7.13 的独立包；不存在版本失配症状。 |

成功配置保持 public Eager worker，只把足够多的同 symbol dispatch 留给 ATT 收尾：

```text
rocprofv3 --advanced-thread-trace true \
  --att-gpu-index 0 --att-target-cu 1 --att-simd-select 0xF \
  --att-consecutive-kernels 20 \
  --kernel-include-regex _qwen_gdn_persistent_recurrence_r4_joint_v4_kernel \
  --output-directory .../att_complete_probe_c20_repeat20/trace -- \
  python bench_qwen_gdn_r4_tail_vs_current_vllm_eager.py \
    --worker --implementation r4_tail --T 2048 --warmup 0 --repeat 20
```

其 link-time code object 的 SHA-256 也为上述 `5ebff…1740c7`，没有 runtime re-JIT 成第二个 code object。

## 归因方法与缩放

完整 ATT raw：

```text
test/examples/linear_attention/rocprof_outputs/
qwen_r4_tail_production_t2048_capture_20260804_v2/
att_complete_probe_c20_repeat20/trace/0364d3a007f9/
757897_47831_shader_engine_0_14.att
```

四个完整 wave 的 instruction record 数为 `54,113, 54,129, 54,113, 54,129`，对应 2 个完整 WG（WG128 即两个 wave/WG）。T=2048 的 production dispatch 是 32 WG，故用完整 trace 的 PC counts 乘 16。所有 workgroup 均执行相同的 persistent recurrence 控制流；更重要的是，缩放后的每一类计数均与 32-WG PMC 精确闭合。

动态类别来自 decoder 的实际 executed-PC category：`FLAT(4)` 计 VMEM、`VALU(6)` 计 VALU、`SALU(2)` 计 SALU、`LDS(5)` 计 LDS；ISA opcode 前缀 `v_mfma` 是 VALU 的子集，另列 MFMA，不从 VALU 中扣除。这正是 `SQ_INSTS_VALU` 和 `SQ_INSTS_MFMA` 可以同时闭合的口径。

下表的 PC 范围是 production ISA 的动态选择；全局 load/store 的精确 opcode 地址被从周围 ALU block 单独分出。`state feedback/final update` 是后端基本块布局名称，含 16 条 update MFMA；并非声称该 range 中每条机器指令都是纯 FP32 feedback。

## PC-range 动态分桶

计数是完整 T=2048 dispatch；括号内是除以 `32 chunks × 32 physical V32 tiles = 1,024` 得到的每 V32/chunk 值。

| bucket | production PC selection | VMEM | VALU | SALU | LDS | MFMA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| W/K prologue | executed `0x1a00–0x21dc` | 1,536 (1.5) | 14,976 (14.625) | 2,176 (2.125) | 1,024 (1) | 0 |
| tail W/K issue + same-bank commit | executed `0x21e0–0x29e0` | 32,768 (32) | 430,080 (420) | 18,432 (18) | 32,768 (32) | 0 |
| U/g load | `global_load_*`: `0x29e4`, `0x38e4–0x3e18`, `0x4240–0x4844` | 67,584 (66) | 0 | 0 | 0 | 0 |
| H/v_new/terminal store | `global_store_*`: `0x2a54–0x36e0`, selected `0x3940–0x4b18`, `0x561c–0x5ac8` | 100,352 (98) | 0 | 0 | 0 | 0 |
| pred operand preparation | executed `0x3700–0x38dc` | 0 | 90,112 (88) | 0 | 24,576 (24) | 16,384 (16) |
| BF16 bridge | executed non-global instructions `0x38e0–0x4140` | 0 | 497,664 (486) | 0 | 47,104 (46) | 0 |
| update operand preparation / remaining update groups | executed non-global instructions `0x4144–0x4b90`, `0x51c4–0x55e0` | 0 | 628,736 (614) | 2,048 (2) | 141,312 (138) | 32,768 (32) |
| state feedback / final update block | executed `0x4b94–0x51c0` | 0 | 180,224 (176) | 4,096 (4) | 69,632 (68) | 16,384 (16) |
| loop/control, output-mask and terminal blocks | remaining non-global PCs: `0x29ec–0x36fc`, `0x55e4–0x5ad0` | 0 | 398,080 (388.75) | 136,320 (133.125) | 65,536 (64) | 0 |
| **sum** | all executed production PCs | **202,240 (197.5)** | **2,239,872 (2,187.375)** | **163,072 (159.25)** | **381,952 (373)** | **65,536 (64)** |

VMEM opcode-width ledger from the same executed PCs:

| path | executed opcode mix / V32/chunk | dynamic VMEM / V32/chunk | interpretation |
| --- | --- | ---: | --- |
| W/K prologue | `global_load_dwordx4` | 1.5 | one-time/current packet setup amortized over chunks |
| tail W/K | `global_load_dwordx4` | 32 | wide b128 packet issue; not the dominant R4 VMEM bucket |
| U/g | 32 × `global_load_ushort` + 34 × `global_load_dword` | 66 | narrow/fractured scalar input accesses |
| H/v_new/terminal | 64 × `global_store_short` + 32 × `global_store_short_d16_hi` + 2 × `global_store_dword` | 98 | narrow output stores dominate R4 VMEM |

`global_store_short_d16_hi` is a real production packed BF16 store in this code object, not an audit-only `v_decay` sink; audit tensor symbols are absent from production ISA.

## PMC 闭合

PMC source：

```text
.../pmc/trace/0364d3a007f9/753325_counter_collection.csv
```

它记录相同 symbol、`Grid_Size=4096` work-items、`Workgroup_Size=128`、LDS=53,248 B、scratch=0、VGPR=128、AccVGPR=160、SGPR=112。`4096 / 128 = 32`，即与 source logical grid 一致的 32 CTA。

| metric | ATT PC buckets（缩放到 32 WG） | PMC | delta |
| --- | ---: | ---: | ---: |
| SQ_INSTS_VMEM | 202,240 | 202,240 | 0 |
| SQ_INSTS_VALU | 2,239,872 | 2,239,872 | 0 |
| SQ_INSTS_SALU | 163,072 | 163,072 | 0 |
| SQ_INSTS_LDS | 381,952 | 381,952 | 0 |
| SQ_INSTS_MFMA | 65,536 | 65,536 | 0 |

因此 production R4 的 PC bucket 覆盖是 VMEM/VALU/SALU/LDS/MFMA 全部 100%，不是 audit ISA static site 乘 trip count。

## 对 R4-tail / current-vLLM 差额的严格含义

T=2048 已有同口径 aggregate current-vLLM 数为每 V32/chunk：VMEM=57、VALU=1,362.875、SALU=63.313、MFMA=64；R4 分别为 197.5、2,187.375、159.25、64。故差额为 VMEM `+140.5`、VALU `+824.5`、SALU `+95.938`、MFMA `0`。（长序列 T=8192 的稳态差额为约 `+138.625/+840.625/+97.484/0`。）

可直接证明的结论：

1. R4 的 `H/v_new/terminal store + U/g load = 164 VMEM/V32/chunk`。由于 Triton 全 kernel VMEM 只有 57，这两个 I/O bucket 对 R4 VMEM 差额的严格下界为 `164 - 57 = 107`，即 T=2048 VMEM 差额的 **77.3%**。tail 的 R4 绝对值仅 32，不能支持“tail issue 是最大 VMEM 根因”。
2. R4 loop/control SALU 是 133.125；Triton 全 kernel SALU 只有 63.313，所以它对 SALU 差额的严格下界是 69.812，即 **72.8%**。这指出 AveLang 输出-mask/loop-control/basic-block work 是 SALU 的主矛盾。
3. R4 VALU 的主要绝对工作在 update preparation、BF16 bridge、tail/commit 和 loop/control 四块。没有 current-vLLM 的同源完整 PC trace 时，无法把它的 1,362.875 VALU 分配给对应阶段；因此不能宣称任一块已经解释 `+824.5` VALU 差额的某个百分比。

这满足“production R4 动态 PC 归因 + PMC 闭合”，但不把单边 trace 误写成双边 phase-delta 审计。若之后需要达到跨实现 VALU/VMEM phase-delta 的 90%/70% 门槛，缺失的唯一证据是 current-vLLM 也做同样的 full-wave ATT，而不是再猜测或立刻修改 layout。

## 本轮决策

不编码。已证实的第一优先机器路径是 production R4 的 **U/g narrow loads 与 H/v_new narrow stores**；它们是 R4 最大 VMEM bucket，且与已存 current-vLLM ISA 中 wider buffer load/store 的差异方向一致。任何后续 full-op I/O packet ownership / wide access 实现仍须先得到 current-vLLM PC trace，或在候选的同源 PMC 中证明实际减少了这些 VMEM 及其地址 VALU/SALU。

本报告不支持下列动作：重开 core-lastuse/distance、K LDS layout 单点替换、第二 LDS，或把 BF16/update 的 R4 绝对 VALU 热点误判为已证实的 Triton phase 差额。
