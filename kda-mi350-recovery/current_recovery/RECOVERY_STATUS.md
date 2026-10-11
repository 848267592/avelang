# 恢复状态

本表描述可用证据，不表示当前可构建或已在 MI350 验证。

| 资产 | 分类 | 当前材料与限制 |
|---|---|---|
| A70 native BW16/W4 K2 | EXACT_RECOVERED | 完整源码 SHA-256 匹配 golden `a496c2b2cefa4bf0ff475049876d6db34091fe64e73a586498e83a61c1174801`。历史报告称 MI300X correctness PASS，未在本归档环境重跑。B66/B67 编译器依赖缺失。 |
| K1 C23，NT≥102 | METADATA_ONLY | 未找到完整源码、host dispatch、workspace ABI 或可重放补丁链。 |
| K1 C26，NT≤101 | PARTIAL | 只有 7,000 字符截断的源码输出，不可执行。 |
| B66/B67 BF16 ordinary-V MFMA 编译器支持 | PARTIAL | 只有 5,000 字符截断的 `amdgpu_module.cc` 检视片段。它不是完整 patch；早期其他 compiler patch 不代表 B66/B67。 |
| A56 BW64 | METADATA_ONLY | 有历史说明，没有 kernel 源码。 |
| A69 persistent K2 | PARTIAL | 保留截断源片段、少量 diff/patch 与 ownership ledger；缺少可完整重放的 base。 |
| A67/A68 | RECONSTRUCTED（辅助材料） | 部分 correctness gate、layout/ownership 与审计工具可供研究；不是 A56/A69 的替代 kernel。 |
| D71 benchmark runner | RECONSTRUCTED | 从完整历史源码字面量重建；远程文件是否包含最后追加片段不能验证。依赖未恢复的 C68 `run_sweep.py` 和原项目布局。 |
| D73 benchmark/joint-sweep runners | EXACT_RECOVERED | 两份完整历史写入内容已逐字保存；它们是调度/benchmark 工具，不是 KDA kernel。依赖 C68 sweep 模块、AITER 与原项目目录，当前不能独立运行。 |
| V0 correctness 记录 | EXACT_RECOVERED（文档） | 从本地 D 盘逐字复制。报告记录历史边界，不是当前重测结果；V0 源码和原始逐形状 JSON/log 未恢复。 |
| Stage 5 performance 文档 | METADATA_ONLY（归档状态） | 原文留在本地恢复目录；静态筛查命中内部机器名，因此未复制到该项目。 |
| C67/D72 machine golden | PARTIAL / METADATA_ONLY | 有报告和摘要，没有完整 golden 包、HSACO、ISA、MLIR 或可独立核验的原始 profiler 包。 |

## 可直接编译/运行性

目前没有文件能在一个干净 checkout 中直接完成构建或端到端运行：A70 缺 Avelang 编译器修改及运行时；K1 源码不完整；benchmark runners 依赖缺失的 C68/AITER/项目模块和原机器环境。D73 文件内容完整不等于运行依赖完整。

本仓库未运行恢复代码。历史 correctness、性能和资源数据仅按原报告引用；不把它们称作当前复验结果。

## B70: BW16/W4 four-wave MFMA layout investigation

B70 is a read-only machine-level layout investigation. The recovered report describes the transposed #mma1 Kd/INV result carrier, the four-wave #mma Kr/H carrier, lane/fragment ownership, LDS slots, and the Kr ds_bpermute_b32 / v_perm_b32 mapping. It explicitly says B70 did not modify A69. These probes are not a complete K2 kernel.

### Files and recovery status

| Artifact | Status | Recovered byte SHA-256 | Evidence |
|---|---|---|---|
| reports/B_20260927_70_LAYOUT_RECOVERY.md | RECONSTRUCTED | 57b715bb3cb6b5bbf27ecf3a21158fae32c0d6a61d8ccda1c384ebbe55973bd8 | Complete direct source body, call exec-bc83d6bd-e5e8-444c-afe0-f68a7dfe8d7b; no independent machine-side hash. |
| analysis/k2/B_20260927_70_LAYOUT_PROBES.py | RECONSTRUCTED | cae3ea07cf76c98e04279b5a6ce6d83265bd87e1675a8c4d4972139967da2a4f | Complete direct source body, call exec-e7ba4ec0-d8bf-4b75-b8ef-749a9d4ebac6. Defines Kd, INV, and Kr probe kernels; harness imports PyTorch and Triton. |
| analysis/k2/B_20260927_70_PROBE_CORRECTNESS.py | RECONSTRUCTED | 8fb6ee8d0307a20c19cdcb699fcffe997ca9c4dba57857c460c2849800298c3c | Complete direct source body, call exec-aa7a496a-fb00-4cfa-af0f-11ac0e842050; not executed during this recovery. |
| analysis/k2/B_20260927_70_WRITE_LAYOUT_REPORT.py | RECONSTRUCTED | 3232d49c16735bbc2d815b4d2d6f8f64924bd61f6eba976cef9767beedff2037 | Last captured 188-byte compatibility shim, call exec-c59c6e97-5d27-41a8-b4bb-7e39712cb751; imports the missing correlation module. |
| analysis/k2/history/B_20260927_70_WRITE_LAYOUT_REPORT.initial.py | RECONSTRUCTED | 9a7244ed1789627d0ec7820743bd625a5a5636bb0c363d363ff632b5b2ae0eff | Earlier complete 6410-byte writer source, call exec-0d39ae1f-1e45-41b2-a552-f043178b9f70; historical version, not final writer. |
| analysis/k2/partials/B_20260927_70_IR_ISA_CORRELATION.py.partial.txt | PARTIAL | 72ff032e5bad3383f9b87219f621dbdd5507958a9b67bfd7d8a99efe0b8fe6fb | 12000-byte fragment ends mid-dictionary; one source event was truncated. Five later diffs are preserved under patches/b70/ir_isa_correlation/. |
| B_20260927_70_IR_ISA_CORRELATION.json | METADATA_ONLY | — | Named in the recovery target, but no JSON byte body or matching file was found in the local recovery inventory. |
| B_20260927_70_PROBE_CORRECTNESS.json | METADATA_ONLY | — | The recovered harness names this output, but the output JSON itself was not recovered. |
| B_20260927_70_LAYOUT_PROBES.json | METADATA_ONLY | — | Expected output path is present in the probe source; JSON bytes are absent. |
| B_20260927_70_LAYOUT_RECOVERY.json | METADATA_ONLY | — | Expected output path is present in the historical report writer; JSON bytes are absent. |

RECONSTRUCTED is used for complete historical write bodies whose original server-side hash is unavailable. SHA-256 values identify the recovered bytes only; they are not claims about an independently verified original checksum. The JSON files are not replaced with generated or inferred content.

### Probe scope and dependency boundary

The probe sources import torch and triton. The correctness harness covers isolated Kd/Qd, INV/Mqk, and Kr shapes and calculates ownership coverage; it was not run during this recovery. The captured sources are research probes and report tooling. They do not supply an Avelang production K2 kernel or a standalone custom Avelang intrinsic. The current report-writer shim depends on the incomplete B_20260927_70_IR_ISA_CORRELATION.py module. Original compiler outputs and JSON results are absent.

The report records MI300X observations including #mma1 with warpsPerCTA=[2,2], instrShape=[16,16,16], transposed operands, and #mma Kr/H with warpsPerCTA=[4,1]. It also records the LDS slot ledger and the Kr ds_bpermute_b32 / v_perm_b32 sequence. These are machine observations from the MI300X build and must be rechecked against MI350 compiler IR, ISA, register assignments, LDS placement, barriers, and correctness.

### Remaining gaps

- No standalone A69_REQUIRED_CHANGES audit file was found in the B70 artifacts. The B70 report says it did not modify A69; that is not a substitute for a separate A69 audit.
- The complete custom Avelang intrinsic implementation and the original compiler-generated TTIR, TTGIR, LLIR, ISA, and HSACO payloads were not recovered as B70 files.
- The correctness and correlation JSON results remain metadata only. No B70 probe or recovered script was run on this machine.