# C_20260927_71 K1 TRUE-BEST MACHINE REALIGNMENT AUDIT

本报告是只读机器/资源/ISA/phase-forensic audit。没有重新运行 C69 candidate sweep，没有 benchmark C23/C26 winner，没有创建 W4 Avelang candidate，也没有修改 kernel、compiler、allocator、runtime 或 Avelang source。

## 0. Identity gate

固定身份：AMD Instinct MI300X/gfx942；B=1,H=12,K=128,V=128,C=32,BC=16；AITER commit `7bb44998274fa679ece0a37bf8426ab780b1837d`。

完整逐 T identity gate 已写入：

`test/examples/kda/reports/C_20260927_71_K1_TRUE_BEST_MACHINE_REALIGNMENT_AUDIT/identity_gate.json`

其中 T32 使用精确 T32 W4/S2 specialized identity；T64..1024 保留每个 T 的 W4 config/stage/Triton hash/grid；T2048/4096 使用 W2/S2 identity。不能仅以 HSACO 判断身份。

## A. AITER_K1_TRUE_BEST_MACHINE_GOLDEN

### A: T32 specialized W4/S2

- config: `w4_s2`, warps/stages `4/2`, grid `[1,12,1]`, block 256
- HSACO: `5c0a24b34425219f94844f615ad731f1d553033d44b0ac94e75ee22cfe5e0e02`
- Triton hash: `42d944a8b207fea563aa684bf867d4ebd2f1e97a65c0d3c1e735a68f77a778fa`
- ordinary VGPR 128; AGPR 0; unified span 128; accum_offset 0
- ELF allocation fields: VGPR 128, SGPR 56; explicit numbered SGPR span `s0..s49` = 50; highest VGPR `v118`; highest explicit SGPR `s49`
- VGPR/SGPR spill 0/0; private/scratch 0; static LDS 0; dynamic LDS 8192 B
- block 256, 4 waves/workgroup
- actual `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(...,256,8192)` = 4 blocks/CU; derived resident waves = 16/CU
- limiter assessment: API sensitivity shows 8 KiB LDS does not reduce this actual result versus LDS=0; the observed ceiling is the register/resource wave ceiling at 16 waves/CU. HIP occupancy does not expose an independent subresource reason, so no unsupported SGPR/block claim is made.

注意：captured filename contains `w4_s1` because the original C69 resource capture used that first winner label, but the frozen T32 metadata identity above is `w4_s2`. The binary SHA and metadata hash are kept separate.

### B: generic W4 family, T64..1024

- unique HSACO: `34419742ff3240a0d60cc3792311ee7b56553a2a3a87f4f280d0aa6b745d8e13`
- block 256, 4 waves/workgroup; dynamic LDS 8192 B
- ordinary VGPR 128; AGPR 0; unified span 128; accum_offset 0
- ELF allocation fields: VGPR 128, SGPR 55; explicit numbered SGPR span `s0..s48` = 49; highest VGPR `v119`; highest explicit SGPR `s48`
- VGPR/SGPR spill 0/0; private/scratch 0; static LDS 0; dynamic LDS 8192 B
- actual occupancy API result = 4 blocks/CU; derived resident waves = 16/CU
- representative config is T512 `w4_s2`, Triton hash `f79d3e95d806477b9709bbee9af94cb57100390cf396fee0384614e8fa07de2d`; every T-specific identity is in `identity_gate.json`
- limiter assessment is the same W4 register/resource wave ceiling; actual 8 KiB LDS is not the limiting factor at the measured block size/API query.

Per-T W4 identities are distinct even though the physical HSACO is shared:

- T64/T128/T256/T384: `w4_s1`, Triton `8dd7fa31ef878420a1f7bf60982721aa2221671908291fc761b6597731d1ac13` where selected
- T127: `w4_s2`, Triton `898d92b726206e3ee6ce8334e06b5ea280fd7d19efef4ea1badb21994987928f`
- T511/T1024: `w4_s1`, Triton `e87c3f2dbe7209cf51cd1998671435013718acf6da1754c00cc5fd91b6e38f82`
- T512: `w4_s2`, Triton `f79d3e95d806477b9709bbee9af94cb57100390cf396fee0384614e8fa07de2d`

### C: long-T W2/S2, T2048/4096

- config `w2_s2`, warps/stages `2/2`, grid `[64,12,1]` at T2048 and `[128,12,1]` at T4096, block 128
- HSACO: `68844813aee59496efb9dd1844dd49ce3e380aa65971907f0cd09d1190b29014`
- Triton hash: `cef0a88f4d9ce72fcbe18b85287dfdb6a441f91b36d5dc9a611ad91e3a5fe477`
- ordinary VGPR 187; AGPR 0; unified span 187; accum_offset 0
- ELF allocation fields: VGPR 187, SGPR 56; explicit numbered SGPR span `s0..s49` = 50; highest VGPR `v186`; highest explicit SGPR `s49`
- VGPR/SGPR spill 0/0; private/scratch 0; static LDS 0; dynamic LDS 8192 B
- actual occupancy API result = 4 blocks/CU; derived resident waves = 8/CU
- this object shares HSACO bytes with AITER_PUBLIC_W2_REFERENCE `w2_s1`, but C is not PUBLIC: config/stage/Triton metadata differ. PUBLIC W2 resources are not reused as W4 resources.
- limiter assessment: the 187-VGPR allocation is consistent with the lower 8-wave/CU ceiling; 8 KiB LDS does not change the actual API result. HIP does not expose an independent SGPR versus block reason.

Raw resource and occupancy evidence:

`test/examples/kda/reports/C_20260927_71_K1_TRUE_BEST_MACHINE_REALIGNMENT_AUDIT/true_best_resource_manifest.json`

`test/examples/kda/reports/C_20260927_71_K1_TRUE_BEST_MACHINE_REALIGNMENT_AUDIT/occupancy_queries.json`

The occupancy sensitivity rows are diagnostic only. Production residency above always uses the actual block size and dynamic LDS=8192 B.

## B. Final physical ISA contract

ISA parser rule: count only tab-indented opcode records matching `^\s+opcode\s`; ignore headers, labels, comments, and blank lines. Each family also contains raw first-BF16-MFMA, first-FP32-MFMA, first-buffer-store, and first-ds_bpermute spot-check lines.

| Family | ISA instruction records | BF16 MFMA | FP32 MFMA | VMEM loads/stores | LDS reads/writes/ds_bpermute | v_perm | DPP | s_waitcnt | s_barrier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A T32 W4/S2 | 3418 | 32 | 128 | 9 / 15 | 105 / 70 / 160 | 48 | 20 | 183 | 29 |
| B generic W4 | 3424 | 32 | 128 | 9 / 15 | 105 / 70 / 160 | 48 | 20 | 174 | 29 |
| C long W2/S2 | 5065 | 32 | 128 | 15 / 21 | 109 / 85 / 176 | 96 | 40 | 236 | 31 |

All three families have the same MFMA topology: 32 `v_mfma_f32_32x32x8_bf16` and 128 `v_mfma_f32_32x32x2_f32`. The W4 reduction is in the surrounding frontend/feed/packing/dependency envelope, not MFMA count.

Exact opcode forms, branch/control-flow envelopes, memory forms, workspace-store forms, and raw ISA spot-checks are in:

`test/examples/kda/reports/C_20260927_71_K1_TRUE_BEST_MACHINE_REALIGNMENT_AUDIT/true_best_isa_manifest.json`

Important physical differences versus C:

- W4: 24 total VMEM records, 175 LDS read/write records plus 160 ds_bpermute records, 48 v_perm, 20 DPP, and 174/183 waitcnt depending on generic/specialized object.
- C: 36 total VMEM records, 194 LDS read/write records plus 176 ds_bpermute records, 96 v_perm, 40 DPP, and 236 waitcnt.
- W4 MFMA issue volume is unchanged, while address/packing/select/permute/synchronization work is materially smaller.
- Exact per-store semantic workspace field ownership is not encoded in the ELF; the manifests record physical buffer/global store forms without guessing field assignment.

## C. W4 versus AITER_PUBLIC_W2_REFERENCE

The public W2 is cited only as `AITER_PUBLIC_W2_REFERENCE`, never as SOTA. C69 whole-kernel observations are:

| T | PUBLIC W2 us | TRUE_BEST W4 us | gap |
|---:|---:|---:|---:|
| 32 | 22.392999380827 | 18.548000603914 | 20.729990574% |
| 128 | 22.593999281526 | 19.269000738859 | 17.255687452% |
| 512 | 23.494500666857 | 20.230000838637 | 16.136923840% |
| 1024 | 35.011999309063 | 31.727001070976 | 10.353951294% |

Machine explanation:

1. W4 doubles the workgroup from 2 to 4 waves and changes the physical wave partition. The exact compiler layout is now verified for the selected cached T32/T512 objects; the hash-specific T128/T1024 layouts remain absent. The final ISA shows a smaller per-thread/physical instruction envelope. The API confirms 16 resident W4 waves/CU versus 8 for the higher-register W2 object.
2. MFMA counts are identical, so the gain is not “more MFMA.” W4 has 6 fewer VMEM loads, 6 fewer VMEM stores, fewer LDS read/write records, 16 fewer ds_bpermute records, half the v_perm/DPP count, and roughly 1.6K fewer instruction records than W2.
3. The exact ISA shows fewer lane-local address/select/packing and dependency-retirement instructions around Q/K/G frontend, normalization/gate preparation, workspace staging, BF16 feed, FP32 merge, and writeback. This is evidence consistent with a physical work repartition and schedule reduction, not proof of a specific logical row/column ownership map and not a block-size-only explanation.
4. Exact W4 compiler ownership attributes are available in the matched TTGIR cache for T32 W4/S2 and generic T512, and the matched PUBLIC W2 cache is available separately. No row/column map is inferred beyond those explicit compiler layout attributes; T128/T1024 hash-specific TTGIR remains unavailable.

Exact generic-W4 versus PUBLIC-W2 physical delta (B versus C) is:

| opcode/resource | generic W4 B | PUBLIC W2 C | delta B-C |
|---|---:|---:|---:|
| instruction records | 3424 | 5065 | -1641 |
| MFMA total | 160 | 160 | 0 |
| BF16 MFMA | 32 | 32 | 0 |
| FP32 MFMA | 128 | 128 | 0 |
| VMEM loads | 9 | 15 | -6 |
| VMEM stores | 15 | 21 | -6 |
| LDS reads | 105 | 109 | -4 |
| LDS writes | 70 | 85 | -15 |
| ds_bpermute | 160 | 176 | -16 |
| v_perm | 48 | 96 | -48 |
| DPP | 20 | 40 | -20 |
| s_waitcnt | 174 | 236 | -62 |
| s_barrier | 29 | 31 | -2 |
| non-MFMA VALU records | 2371 | 3917 | -1546 |
| SALU records | 694 | 742 | -48 |
| s_nop | 391 | 388 | +3 |

The T32 specialized W4 object is separately recorded: 3418 instruction records, 183 waitcnt, 29 barriers, and 385 s_nop. These are physical object deltas, not phase timings.

### C.1 Compiler layout cache audit

The read-only search found exact compiler artifacts for the selected T32 W4/S2 HSACO (`42d944…` → `5c0a24…`), the selected generic W4/T512 HSACO (`f79d3e…` → `344197…`), and PUBLIC W2/S1 (`3e8b…`/`3071…` → `688448…`). Their TTIR/TTGIR/LLVM IR/source sidecars are recorded in `compiler_layout_search.json`.

The matched W4 TTGIR explicitly records `sizePerThread=[1,8]`, `threadsPerWarp=[4,16]`, `warpsPerCTA=[4,1]`, `#mma` `[32,32,8]`, `#mma1` `[32,32,2]`, `#mma2` `[32,32,2]`, and the two swizzled shared layouts. PUBLIC W2 records the corresponding `[2,1]`/`[1,2]` warp ownership. This resolves the prior “no layout metadata exists” gap for those exact cached objects, but not for hash-specific generic T128 (`8dd7…`) or T1024 (`e87c…`).

### What the Triton source proves, and what it does not

The pinned AITER source proves the logical mapping: program_id(0) is the chunk index, program_id(1) is B*H; `o_c=arange(0,C)`, `o_k=arange(0,K)` define the logical C=32 by K=128 tiles; q/k/g use the same logical `qk_off`; and workspace indexing is `ws_idx=i_h*TOTAL_TILES+g_tile`. The source autotune list changes `num_warps` and `num_stages`, not the logical tensor expressions.

The C69 metadata and matched cache sidecars prove the physical config/layout attributes for the selected T32/T512 W4 objects; the final ISA proves the opcode deltas above. They still do not provide marker-safe semantic phase endpoints, and the hash-specific T128/T1024 sidecars were not found. Therefore the statements “W4 reduces repeated lane-local work” and “W4 changes physical wave scheduling” are machine-supported; exact phase attribution remains unavailable because the C71 diagnostic correctness launch failed. No phase timing is claimed.

Full identity-aware machine diff:

`test/examples/kda/reports/C_20260927_71_K1_TRUE_BEST_MACHINE_REALIGNMENT_AUDIT/w4_vs_public_w2_machine_diff.json`

## D. Phase timeline status

Required matched phase cuts at T32/T128/T512/T1024 are explicitly unavailable. No phase latency is fabricated.

The existing A31/A30 probe harness is anchored to the formal A30 object and old PUBLIC-W2-style probes. Although exact TTIR/TTGIR sidecars are now present for selected W4 objects, the harness has no marker-safe semantic probe for C69 W4 HSACO `5c0a...` or `344197...`. The existing C31 measurements also use old/asymmetric boundaries and cannot be relabeled as C69 W4 phase data without rebuilding and revalidating the object.

A C71-only feasibility check successfully reassembled diagnostic copies from the C26 objdump and both C69 W4 ISA listings, but those copies are not the exact C69 identities after instrumentation/reassembly. The initial raw prefix copy stopped at `hipModuleLoad` rc=209; metadata-preserved prefix smoke values were diagnostic-only and are explicitly invalid.

The subsequent full timestamp diagnostic was allowed one guarded T32/W4_A correctness attempt at `frontend_to_first_bf16`. The final copy passed MessagePack-note, kernel-descriptor, raw-kernarg, ISA-offset, allocation-range, and module-load checks, but the launch still stopped with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`. No output/workspace correctness pass was established, so all timestamp values and all phase timings are N/A; T128/T512/T1024 were not launched.

Failure evidence is `timestamp_correctness_failure.json`; assembly-only evidence is `phase_probe_feasibility.json`. `prefix_probe_timing.json` is retained for provenance but its smoke medians must not be interpreted as phase measurements.

The final failed HSACO was statically disassembled. Its exact SHA is `92e2d0b1025c2a2473651df93e167b843c4412ce90e576a2ecad4c485d4df7d8`. The timer pointer load at PC `0x194c` is captured in `v128:v129`, and both stores (`0x19ac` and `0x5024`) use that captured pointer immediately before the store; no pre-store pointer overwrite is proven. A separate real save-slot bug is proven: `v130:v131` initially save `s46:s47` (`0x1944/0x1948`), then are overwritten with `s42:s43` (`0x1980/0x1984`, repeated at `0x4ff8/0x4ffc`), but are later used to restore `s46:s47` (`0x19c4/0x19c8`, `0x503c/0x5040`). The statically safe diagnostic-only repair is to use the unused `v132:v133` pair for the entry `s46:s47` save/restore; the original W4 object reaches only `v118`, and the diagnostic descriptor already allocates through `v135`. This does not by itself attribute the aperture failure to the store and was not rerun.

The complete pointer/liveness/PC audit is `timestamp_pointer_static_audit.json`. No production object, compiler, allocator, runtime, or Avelang source was modified.

The isolated C71 probe was then rebuilt with the proven save-slot repair: `v130:v131` remains the entry `s46:s47` pair, while `v132:v133` carries endpoint `s42:s43`. Reassembly, metadata, kernel-descriptor, allocation-bound, and original-object identity checks passed. The corrected HSACO is `c0d4c032f52489d54cf0dbaa2bfc5703456876eaa35cb8baaf02c108d98ed77d`. The single authorized T32/W4_A correctness gate completed the original baseline launch but failed at corrected diagnostic launch with the same `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`; no phase timing or further launch was attempted. Evidence is `timestamp_corrected_static_manifest.json` and `timestamp_corrected_correctness_failure.json`.

The no-launch profiler feasibility check found that installed `rocprof`/`rocprofv2` provide dispatch traces, timestamps, and counters but no internal semantic phase endpoints. The installed rocprof-compute sources expose stochastic/host-trap PC sampling (block 21), which can attribute samples by original-HSACO code-object offset, instruction index, stall reason, and optional source-line comment; it cannot produce exact semantic phase start/end timestamps or phase wall-clock durations. Its runtime is also unavailable here because required Python packages are missing. The final timestamp probe's proven chain is recorded in `phase_method_feasibility.json`; its aperture failure makes the altered object invalid for timing.

The first raw A29-style prefix attempt stopped at `hipModuleLoad` for `C26_frontend_to_first_bf16.hsaco` with runtime return code 209. A later metadata-preserved prefix smoke was still only diagnostic smoke and is excluded from phase results. Complete initial exception evidence is `prefix_probe_smoke_failure.json`.

C69 whole-kernel latencies are retained only as whole-kernel references; they are not phase timings. Evidence:

`test/examples/kda/reports/C_20260927_71_K1_TRUE_BEST_MACHINE_REALIGNMENT_AUDIT/true_best_phase_timeline.json`

`test/examples/kda/reports/C_20260927_71_K1_TRUE_BEST_MACHINE_REALIGNMENT_AUDIT/phase_probe_feasibility.json`

`test/examples/kda/reports/C_20260927_71_K1_TRUE_BEST_MACHINE_REALIGNMENT_AUDIT/timestamp_correctness_failure.json`

`test/examples/kda/reports/C_20260927_71_K1_TRUE_BEST_MACHINE_REALIGNMENT_AUDIT/phase_method_feasibility.json`

## E. C26 versus TRUE_BEST W4 realignment

C26 remains the frozen Avelang production policy for NT<=101; C23 remains frozen for NT>=102. This policy was not revalidated in C71.

C26 evidence shows a 128-thread/2-wave organization with explicit lane mapping, 174 VGPRs/Tier192, 8 KiB LDS, and historical 4-WG/CU residency. W4 uses 256 threads/4 waves, 128 VGPRs, and 4 API blocks/CU/16 resident waves for the K1 prepare object.

The C26 source-level ownership results remain useful: eliminating half-wave idling, repeated Q/K reloads, transpose barriers, and serial gate recurrence is an internal causal result. What must change is the external reference label: old AITER comparisons were PUBLIC W2 or older AITER objects, not the C69 W4 TRUE_BEST family.

The exact compiler layout attributes for selected W4 objects are now recorded in `compiler_layout_search.json`; T128/T1024 hash-specific layouts, semantic phase endpoints, LDS slot ownership beyond explicit TTGIR attributes, and phase-level fragment attribution remain unavailable rather than guessed.

C69 extended `w1_s1` remains appendix-only and is not a C71 target or resource target. C23/C26 production selection and the C29 dispatch policy are frozen inputs; C71 does not revalidate them.

Detailed comparison:

`test/examples/kda/reports/C_20260927_71_K1_TRUE_BEST_MACHINE_REALIGNMENT_AUDIT/avelang_c26_vs_true_best_w4.json`

## F. Historical reference reclassification

The internal causal results are preserved. AITER-relative and SOTA claims based on PUBLIC W2 are reclassified; PUBLIC W2 is now labeled `AITER_PUBLIC_W2_REFERENCE`. The covered historical IDs are explicitly `C17`, `C23`, `C26`, `C28`, `C29`, `C30`, `C31`, `C32`, `C33`, `C34`, `C35`, and `C36`. C23/C29 long-T strategy remains open/frozen as instructed.

`test/examples/kda/reports/C_20260927_71_K1_TRUE_BEST_MACHINE_REALIGNMENT_AUDIT/historical_reference_reclassification.json`

## Required conclusions

A. The complete machine golden is frozen for specialized W4 A, generic W4 B, and long W2/S2 C, with exact metadata identities, ELF resources, physical ISA counts, and actual occupancy API results.

B. W4 is faster than PUBLIC W2 at short/mid T because the exact machine object performs the same MFMA work with a smaller surrounding instruction/VMEM/LDS/permute/wait envelope and a four-wave ownership/scheduling partition. This is not a claim based on block size alone.

C. Realignment plan: retain C26’s proven internal frontend/ownership causal work and frozen NT policy; stop using PUBLIC W2 as the SOTA target; compare short/mid-T work against W4 and long-T work against W2/S2. Exactly one next optimization direction is prioritized:

“Design a read-only four-wave C26 frontend-to-BF16-feed ownership-repartition probe, preserving the C26 production policy; do not implement or tune it in C71.”

No W4 Avelang candidate was created. No performance tuning was performed.
