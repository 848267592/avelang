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
