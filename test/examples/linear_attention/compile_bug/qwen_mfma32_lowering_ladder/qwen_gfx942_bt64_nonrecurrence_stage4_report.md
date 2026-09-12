# Qwen gfx942 BT64 Non-Recurrence Stage 4

## Summary

Stage 4 is complete as an opt-in, high-level Avelang experiment. It replaces
the Stage 3 BT64 KKT, W/U, and chunk-o schedules while reusing cumsum, v18
solve, and the frozen gfx942 asm-v0 recurrence unchanged.

At T=2048, median of three independent warmup-10/repeat-50 sessions:

| implementation | full ms | relative to Stage 4 |
|:--|--:|--:|
| Stage 3 BT64 | `1.089079` | `2.2887x` slower |
| v24 BT16 | `0.587173` | `1.2339x` slower |
| **Stage 4 BT64** | **`0.475867`** | `1.0000x` |
| vLLM BT64 | `0.365043` | Stage 4 is `1.3036x` slower |

The main `<0.75 ms`, v24-beating, and `<=1.5x vLLM` targets all pass. This
does not change production dispatch. The experimental public entry is
`qwen_gdn_full_bt64_stage4_all_s0`; the historical name contains KKT-S0,
residual-MFMA WU-S1, and chunk-o-S0.

## Historical Audit

The audit covered v14 W/U, v18 parallel solve, v20 BT32 MFMA, v24 KKT/full,
Stage 2 scalar BT64, Stage 3 native W/U/chunk-o, and the frozen asm-v0 path.
Detailed successful/failed method mapping is in
`codex_qwen_bt64_nonrecurrence_stage4/historical_method_matrix.md/json`.

- KKT reused v24's verified MFMA16 token-dot primitive.
- W/U reused v14's main-product/residual idea but made both terms MFMA.
- chunk-o reused Stage 3/v24 MFMA16 math with BT64 CTA ownership.
- solve retained v18's correctness-proven parallel recurrence after audit.
- recurrence reused asm-v0 byte-for-byte.

Stage 3 was reproduced with the same full-path methodology. Three new vLLM
full sessions were obtained. A complete isolated vLLM KKT/W/U/chunk-o split
was not captured; only full latency and a direct solve comparison are claimed.

## Native KKT

Kernel: `_qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0`.

Each 64-thread CTA computes one of the 4x4 token16 tiles of a BT64
chunk/value-head matrix. Lower/diagonal tiles stage two `[16,128]` K tiles,
run eight MFMA16 reductions, and apply causal mask, beta, and FP32 decay at
writeback. The output remains layout-compatible with v18 solve.

| T | Stage 3 ms | Stage 4 ms | speedup |
|--:|--:|--:|--:|
| 512 | `0.145897` | `0.034731` | `4.2007x` |
| 1024 | `0.213637` | `0.035513` | `6.0157x` |
| 2048 | `0.351423` | `0.046589` | `7.5430x` |

KKT maximum absolute error was `4.47e-8`; solve-after-KKT was `3.73e-8`.

## Native W/U

Kernels: `_qwen_gdn_w_bf16_kernel_bt64_mfma_v2_s0` and
`_qwen_gdn_u_bf16_kernel_bt64_mfma_v2_s0`; final wrapper:
`qwen_gdn_w_u_bt64_mfma_v2_s1`.

One 256-thread CTA owns a full BT64 row extent and a 16-column output tile;
four waves own four token16 rows and reuse shared A/K-or-V tiles. This cuts
CTA count fourfold. The main BF16 product is followed by a second BF16 MFMA
of `A_fp32 - bf16(A)`, replacing the exact S0 scalar 64-term correction.

| T | Stage 3 ms | exact S0 ms | residual-MFMA S1 ms | S1 speedup |
|--:|--:|--:|--:|--:|
| 512 | `0.123203` | `0.095522` | `0.060971` | `2.0207x` |
| 1024 | `0.149723` | `0.146318` | `0.064196` | `2.3323x` |
| 2048 | `0.248470` | `0.229160` | `0.084165` | `2.9522x` |

S1 differs from exact S0 by at most `1.88e-6`. The faster no-correction
candidate failed the frozen full contract and remains named as a failed
diagnostic; it is not called by the final path.

## Native chunk-o

Kernel: `_qwen_gdn_chunk_o_bf16_kernel_bt64_mfma_v2_s0`.

One 256-thread CTA owns `(chunk,value-head,V16)` and its four waves own all
four output token16 tiles. It stages/reuses Q, H, K, and V-new across the
BT64 tile. Inter-state and causal intra-chunk accumulators remain separate in
FP32 and are summed once at writeback. A single merged accumulator was tested
but rejected after numerical error reached roughly `2e-2` to `3.4e-2`.

| T | Stage 3 ms | Stage 4 ms | speedup |
|--:|--:|--:|--:|
| 512 | `0.087570` | `0.044406` | `1.9720x` |
| 1024 | `0.141130` | `0.062573` | `2.2554x` |
| 2048 | `0.228620` | `0.095342` | `2.3979x` |

Random T=64/128/512 and dedicated inter/intra/source0/source1/source2
cross-token16 checks are bit-exact against Stage 3.

## Solve Audit

| T | v18 FP32 ms | vLLM FP32 ms | ratio | FP32 max abs |
|--:|--:|--:|--:|--:|
| 512 | `0.121140` | `0.053359` | `2.270x` | `2.98e-8` |
| 2048 | `0.126888` | `0.053779` | `2.359x` | `2.98e-8` |

The evidence permits a future solve experiment, but vLLM uses a distinct
hierarchical block inverse/dot schedule. Stage 4 does not add a rushed solve
variant after already meeting every full target. A separately gated FP32
hierarchical solve is the next high-level action.

## Incremental Integration

T=2048, three-session median:

| graph | full ms | gain from previous |
|:--|--:|--:|
| Stage 3 | `1.089079` | baseline |
| + KKT-S0 | `0.795162` | `1.3696x` |
| + WU-S1 | `0.610087` | `1.3034x` |
| + chunk-o-S0 | `0.475867` | `1.2821x` |

## Stage Breakdown

| T | cumsum | KKT | solve | W/U | asm recurrence | chunk-o |
|--:|--:|--:|--:|--:|--:|--:|
| 512 | 0.0311 | 0.0331 | 0.1193 | 0.0574 | 0.0959 | 0.0429 |
| 2048 | 0.0310 | 0.0453 | 0.1257 | 0.0812 | 0.2015 | 0.0928 |
| 8192 | 0.0313 | 0.1473 | 0.2326 | 0.2038 | 0.6238 | 0.2574 |
| 16384 | 0.0312 | 0.2699 | 0.3394 | 0.3725 | 1.2233 | 0.4793 |

Stage timings are independent dispatch measurements and do not sum exactly
to cached full latency.

## Full Scaling

| T | Stage 4 | Stage 3 | v24 | vLLM |
|--:|--:|--:|--:|--:|
| 512 | `0.313706` | `0.490168` | `0.300726` | `0.304873` |
| 2048 | `0.475867` | `1.089079` | `0.587173` | `0.365043` |
| 8192 | `1.399619` | `3.655211` | `2.122734` | `0.725157` |
| 16384 | `2.632552` | `7.213319` | `4.252098` | `1.268545` |

Stage 4 is slightly slower than v24/vLLM at T=512, but clearly beats v24
from T=2048 onward. Its long-sequence gap to vLLM remains about 1.9x to 2.1x.

## Correctness

The frozen 37-case matrix covers random T=64/128/512/2048, zero/nonzero
initial state, neutral gate, high dynamic range, cancellation, small values,
and T=8192 smoke cases.

| public quantity | maximum abs | threshold | result |
|:--|--:|--:|:--|
| BF16 output | `0.001953125` | `0.0078125` | pass |
| FP32 final state | `0.013475478` | `0.020000000` | pass |

All 37 cases passed. Standalone Stage 4 pytest was `29 passed in 23.06s`.
The unchanged asm-v0, external bridge, P16, gfx942 smoke, Stage 2, and Stage 3
regressions were `80 passed, 1 skipped in 79.38s`.

## rocprof and ISA

T=2048 counter-instrumented tail medians:

| kernel | trace us | WG | LDS B | VGPR | AccVGPR | scratch | MFMA | VALU | VMEM |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| KKT | 40.941 | 64 | 8192 | 20 | 4 | 0 | 20480 | 1565696 | 210944 |
| W | 27.722 | 256 | 2560 | 52 | 4 | 0 | 262144 | 5160960 | 458752 |
| U | 24.797 | 256 | 2560 | 48 | 8 | 0 | 262144 | 4005888 | 393216 |
| chunk-o | 68.622 | 256 | 27136 | 112 | 64 | 0 | 458752 | 12918784 | 851968 |

Every code object reports private segment `0`, VGPR spill count `0`, and SGPR
spill count `0`. Static ISA contains MFMA16 counts KKT/W/U/chunk-o =
`8/8/8/56`; all contain zero MFMA32. No v29-style scratch/high-AGPR cliff
appeared.

## Decision

Stage 4 validates the hypothesis that the remaining Stage 3 gap was primarily
high-level ownership, reuse, and correction schedule. Only experimental
high-level Python/Avelang code and evidence files were added; asm-v0,
compiler/RA, and production v23/v24/v26/v27/v28 were not modified.

The largest absolute T=2048 stage is the frozen recurrence (`0.2015 ms`). The
largest mutable non-recurrence stage is solve (`0.1257 ms`), and its measured
2.36x gap to vLLM makes a hierarchical FP32 BT64 solve the next single action.
No compiler or new assembly work is justified by Stage 4 data.

All evidence, exact commands, raw sessions, tests, HSACOs, ISA, and rocprof
CSVs are under `codex_qwen_bt64_nonrecurrence_stage4/`; the structured result
is `final_decision.json`.
