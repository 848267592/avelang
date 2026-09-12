# Direct-K64 Three-Way Recurrence-Update Dataflow Audit

## 1. Scope and Decision Boundary

This is an isolated, diagnostic recurrence-update suffix experiment on
gfx942. It deliberately does **not** modify a production path, the current
external recurrence bridge, allocator/register allocation, broad-K/compact-K,
or the unresolved full-v29 nonzero-W pred path.

All arms consume preallocated tensors with the current-vLLM storage boundary:

| input/output | dtype | layout |
|:--|:--|:--|
| K | BF16 | `[1, T, 4, 128]` |
| V-new input | BF16 | `[1, T, 8, 128]` |
| g | FP32 | `[1, T, 8]` |
| initial state / final state | FP32 | `[1, 8, 128, 128]` |
| per-chunk H snapshot | BF16 | `[1, T/64, 8, 128, 128]` |

The mathematical suffix is:

```text
H <- exp(g_last) * H + K^T @ (V-new * exp(g_last - g_token))
```

The test has three arms:

1. `current_triton_w0_control`: captured current-vLLM Triton recurrence
   HSACO. It is invoked with `w=0` and `v=V-new`, so its update receives the
   same BF16 V-new values. Triton has no exported update-only entry, therefore
   this control still executes its fused pred pipeline. Its timing is thus a
   conservative native control, not a pure update-only body.
2. `direct_k64_mfma32`: Avelang direct-K64 update, using BF16
   `mfma_32x32x8_bf16_f32`.
3. `direct_k64_mfma16`: same Avelang ABI, launch, state/output layout and
   update math, but a BF16 `mfma_16x16x16_bf16_f32` decomposition.

The two Avelang arms share `BT=64`, `BV=64`, `K=128`, `WG=128`, 16 persistent
CTAs and the same `[V64, K128]` state per CTA. They are a strict geometry
control. Current Triton instead uses `BV=32` and 32 CTAs. Consequently this is
strong dataflow/lowering evidence, but it is **not** a compiler-only,
same-CTA-ownership A/B between Avelang and Triton.

## 2. Native Current-vLLM Audit

The audited artifacts are frozen under:

`codex_qwen_bt64_recurrence_reconciliation_stage6r/current_kernels/vllm/`

| property | current Triton evidence |
|:--|:--|
| symbol | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` |
| launch | grid `(4, 8, 1)` at T=2048, 32 CTAs; WG=128, two warps, two stages |
| tile | `BT=64`, `BV=32` |
| persistent state | FP32 `H1[32,64]` and `H2[32,64]` per CTA |
| K update blocks | two BF16 `K[64,64]` blocks, one for each K64 half |
| update dot | BF16 `K[64,64] @ V[64,32] -> FP32 [64,32]`, then transpose/add into H1/H2 |
| final MFMA ISA | `v_mfma_f32_32x32x8_bf16` |
| fixed ABI LDS | 40,960 B dynamic LDS |

The TTIR records two update loads and dots:

```mlir
%k0 = tt.load ... : tensor<64x64x!tt.ptr<bf16>>
%h1 = tt.dot %k0, %v, ... : tensor<64x64xbf16> * tensor<64x32xbf16>
                              -> tensor<64x32xf32>
%k1 = tt.load ... : tensor<64x64x!tt.ptr<bf16>>
%h2 = tt.dot %k1, %v, ... : tensor<64x64xbf16> * tensor<64x32xbf16>
                              -> tensor<64x32xf32>
```

The TTGIR retains this as block operations: two `64x64` BF16 shared K
memdescs, two `64x64` BF16 W memdescs and `ttg.local_load` directly into
`ttg.dot_op` operands. The captured ISA has 64 static
`v_mfma_f32_32x32x8_bf16` instructions, 32 static `s_barrier`, 150 static
`ds_read` and 183 static `ds_write` instructions. The static barrier count
cannot be compared directly with Avelang's count because the loop bodies and
unrolling differ.

`rocprofv3` reports `LDS_Block_Size=0` for this external HSACO; the captured
launch ABI, not that collector field, is authoritative for its 40,960 B
dynamic LDS allocation.

## 3. Avelang Controls

### 3.1 Direct-K64 MFMA32

`repro_qwen_gdn_direct_k64_update_current_abi.py` has two waves per CTA. Each
wave owns four FP32 32x32 accumulator fragments covering `V32 x K128`:

```text
h1_lo[V32,K0:32], h1_hi[V32,K32:64]
h2_lo[V32,K64:96], h2_hi[V32,K96:128]
```

It stages only immediate operands, not `k_all_t` or a broad transposed K
view:

```python
a_stage = al.make_shared((2, 32, 32), al.bf16)  # V-decay
b_stage = al.make_shared((32, 32), al.bf16)     # direct K
a_vec = al.view(a_stage, al.i32, ...)
b_vec = al.view(b_stage, al.i32, ...)
update_acc = al.amdgpu.mfma_32x32x8_bf16_f32(...)
```

The shared footprint is exactly `2*32*32*2 + 32*32*2 = 6,144 B`, matching
the profiler.

### 3.2 Direct-K64 MFMA16 decomposition

`repro_qwen_gdn_direct_k64_update_mfma16_current_abi.py` changes only the
update-dot decomposition. It stages immediate `[V16,T16]` and `[K16,T16]`
inputs and uses the proven MFMA16 lane mapping. Its output ownership differs
from the persistent MFMA32 H layout, so an explicitly shared
`delta_stage[64,128]` FP32 tile converts it back before the same H1/H2 update.

```python
a_stage = al.make_shared((2, 16, 16), al.bf16)
b_stage = al.make_shared((16, 16), al.bf16)
delta_stage = al.make_shared((64, 128), al.f32)
update_acc = al.amdgpu.mfma_16x16x16_bf16_f32(...)
```

This is `1,024 + 512 + 32,768 = 34,304 B` LDS, also exactly matching the
profiler. It is intentionally a geometry control, not a proposed optimized
implementation.

## 4. Correctness

The reference is the direct update recurrence, not full v29. Both Avelang
arms passed the T=64/T=512 pytest suite (`4 passed`) and the three-way
preallocated benchmark remained finite at every length. Native V-new is
bit-exact to the supplied BF16 input in the `w=0` control.

| T | arm(s) | H max abs vs update reference | final-state max abs vs update reference |
|--:|:--|--:|--:|
| 512 | native / MFMA32 / MFMA16 | `0.25` | native `1.53e-05`; Avelang `1.53e-05` |
| 1024 | native / MFMA32 / MFMA16 | `0.25` | native `5.34e-05`; Avelang `3.05e-05` |
| 2048 | native / MFMA32 / MFMA16 | `0.50` | native `9.16e-05`; Avelang `4.58e-05` |

The H difference is expected BF16 snapshot rounding. The direct recurrence
has no NaN/Inf and the FP32 final-state differences remain small. This does
not make a claim about the separate full-v29 nonzero-W pred correctness issue.

## 5. Preallocated Body Timing

These are medians of five session medians. Each session used preallocated
inputs/outputs, precompiled modules, warmup=5, repeat=20 and an `ABCCBA`
balanced order. HIP events are the body-latency authority; rocprof trace is
used only for resources/counters.

| T | chunks | Triton W0 control ms | Avelang MFMA32 ms | MFMA32 / Triton | Avelang MFMA16 ms | MFMA16 / MFMA32 |
|--:|--:|--:|--:|--:|--:|--:|
| 512 | 8 | `0.038978` | `0.294818` | `7.563x` | `0.549116` | `1.863x` |
| 1024 | 16 | `0.065217` | `0.567443` | `8.701x` | `1.080847` | `1.905x` |
| 2048 | 32 | `0.116653` | `1.133085` | `9.714x` | `2.163236` | `1.909x` |

From T=512 to T=2048, endpoint slopes are approximately `3.24 us/chunk` for
the native W0 control, `34.93 us/chunk` for Avelang MFMA32 and
`67.26 us/chunk` for Avelang MFMA16. Since the native arm still performs pred
work, its pure update-only slope can only be lower than this measured native
control; the source-level update gap is not understated by omitting pred from
the Avelang arms.

## 6. T=2048 rocprof Counters

Each row is the median of eight matching dispatches from the same repeat
profile. `trace_us` is profiler-perturbed and should not replace Section 5.

| arm | CTAs | WG | trace us | VGPR | AccVGPR | SGPR | LDS B | scratch | occupancy | MFMA | VALU | SALU | VMEM | LDS inst |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| native W0 control | 32 | 128 | `113.429` | 104 | 160 | 96 | ABI `40960` | 0 | `0.5943` | 65,536 | 1,395,584 | 64,832 | 58,368 | 305,472 |
| Avelang MFMA32 | 16 | 128 | `1121.527` | 24 | 144 | 112 | 6,144 | 0 | `0.3243` | 32,768 | 6,336,416 | 1,087,456 | 354,304 | 229,376 |
| Avelang MFMA16 | 16 | 128 | `2149.916` | 52 | 212 | 112 | 34,304 | 0 | `0.3257` | 262,144 | 5,763,200 | 598,016 | 673,792 | 966,656 |

MFMA32 has half the *grid-wide* MFMA count because it owns `BV=64` with 16
CTAs, while native owns `BV=32` with 32 CTAs. Normalized per CTA, both native
and Avelang MFMA32 execute exactly `2,048` dynamic MFMA instructions:

| T=2048 per CTA | native | MFMA32 | MFMA32 / native |
|:--|--:|--:|--:|
| MFMA | 2,048 | 2,048 | `1.00x` |
| VMEM | 1,824 | 22,144 | `12.14x` |
| VALU | 43,612 | 396,026 | `9.08x` |
| SALU | 2,026 | 67,841 | `33.49x` |
| LDS instructions | 9,546 | 14,336 | `1.50x` |

MFMA16 is materially worse: it runs `16,384` MFMAs per CTA, eight times the
MFMA32/native per-CTA count, and its `delta_stage` raises both LDS footprint
and dynamic LDS work. It is not a viable fallback geometry.

## 7. ISA and Spill Evidence

| ISA property | native Triton | Avelang MFMA32 | Avelang MFMA16 |
|:--|--:|--:|--:|
| MFMA mnemonic | `v_mfma_f32_32x32x8_bf16` | same | `v_mfma_f32_16x16x16_bf16` |
| static MFMA | 64 | 16 | 16 |
| static `s_barrier` | 32 | 9 | 10 |
| static `ds_read` / `ds_write` | 150 / 183 | 16 / 96 | 48 / 26 |
| static `global_load` / `global_store` | 12 / 0 | 176 / 32 | 56 / 32 |
| code-object private segment | not re-audited here | 0 B | 0 B |
| code-object VGPR/SGPR spills | not re-audited here | 0 / 0 | 0 / 0 |

The static instruction counts are control-flow/unrolling evidence, not a
substitute for the dynamic PMC values. The AMDGPU code-object resource
enumeration and rocprof's `Accum_VGPR_Count` use different accounting
domains, so they should not be equated one-for-one. The stable conclusions are
that neither Avelang arm has private scratch or a register spill, and that
MFMA16's `AccVGPR=212` is not a spill event.

Artifacts:

- Avelang disassembly and code-object notes:
  `rocprof_outputs/qwen_direct_k64_three_way/isa/{mfma32,mfma16}/`
- PMC/trace CSVs:
  `rocprof_outputs/qwen_direct_k64_three_way/{mfma32,mfma16,native}/`
- captured native TTIR/TTGIR/ISA:
  `codex_qwen_bt64_recurrence_reconciliation_stage6r/current_kernels/vllm/`

## 8. What the Data Does and Does Not Prove

### Supported evidence

1. Direct K64 plus the correct MFMA32 instruction is insufficient by itself.
   Avelang MFMA32 has exactly the same dynamic MFMA work per CTA as native,
   but is `9.714x` slower at T=2048 and has much more VMEM/VALU/SALU.
2. The gap is not explained by Avelang scratch/spilling in this suffix:
   both variants have zero private segment and zero VGPR/SGPR spills.
3. The MFMA16 decomposition is not a remedy. Maintaining the same persistent
   H layout forces an explicit FP32 `delta_stage`, raising LDS to 34,304 B,
   AccVGPR to 212 and dynamic MFMA to eight times the MFMA32 path.
4. The candidate source-level cost centers are visible: explicit scalar fill
   loops for `a_stage`/`b_stage`, repeated `[k_half, col_half, token_half]`
   staging, `al.view(... i32 ...)` fragment construction, BF16 conversion and
   scalar address/decay arithmetic. Native TTGIR represents the equivalent
   dataflow as typed `local_alloc`/`local_load` dot operands instead.

### Dataflow attribution, with its limits

The aggregate PMCs cannot attribute every instruction to a single source
line. The following table therefore distinguishes source-visible facts from
an inference about their counter contribution.

| dataflow region | MFMA32 source fact | current Triton IR fact | likely observed effect |
|:--|:--|:--|:--|
| V-decay operand | `a_stage[2,32,32]` is refilled inside every `k_half x col_half x token_half` iteration. Per CTA/chunk this writes 16,384 BF16 A elements, while the unique V64-by-T64 values are only 4,096. | A typed BF16 `V[64,32]` local operand is consumed by the two K64 update dots in the TTGIR loop. | repeated V-new/g loads, decay evaluation and shared writes; a strong candidate for extra VMEM/VALU/LDS. |
| direct K operand | `b_stage[32,32]` is filled for each immediate K32/token32 tile, then packed through `b_vec`. | K remains a typed `64x64` shared memdesc and is locally loaded as a dot operand. | Avelang pays scalar tile address and packed-view construction; native retains a dot layout. |
| fragment construction | `al.view(shared, i32, ...)`, scalar word indexing and `al.view(word, Tensor((2,4,1), bf16))` occur in the MFMA loop. | `ttg.local_load` produces `#ttg.dot_op` directly. | a source-visible explanation for the SALU/VALU asymmetry, not for MFMA count. |
| state snapshot/final state | Avelang writes the BF16 H snapshot and FP32 final state explicitly. | Native has the same public H/final-state ABI. At grid scope, Avelang `16 x V64` and native `32 x V32` move the same state element count. | this interface traffic is necessary and does not by itself explain the large grid-wide difference. |
| CTA ownership | Avelang independently runs two V32 wave-local paths inside a BV64 CTA. | Two native waves cooperate on one BV32 CTA. | affects scalar scheduling, occupancy and reuse; it prevents a backend-only attribution. |

Thus the strongest present statement is: the extra VMEM/VALU/SALU is
consistent with the explicit Avelang staging/view/address dataflow, and the
direct-K MFMA32 experiment removes broad-K as the immediate explanation. It
is not yet a per-instruction proof that any one of those source constructs is
the sole cause.

### Not established by this experiment

This is not a same-source, same-ownership compiler A/B. Native has `BV=32`,
two warps cooperatively owning each V32 state, and an unavoidable fused pred
pipeline; Avelang owns V64 per CTA and evaluates only the update suffix.
Therefore the data supports a focused high-level/lowering investigation, but
does **not** yet prove that every counter difference is caused solely by the
Avelang backend rather than the explicit source schedule and CTA ownership.

## 9. Proposed Next Experiment, Not Implemented Here

The evidence is strong enough to justify a narrowly scoped
`block_dot_bf16_f32` experiment, but not to add it before a compiler-only
control exists. The proposed API would preserve the explicit shared tiles and
emit a logical block dot rather than an exposed `view(i32)` plus manually
iterated MFMA fragment sequence:

```python
acc = al.amdgpu.block_dot_bf16_f32(
    k_shared_64x64,
    v_shared_64x32,
    acc,
    m=64,
    n=32,
    k=64,
    operand_layout="gfx942_mfma32",
)
```

The required proof is a **same-source generic/specialized lowering A/B**:

- identical Python/AveLang source, shared allocation, CTA/WG ownership,
  barriers, global loads/stores and numerical result;
- generic lowering expands to the present scalar/view/fragment sequence;
- gfx942 specialized lowering preserves the logical dot until AMDGPU lowering
  and creates the native-style dot operand load sequence;
- pre-branch MLIR hash must match; post-lowering differences must be confined
  to the operation; bit-exactness, dynamic counters, scratch/spill, and
  per-CTA work must be rechecked.

Until that experiment exists, do not modify allocator/RA or restart
broad-K/compact-K tuning. The next control point is the representation of
the immediate K64/V32 block operand and its lowering, not a blanket
register-allocation intervention.

## 10. Reproduction

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:.

python3 -m pytest -q \
  test_qwen_gdn_direct_k64_update_current_abi.py \
  test_qwen_gdn_direct_k64_update_mfma16_current_abi.py -s

python3 bench_qwen_gdn_direct_k64_three_way.py \
  --T 512 1024 2048 --warmup 5 --repeat 20 --sessions 5 --json

/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex _qwen_gdn_direct_k64_update_current_abi_kernel \
  -d ../rocprof_outputs/qwen_direct_k64_three_way/mfma32 -o counters -f csv -- \
  python3 profile_qwen_gdn_direct_k64_three_way.py --implementation mfma32 --T 2048 --warmup 2 --repeat 5
```

Use the analogous kernel regex/`--implementation` for `mfma16` and `native`.
