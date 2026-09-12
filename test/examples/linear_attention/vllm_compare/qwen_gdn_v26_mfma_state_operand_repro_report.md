# v26 MFMA State-Operand Minimal Repro

## Purpose

This repro isolates the v26 backend performance question:

```text
Does Avelang generate excessive VGPRs, accumulator registers, moves, or scratch
when persistent accumulator/register state is later converted/reused as the B
operand of another BF16 MFMA?
```

It intentionally removes full GDN structure:

- no `g` or `gk`
- no `h` store
- no `vn` store
- no `final_state`
- no real Qwen indexing

## Files

- `repro_v26_mfma_state_operand.py`
- `bench_repro_v26_mfma_state_operand.py`
- `qwen_gdn_v26_mfma_state_operand_repro_report.md`

## Shape

- `BT=64`
- `Kq=32`
- `NT=32` chunks inside one persistent CTA
- `BV=32` and `BV=16`
- default benchmark grid: `num_blocks=64`
- dtype: BF16 operands, FP32 accumulators/state

## Variants

| Variant | Meaning |
|:---|:---|
| `persistent` | state lives in accumulator/register fragments; every chunk converts stages state to BF16 LDS and uses it as pred MFMA B operand |
| `lds` | state lives in FP32 LDS; every chunk converts stages LDS state to BF16 LDS and uses it as pred MFMA B operand |
| `constant` | state lives in accumulator/register fragments, but pred uses a constant BF16 B operand |
| `update_only` | state lives in accumulator/register fragments; skips pred and state-to-B-operand conversion |

## Commands

Correctness here is only a smoke/run check because variants intentionally do
different math.  The output checksum prevents dead-code elimination.

```bash
python bench_repro_v26_mfma_state_operand.py --bv 32 16 --variant all --warmup 10 --repeat 30
```

Profile one variant:

```bash
/opt/rocm/bin/rocprofv3 \
  --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex repro_v26_mfma_state_operand \
  -d test/examples/linear_attention/rocprof_outputs/repro_v26_mfma_state_operand_persistent_bv32 \
  -o repro_v26_mfma_state_operand_persistent_bv32 \
  -f csv \
  -- python test/examples/linear_attention/vllm_compare/bench_repro_v26_mfma_state_operand.py \
       --bv 32 --variant persistent --warmup 2 --repeat 5
```

## Benchmark

Command:

```bash
docker exec ac739c57a0bf sh -lc \
  'cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare && \
   python bench_repro_v26_mfma_state_operand.py --bv 32 16 --variant all --warmup 10 --repeat 30'
```

Results:

| BV | variant | latency ms | checksum |
|---:|:---|---:|---:|
| 32 | `persistent` | 0.264353 | 0.014194695 |
| 32 | `lds` | 0.292774 | 0.014194695 |
| 32 | `constant` | 0.256542 | 0.018248774 |
| 32 | `update_only` | 0.257402 | 0.014142111 |
| 16 | `persistent` | 0.185796 | 0.015065949 |
| 16 | `lds` | 0.197353 | 0.015065949 |
| 16 | `constant` | 0.179867 | 0.018538345 |
| 16 | `update_only` | 0.176122 | 0.014990628 |

Latency deltas:

| BV | comparison | delta ms | delta |
|---:|:---|---:|---:|
| 32 | persistent - constant | +0.007811 | +3.0% |
| 32 | persistent - update_only | +0.006951 | +2.7% |
| 32 | lds - persistent | +0.028421 | +10.8% |
| 16 | persistent - constant | +0.005929 | +3.3% |
| 16 | persistent - update_only | +0.009674 | +5.5% |
| 16 | lds - persistent | +0.011557 | +6.2% |

## rocprof

Command template:

```bash
cd /workspace/project/avelang
for bv in 32 16; do
  for variant in persistent lds constant update_only; do
    outdir=test/examples/linear_attention/rocprof_outputs/repro_v26_mfma_state_operand_${variant}_bv${bv}
    /opt/rocm/bin/rocprofv3 \
      --kernel-trace \
      --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
      --kernel-include-regex repro_v26_mfma_state_operand \
      -d ${outdir} \
      -o repro_v26_mfma_state_operand_${variant}_bv${bv} \
      -f csv \
      -- python test/examples/linear_attention/vllm_compare/bench_repro_v26_mfma_state_operand.py \
           --bv ${bv} --variant ${variant} --warmup 2 --repeat 5
  done
done
```

Output dirs:

```text
test/examples/linear_attention/rocprof_outputs/repro_v26_mfma_state_operand_{variant}_bv{16,32}
```

Median kernel counters and trace metadata:

| BV | variant | trace us | WG | grid | LDS block | scratch | VGPR | AccVGPR | SGPR | MFMA | VALU | SALU | VMEM | LDS inst | occupancy |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | `persistent` | 268.099 | 64 | 4096 | 11264 | 0 | 96 | 144 | 112 | 163840 | 1106240 | 269056 | 197632 | 385024 | 0.6348 |
| 32 | `lds` | 307.236 | 64 | 4096 | 15360 | 0 | 92 | 140 | 112 | 163840 | 1110016 | 335872 | 197632 | 477184 | 0.6359 |
| 32 | `constant` | 272.786 | 64 | 4096 | 11264 | 0 | 92 | 140 | 112 | 163840 | 911680 | 272640 | 197632 | 353280 | 0.6325 |
| 32 | `update_only` | 236.011 | 64 | 4096 | 8192 | 0 | 128 | 136 | 64 | 131072 | 419008 | 331008 | 132096 | 311296 | 0.6304 |
| 16 | `persistent` | 201.680 | 64 | 4096 | 11264 | 0 | 64 | 136 | 112 | 81920 | 601088 | 138176 | 164352 | 237568 | 0.6268 |
| 16 | `lds` | 213.197 | 64 | 4096 | 15360 | 0 | 68 | 140 | 112 | 81920 | 599808 | 173632 | 164352 | 271360 | 0.6284 |
| 16 | `constant` | 196.072 | 64 | 4096 | 11264 | 0 | 64 | 136 | 112 | 81920 | 506304 | 140096 | 164352 | 221696 | 0.6222 |
| 16 | `update_only` | 157.294 | 64 | 4096 | 8192 | 0 | 84 | 132 | 64 | 65536 | 224832 | 200000 | 98816 | 163840 | 0.6194 |

Important counter deltas:

| BV | comparison | VGPR delta | AccVGPR delta | scratch delta | VALU delta | LDS inst delta | trace delta us |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 32 | persistent - constant | +4 | +4 | 0 | +194560 | +31744 | -4.687 |
| 32 | persistent - update_only | -32 | +8 | 0 | +687232 | +73728 | +32.088 |
| 32 | lds - persistent | -4 | -4 | 0 | +3776 | +92160 | +39.137 |
| 16 | persistent - constant | 0 | 0 | 0 | +94784 | +15872 | +5.608 |
| 16 | persistent - update_only | -20 | +4 | 0 | +376256 | +73728 | +44.386 |
| 16 | lds - persistent | +4 | +4 | 0 | -1280 | +33792 | +11.517 |

## Interpretation

This minimal repro does not show a catastrophic backend failure for the
persistent accumulator/register state pattern by itself.

Observed:

- `persistent` has no scratch for both BV=32 and BV=16.
- `persistent` VGPR/AccVGPR are moderate:
  - BV=32: `VGPR=96`, `AccVGPR=144`
  - BV=16: `VGPR=64`, `AccVGPR=136`
- Compared with `constant`, the persistent state-to-B-operand path adds only a
  small normal benchmark cost:
  - BV=32: `+0.007811 ms`
  - BV=16: `+0.005929 ms`
- The extra cost mainly appears as VALU and LDS movement:
  - BV=32 persistent vs constant: `+194560 VALU`, `+31744 LDS`
  - BV=16 persistent vs constant: `+94784 VALU`, `+15872 LDS`
- `lds` is slower than `persistent`, mostly due to larger LDS footprint and more
  LDS instructions:
  - BV=32 LDS block: `15360` vs `11264`
  - BV=16 LDS block: `15360` vs `11264`
- `update_only` is much cheaper in trace because it removes the pred path:
  fewer MFMA, VMEM, VALU, and LDS instructions. That confirms pred/state operand
  construction is measurable, but not uniquely disastrous in this reduced setup.

Conclusion:

The reduced BT=64/BV={32,16}/Kq=32/NT=32 repro does not reproduce the large v26
regression as a standalone "persistent accumulator reused as MFMA B operand"
backend pathology. The persistent variant is only a few microseconds slower than
the constant-state variant and does not spill.

This suggests the full v26 issue likely needs one or more additional ingredients:

- larger real GDN state/reduction shape, especially K=128 instead of Kq=32
- more waves or more live accumulator fragments than this 64-thread repro
- pred partial reduction and distributed token/value coordination
- real g/decay and output/writeback pressure
- interaction between persistent state fragments and full chunk_gdr scheduling

Recommended follow-up repros for the backend team:

1. Increase Kq from `32` to `64` and then `128`, keeping the same four variants.
2. Add a 4-wave or 8-wave version with multiple live pred/update accumulator
   groups, closer to v23/v26 chunk_gdr pressure.
3. Keep the same `persistent` / `constant` / `update_only` split, because it
   cleanly separates B-operand construction from the rest of the MFMA work.
