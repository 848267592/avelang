# Qwen GDN Next Decision After Direct-K64 BV32 B2 Lookahead

## Decision

Do not retain B2 as a native recurrence baseline. Keep B0 as the correct
native Direct-K64 BV32 baseline and keep the current-vLLM external HSACO
bridge as the fastest correctness-qualified recurrence path.

B2 proved that a single-LDS-bank one-chunk lookahead can preserve the P2
feedback contract, but its source representation is not resource-safe:

| T | B0 ms | B2 ms | B2 / B0 |
|---:|---:|---:|---:|
| 512 | `0.157354` | `0.245926` | `1.563x` |
| 1024 | `0.357832` | `0.557949` | `1.559x` |
| 2048 | `0.676045` | `1.084111` | `1.604x` |
| 8192 | `2.778029` | `4.248433` | `1.529x` |

The body slope rises from `21.763744` to `33.150944 us/chunk`. B2 still has
the same `65,536` dynamic MFMA at T=2048 and the same 13 static barriers, so
the regression is not purchased by different math.

## Why B2 Stops

The lookahead creates two full per-thread local arrays, `w_next[64]` and
`k_next[64]`, which remain live across current pred and both update K halves.
Exact full-LTO MIR shows:

```text
35 SI_SPILL_AV32_SAVE + 87 SI_SPILL_AV64_SAVE
= 209 VGPR spill words
= 840 B private segment
```

At T=2048 B2 therefore has scratch `840 B`, VGPR spill count `209`,
Accum_VGPR `384`, VMEM `641,824` and trace median `1,089.459 us`; B0 has no
scratch/spills, Accum_VGPR `336`, VMEM `315,904` and trace median `689.745 us`.

B2 does eliminate the forbidden V-new global reload and uses the current
preloaded K LDS bank. The failure is specifically the materialized
next-W/K local-register window, not a recurrence correctness issue.

## Only Valid Future Direction

Do not add ping-pong, double buffering, further lookahead, a new LDS layout,
or an RA tweak to B2. A future pipeline effort must first demonstrate an
experimental typed global-load-to-LDS streaming/transaction representation
that survives to late AMDGPU lowering without keeping an entire next W/K block
as ordinary thread-local values. It needs its own isolated spill and latency
gate before returning to full recurrence.

Detailed evidence: [B2 report](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_direct_k64_bv32_one_chunk_lookahead_b2_report.md).
