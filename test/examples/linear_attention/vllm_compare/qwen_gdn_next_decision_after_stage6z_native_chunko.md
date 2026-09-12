# Next Decision After Stage 6Z Native Chunk-O

## Decision

**Close Stage 6Z at Z1. Keep Stage6X X2 as the Avelang BT64 experimental
baseline and keep v24 as production/default.**

Z0 completed its native vLLM audit and Z1 passed isolated correctness plus
showed a positive body effect. However, Z1's captured ISA contains 41 static
`s_barrier` instructions, violating the Stage 6Z hard resource gate
`barrier < 19`. The experiment must not proceed to Z2 full integration.

| gate | result |
|:--|:--|
| Z0 final native specializations captured | pass |
| Z1 BF16 isolated correctness T64 to T8192 | pass, max abs <= 1.526e-5 |
| Z1 zero V-new/reuse/rejection checks | pass |
| Z1 scratch/spill | pass, zero |
| Z1 AccVGPR / LDS | pass, 172 / 28672 B |
| Z1 barrier | **fail, 41 >= 19** |
| Z1 T2048/T8192 isolated body | positive, 1.220x / 1.284x |
| Z2 full graph | not created |

## What The Evidence Means

The native-style V64 ownership is a real source of recoverable work: it
reduces Stage6W's Q/K score repetition from eight V16 blocks per chunk-head
to two V64 blocks and reduces caller-owned body slope by about 0.407
us/chunk in this prototype. But the Avelang Z1 expression implements that
schedule with an unsafe resource shape: source staging and score/V phases
need 41 static barriers. The correct action is to preserve this evidence, not
to accept the body win and create a full graph with an unbounded long-text
risk.

## Explicit Non-Actions

- No Stage6Z Z2 full wrapper or Eager public benchmark.
- No change to X2, Stage6U W/U, immutable recurrence, compiler, or selector.
- No Z1b tile/workgroup sweep and no revival of O0.
- No production/default promotion.

The next optimization must be chosen by a new, separately audited decision.
It cannot be a continuation of this failed barrier envelope under a different
name.

## Fixed-source re-evaluation decision

The earlier decision predates the repair of the Phase-B dead K overread. The
repaired Z2/Z3 were subsequently checked in independent fresh processes at
T64/512/1024/2048/4096/8192/16384 and both passed the BF16 correctness contract;
Z3 was byte-exact with Z2. The old Z2 timing and PMC remain historical and are
not used below.

The valid caller-owned body results are:

| T | Z2 WG256 | Z3 WG128 | native selected | Z3/Z2 |
|---:|---:|---:|---:|---:|
| 512 | 0.061431 ms | 0.074491 ms | 0.036194 ms | 1.213x |
| 1024 | 0.063634 ms | 0.078456 ms | 0.037756 ms | 1.233x |
| 2048 | 0.077475 ms | 0.094661 ms | 0.042763 ms | 1.222x |
| 4096 | 0.110724 ms | 0.126388 ms | 0.056224 ms | 1.142x |
| 8192 | 0.177203 ms | 0.220769 ms | 0.091917 ms | 1.246x |
| 16384 | 0.317692 ms | 0.393445 ms | 0.141591 ms | 1.238x |

Z3 also has static barrier=21 versus Z2=9, despite zero scratch/spill. It is
slower at every length and fails the pre-registered `barrier < 19` and long-text
speed gates. Therefore the fixed-source decision is unchanged in substance:
do not create a selector, do not connect X2, and do not run X2 public Eager.
The current Avelang isolated baseline is repaired Z2; native selected chunk-o is
the faster external diagnostic reference.
