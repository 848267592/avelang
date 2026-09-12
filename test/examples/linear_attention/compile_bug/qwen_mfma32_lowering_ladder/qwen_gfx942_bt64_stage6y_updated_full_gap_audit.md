# Qwen gfx942 BT64 Stage 6Y: Updated Five-Dispatch Full-Gap Audit

## Scope

Stage 6X X2 is frozen as the new Avelang BT64 experimental baseline. This
audit changes no kernel, compiler, recurrence HSACO, assembly ABI, or default
selector. Its purpose is to prevent a Stage 6W-era six-dispatch accounting
from choosing the next experiment after X2 has already removed the KKT FP32
`a` materialization and standalone solve dispatch.

The authority remains the uncaptured Eager public API. CUDA Graph replay,
stage bodies and rocprof are diagnostic tools only.

## What Is Already Established

The manually completed formal X2 confirmation is recorded in
`codex_qwen_bt64_kkt_solve_handoff_stage6x_manual_confirmation/`.

- X2 is bit-exact to Stage 6W for the frozen full contract.
- NaN/reuse and non-default stream gates pass.
- In 5 sessions and 50 paired Williams blocks, X2 beats W1 at every measured
  length T=512 through T=16384.
- At T=2048, the gain is 26.020 us with event CI [23.304, 28.563] us.
- At T=8192, the gain is 82.823 us with event CI [80.432, 85.154] us.
- W1/X2 slopes are 6.341/5.694 us per chunk.

This is sufficient for X2 experimental promotion. It is not a Stage 6Y
per-stage explanation. Promotion alone does not select the next optimization;
the updated body and dispatch evidence below makes that selection.

## Frozen Graph

```text
cumsum
  -> fused KKT+solve
  -> fused W/U
  -> immutable recurrence HSACO
  -> chunk-o
```

X2 has five Avelang-side logical dispatches. The removed Stage 6W boundaries
are the global FP32 `a` write, its global read by solve, and the standalone
solve dispatch. Native vLLM must be traced anew: it is not assumed to have the
same count or W/U/KKT/solve grouping.

## Current Global Intermediate Accounting

| tensor | dtype | T=2048 size / round trip | T=8192 size / round trip | producer -> consumer | status |
|:--|:--|--:|--:|:--|:--|
| `g_cumsum` | FP32 | 0.0625 / 0.3125 MiB | 0.25 / 1.25 MiB | cumsum -> four consumers | multi-use |
| `a_solved_bf16` | BF16 | 2 / 4 MiB | 8 / 16 MiB | fused KKT+solve -> fused W/U | source-native, one use |
| `w_bf16` | BF16 | 4 / 8 MiB | 16 / 32 MiB | fused W/U -> recurrence | frozen recurrence ABI |
| `u_bf16` | BF16 | 4 / 8 MiB | 16 / 32 MiB | fused W/U -> recurrence | frozen recurrence ABI |
| `h_bf16` | BF16 | 8 / 16 MiB | 32 / 64 MiB | recurrence -> chunk-o | frozen recurrence ABI |
| `v_new_bf16` | BF16 | 4 / 8 MiB | 16 / 32 MiB | recurrence -> chunk-o | frozen recurrence ABI |

`a` no longer appears in this table. Traffic alone makes `h_bf16` plus
`v_new_bf16` the largest boundary, but it crosses the immutable recurrence
ABI. `a_solved_bf16` is the largest remaining source-native one-use edge, but
fusing it would combine the X2 40 KiB LDS / AccVGPR=164 kernel with W/U. That
is a hypothesis, not a selected optimization.

## Measurement Tooling

The new runner is
`vllm_compare/audit_qwen_gdn_bt64_stage6y_full_gap.py`.

It measures two scopes with identical inputs and current stream:

1. `full_eager_public_api`: X2 public entry against vLLM public entry. This
   is the ranking authority.
2. `body_wrapper:<stage>`: cumsum, logical KKT+solve, W/U, recurrence and
   chunk-o. These are Eager wrapper diagnostics and must never be summed into
   a fake full latency. The vLLM logical KKT+solve body deliberately runs its
   native KKT followed by native solve so it is comparable to the one X2
   composite stage.

For real dispatch counts and counter identities, launch the same runner under
`rocprofv3 --kernel-trace` once for X2 and once for vLLM, identify matching
kernel names, then rerun counter collection with narrow includes. Do not use
the old Stage 6A captured-graph trace: it has a different seven/eight-dispatch
graph and cannot explain X2.

## Body Sweep Results

The Stage 6Y runner completed a process-isolated diagnostic sweep at
T=512/1024/2048/4096/8192/16384. Every T uses three sessions, five warmups
and 20 ABBA repeats per session. The table gives the median of the three
session medians, in milliseconds. This is deliberately a wrapper/body
diagnostic under a shared GPU, not a substitute for the formal X2 full Eager
ranking already completed by the user.

| T | cumsum X2 / vLLM | KKT+solve X2 / vLLM | W/U X2 / vLLM | recurrence X2 / vLLM | chunk-o X2 / vLLM |
|---:|:--|:--|:--|:--|:--|
| 512 | .033911 / .039159 | .049053 / .095983 | .071046 / .056444 | .054882 / .081982 | .054842 / .051557 |
| 1024 | .032809 / .036635 | .046950 / .092198 | .070665 / .054762 | .078617 / .106699 | .070225 / .050115 |
| 2048 | .033570 / .038277 | .048853 / .095422 | .073430 / .056464 | .129653 / .159117 | .103414 / .055703 |
| 4096 | .034571 / .039158 | .062373 / .096284 | .080239 / .067741 | .231044 / .258765 | .155111 / .070305 |
| 8192 | .034452 / .038497 | .089614 / .104415 | .124365 / .082663 | .429519 / .459283 | .264794 / .102753 |
| 16384 | .035333 / .040100 | .130374 / .113569 | .191064 / .115131 | .829675 / .861983 | .481958 / .153148 |

Linear fits across the six lengths separate fixed intercept from growth:

| body | X2 slope us/chunk | vLLM slope us/chunk | X2-vLLM us/chunk | interpretation |
|:--|--:|--:|--:|:--|
| cumsum | .008 | .008 | .000 | tied slope; X2 has lower fixed cost |
| logical KKT+solve | .348 | .082 | +.266 | X2 loses long-text slope after its strong fixed-cost win |
| W/U | .504 | .247 | +.257 | real runner-up gap |
| recurrence | 3.126 | 3.144 | -.018 | no standalone device-kernel opportunity |
| chunk-o | 1.713 | .427 | **+1.286** | dominant recoverable gap |

The bodies must not be summed: their allocations, launch boundaries and
runtime behavior differ from the complete public graph. They do agree with
the formal full result's long-text direction: chunk-o is the only component
with an approximately 330 us deficit at T=16384, larger than the complete
X2-vLLM gap of about 360 us.

## Actual T=2048 Dispatch Graph

One trace-only run was used solely to identify the final stable replay
sequence. The trace contains a large Triton autotune prefix; that prefix is
not counted. The final repeated sequences are:

```text
X2 (5):
  _qwen_gdn_chunk_cumsum_kernel_v6_standalone
  _qwen_gdn_kkt_solve_bf16_kernel_bt64_one_cta_stage6x_x2
  _qwen_gdn_wu_kernel_bt64_fused_bf16_solved_stage6u
  chunk_gated_delta_rule_fwd_kernel_h_blockdim64
  _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6w

native vLLM (7):
  chunk_local_cumsum_scalar_kernel
  chunk_scaled_dot_kkt_fwd_kernel
  BF16 FillFunctor
  merge_16x16_to_64x64_inverse_kernel
  recompute_w_u_fwd_kernel
  chunk_gated_delta_rule_fwd_kernel_h_blockdim64
  chunk_fwd_kernel_o
```

Thus X2 has fewer dispatches, not more. In particular, the long-text deficit
cannot be blamed on an X2 KKT--solve handoff or on a separate final cast.

The stable trace resource metadata makes the chunk-o ownership mismatch
concrete:

| chunk-o T=2048 | workgroup | global grid | CTA count | LDS | VGPR / AccVGPR | scratch |
|:--|--:|:--|--:|--:|:--|--:|
| X2 Stage 6W | 256 | 524288 x 1 x 1 | 2048 | 27136 B | 112 / 64 | 0 |
| native vLLM | 256 | 512 x 32 x 8 | 512 | 0 B | 100 / 36 | 0 |

At the same 256-thread workgroup, X2 launches four times as many CTAs and
uses a 27 KiB LDS tile where the selected native kernel reports no LDS. This
does not prove that CTA count alone causes the gap, but together with the
body slope it is a strong ownership/layout hypothesis.

The recurrence dispatch has the exact same symbol and resource tuple on both
paths: WG128, grid `512 x 1 x 1`, VGPR/AccVGPR `104/160`, scratch zero. The
different wrapper-body medians therefore do not justify a recurrence kernel
rewrite.

## Counter Scope

A narrow fresh-process PMC collection for the X2 chunk-o symbol was attempted
after graph identification. rocprof completed its process but did not leave a
counter CSV, so MFMA/VALU/SALU/VMEM/LDS instruction counts from that attempt
are recorded as **N/A**, not estimated. The resource metadata above comes
directly from the final kernel-trace rows. This is enough to select the next
ownership audit; any Stage 6Z promotion must rerun narrow PMC collection with
verified output artifacts.

## Decision: Stage 6Z Is Native BT64 Chunk-O Ownership

Stage 6Y selects exactly one next action:

```text
Stage 6Z: native BT64 chunk-o ownership redesign
```

The first 6Z action is an X0 source/trace ownership audit of native
`chunk_fwd_kernel_o`, followed by one isolated BT64 chunk-o prototype. It
must preserve the Stage 6X BF16 `v_new`, BF16 `h`, FP32 accumulation and BF16
public-output contract. It must not change recurrence HSACO, X2 KKT+solve,
W/U, compiler, or the default selector. W/U remains the runner-up and is
explicitly deferred.

## Reproduction

```bash
cd /workspace/project/avelang
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  test/examples/linear_attention/vllm_compare/audit_qwen_gdn_bt64_stage6x_intermediates.py \
  test/examples/linear_attention/vllm_compare/audit_qwen_gdn_bt64_stage6y_full_gap.py

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/audit_qwen_gdn_bt64_stage6x_intermediates.py \
  --T 512 1024 2048 4096 8192 16384 \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6y_updated_full_gap_audit

# Eager authority and body diagnostics; run each T in a fresh process.
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/audit_qwen_gdn_bt64_stage6y_full_gap.py \
  --mode all --T 2048 --sessions 5 --warmup 20 --repeat 100 \
  --out-dir /tmp/qwen_stage6y_t2048
```

Trace-only graph identification:

```bash
/opt/rocm/bin/rocprofv3 --kernel-trace \
  -d /tmp/qwen_stage6y_trace_t2048 -o stage6y_graph -f csv -- \
  python3 test/examples/linear_attention/vllm_compare/audit_qwen_gdn_bt64_stage6y_full_gap.py \
  --mode full --T 2048 --sessions 1 --warmup 1 --repeat 1 \
  --out-dir /tmp/qwen_stage6y_full_t2048
```
