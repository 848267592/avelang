# Qwen GDN Next Decision After Stage 6W Confirmation

## Decision

Promote Stage 6W / W1 as the current **Avelang experimental baseline** for
paired shared-environment Eager public-API timing. Do not classify it as a
production/default selector change or as an across-host absolute claim.

The frozen-code, Eager-public-API confirmation used 12 independent sessions
and 72 randomized Williams blocks at each of T=2048 and T=8192. The required
block, session, and nested-cluster bootstrap lower bounds were not all
positive at either length. That historical run was polluted by long tails.

A fresh shared-environment paired retest then ran 8 independent sessions and
64 complete Williams blocks per length. Its pre-registered primary estimator,
the session-level paired-median HIP-event gain, passed at both lengths:
T=2048 `+8.994 us`, 95% CI `[+7.246, +10.545]`; T=8192 `+27.352 us`, 95% CI
`[+25.398, +29.163]`. Wall-clock and nested sensitivity analyses agree, and
every session favors W1 over U1.

Stage 6W remains an opt-in candidate. No kernel is changed by this decision.

## Measurement Scope

The benchmark still defaults to exclusive-GPU preflight for an absolute claim,
but `--allow-shared-gpu` is valid for the paired relative-ranking contract used
here. API-internal allocation remains in scope for this public-API metric.
The external PID 83597 remains recorded in telemetry; it does not invalidate a
same-block U1/W1/vLLM comparison, but it limits the result's scope.

## One Registered Follow-Up, Not Yet An Implementation

The only next source-level graph experiment to plan is elimination of the
source-native `KKT FP32 a -> solve` global handoff. It is the largest feasible
one-use upstream edge after Stage 6W: 8 MiB write+read traffic at T=2048 and
32 MiB at T=8192.

Begin only the source/DAG/CTA-ownership audit of this candidate: `a` is 4 MiB
at T=2048, so it cannot simply be retained as one private tile. Do not work on
`h`, `v_new`, W/U, cumsum, recurrence HSACO, or a second cast change in
parallel.

## Measurement Prerequisite

Before making a cross-host absolute-latency claim or changing the production
default selector, complete the exclusive-GPU Eager Williams protocol with
complete GPU-context telemetry. Existing public APIs intentionally include
their internal allocations in this metric; caller-owned preallocation is a
separate device-pipeline diagnostic and must not replace public-API timing.
Do not replace the public-API authority with CUDA Graph replay.
