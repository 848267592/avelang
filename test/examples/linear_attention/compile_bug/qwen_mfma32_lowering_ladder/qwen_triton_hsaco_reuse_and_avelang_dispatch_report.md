# Qwen Triton HSACO Reuse and External Dispatch

## Result

The real vLLM/Triton BT64 `chunk_delta_h` specialization was frozen, extracted,
loaded through HIP without the Triton Python launcher, and rebuilt from
Triton's compiler-stage AMDGCN assembler input. Both original and rebuilt
objects match vLLM bit-exactly on 55 fixed single-step cases. Their device
kernel traces and dynamic counters match the selected wrapper within 3%.

The new bridge is an audit-only external dispatch. It does not lower an HSACO
into AveLang IR and is not evidence of a new Avelang code generator.

## Frozen Specialization

| item | value |
|:--|:--|
| source/kernel | vLLM `chunk_delta_h.py` / `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` |
| target | `gfx942:sramecc+:xnack-` |
| shape/layout | `B=1,T=64,Hk=4,Hv=8,K=V=128`, `[B,T,H,D]` |
| specialization | `BT=64,BV=32`, G + initial state + final state + v_new; no gk/varlen/head-first |
| launch | grid `(4,8,1)`, workgroup `(256,1,1)`, four warps, two stages |
| dynamic LDS | `57,344 B` |
| cache key | `JM5FXOJP4XDF5LGYXIQP2FVT7DVYQMES3VOVFQDZJT7WNOWXH2MQ` |
| metadata hash | `4b3a5bb92fe5c65eacd8ba20fd16b3f8eb883092dd5d52c0794cff66bad73e99` |
| original HSACO hash | `cb1811c318652b89989dbf63ecda595acb82324561084f7b825291b6b56f03f9` |

The actual generated HIP launcher uses 11 `kernelParams`: eight data pointers,
`T=64`, then null global/profile scratch pointers. Code-object metadata shows
zero **static** LDS, but the launch requires 57,344 bytes of dynamic LDS.

## Independent HIP Gate

The C++ harness uses `hipModuleLoad`, `hipModuleGetFunction`, and
`hipModuleLaunchKernel` directly with the frozen kernelParams ABI.

| comparison | cases | h / v_new / final-state max abs |
|:--|--:|:--|
| extracted HSACO vs vLLM | 55 | `0 / 0 / 0` |
| rebuilt HSACO vs vLLM | 55 | `0 / 0 / 0` |

The set is 50 seeded random cases plus zero-W, zero-state, unit-decay,
high-dynamic-range, and cancellation cases.

| path | trace median us | WG | VGPR | AccVGPR | SGPR | scratch |
|:--|--:|--:|--:|--:|--:|--:|
| selected vLLM Triton dispatch | `9.614` | 256 | 128 | 192 | 80 | 0 |
| extracted HSACO via HIP module | `9.575` | 256 | 128 | 192 | 80 | 0 |
| rebuilt AMDGCN via HIP module | `9.535` | 256 | 128 | 192 | 80 | 0 |

Extracted-versus-wrapper is `-0.406%`; rebuilt-versus-extracted is `-0.418%`.
All three report `MFMA=6144`, `VALU=148352`, `SALU=17664`, `VMEM=3840`, and
`LDS=25472`. The C++ single-launch HIP-event result has host-submit jitter and
is retained only as a diagnostic JSON, not used as the device-kernel gate.

## Editable Assembly

`editable_assembly/original_from_triton.s` is copied from Triton's cached
compiler-stage `.amdgcn` input, not an objdump reconstruction. Reassembling it
unchanged for gfx942 produced SHA256
`98b503bc74ef233bf54bdf439d89c622fd9a573dd0b9c87e116550ccde751622`. The
disassembly differs only in the input file-name header; semantic, resource,
counter, and device-trace checks all pass. It is a valid exact-version golden
reference, not a modified production Qwen kernel.

## External Dispatch

`avelang_integration/qwen_gdn_bt64_gfx942_external.py` defines:

```python
qwen_gdn_bt64_gfx942_external(k, w, u, g, initial_state, fallback=...)
```

It validates gfx942, the exact fixed shape/dtypes/layout/contiguity and initial
state, validates the local HSACO SHA256, forwards the current HIP stream, and
caches the module/function in a small C++ HIP bridge. Five GPU pytest checks
passed:

1. external output equals real vLLM and repeated launch reuses the module;
2. guard failure and missing artifact call the explicit fallback;
3. hash mismatch is rejected.
4. a simulated non-gfx942 target is rejected before module launch;
5. a non-default current HIP stream produces the same result.

The callback is intentionally generic: this single-step surface takes `w/u/g`
intermediates while v24 is a distinct full-forward contract. A production
caller must provide its v24 fallback at its full-forward boundary; the bridge
will not invent an unsafe v24 call or dispatch a failed guard to HSACO.

## License and Deployment

Installed vLLM is Apache-2.0; the source header also carries explicit
flash-linear-attention MIT attribution. Installed Triton carries an MIT license
text. Cache/HSACO/assembly are local audit artifacts only, not vendored assets.
The first deployment policy is explicit local path plus SHA256. Redistribution
requires a separate provenance and attribution decision.

## Reproduction and Decision

All audit artifacts, hashes, compiler stages, direct HIP harness, assembly
build, bridge, and commands are in
`compile_bug/qwen_mfma32_lowering_ladder/codex_triton_hsaco_reuse_audit/`.
Run `commands.sh` inside the known ROCm container.

The single-step reuse gate passes. Keep v24 as the Avelang production baseline.
Use the bridge only as an opt-in experimental comparator and the compiler-stage
AMDGCN source as a fixed gfx942 reference; do not call this Avelang lowering or
general-shape support.
