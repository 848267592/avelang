# gfx942 Assembly Environment And ABI

## Verified Environment

- Container: `ljd_qwen_vllm_avelang_rocm722`
- GPU: AMD MI300X, `gfx942:sramecc+:xnack-`
- ROCm: 7.2.2
- Device assembler: `/opt/rocm/llvm/bin/clang`
- Linker: `/opt/rocm/llvm/bin/ld.lld`
- Runtime compiler: `/opt/rocm/bin/hipcc`

`detected_toolchain.json` records the exact paths and versions used.

## Raw Assembly Build Path

The successful smoke build is deliberately independent of the AveLang JIT:

```sh
/opt/rocm/llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx942 \
  -x assembler -c source.s -o source.o
/opt/rocm/llvm/bin/ld.lld -shared source.o -o kernel.hsaco
/opt/rocm/bin/hipcc -std=c++17 -O2 smoke_harness.cpp -o smoke_harness
```

The required runtime-loadable code-object components are all present in
`smoke/source.s`:

1. `.amdgcn_target "amdgcn-amd-amdhsa--gfx942"`;
2. `.amdhsa_code_object_version 6`;
3. a kernel descriptor generated with `.amdhsa_kernel`;
4. an `AMDGPU` metadata note emitted by `.amdgpu_metadata`.

The first two descriptor-only attempts assembled but HIP rejected the object
with `hipModuleLoad(...): no kernel image is available for execution on the
device`. Adding the metadata note made the same code object load and run.
This is an ABI requirement, not a Qwen math issue.

## Verified Kernarg ABI

`qwen_gfx942_asm_smoke` uses a 24-byte host struct:

| Offset | Field | Assembly use |
|---:|:---|:---|
| 0 | `void* input` | `s[4:5]` |
| 8 | `void* output` | `s[6:7]` |
| 16 | `uint32_t addend` | `s8` |
| 20 | padding | none |

The code obtains the kernarg pointer from `s[0:1]`, workgroup ID from `s2`,
and workitem ID from `v0`. The hand-written kernel launches four 128-thread
workgroups and verifies all 512 outputs.

## Static Resource Result

The code object has `private_segment_fixed_size=0` and group segment size zero.
The T=512 smoke rocprof dispatch reports `Scratch_Size=0`, `VGPR_Count=4`,
`Accum_VGPR_Count=4`, and `LDS_Block_Size=0`. This is only an ABI smoke result;
it is not evidence about the eventual recurrent-step resources.
