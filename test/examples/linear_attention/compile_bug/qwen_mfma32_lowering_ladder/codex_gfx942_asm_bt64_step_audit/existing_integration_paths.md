# Existing Code-Object Integration Paths

## AveLang AMDGPU Runtime

`python/avelang/backends/amdgpu/driver.c` loads compiled bytes with
`hipModuleLoadDataEx`, resolves symbols through `hipModuleGetFunction`, and
launches with `hipModuleLaunchKernel`. `driver.py` packs runtime arguments in
a C struct and supplies it through:

```c
HIP_LAUNCH_PARAM_BUFFER_POINTER
HIP_LAUNCH_PARAM_BUFFER_SIZE
HIP_LAUNCH_PARAM_END
```

This matches the smoke harness exactly. A future successful external HSACO
integration can therefore use an existing ABI path; no production runtime
change is justified until the Qwen single-step gate passes.

## Backend Code-Object Link Path

`lib/Target/AMDGPU/amdgpu_backend.cc` produces LLVM bitcode then invokes ROCm
`ld.lld` with discovered device libraries. When
`AVELANG_AMDGPU_LINK_DEBUG_DIR` is set it preserves the pre-link bitcode,
linked code object, and replayable linker argv. That is useful for generated
AveLang kernels but unnecessary for the raw smoke build, which directly calls
clang assembler plus `ld.lld`.

## Existing C++ Module Test

`lib/Target/AMDGPU/amdgpu_codegen_test.cc` is a checked C++ precedent for
module loading and struct kernarg launch. It calls `hipModuleLoadDataEx`,
`hipModuleGetFunction`, and `hipModuleLaunchKernel` exactly as the runtime
driver does.

## Existing Qwen / ISA Evidence

- `vllm_compare/repro_qwen_v31_active_v16_pred_primitives.py`: verified P16
  direct MFMA16 two-K64-partial mapping.
- `vllm_compare/qwen_gdn_chunked_avelang_v31_bt64_bv32_hierarchical_mfma16_pred.py`:
  BT64, BV32, 128-thread / two-wave experimental kernel and authoritative
  BT64 reference routine.
- `vllm_compare/profile_vllm_triton_chunk_delta_h.py`: actual vLLM Triton
  wrapper with `chunk_size=64` and `head_first=False`.

There was no pre-existing source-level raw AMDGCN assembly kernel in this
repository. The two `issue_evidence/*.s` files are disassemblies, not input
assembly sources, and must not be used as ABI templates without metadata.
