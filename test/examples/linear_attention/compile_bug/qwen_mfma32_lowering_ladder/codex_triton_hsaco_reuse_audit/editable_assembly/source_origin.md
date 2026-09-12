# Editable Assembly Origin

`original_from_triton.s` is copied from Triton cache key
`JM5FXOJP4XDF5LGYXIQP2FVT7DVYQMES3VOVFQDZJT7WNOWXH2MQ`'s compiler-stage
`.amdgcn` artifact. It is assembler input emitted before HSACO linking, not
an `llvm-objdump` reconstruction. It preserves target directives, kernel
descriptor, labels, waits, barriers, register numbering, and AMDGPU metadata.

`build_original.sh` only assembles and links this unchanged compiler-stage
source for `gfx942`. The rebuilt code object has a different ELF hash but
passed the same standalone semantic and device-trace gates.
