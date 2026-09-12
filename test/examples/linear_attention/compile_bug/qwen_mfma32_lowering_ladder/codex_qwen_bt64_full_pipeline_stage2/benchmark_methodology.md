# Benchmark Methodology

Every row is a HIP-event median with configured warmup/repeat and a current stream. `asm_recurrence_preallocated` excludes output allocation/module loading. The generic historical Avelang wrappers allocate public output tensors internally; their full row is therefore labelled `cached_allocator`, not a production allocation-free claim.
