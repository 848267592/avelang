# ABI Difference

Both code objects use an 88-byte, 8-byte-aligned kernarg segment with eight pointer slots at offsets 0..56, a 32-bit T value at offset 64, and two runtime scratch pointers at 72/80. Thus the *physical kernarg layout* is compatible.

The semantic ABI is not compatible: current vLLM pointers for `v`, `w`, and `v_new` are BF16; asm-v0 requires FP32 for all three. `k` and `h` are BF16; `g`, initial state, and final state are FP32 in both. Current vLLM uses `WG=128` and 40,960 B dynamic shared memory, while asm-v0 uses `WG=256` and 57,344 B.
