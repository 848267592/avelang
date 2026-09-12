# Frozen Triton Call

The selected JIT function is
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`. The actual vLLM wrapper
calls it with `k`, `v=u`, `w`, `v_new`, `g`, `gk=None`, `h`, `h0`, `ht`,
`cu_seqlens=None`, `chunk_offsets=None`, `T=64`, `H=8`, `Hg=4`, `K=128`,
`V=128`, and `BT=64`.

The frozen specialization has `BV=32`, four Triton warps, two stages, and
one CTA. Its logical launch grid is `(ceil(128 / 32), 1 * 8) = (4, 8)`;
the HIP workgroup is 256 threads with 57,344 bytes of dynamic shared memory.

The fixed tensors use `[B,T,H,D]` layout: `k=[1,64,4,128]` BF16;
`v/w=[1,64,8,128]` FP32; `g=[1,64,8]` FP32; and
`h0/ht=[1,8,128,128]` FP32. The output `h` is BF16
`[1,1,8,128,128]`, and `v_new` is FP32 `[1,64,8,128]`.

The HIP loader must use a `kernelParams` pointer array. The last two ABI
pointers are null global/profile scratch pointers. Metadata reports zero
static LDS, but the launch requires 57,344 bytes of **dynamic** LDS; omitting
that dynamic allocation changes the ABI and is invalid.
