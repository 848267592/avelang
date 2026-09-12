# Stage 6R Actual-vLLM Bridge Probe

This directory documents the audit-only external bridge, not a production
integration. The bridge library is built from `../stage6r_external_bridge.cpp`
by `../build_bridge.sh` and launches the unchanged extracted current-vLLM
HSACO from `../current_kernels/vllm/kernel.hsaco`.

The probe validates the extracted kernel's physical ABI, current stream,
grid `(4, 8, 1)`, workgroup `128`, and dynamic LDS `40960` bytes. It rejects
the fixed-shape contract if K is not BF16, if W/U/V-new do not share a BF16 or
FP32 dtype, or if tensors are non-contiguous. The Stage 6R candidate uses the
native BF16 W/U/V-new contract and is bit-exact against the vLLM source launch
at T=512/2048/8192/16384; it is intentionally not connected to a full graph.
