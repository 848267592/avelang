# Device Kernel Trace Comparison

These are rocprof kernel-trace medians for the same fixed dispatch, not Python wrapper wall time. The standalone C++ event loop has host submission jitter and is intentionally not used as the ABI/performance gate.

| path | median trace us | delta |
|:--|--:|--:|
| vLLM selected Triton dispatch | 9.614 | baseline |
| extracted HSACO via HIP module | 9.575 | -0.406% |
| rebuilt compiler-stage AMDGCN via HIP module | 9.535 | -0.418% vs extracted |
