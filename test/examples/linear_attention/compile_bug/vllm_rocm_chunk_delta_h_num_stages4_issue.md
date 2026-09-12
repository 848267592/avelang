# vLLM Issue Evidence: ROCm Triton compile failure in `chunk_delta_h` with `num_stages=4`

## Suggested Issue Title

```text
[Bug][ROCm] FLA chunk_gated_delta_rule Triton compilation fails on MI210/gfx90a with num_stages=4
```

## Summary

`vllm.model_executor.layers.fla.ops.chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
fails during Triton AMD compilation on ROCm 7.2.2 / Triton 3.6 / MI210 (`gfx90a`) when the autotuner tries a
`num_stages=4` configuration.

Filtering out `num_stages=4` from this kernel's autotune configs avoids the compiler failure. The same first
benchmark case then runs successfully with the remaining 8 configs.

## Environment

From `python_versions.txt`:

```text
torch: 2.10.0+rocm7.2.2.git40d237bf
triton: 3.6.0+rocm7.2.2.git4ed88892
vllm: 0.16.0rc2.dev119+g31d992d21
cuda available: True
hip: 7.2.53211
device: AMD Instinct MI210
```

From `rocminfo_head.txt`:

```text
Name:                    gfx90a
Marketing Name:          AMD Instinct MI210
Name:                    amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
```

From `vllm_collect_env_module.txt`:

```text
OS                           : Ubuntu 24.04.4 LTS (x86_64)
Clang version                : 22.0.0git (ROCm roc-7.2.2)
PyTorch version              : 2.10.0+rocm7.2.2.git40d237bf
ROCM used to build PyTorch   : 7.2.53211
GPU models and configuration : AMD Instinct MI210 (gfx90a:sramecc+:xnack-)
HIP runtime version          : 7.2.53211
MIOpen runtime version       : 3.5.1
vLLM Version                 : 0.16.0rc2.dev119+g31d992d21 (git sha: 31d992d21)
PYTORCH_ROCM_ARCH            : gfx90a
VLLM_TARGET_DEVICE           : rocm
```

Note: `vllm collect-env` failed in this container before dispatching the collect-env subcommand due to device
inference during CLI setup. `python -m vllm.collect_env` worked and is saved as `vllm_collect_env_module.txt`.
The failed CLI output is saved as `vllm_collect_env.txt`.

## Reproduction Case

The failing benchmark case is:

```text
B=1, T=64, Hk=2, Hv=4, K=32, V=32, chunk_size=64, dtype=torch.bfloat16
```

The minimal failing repro was run from the validated long-term container:

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare/issue_evidence
HIP_VISIBLE_DEVICES=0 PYTHONPATH=/home/jiandongliu/project/vllm_stageb_snapshot:$PYTHONPATH python <repro> 2>&1 | tee fail_num_stages4.log
```

The repro uses:

```python
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule
```

with:

```python
output_final_state=True
cu_seqlens=None
head_first=False
use_qk_l2norm_in_kernel=False
```

## Failing Autotune Configs

Before applying the workaround, the kernel has 12 autotune configs:

```text
autotune configs: 12
config: {'BV': 32} num_warps= 2 num_stages= 2
config: {'BV': 64} num_warps= 2 num_stages= 2
config: {'BV': 32} num_warps= 2 num_stages= 3
config: {'BV': 64} num_warps= 2 num_stages= 3
config: {'BV': 32} num_warps= 2 num_stages= 4
config: {'BV': 64} num_warps= 2 num_stages= 4
config: {'BV': 32} num_warps= 4 num_stages= 2
config: {'BV': 64} num_warps= 4 num_stages= 2
config: {'BV': 32} num_warps= 4 num_stages= 3
config: {'BV': 64} num_warps= 4 num_stages= 3
config: {'BV': 32} num_warps= 4 num_stages= 4
config: {'BV': 64} num_warps= 4 num_stages= 4
```

## Error

From `fail_num_stages4.log`:

```text
/home/jiandongliu/project/vllm_stageb_snapshot/vllm/model_executor/layers/fla/ops/chunk_delta_h.py:177:22:
error: 'tt.load' op operation destroyed but still has uses
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v
                     ^
LLVM ERROR: operation destroyed but still has uses
```

The generated MLIR reproducer pipeline includes:

```text
tritonamdgpu-schedule-loops{num_stages=4}
```

The Python stack ends with:

```text
RuntimeError: PassManager::run failed
```

## Workaround Tested

Filtering out `num_stages=4` configs for
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` avoids the Triton AMD compiler failure:

```python
kernel = chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
autotuner = kernel.fn
autotuner.configs = [
    config for config in autotuner.configs
    if config.num_stages != 4
]
autotuner.cache.clear()
```

The benchmark prints:

```text
patched vLLM ROCm autotune configs: 12 -> 8 (disabled num_stages=4 for chunk_delta_h)
```

The first benchmark case then passes:

```text
output shapes: (1, 64, 4, 32) (1, 64, 4, 32)
state  shapes: (1, 4, 32, 32) (1, 4, 32, 32)
output max abs err: 0.0010410994291305542
state  max abs err: 0.004581227898597717
vLLM ms: {'mean': 0.4414961338043213, 'median': 0.438401997089386, 'min': 0.4179210066795349, 'max': 0.664322018623352}
```

## Expected Behavior

The vLLM FLA `chunk_gated_delta_rule` kernel should compile successfully on ROCm/gfx90a, or the autotuner should
avoid configs known to fail compilation on this backend.

## Actual Behavior

When the autotuner tries a `num_stages=4` candidate, Triton AMD compilation fails with:

```text
LLVM ERROR: operation destroyed but still has uses
RuntimeError: PassManager::run failed
```

## Evidence Files

- `python_versions.txt`: Python package versions and GPU visibility
- `rocminfo_head.txt`: ROCm GPU target information
- `vllm_collect_env_module.txt`: successful `python -m vllm.collect_env` output
- `vllm_collect_env.txt`: failed `vllm collect-env` CLI attempt
- `fail_num_stages4.log`: full failure log with original 12 configs and MLIR/LLVM error
- `pass_without_num_stages4_first_case.log`: successful first-case run after filtering out `num_stages=4`

## Suggested Fix Direction

If vLLM wants to handle this locally, consider filtering `num_stages=4` for this specific FLA kernel on ROCm
and affected architectures/versions, rather than removing it globally. This keeps CUDA and other ROCm targets
unchanged unless they are also affected.
