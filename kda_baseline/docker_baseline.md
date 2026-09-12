# KDA baseline Docker Stage 2

Checked: 2026-09-10. No existing container was stopped or modified. Both new
containers were created without `--rm` and with `--restart unless-stopped`.

## Host

- Host: `nimrodmi300`
- GPUs: 8 AMD MI300X-class devices
- GFX: `gfx942`
- Host driver reported by `rocm-smi`: `6.16.13`
- Docker root: `/data01/docker`

## SGLang container

- Name: `ljd_sglang_kda`
- Image: `lmsysorg/sglang-rocm:v0.5.19-rocm724-mi30x-20260909`
- Pulled digest: `sha256:16e1ae7e199418fe4df58f6dffb7b2d6b8382cc998aae0d525f8d8b2f7959c98`
- Image base label: `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`
- Python: `3.12.3`
- PyTorch: `2.11.0+rocm7.2`
- HIP: `7.2.26015`
- Triton: `3.7.0`
- SGLang: `0.5.19.dev20260909+gffe98a4279`
- `torch.cuda.is_available()`: `True`
- `torch.cuda.get_device_name(0)`: `AMD Instinct MI300X`
- `torch.cuda.get_device_capability(0)`: `(9, 4)`
- Image environment: `GPU_ARCH=gfx942-rocm724`, `GPU_ARCH_LIST=gfx942`

## vLLM container

- Name: `ljd_vllm_kda`
- Image: `vllm/vllm-openai-rocm:kimi-k3`
- Pulled digest: `sha256:5aa7e626ff73672f5ca7aae46754570488c23d33ca1ac90756a1d2d1a3fe099b`
- Python: `3.12.13`
- PyTorch: `2.11.0+gitd0c8b1f`
- HIP: `7.2.53211`
- Triton: `3.7.0`
- vLLM: `0.1.dev19253+g5f76ae224.d20260727`
- `torch.cuda.is_available()`: `True`
- `torch.cuda.get_device_capability(0)`: `(9, 4)`
- `torch.cuda.get_device_name(0)`: empty string in this image; `rocm-smi`
  reports Card SKU `M3000100` and GFX `gfx942`.

## Common container configuration

- Status after creation: `running`
- Restart policy: `unless-stopped`
- Host bind: `/home/jiandongliu/project/avelang`
- Container bind: `/workspace/project/avelang`
- Devices: `/dev/kfd`, `/dev/dri`
- Groups: `video`, `render`
- IPC: host; shared memory: `32g`
- Capability: `SYS_PTRACE`
- Seccomp: `unconfined`
- Main process: `tail -f /dev/null`

Both containers report `/workspace/project/avelang`,
`third_party/sglang`, and `third_party/vllm` as present. The only warning seen
was a harmless `libtinfo.so.6` version-information warning from bash in the
SGLang image; Python, PyTorch, Triton, and GPU initialization succeeded.
