# Stage 5B — AMD ROCm/AITER Kimi KDA Prefill

本阶段已完成并停止在 Avelang kernel 实现之前。没有修改 SGLang/vLLM
源码，也没有修改已有 Stage 5 结果。

## 结论

- AITER public wrapper 在 MI300X/gfx942 上成功进入 merged FlashKDA-style
  two-kernel Triton path；未接受 fallback pipeline 结果。
- 统一 benchmark correctness：全部通过，包括 nonzero initial state。
- 在本次 public-call latency 测试中，AITER 在所有 8 个 T 上最快。
- 按后续 Avelang Kimi Prefill 的性能方向，AITER FlashKDA 是最强的参考
  baseline；SGLang ordinary Triton 仍适合作为普通-KDA correctness/reference
  baseline。

## Hardware and software

| 项目 | 值 |
|---|---|
| GPU | AMD Instinct MI300X |
| Architecture | gfx942 (`device_capability=[9,4]`) |
| GPU selection | `HIP_VISIBLE_DEVICES=0`, `ROCR_VISIBLE_DEVICES=0` |
| Container | `ljd_sglang_kda` |
| PyTorch | `2.11.0+rocm7.2` |
| HIP/ROCm runtime | `7.2.26015` |
| Triton | `3.7.0` |
| AITER commit | `7bb44998274fa679ece0a37bf8426ab780b1837d` |
| AITER flags | `AITER_TRITON_ONLY=1`, `CHUNK_DELTA_ATTN_USE_FLASH_KDA=1` |

源码位置：

```text
/home/jiandongliu/project/avelang/third_party/aiter
```

## Kimi K3 TP8-style contract

```text
B = 1
H = 12
K = V = 128
input = BF16
recurrent state = FP32, [N,H,V,K]
lower_bound = -5.0
scale = 1/sqrt(128)
chunk_size = 32
state_v_first = True
```

调用入口：

```text
aiter.ops.triton.kimi_delta_attn.chunk_kimi_delta_attn
  -> chunk_delta_attn_fwd
  -> flash_kda_fwd
```

调用中启用了 `use_qk_l2norm_in_kernel=True`、`use_gate_in_kernel=True`、
`use_beta_sigmoid_in_kernel=True` 和 `safe_gate=True`。adapter 在每个形状
正式测试前执行未计时 dispatch probe，并对 `flash_kda_fwd` 做一次进程内
调用计数；若没有实际调用该函数会直接失败。

## Stage B 官方 AITER benchmark（单独记录）

官方脚本：

```text
third_party/aiter/op_tests/op_benchmarks/triton/bench_flash_kda.py
```

形状 `B=1,T=8192,H=12,K=V=128` 的官方结果：

| Time | TFLOPS | BW |
|---:|---:|---:|
| 0.5539 ms | 14.54 | 227.5 GB/s |

该结果只计官方 `flash_kda_fwd` 测试边界，不能直接与下面统一 Stage 5
表格混用。原始日志见 `results/aiter_flash_kda_official.log`。

## Stage D correctness

使用 `bench_common.py` 现有 FP32 recurrence reference 和原有
`atol=3e-2, rtol=3e-2`，没有放宽 tolerance。下表为最大绝对误差：

| T | output max abs | final state max abs | 结果 |
|---:|---:|---:|---|
| 64 | 1.221e-4 | 9.149e-4 | PASS |
| 128 | 1.221e-4 | 1.272e-3 | PASS |
| 256 | 1.221e-4 | 1.051e-3 | PASS |
| 512 | 1.221e-4 | 1.711e-3 | PASS |
| 1024 | 1.221e-4 | 1.004e-3 | PASS |
| 2048 | 2.441e-4 | 9.768e-4 | PASS |
| 4096 | 2.441e-4 | 1.040e-3 | PASS |
| 8192 | 2.441e-4 | 8.903e-4 | PASS |
| 128，nonzero initial state | 1.221e-4 | 9.959e-4 | PASS |

因此 correctness 结论为 **PASS**。

## Stage E unified Prefill latency

SGLang/vLLM 数值取已有 Stage 5 三轮 median 的中位数；AITER 也使用三轮
median 的中位数。每轮为 warmup=50、rep=300，GPU event 只包住 operator
调用；显式 state reset、`v` reset、输入构造均在 event 外。

| T | SGLang (us) | vLLM AMD (us) | AITER (us) | winner | best old / AITER |
|---:|---:|---:|---:|---|---:|
| 64 | 310.662 | 356.950 | 202.421 | AITER | 1.535x |
| 128 | 311.544 | 358.713 | 202.781 | AITER | 1.536x |
| 256 | 311.103 | 363.962 | 204.524 | AITER | 1.521x |
| 512 | 309.601 | 361.457 | 215.340 | AITER | 1.438x |
| 1024 | 336.420 | 393.986 | 246.005 | AITER | 1.368x |
| 2048 | 496.318 | 466.233 | 307.677 | AITER | 1.515x |
| 4096 | 637.948 | 605.841 | 431.361 | AITER | 1.404x |
| 8192 | 930.503 | 936.993 | 643.817 | AITER | 1.445x |

AITER 相对已有最快 baseline 的优势范围为 **1.368x–1.536x**；本 sweep
中 AITER 在 **8/8** 个形状获胜。

## Fairness / timing caveat

以下条件已保持一致：GPU0、顺序执行、`B=1,H=12,K=V=128`、BF16 输入、
FP32 state、`lower_bound=-5`、warmup/rep、输入/reference contract 和
operator boundary。SGLang/vLLM 两边仍是 Stage 5 的 ordinary Triton path；
AITER 则按本阶段目标明确使用 FlashKDA path。

需要明确记录一个 AITER 实现细节：AITER public wrapper 在
`state_v_first=True` 时，会在 `flash_kda_fwd` 内部把输入 state 从
`[N,H,V,K]` 转为 kernel 内部的 `[N,H,K,V]`，并调用 `.contiguous()`。这一步
不是 adapter 的 reset/copy，但属于 public AITER 调用并落在 GPU event 区间内。
因此本表是 **统一 public operator-call latency**，不是完全排除内部 state
layout normalization 的纯 kernel-only latency。若下一步需要严格的纯 kernel
比较，应另做一个不在 event 内执行该内部 transpose/contiguous 的 AITER
低层入口实验；本阶段没有修改官方 AITER 实现来规避它。

在这个限定下，报告中的 measured winner 是 AITER；若把“禁止任何内部
transpose/contiguous”作为硬性 fairness gate，则该 gate 的状态应记录为
**未完全满足，不能把本表解释成纯 kernel-only 最终定论**。

## 生成文件

- `kda_baseline/bench_kda_prefill_aiter.py`
- `kda_baseline/results/aiter_flash_kda_official.log`
- `kda_baseline/results/aiter_kimi_prefill_correctness.log`
- `kda_baseline/results/aiter_kimi_prefill_bench.log`
- `kda_baseline/results/aiter_kimi_prefill.json`
- `kda_baseline/STAGE5B_AITER_KIMI.md`

已有 Stage 5 文件未修改；本阶段未开始 Avelang kernel 实现。
