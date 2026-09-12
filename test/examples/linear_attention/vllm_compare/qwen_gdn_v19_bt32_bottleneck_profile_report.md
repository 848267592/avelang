# Qwen GDN v19 BT32 Bottleneck Profile Report

## Target

- Shape: `B=1`, `Hk=4`, `Hv=8`, `K=128`, `V=128`, layout `[B,T,H,D]`
- Dtypes: BF16 `q/k/v`, FP32 `g/beta/w/u/output/final_state`
- Baseline: v17 predecay BT16
- Candidate: v19 BT32
- Scope: profiling/report only. No v17/v18/v19 baseline kernels were modified.

## Changed Files

- `profile_qwen_gdn_v19_bt32_bottleneck.py`
- `qwen_gdn_v19_bt32_bottleneck_profile_report.md`

## Commands

Benchmark/delta:

```bash
docker exec ac739c57a0bf bash -lc \
'cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare && \
 env PYTHONDONTWRITEBYTECODE=1 python profile_qwen_gdn_v19_bt32_bottleneck.py \
 --mode benchmark --T 512 1024 2048 --warmup 5 --repeat 20'
```

Targeted rocprof used `T=2048`, `warmup=1`, `repeat=5`, and:

```bash
/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex <target_kernel_regex> \
  -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v19_bt32_bottleneck/<version_stage> \
  -o <version_stage>_counters -f csv -- \
  python test/examples/linear_attention/vllm_compare/profile_qwen_gdn_v19_bt32_bottleneck.py \
  --mode profile --version <v17|v19> --stage <stage> --T 2048 --warmup 1 --repeat 5
```

## Stage Delta

Times are median ms. Contribution is `stage_delta / full_delta`; stage timings are measured independently, so percentages are directional rather than expected to sum to exactly 100%.

| T | stage | v17 predecay | v19 BT32 | delta | contribution |
|---:|:---|---:|---:|---:|---:|
| 512 | full | 0.331513 | 0.349920 | +0.018407 | 100.00% |
| 512 | cumsum | 0.032047 | 0.031006 | -0.001041 | -5.66% |
| 512 | KKT | 0.068121 | 0.096603 | +0.028482 | 154.73% |
| 512 | solve | 0.048372 | 0.036093 | -0.012279 | -66.71% |
| 512 | w_u | 0.059228 | 0.100309 | +0.041081 | 223.18% |
| 512 | gdr_decay | 0.031727 | 0.000000 | -0.031727 | -172.36% |
| 512 | chunk_gdr | 0.138045 | 0.128992 | -0.009053 | -49.19% |
| 512 | chunk_o | 0.054702 | 0.063995 | +0.009293 | 50.49% |
| 1024 | full | 0.468857 | 0.503909 | +0.035052 | 100.00% |
| 1024 | cumsum | 0.032368 | 0.032128 | -0.000240 | -0.68% |
| 1024 | KKT | 0.087290 | 0.129733 | +0.042443 | 121.09% |
| 1024 | solve | 0.049333 | 0.035713 | -0.013620 | -38.86% |
| 1024 | w_u | 0.064636 | 0.117514 | +0.052878 | 150.86% |
| 1024 | gdr_decay | 0.032508 | 0.000000 | -0.032508 | -92.74% |
| 1024 | chunk_gdr | 0.230022 | 0.215581 | -0.014441 | -41.20% |
| 1024 | chunk_o | 0.077195 | 0.090355 | +0.013159 | 37.54% |
| 2048 | full | 0.698938 | 0.878745 | +0.179807 | 100.00% |
| 2048 | cumsum | 0.032608 | 0.032368 | -0.000240 | -0.13% |
| 2048 | KKT | 0.123503 | 0.199516 | +0.076013 | 42.27% |
| 2048 | solve | 0.051416 | 0.036394 | -0.015022 | -8.35% |
| 2048 | w_u | 0.086649 | 0.192827 | +0.106178 | 59.05% |
| 2048 | gdr_decay | 0.031887 | 0.000000 | -0.031887 | -17.73% |
| 2048 | chunk_gdr | 0.412473 | 0.388558 | -0.023915 | -13.30% |
| 2048 | chunk_o | 0.113769 | 0.141430 | +0.027661 | 15.38% |

At `T=2048`, v19 loses `+0.179807 ms` full latency. The largest positive regressions are:

- `w_u`: `+0.106178 ms`
- `KKT`: `+0.076013 ms`
- `chunk_o`: `+0.027661 ms`

The improvements are smaller:

- `solve`: `-0.015022 ms`
- `chunk_gdr`: `-0.023915 ms`
- removing v17 predecay launch: `-0.031887 ms`

## rocprof: KKT

Both versions use `_qwen_gdn_kkt_bf16_kernel_v6_standalone`; the difference is chunk size.

| metric | v17 BT16 | v19 BT32 | ratio |
|:---|---:|---:|---:|
| Workgroup_Size | 1 | 1 | 1.00x |
| Grid_Size | 16384 | 16384 | 1.00x |
| LDS_Block_Size | 0 | 0 | n/a |
| Scratch_Size | 0 | 0 | n/a |
| VGPR_Count | 52 | 52 | 1.00x |
| Accum_VGPR_Count | 132 | 132 | 1.00x |
| SGPR_Count | 32 | 32 | 1.00x |
| SQ_INSTS_MFMA | 0 | 0 | n/a |
| SQ_INSTS_VALU | 51174400 | 103472640 | 2.02x |
| SQ_INSTS_SALU | 3875840 | 6857216 | 1.77x |
| SQ_INSTS_VMEM | 2796544 | 5547520 | 1.98x |
| SQ_INSTS_LDS | 0 | 0 | n/a |
| OccupancyPercent | 18.3347 | 19.4311 | 1.06x |
| median trace us | 93.138 | 170.453 | 1.83x |

KKT regression is structural for the current v6 scalar mapping: same one-thread workgroup, no MFMA, same total grid work-items, but BT32 roughly doubles VALU and VMEM work.

## rocprof: w_u W Kernel

| metric | v17 W | v19 W | ratio |
|:---|---:|---:|---:|
| Workgroup_Size | 64 | 64 | 1.00x |
| Grid_Size | 524288 | 524288 | 1.00x |
| LDS_Block_Size | 1024 | 2048 | 2.00x |
| Scratch_Size | 0 | 0 | n/a |
| VGPR_Count | 68 | 108 | 1.59x |
| Accum_VGPR_Count | 4 | 4 | 1.00x |
| SGPR_Count | 112 | 112 | 1.00x |
| SQ_INSTS_MFMA | 32768 | 65536 | 2.00x |
| SQ_INSTS_VALU | 11272192 | 25206784 | 2.24x |
| SQ_INSTS_SALU | 1433600 | 2998272 | 2.09x |
| SQ_INSTS_VMEM | 1556480 | 3637248 | 2.34x |
| SQ_INSTS_LDS | 163840 | 327680 | 2.00x |
| OccupancyPercent | 35.8391 | 36.9437 | 1.03x |
| median trace us | 33.610 | 100.389 | 2.99x |

## rocprof: w_u U Kernel

| metric | v17 U | v19 U | ratio |
|:---|---:|---:|---:|
| Workgroup_Size | 64 | 64 | 1.00x |
| Grid_Size | 524288 | 524288 | 1.00x |
| LDS_Block_Size | 1024 | 2048 | 2.00x |
| Scratch_Size | 0 | 0 | n/a |
| VGPR_Count | 68 | 104 | 1.53x |
| Accum_VGPR_Count | 4 | 8 | 2.00x |
| SGPR_Count | 80 | 112 | 1.40x |
| SQ_INSTS_MFMA | 32768 | 65536 | 2.00x |
| SQ_INSTS_VALU | 3801088 | 8183808 | 2.15x |
| SQ_INSTS_SALU | 1064960 | 2359296 | 2.22x |
| SQ_INSTS_VMEM | 1155072 | 2555904 | 2.21x |
| SQ_INSTS_LDS | 163840 | 327680 | 2.00x |
| OccupancyPercent | 41.1907 | 35.5272 | 0.86x |
| median trace us | 25.117 | 67.220 | 2.68x |

BT32 w_u is slower because it does not reduce launch grid or workgroup count versus BT16. Instead, it keeps the same effective grid while doubling source-token work inside each program. MFMA, LDS instructions, LDS allocation, and most VMEM/VALU work roughly double; VGPR pressure rises sharply. Occupancy does not collapse, so occupancy is not the primary cause.

## rocprof: chunk_gdr

| metric | v17 4-wave | v19 BT32 fallback | ratio |
|:---|---:|---:|---:|
| Kernel | `_qwen_gdn_chunk_gdr_bf16_kernel_v17_4wave_mfma` | `_qwen_gdn_chunk_gdr_bf16_kernel_v19_bt32_4wave_mfma` |  |
| Workgroup_Size | 256 | 256 | 1.00x |
| Grid_Size | 16384 | 16384 | 1.00x |
| LDS_Block_Size | 25088 | 29696 | 1.18x |
| Scratch_Size | 0 | 0 | n/a |
| VGPR_Count | 40 | 80 | 2.00x |
| Accum_VGPR_Count | 136 | 136 | 1.00x |
| SGPR_Count | 112 | 112 | 1.00x |
| SQ_INSTS_MFMA | 327680 | 327680 | 1.00x |
| SQ_INSTS_VALU | 5576960 | 5474560 | 0.98x |
| SQ_INSTS_SALU | 830976 | 717824 | 0.86x |
| SQ_INSTS_VMEM | 921600 | 774144 | 0.84x |
| SQ_INSTS_LDS | 2371584 | 1847296 | 0.78x |
| OccupancyPercent | 2.5341 | 2.5326 | 1.00x |
| median trace us | 397.711 | 380.285 | 0.96x |

v19 chunk_gdr is not the intended 512-thread 8-wave direction in the measured path. The actual kernel is the 256-thread 4-wave fallback, and `Workgroup_Size=256` confirms it. MFMA count is identical to v17. VALU/VMEM/LDS are lower, but VGPR doubles and the fallback still serializes BT32 token halves enough that chunk_gdr only improves about `0.024 ms` in stage timing at `T=2048`.

## rocprof: chunk_o

| metric | v17 chunk_o | v19 BT32 chunk_o | ratio |
|:---|---:|---:|---:|
| Workgroup_Size | 64 | 64 | 1.00x |
| Grid_Size | 524288 | 524288 | 1.00x |
| LDS_Block_Size | 13312 | 18432 | 1.38x |
| Scratch_Size | 0 | 0 | n/a |
| VGPR_Count | 36 | 44 | 1.22x |
| Accum_VGPR_Count | 68 | 92 | 1.35x |
| SGPR_Count | 112 | 112 | 1.00x |
| SQ_INSTS_MFMA | 163840 | 262144 | 1.60x |
| SQ_INSTS_VALU | 13148160 | 11182080 | 0.85x |
| SQ_INSTS_SALU | 1564672 | 1482752 | 0.95x |
| SQ_INSTS_VMEM | 950272 | 1277952 | 1.34x |
| SQ_INSTS_LDS | 1007616 | 1458176 | 1.45x |
| OccupancyPercent | 10.3779 | 8.0646 | 0.78x |
| median trace us | 83.303 | 121.801 | 1.46x |

chunk_o is not the main requested optimization target, but BT32 chunk_o is a real regression: more MFMA/LDS/VMEM work, larger LDS, more VGPR/accumulator pressure, and lower occupancy.

## Answers

### 1. Why does BT32 w_u go from about 0.082 ms to about 0.189 ms?

The profile points to structural doubled work plus register pressure:

- W and U MFMA counts both double: `32768 -> 65536`.
- LDS instruction count and LDS allocation both double.
- VMEM grows `2.34x` for W and `2.21x` for U.
- VALU grows `2.24x` for W and `2.15x` for U.
- Workgroup and grid are unchanged, so BT32 does not buy lower launch/grid work here.
- VGPR rises from `68` to `108` for W and `104` for U.
- U occupancy drops from `41.19` to `35.53`; W occupancy is roughly flat.

So the main issue is not occupancy collapse. It is that v19 w_u processes the BT32 source dimension by doing about twice the work with similar grid shape, plus higher VGPR pressure.

### 2. Why does BT32 KKT go from about 0.115 ms to about 0.192 ms?

KKT is still the v6 standalone scalar kernel:

- Workgroup size stays `1`.
- MFMA count stays `0`.
- VALU doubles: `51.17M -> 103.47M`.
- VMEM roughly doubles: `2.80M -> 5.55M`.
- Median trace grows `93.1 us -> 170.5 us`.

This is mostly a chunk_size=32 structural cost in the current scalar mapping. It is not a new MFMA/tiled implementation, and the profile does not show a grid/workgroup improvement that offsets the larger chunk-local work.

### 3. Why does BT32 chunk_gdr only improve from about 0.4097 ms to about 0.3886 ms?

The measured v19 path is not an 8-wave 512-thread kernel:

- Actual kernel: `_qwen_gdn_chunk_gdr_bf16_kernel_v19_bt32_4wave_mfma`
- Workgroup size: `256`, same as v17
- MFMA count: unchanged at `327680`
- Occupancy: essentially unchanged
- VGPR doubles: `40 -> 80`

BT32 reduces loop/chunk overhead and lowers VMEM/LDS/VALU somewhat, but the fallback still handles BT32 through 16-token halves rather than exposing token halves as a truly parallel 8-wave path. The result is a small chunk_gdr gain, not enough to offset w_u/KKT/chunk_o regressions.

## v20 Recommendation

Priority 1: optimize BT32 w_u.

It contributes the largest `T=2048` regression: `+0.106 ms`, about `59%` of full regression. The data says v19 w_u doubles MFMA/LDS and more than doubles VMEM/VALU without reducing grid work. A useful v20 BT32 path must change that mapping, not just polish the current one.

Priority 2: optimize BT32 KKT.

KKT contributes `+0.076 ms`, about `42%` of full regression. The current v6 scalar KKT is workgroup-size 1 and roughly doubles VALU/VMEM at BT32. If BT32 is kept, KKT needs a parallel/tiled rewrite or a different formulation.

Priority 3: only after w_u and KKT improve, revisit BT32 chunk_gdr.

The current 4-wave fallback is correct but does not deliver the larger-chunk payoff. A true token-half-parallel chunk_gdr could still matter, but current chunk_gdr saves only `0.024 ms` while w_u+KKT add `0.182 ms`.

Do not prioritize chunk_o right now.

BT32 chunk_o regresses `+0.028 ms`, but it is smaller than w_u/KKT and the current project direction says not to keep optimizing chunk_o unless it becomes the main bottleneck.

Decision:

- Do not abandon BT32 solely on this result; solve is fixed and chunk_gdr is slightly faster.
- Do not write more chunk_gdr first; v19 loses before chunk_gdr can pay for itself.
- v20 should start with BT32 w_u, then BT32 KKT. If those cannot recover roughly `0.15 ms` at `T=2048`, v17 BT16 remains the better production baseline and larger chunks should be paused.
