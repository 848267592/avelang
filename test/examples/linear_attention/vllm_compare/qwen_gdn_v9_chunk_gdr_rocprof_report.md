# Qwen GDN v9 chunk_gdr rocprof report

Date: 2026-06-13

## Target

Primary target is the single-GPU vLLM Qwen3Next TP4 per-rank operator shape:

```text
GPU: MI300 / gfx942
B=1, T=512
Hk=4, Hv=8
K=128, V=128
dtype=BF16
layout: q/k/v = [B,T,H,D]
initial_state/final_state = [B,Hv,V,K]
chunk_size=4
chunk_gdr block_v=4, block_k=64
chunk_o block_v=4, block_k=16
```

Kernel profiled:

```text
_qwen_gdn_chunk_gdr_bf16_kernel_v9_vk
```

## Outputs

rocprof output directories:

```text
/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_profile_v9_chunk_gdr_trace
/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_profile_v9_chunk_gdr_counters_core
/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_profile_v9_chunk_gdr_counters_lds
/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_profile_v9_chunk_gdr_counters_l2
/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_profile_v9_chunk_gdr_counters_barrier
```

One attempted combined memory profile failed because the requested counter set exceeded hardware collection capability on gfx942:

```text
Unable to find all counters for agent 2 (gpu-0, gfx942) in [ALUStalledByLDS, FetchSize, MemUnitStalled, TCC_HIT_sum, TCC_MISS_sum, VALUBusy, VALUUtilization, VFetchInsts, VWriteInsts, WriteSize].
Found: [FetchSize, MemUnitStalled, TCC_HIT_sum, TCC_MISS_sum, VALUBusy, VALUUtilization, VFetchInsts, VWriteInsts, WriteSize].
Missing: [ALUStalledByLDS]
Could not construct profile cfg failed with error code 38: Request exceeds the capabilities of the hardware to collect
```

I then split the profile into smaller groups.

## Trace Summary

From `v9_chunk_gdr_trace_kernel_stats.csv`:

| Kernel | Calls | Average ns | Min ns | Max ns | Kernel time share |
|---|---:|---:|---:|---:|---:|
| chunk_gdr v9 vk | 13 | 922,906 | 890,002 | 992,113 | 53.51% |
| chunk_o v6 fallback | 9 | 871,477 | 862,161 | 894,208 | 34.98% |
| chunk_o v9 vk | 9 | 125,053 | 123,584 | 130,393 | 5.02% |
| w_u | 13 | 74,813 | 69,663 | 85,487 | 4.34% |

The trace confirms the bottleneck after v9 `chunk_o` is still `chunk_gdr`.

## Kernel Metadata

All counter runs report the same kernel launch/resource metadata:

| Metric | Value |
|---|---:|
| Grid_Size | 65,536 |
| Workgroup_Size | 256 |
| LDS_Block_Size | 1,536 bytes |
| Scratch_Size | 0 |
| VGPR_Count | 28 |
| Accum_VGPR_Count | 4 |
| SGPR_Count | 64 |

This means the v9 VK mapping is really launching a 256-work-item workgroup. There is no scratch spill in the profiled configuration.

## Core Counters

From `v9_chunk_gdr_core_counter_collection.csv`, median over 4 profiled dispatches:

| Counter | Median |
|---|---:|
| OccupancyPercent | 10.23% |
| SQ_INSTS | 65,634,304 |
| SQ_INSTS_VALU | 21,147,648 |
| SQ_INSTS_SALU | 15,147,008 |
| SQ_INSTS_VMEM | 3,280,896 |
| SQ_WAVES | 1,024 |
| Dispatch duration | 969,981 ns |

Instruction mix:

| Category | Share of SQ_INSTS | Per wave |
|---|---:|---:|
| VALU | 32.22% | 20,652 |
| SALU | 23.08% | 14,792 |
| VMEM | 5.00% | 3,204 |
| LDS | 4.79% | 3,072 |
| Other/control/wait/etc. | 34.91% | 22,376 |

The kernel is not primarily VMEM-instruction-heavy. VALU + SALU + other/control dominate the instruction stream.

## LDS And Barrier

From `v9_chunk_gdr_lds_counter_collection.csv`, median over 4 dispatches:

| Counter | Median |
|---|---:|
| SQ_INSTS_LDS | 3,145,728 |
| SQ_ACTIVE_INST_LDS | 4,718,592 |
| SQ_WAIT_INST_LDS | 114,425 |
| SQ_LDS_BANK_CONFLICT | 15,335,424 |
| SQ_LDS_ADDR_CONFLICT | 0 |

From `v9_chunk_gdr_barrier_counter_collection.csv`:

| Counter | Median |
|---|---:|
| SPI_RA_BAR_CU_FULL_CSN | 0 |

Interpretation:

- There are many LDS bank conflict events, so LDS reduction layout is not perfect.
- But direct LDS wait instruction count is much smaller than total SQ instructions, and barrier allocation stall is 0.
- Current evidence does not support "barrier resource stall" as the main bottleneck.

## L2

From `v9_chunk_gdr_l2_counter_collection.csv`, median over 4 dispatches:

| Counter | Median |
|---|---:|
| TCC_HIT_sum | 1,196,224 |
| TCC_MISS_sum | 957,423 |
| Approx. L2 hit rate | 55.54% |

The L2 hit rate is not great, but VMEM instructions are only about 5% of SQ instructions. So this does not look like a pure global-memory bandwidth bottleneck. Memory behavior still matters, but it is not the first explanation for the 0.9-1.36 ms `chunk_gdr` time.

## Benchmark Sanity During rocprof

The rocprof runs printed the expected TP4 per-rank shape:

```text
q=(1, 512, 4, 128)
k=(1, 512, 4, 128)
v=(1, 512, 8, 128)
vn=(1, 512, 8, 128)
h=(1, 128, 8, 128, 128)
out=(1, 512, 8, 128)
chunk=4
```

Representative stage breakdown printed under rocprof:

| Stage | Median ms |
|---|---:|
| cumsum | 0.22-0.28 |
| KKT | 0.20-0.22 |
| solve | 0.11 |
| w_u | 0.21-0.22 |
| chunk_gdr | 1.36 |
| chunk_o_v9_vk | 0.25 |

The absolute timings under rocprof are slower than normal benchmark mode, but the bottleneck ordering is consistent: `chunk_gdr` dominates.

## Conclusion

I had not collected this full counter set before; this report is the new `chunk_gdr` profile.

Current bottleneck classification:

```text
Not scratch spill:      Scratch_Size = 0
Not VGPR explosion:     VGPR_Count = 28
Not barrier alloc stall: SPI_RA_BAR_CU_FULL_CSN = 0
Not pure VMEM-bound:    SQ_INSTS_VMEM ~= 5% of SQ_INSTS
Some LDS pressure:      bank conflicts high, but LDS wait is modest
Main signature:         low occupancy + high VALU/SALU/control instruction count
```

So the bottleneck is most likely the algorithmic structure of `chunk_gdr`: each lane still performs a lot of serial recurrence/control work over chunks/tokens, plus local reductions. The v9 VK mapping fixed the worst workgroup-size problem, but did not remove the long dependency chain inside state update.

## Next Optimization Direction

The next useful direction is not just increasing `block_v` or `block_k`.

Recommended next steps:

1. Inspect generated ISA for `chunk_gdr_v9_vk` to separate real arithmetic from control/wait instructions.
2. Reduce the amount of per-lane recurrence work, ideally with a real parallel scan / associative state transition across chunks.
3. Rework LDS reduction layout if keeping VK reduction, because `SQ_LDS_BANK_CONFLICT` is high.
4. Avoid larger workgroups until proven useful: occupancy is already only about 10%, and larger blocks may reduce resident waves or trigger launch/runtime failures.
5. After `chunk_gdr` improves, rerun stage profile; `chunk_o_v9_vk` is now much smaller, so the next bottleneck will likely be `w_u`, `KKT`, or cumsum/solve overhead rather than `chunk_o`.

