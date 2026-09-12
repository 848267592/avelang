# Cache/Execution-State Control

所有控制均在相同 canonical downstream pointer 上运行，cache 操作放在 tail event
之前。这里的 512 MiB `add_` 只能称 controlled cache perturbation；它同时可能改变
clock/power 状态，不能宣称精确清空某一级 cache。

| T | mode | after v18 ms | after v1 ms | v1 penalty |
|--:|:--|--:|--:|--:|
| 2048 | none | 0.267898 | 0.336520 | 68.622 us |
| 2048 | warm tail once | 0.267638 | 0.267758 | 0.120 us |
| 2048 | 512 MiB perturb | 0.274528 | 0.274448 | -0.080 us |
| 2048 | reduction prime | 0.280777 | 0.356189 | 75.412 us |
| 8192 | none | 0.993256 | 1.009660 | 16.404 us |
| 8192 | warm tail once | 0.993717 | 0.993937 | 0.220 us |
| 8192 | 512 MiB perturb | 1.011784 | 1.012064 | 0.280 us |
| 8192 | reduction prime | 1.001168 | 1.020457 | 19.289 us |

warm tail 和大 buffer perturb 都把差距压到亚微秒，说明根因是可被一致 predecessor
归一化的瞬态状态。reduction prime 没有归一化，说明“读过输入”本身不充分。
没有新 cache counter 和 clock 采样，故当前只能把机制缩小为 cache residency、
frequency/power ramp 或二者组合，不能进一步定性。
