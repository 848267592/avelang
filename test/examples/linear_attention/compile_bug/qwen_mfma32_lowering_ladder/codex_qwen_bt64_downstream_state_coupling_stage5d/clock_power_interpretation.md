# Clock/Power Interpretation

N/A。准备了只读 `amd-smi`/`rocm-smi` sampler，但新 Docker 调用被平台执行额度
策略拒绝，未采集 clock、power、temperature 或 throttling。控制实验中“长 v18
predecessor、warm tail、512 MiB perturb 可归一化，而短 v1/dummy/reduction prime
不能”的模式与 frequency ramp 相容，但在无 telemetry 时只属于 supported
hypothesis，不能写成事实。没有修改任何时钟、power cap 或系统配置。
