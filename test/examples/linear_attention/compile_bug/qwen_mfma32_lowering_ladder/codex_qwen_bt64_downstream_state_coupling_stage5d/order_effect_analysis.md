# Order Effect Analysis

连续 downstream tail 的 ABAB、BABA 和 RANDOM_ABBA 在 T=2048 高度一致：v18
predecessor 后约 0.2679 ms，v1 后约 0.3322 ms，差约 64.3 us。T=8192 三种顺序
也一致，差约 12.8 us。因此 tail 抵消不是固定 A-first/B-first 顺序造成。

固定 buffer full 在 T=8192 各顺序稳定获得约 195 us；T=2048 的 RANDOM_ABBA
session 出现一次反转和明显双峰，而其余四个 session 获得 75--96 us。该异常说明
T=2048 fixed-buffer full 对稳态/执行状态敏感，不能用单个 session 取代 Stage 5C
public full 的约 18--20 us 结论。

所有 raw sample、顺序和 session 号在 `full_ab_raw_samples.csv` 与
`downstream_tail_raw.csv`。
