# Dispatch Difference

固定 buffer 图 A/B 的 dispatch 数量和顺序一致，均为 8。ordinal 0/1/3/4/5/6/7
的 symbol、grid、WG 与参数契约一致；ordinal 2 的 solve symbol/WG 不同。这与
Stage 5C 的显式 selector 源码相符，不存在 silent fallback。

没有发现额外 copy、zero 或 init dispatch。固定图的 BF16 cast 在两边均显式保留。
canonical 和 cache 控制会在 tail event 之前有意增加 solve/copy/perturb/prime
dispatch；这些不是正式 graph A/B 的隐藏差异。

whole-graph rocprof timeline 尚未执行，因此 dispatch 间 gap 的精确纳秒数据为 N/A。
