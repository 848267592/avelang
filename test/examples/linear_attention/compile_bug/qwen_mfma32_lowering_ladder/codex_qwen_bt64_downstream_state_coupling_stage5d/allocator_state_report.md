# Allocator State Report

审计固定图预分配所有 stage 输出，event 内没有 torch allocation。采样结束时
PyTorch allocator 为：20 active allocations、823919104 allocated bytes、
983564288 reserved bytes、23 segments、0 allocation retry、0 OOM。

fixed-buffer full 在多数 T=2048 sessions 恢复了比 public Stage 5C 更大的 solve
收益，但 T=2048 存在一次 RANDOM_ABBA 双峰反转，不能据此宣称 allocator 是唯一
根因。canonical 下游指针固定后仍有稳定 tail penalty，也反对“仅 allocator”解释。

原 public wrapper 的逐调用 allocator block ID 没有可靠公开 API；两种 solve 直接
写同一预分配 out pointer 尚未执行。allocator 不是当前已证实主因，但仍是未完全
关闭的次要假设。
