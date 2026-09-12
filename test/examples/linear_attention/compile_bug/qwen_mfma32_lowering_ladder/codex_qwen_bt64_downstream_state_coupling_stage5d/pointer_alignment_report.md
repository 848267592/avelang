# Pointer And Alignment Report

solve A、solve B 与 canonical 都是 contiguous FP32 `[1,T,8,64]`，stride
`[T*512,512,64,1]`，storage offset 0。T=2048 三个地址均 4 KiB 和 64 KiB
对齐；T=8192 也相同。因此可见的 16/64/128/256/4K/64K 低位 alignment 不同
不能解释差距。

canonical control 已保证 downstream 消费 data_ptr 完全相同，差距仍存在。
当前 `same_pointer_control.csv` 使用“各 solve 写自己的输出，再 copy 到同一
canonical pointer”的安全控制；它在 T=2048 仍有 68.181 us penalty。两种 solve
**直接写同一个预分配 out pointer** 尚未运行，不能排除原 solve store 地址造成的
cache-set/physical-address影响。该项是 Stage 5E 的唯一建议。

copy 本体 T=2048 中位约 9.8--10.0 us，位于 tail event 外，不计入 tail latency。
