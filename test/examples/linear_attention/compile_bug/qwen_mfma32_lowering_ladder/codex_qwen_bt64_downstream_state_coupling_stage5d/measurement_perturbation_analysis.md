# Measurement Perturbation

Stage 5C public full 单 event 的 T=2048 gain 为 18.808 us。每 stage 插 event 后，
v18 total 增加 15.623 us，v1 增加 29.844 us，观察到的 gain 被压到 4.587 us；
即 instrumentation 对 A/B 差值造成 14.221 us 扰动。因此 per-stage 表只能定位
现象，不能精确分摊 78 us。

本轮 continuous tail 只在整个 W/U->asm->chunk-o->cast 外放一个 event，仍稳定
复现 T=2048 的 64.255 us penalty，证明 downstream coupling 不是 per-stage event
制造的。whole-graph rocprof 新数据为 N/A，不能量化其扰动。
