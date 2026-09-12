# Single Fused Schedule

一个 CTA 拥有一个 `(chunk,value_head)`；WG=256 即四 wave。每个 phase 依次处理四个 32-column pair，并且每个 lane 仅保持两个 16-column accumulator fragment。W 完整 store 后才开始 U，避免两套完整 W/U accumulator 同时存活。F0/F1 的 grid、WG、lane mapping、MFMA、LDS 和 barrier 顺序相同。
