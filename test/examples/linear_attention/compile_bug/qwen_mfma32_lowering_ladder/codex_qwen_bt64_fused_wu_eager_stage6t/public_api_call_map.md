# Timed Call Map

每个 timed callable 都是上述 public full API。F0/F1 内部才调用 cumsum、KKT、冻结 solve bridge、fused W/U、冻结 recurrence bridge、V-new cast、chunk-o 和最终 cast；benchmark 不直接 launch private kernel。
