# Memory Traffic Model

Both paths materialize BF16 W/U. F1 does residual passes and 16x16 shared staging; native vLLM uses BF16 A/K/V, 64-column blocks and packed stores. Diagnostic counters report F1 VMEM 491520 versus native vLLM 59392, and LDS instructions 1114112 versus 38912; profiler configuration differences are documented in counter_normalization.md.
