# Dynamic Work Analysis

F0: `524288` MFMA, `491520` VMEM, `1114112` LDS. F1 preserves MFMA/VMEM/LDS, but VALU rises from `4518912` to `4846592`. The CTA reduction is real, but this schedule still has much larger MFMA/VMEM work than the Stage 6A vLLM trace context.
