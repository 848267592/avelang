# Ownership Gap

The four-fold CTA gap is structural: Avelang Stage 4 maps one CTA to V16,
whereas vLLM's selected kernel maps one CTA to V64. For one chunk/head,
Avelang repeats Q/K staging and QK score production in eight CTAs; vLLM uses
two CTAs. This also explains why the Stage 4 source recalculates QK tiles for
each V range.

O0 changes Avelang to the same CTA count as vLLM: two V64 CTAs per
chunk/head. It computes only the ten lower-triangular token16 score tiles and
keeps them in 8 KiB LDS while processing four sequential V16 output subtiles.

This does not mean all measured work can fall by 4x: H/V-new and final output
remain V-specific, while O0 pays additional barriers, a larger LDS allocation,
and higher accumulator pressure. The measured counts, not an ownership ratio,
are the decision authority.
