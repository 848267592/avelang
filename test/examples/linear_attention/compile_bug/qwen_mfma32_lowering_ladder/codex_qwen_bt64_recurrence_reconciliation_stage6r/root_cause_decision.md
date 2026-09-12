# Root Cause Decision

**CASE B: current vLLM uses a different, faster specialization.** The raw MFMA discrepancy is not aggregation; the fresh native-vLLM-to-extracted-HSACO bridge is bit-exact and stays within 5% at every measured length. The historical asm-v0 has not regressed. It remains bit-exact with historical original/rebuilt code on its FP32 ABI.

No assembly or compiler change is justified by this audit. The measured body gap belongs to the explicit current-BF16 versus historical-FP32 specialization boundary plus its associated geometry/lowering; this audit does not assign all of it to dtype alone.
