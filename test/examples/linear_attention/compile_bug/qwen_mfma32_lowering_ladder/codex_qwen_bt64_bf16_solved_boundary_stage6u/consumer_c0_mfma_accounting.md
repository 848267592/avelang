# C0 MFMA Accounting

Source and PMC agree at T=2048: C0 executes 512 W-main plus 512 U-main MFMA
per CTA, or 1024 total. With 256 CTAs this is 262144 MFMA/dispatch, exactly
half F1's 524288. Native vLLM remains at 128/CTA, so residual removal solves
only the proven 2x main/residual factor.
