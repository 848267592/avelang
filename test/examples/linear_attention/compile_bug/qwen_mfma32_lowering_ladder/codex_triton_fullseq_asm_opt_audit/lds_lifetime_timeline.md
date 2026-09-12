# LDS Lifetime Timeline

For each runtime `i_t` chunk iteration, the assembly follows this sequence:

1. state/W operand preparation in low LDS and the `49152+` fragment band;
2. barrier, DS reads, and pred MFMA;
3. corrected-V materialization and global `v_new` stores;
4. K/update repacking in `32768+` and corrected-V fragments in `49152+`;
5. barrier, DS reads, and update MFMA into persistent register state;
6. next `i_t` iteration reuses the same LDS addresses.

The same `49152+` addresses appear under different source locations only
after intervening producer/consumer barriers. That is evidence that Triton
already performs phase-local reuse. It is not evidence that either low LDS or
the K band can be moved into that range: both have dynamically formed LDS
addresses and loop-carried consumers.
