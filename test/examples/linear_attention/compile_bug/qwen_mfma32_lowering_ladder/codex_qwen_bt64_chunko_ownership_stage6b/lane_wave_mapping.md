# Lane and Wave Mapping

For the current kernel and both experimental kernels:

```text
wave_id = tid >> 6              # four waves, token16 row tile
lane = tid & 63
lane_col = lane & 15           # column inside the current V16/K16 tile
lane_group = lane >> 4
token = wave_id*16 + lane_group*4 + r, r in [0,4)
value = current_v16_base + lane_col
```

O0 keeps this proven V16 microtile mapping and runs it four times under the
same V64 CTA. O1 runs it twice under V32. This avoids changing the passing
MFMA16 fragment convention.

vLLM uses the selected `BV=64`, four-warp Triton block-dot layout. Its exact
lane fragment permutation is compiler-generated; no unsupported lane mapping
is claimed here.
