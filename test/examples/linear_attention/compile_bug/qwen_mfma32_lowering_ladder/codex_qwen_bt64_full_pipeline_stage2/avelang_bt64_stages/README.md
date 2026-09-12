# Experimental BT64 Stage Ownership

The Stage 2 full path reuses isolated existing BT64-capable implementations:

- `preprocessing`: v6 local cumsum;
- `kkt`: v6 BT64 KKT;
- `solve`: v18 BT64 solve;
- `w_u`: v6 BT64 W/U;
- `adapters`: FP32 zero state for `initial_state=None` and BF16-H to FP32-view conversion;
- `chunk_o`: generic v6 chunk-o invoked with `chunk_size=64`.

The source locations and casts are frozen in `../stage_source_map.md` and
`../bt64_numerical_contract.md`. These are experimental adapters only; no
production v24 implementation or dispatch path is changed.
