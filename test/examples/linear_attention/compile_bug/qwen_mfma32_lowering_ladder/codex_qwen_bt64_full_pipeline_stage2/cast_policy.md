# Cast Policy

| Boundary | vLLM | Candidate |
|:--|:--|:--|
| input q/k/v | BF16 | BF16 |
| g local cumsum | FP32 | FP32 |
| KKT accumulation | FP32 | FP32 |
| solved transform | BF16 | FP32 (v18 compatibility adapter) |
| w/u materialization | BF16 | FP32 to satisfy frozen asm ABI |
| recurrence pred | Triton dot with runtime input; Stage 1 frozen XF32 contract | frozen asm XF32 |
| h snapshot | BF16 | BF16 exact asm output |
| h to candidate chunk_o | BF16 load | explicit BF16-to-FP32 widening, no value recomputation |
| public output | `o.to(q.dtype)` BF16 | `output_fp32.to(q.dtype)` BF16 |

The candidate’s FP32 solve/W/U are deliberate CASE-C adapter boundaries.  The
full correctness matrix measures their consequence against the authoritative
vLLM public output rather than pretending the intermediate dtypes match.

