# Direct-K64 C0.5S: Source-Level Vectorized LDS Store Control

## 1. Question

This experiment answers a narrower question than C0.5:

> Can a programmer obtain the same packed LDS producer store by changing only
> AveLang source, without changing `block_dot_bf16_f32` lowering or any other
> compiler code?

The answer is **yes for the producer store itself**, but **no evidence shows
that this source change can reproduce the useful full direct-K64 operand
path**. In this control the explicit packed-source arm is not faster; the
scalar-looking source arm is already combined into the same `ds_write_b128`
instructions by the existing compiler/backend.

This is not a production kernel, a replacement for C0/C0.5, or a claim about
full-v29 correctness. It is a minimal source-language expressibility control.

## 2. Frozen Control

New source and test:

- `test/examples/linear_attention/vllm_compare/repro_qwen_direct_k64_source_vectorized_lds_c05s.py`
- `test/examples/linear_attention/vllm_compare/test_qwen_direct_k64_source_vectorized_lds_c05s.py`

Both arms have the same:

| Property | Frozen value |
|:--|:--|
| input | BF16 `V[64,32]`, BF16 direct K `[64,64]` |
| grid | one CTA per BT64 token block |
| workgroup | 128 threads / two waves |
| global loads | `raw_buffer_load_x4`, six static `buffer_load_dwordx4` |
| LDS allocation | BF16 V `[64,32]` plus BF16 K `[64,64]` = 12 KiB |
| barrier | one producer-to-consumer barrier |
| global output | copy both staged tensors back as BF16 |
| mathematical result | identity staging/copy; no rounding beyond the BF16 inputs |

Only the source expression of the producer store changes:

```python
# scalar arm
frag = al.view(packed_v, al.Tensor((8,), al.bf16))
for element in al.range(8):
    v_stage[token, row_base + element] = frag[element]

# packed arm
v_words = al.view(v_stage, al.u32, al.make_layout((64, 4, 4), (16, 4, 1)))
v_words[token, row_group] = packed_v
```

The K path uses the analogous two layouts. No `block_dot`, special lowering
option, allocator change, register-allocation change, MFMA geometry, or
production Qwen file participates in this control.

## 3. Why This Stops Before MFMA

The global BF16x8 vector is contiguous along V/K for one token. The current
direct-K64 MFMA32 consumer fragment is contiguous along eight tokens for one
V/K row. These directions are transposed.

An initial full-source attempt did the natural thing after the packed
token-major LDS write: gather eight BF16 values into a local
`Tensor((2,4,1), bf16)` then feed its two slices to
`mfma_32x32x8_bf16_f32`. It failed AveLang-to-LLVM translation with four
`builtin.unrealized_conversion_cast` blockers at the MFMA operands. The
public frontend accepts packed shared views, but it does not provide a
lowerable source construct for this non-contiguous LDS gather to become the
typed packed MFMA operand.

That failure is not evidence that the compiler cannot generate such a path:
C0.5's compiler lowering can generate the token-major producer layout and
the scalar consumer gather. It is evidence that, in the currently exposed
source API, users cannot express the complete packed-scatter plus packed-MFMA
fragment construction without a dedicated layout/fragment primitive.

Therefore this control deliberately isolates the part that *is* expressible
in ordinary source: raw BF16x8 load to packed LDS store.

## 4. Correctness

Both arms round-trip the original input bytes exactly and match each other.

| T | scalar input equality | packed input equality | cross-arm equality | common SHA256 |
|--:|:--|:--|:--|:--|
| 64 | yes | yes | yes | `05ef96ccc2a8df1e27ca121058000fa3770dc3ec46b2e37dbd36f3fce9728b3a` |
| 512 | yes | yes | yes | `4872a583310dbd3e2ce4b26e557213e7edf49bf53095d62af8b3dc7666fa06e3` |
| 2048 | yes | yes | yes | `eeddb6efc10447eaef2da4a0f9977b5c818fe1c97d5cb36eee9055337c57ce5c` |

`pytest` coverage executes the same assertions at all three lengths.

## 5. ISA and Resource Evidence

The HSACO files are in:

`test/examples/linear_attention/rocprof_outputs/qwen_block_dot_bv32_source_lds_c05s/isa/`

| static ISA / metadata | scalar source | explicit packed source |
|:--|--:|--:|
| `buffer_load_dwordx4` | 6 | 6 |
| `ds_write_b128` | 6 | 6 |
| `ds_write_b16` | 0 | 0 |
| `ds_read_u16` | 48 | 48 |
| `s_barrier` | 1 | 1 |
| `.group_segment_fixed_size` | 12288 B | 12288 B |
| `.private_segment_fixed_size` | 0 B | 0 B |
| `.vgpr_count` | 35 | 35 |
| `.agpr_count` | 0 | 0 |
| `.sgpr_count` | 23 | 23 |

The critical observation is that the scalar source loop also lowers to real
`ds_write_b128` instructions. For example, its disassembly contains:

```text
buffer_load_dwordx4 v[2:5], ...
ds_write_b128 v10, v[2:5]
```

Thus the existing compiler/backend already recognizes and packs this
contiguous scalar-store pattern. Explicitly selecting the `u32` view does
not unlock a producer-store instruction that scalar source was unable to
obtain.

The two HSACO byte hashes differ because the surrounding address/control
instruction schedules differ slightly, but the resource metadata and every
relevant static instruction-family count are equal. This is sufficient to
rule out a producer `ds_write_b16 -> ds_write_b128` explanation for any
timing difference in this control.

## 6. Timing

These are diagnostic body timings, not a full recurrence benchmark.

Single process run, warmup=10/repeat=50:

| T | scalar source ms | packed source ms |
|--:|--:|--:|
| 64 | `0.032529` | `0.031267` |
| 512 | `0.032048` | `0.031907` |
| 2048 | `0.032047` | `0.032769` |

Because these values are launch-dominated, no conclusion should be drawn
from a single line. An alternating-order T=2048 confirmation used 12
sessions, each with warmup=10/repeat=50; `scalar,packed` and `packed,scalar`
orders alternated. Every paired session had packed source slower. The
session-median result was:

| metric | value |
|:--|--:|
| scalar median | `0.031126 ms` |
| packed median | `0.032017 ms` |
| packed minus scalar | `+0.891 us` / `+2.85%` |
| paired direction | packed slower in `12 / 12` sessions |

This does not establish a universal source-vectorization slowdown. It does
show that explicit source packing is not a recoverable performance win in
this aligned producer-only control, where the scalar source already compiles
to the same packed LDS store family.

## 7. Relation to C0.5 Compiler Lowering

C0.5 compared two lowerings of the same high-level `block_dot_bf16_f32`:
the scalar-transpose arm had 80 static `ds_write_b16`; the specialized
token-major arm had 10 `ds_write_b128` but then 192 `ds_read_u16` gathers.
It gained only about 3% at T=2048 and remained far from Triton.

C0.5S establishes two narrower facts:

1. AveLang source is capable of requesting packed `u32` shared views and
   raw BF16x8-to-LDS vector stores. So it is incorrect to say that source
   programmers cannot write any vectorized LDS staging at all.
2. For a contiguous producer layout, even source that looks scalar is already
   packed automatically. The missing control is not just “write a vector
   assignment in Python.” It is the transposed producer-to-MFMA-consumer
   layout/fragment bridge.

The full C0.5 arrangement requires a packed global-contiguous producer and a
non-contiguous token-contiguous MFMA consumer. Public source lacks a
lowerable typed fragment-gather or scatter layout abstraction to carry that
relationship. The specialized lowering was introduced precisely at this
semantic boundary.

## 8. Conclusion

The experiment does **not** support either extreme claim:

- Not "AveLang source cannot write a good vectorized kernel": source can
  obtain `buffer_load_dwordx4` plus `ds_write_b128`, and scalar code is
  already automatically vectorized for contiguous stores.
- Not "the C0.5 compiler lowering was unnecessary and source code alone can
  reproduce it": the full producer/consumer transpose cannot currently be
  expressed as a lowerable source-level MFMA fragment path, and the isolated
  producer control produces no timing advantage from explicit syntax.

The defensible conclusion is that the current obstacle is **an API/lowering
composition boundary**, not merely poor high-level source style and not a
general inability of the compiler to vectorize scalar stores. A future
source-facing solution would need a first-class swizzled LDS layout or typed
MFMA-fragment local-load primitive; then it could be compared fairly against
the C0.5 specialized lowering using the same direct-K64 update schedule.

## 9. Reproduction

```bash
cd /workspace/project/avelang
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:.

PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_direct_k64_source_vectorized_lds_c05s.py -s

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/repro_qwen_direct_k64_source_vectorized_lds_c05s.py \
  --T 64 512 2048 --warmup 10 --repeat 50 --json \
  --dump-hsaco-dir test/examples/linear_attention/rocprof_outputs/qwen_block_dot_bv32_source_lds_c05s/isa

for f in test/examples/linear_attention/rocprof_outputs/qwen_block_dot_bv32_source_lds_c05s/isa/*.hsaco; do
  /opt/rocm/llvm/bin/llvm-objdump -d "$f" | \
    grep -Eo 'ds_write_[A-Za-z0-9_]+|ds_read_[A-Za-z0-9_]+|buffer_load_[A-Za-z0-9_]+' | \
    sort | uniq -c
done
```
