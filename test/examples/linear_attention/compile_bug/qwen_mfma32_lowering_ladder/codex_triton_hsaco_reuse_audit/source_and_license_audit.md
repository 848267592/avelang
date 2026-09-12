# Source and License Audit

The fixed kernel source is the installed vLLM file
`vllm/model_executor/layers/fla/ops/chunk_delta_h.py`. Its header declares
`SPDX-License-Identifier: Apache-2.0`, attributes vLLM contributors and
Songlin Yang/Yu Zhang, and states that code was copied from
flash-linear-attention under its original MIT license.

The installed vLLM wheel is `Apache-2.0`. The installed Triton wheel ships an
MIT license text (Philippe Tillet and OpenAI copyright). The audit retains the
source path and headers above; no installed package was changed.

`triton_cache_exact/`, `extracted/*.hsaco`, and rebuilt code objects are local
audit artifacts only. They are not a production source dependency and must
not be committed or redistributed until the project decides on a compliant
asset policy with the required vLLM/flash-linear-attention attribution. The
checked-in Python/C++ bridge contains no embedded binary and requires an
explicit local path plus an SHA256 match.
