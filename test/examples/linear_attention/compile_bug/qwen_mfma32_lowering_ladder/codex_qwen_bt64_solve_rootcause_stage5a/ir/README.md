# IR Capture Scope

This audit preserves the exact Avelang v18 source and the installed vLLM
`solve_tril.py` source under `avelang/` and `vllm/`, plus the corresponding
final HSACOs and disassembly under `../isa/`.

No reliable Avelang high-level IR or Triton TTG/LLVM IR dump was exposed by
the existing JIT/cache path without changing compiler debug behavior. Stage
5A therefore does not pretend to contain those missing intermediate forms.
The final ISA, code-object metadata, source-level launch mapping, and rocprof
counters are the evidence used for conclusions. A future compiler audit may
add IR capture separately; it is not required to distinguish the observed
algorithmic schedules here.
