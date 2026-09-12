# BT64 Single-Step Harness Notes

The harness uses exactly one `BT=64` recurrent chunk with `B=1`, `Hk=4`,
`Hv=8`, `K=V=128`, and a nonzero FP32 initial state. `W` is FP32 at the
current Qwen interface but is converted to BF16 at the MFMA boundary, matching
the v31 reference and P16 kernel. K is BF16; U, decay, scale, and state are
FP32. The recurrence reference is:

```text
pred[t, v] = bf16(W[t, :]) @ bf16(state[v, :])
v_decay[t, v] = fp32_to_bf16((U[t, v] - pred[t, v]) * decay[t])
update[v, k] = sum_t(v_decay[t, v] * bf16(K[t, k]))
new_state[v, k] = scale * state[v, k] + update[v, k]
```

The harness records reference, v31 P16, and actual vLLM Triton data from the
same generated tensors. It uses warmup 10 and repeat 50 and reports median,
p10, and p90. The current run is a small fixed production-shaped batch with
the v31 grid (`32` independent head/value blocks). A future assembly harness
must launch the same independent blocks, not repeat arithmetic inside a block
to hide launch cost.

The vLLM wrapper's BF16 accumulation/order differs slightly from the v31
reference; the recorded vLLM `v_new` and final-state errors are expected
BF16-scale differences, not substitutions for the reference.
