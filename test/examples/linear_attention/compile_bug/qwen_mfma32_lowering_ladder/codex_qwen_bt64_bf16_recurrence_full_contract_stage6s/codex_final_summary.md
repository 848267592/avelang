# Stage 6S Final Summary

Stage 6S is Case A. The opt-in BF16 recurrence bridge passes the full correctness contract and improves the T=2048 same-harness full graph by 26.647 us, from 0.335679 ms to 0.309019 ms. The direct recurrence-body gain is 40.099 us; the materialized W/U/V-new conversion graph costs 20.750 us together.

The current-vLLM recurrence HSACO hash is 632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e. The bridge remains bit-exact to native vLLM recurrence at equal BF16 inputs. Public output max abs is 0.001953125 and final-state max abs is 0.0172200203, both within the frozen thresholds.

Graph dispatch counts at T=2048 are A=8, B=11 and native vLLM C=7. Graph B adds three explicit numeric casts and does not use reinterpretation or fallback. Its native-vLLM gap slope improves from 4.396726 to 3.284716 us/chunk, with no T=8192 or T=16384 regression.

Keep this path experimental-only. The next single action is end-to-end BF16 storage-boundary propagation across W/U producers and chunk-o V-new consumption, eliminating the three explicit casts while retaining the recurrence HSACO and all other frozen components.
