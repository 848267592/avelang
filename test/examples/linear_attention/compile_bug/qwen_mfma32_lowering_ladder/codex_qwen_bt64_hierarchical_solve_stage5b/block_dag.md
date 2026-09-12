# 4x16 Block DAG

```text
             X11    X22    X33    X44
               \     |      |     /
        X21 = -X22 A21 X11
        X32 = -X33 A32 X22
        X43 = -X44 A43 X33
               \     |      |     /
 X31 = -X33 (A31 X11 + A32 X21)
 X42 = -X44 (A42 X22 + A43 X32)
                         \
 X41 = -X44 (A41 X11 + A42 X21 + A43 X31)
```

`L0` diagonal inverses are independent.  `L1` has three independent block
results, `L2` has two, and `L3` has one.  A barrier is allowed only after a
level publishes a block required by a later level.  A 63-row whole-CTA
recurrence is explicitly outside this design.
