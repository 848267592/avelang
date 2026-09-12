# Occupancy Limit Analysis

The launch is permanently `(4,8,1)`: `8192` grid work-items / `256` threads
equals only `32` workgroups. T is runtime and the kernel performs all
`ceil(T/64)` chunks inside those 32 persistent programs. Long T therefore
does not increase the amount of independent grid work.

Using the gfx942 64 KiB LDS capacity, 57,344 B permits one workgroup per CU;
48 KiB would still permit one and 32 KiB could permit two by LDS alone. That
hypothetical second workgroup cannot improve this launch: there are only 32
total workgroups to distribute. The captured runtime resource tuple is also
already heavy: `VGPR=128`, `AccVGPR=192`, `SGPR=80`, `Scratch=0`.

The profiler's `OccupancyPercent` varies between captures because it is a
whole-device sampled metric on a 32-workgroup launch; it is not evidence that
an unproven LDS alias would add work. Thus LDS is not a demonstrated first
occupancy limit for this fixed long-sequence geometry.

**Gate result:** no strictly safe alias and no demonstrated residency threshold
that would improve the real grid. `lds_alias_v1` is deliberately not created.
