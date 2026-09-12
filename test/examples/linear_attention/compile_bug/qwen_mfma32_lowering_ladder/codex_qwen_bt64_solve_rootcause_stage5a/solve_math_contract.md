# BT64 FP32 Solve Contract and Algorithm

## Common contract

For every local 64x64 strictly lower-triangular matrix `A`, both paths produce

```text
X = (I + A)^-1
```

in the original `[1,T,8,64]` layout.  This is an inverse, rather than a
right-hand-side-specific solve.  The Stage 4 W/U stage consumes `X` as
`a_solved` without a layout conversion.  There is no BF16 cast in either
audited solve body.

The 54-case matrix explicitly validates the shared contract against
`torch.linalg.solve_triangular(I + A, I, upper=False)`.

## v18: row-wise forward-substitution recurrence

v18 first stages `M=-A` into `mat[64,64]`.  For each `r=1..63` and each
`c<r`, it performs the recurrence

```text
M[r,c] <- -A[r,c] + sum(i=0..r-1, -A[r,i] * M[i,c])
X[r,c] = M[r,c],  X[r,r] = 1.
```

The two 64-lane groups reduce the `i` dimension for each `c`; group 0 adds
the two partials and commits the row.  `r+1` cannot begin until row `r` is
visible.  There are therefore 63 global row-dependency stages, with three
workgroup barriers per stage and 189 recurrence barriers (190 including the
post-load barrier).  A scalar operation-count model is
`sum(r=1..63, r^2) = 85,344` multiply-accumulate terms per matrix, before
counting masked/control work.

## vLLM: 4x4 hierarchical 16x16 block inverse

Partition `I+A` into four 16x16 diagonal blocks `D1..D4` and strict-lower
blocks `Aij`.  The actual installed kernel first forms four local inverses
`Xii=D_i^-1` using the 16-row recurrence at source lines 302-317.  It then
computes lower blocks using FP32 `tl.dot`:

```text
X21 = -X22 A21 X11
X32 = -X33 A32 X22
X43 = -X44 A43 X33
X31 = -X33 (A31 X11 + A32 X21)
X42 = -X44 (A42 X22 + A43 X32)
X41 = -X44 (A41 X11 + A42 X21 + A43 X31)
```

The first three equations form one dependency level, `X31/X42` the next,
and `X41` the last.  The implementation has four local 14-step loop bodies
at source level, rather than one 63-step 64-row recurrence.  Its 16
matrix-dot expressions represent `16 * 16^3 = 65,536` nominal FP32
FMA-equivalent operations, plus the local inverse recurrences.  That number
is not directly comparable with v18 scalar instructions: the observed ISA
uses MFMA for these dots, which changes the critical path and machine issue
rate.

## Numerical order

Both algorithms are algebraically equivalent for a strict-lower `A`, but
their summation and association order differ.  v18 accumulates each 64-wide
row result in a group-partial order.  vLLM forms 16x16 products and combines
block products in the equations above.  Neither path uses an approximation,
but bitwise equality is not expected.  The measured errors remain within the
predeclared FP32 tolerance; see `numerical_order_comparison.md`.
