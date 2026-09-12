# Remaining MFMA Gap Source Audit

For W, a 64x128 output using MFMA16 has 4 row tiles x 8 column tiles x 4
K-reduction steps = 128 mathematical MFMA. W+U therefore needs 256 MFMA/CTA.
C0 measures 1024/CTA. The 4x excess is produced by four divergent
`lane_group` MFMA call sites: each wave serially executes all four branch
regions and discards lanes outside each predicate.

Native vLLM measures 128/CTA because its MFMA32 geometry covers four times
the output area while using half-width K steps, a net 2x reduction relative
to ideal MFMA16 geometry. Thus `1024 = 4 predicate x 2 geometry x 128`.

The predicate factor is visible in high-level source and therefore is a
source-schedule opportunity. It was not changed here because no isolated
branch-free fragment-selection proof yet shows one wave-uniform MFMA call
with identical AveLang fragment semantics and acceptable resources. U2 is
therefore N/A, not a hidden failed full candidate.
