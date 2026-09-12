# Predicate Duplication Map

C0 source lines 133-140 and 171-178 contain four dynamic `lane_group`
branches. Every branch contains two MFMA calls for adjacent 16-column output
tiles. Since `lane_group` varies within a wave, all four regions execute
serially under lane masks. This is a lane-fragment predicate, not a wave,
head or boundary predicate, and accounts for the measured 4x factor.
