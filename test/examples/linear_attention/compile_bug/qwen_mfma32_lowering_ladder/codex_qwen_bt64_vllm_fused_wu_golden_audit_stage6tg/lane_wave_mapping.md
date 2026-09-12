# Lane/Wave Mapping

TTGIR proves native vLLM normal config uses `warpsPerCTA=[2,2]`, i.e. four waves, with `instrShape=[32,32,8]`. F1 source proves four waves and `lane_group=lane>>4`. A precise per-lane output permutation is not claimed.
