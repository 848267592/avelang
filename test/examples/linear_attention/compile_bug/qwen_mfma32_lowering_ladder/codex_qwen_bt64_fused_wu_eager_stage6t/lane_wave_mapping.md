# Lane/Wave Mapping

`wave_id=tid>>6` 选择 16-token row block；`lane_col=lane&15` 选择 16-column position；`lane_group=lane>>4` 选择该 MFMA fragment 内的四行。每 CTA 覆盖 BT64 的全部 token row 与一个 value head。
