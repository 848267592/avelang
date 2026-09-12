	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 5
	.text
	.globl	qwen_gdn_bt64_gfx942_asm_v0 ; -- Begin function qwen_gdn_bt64_gfx942_asm_v0
	.p2align	8
	.type	qwen_gdn_bt64_gfx942_asm_v0,@function
qwen_gdn_bt64_gfx942_asm_v0: ; @qwen_gdn_bt64_gfx942_asm_v0
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.95:
	.file	1 "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fla/ops" "chunk_delta_h.py"
	.loc	1 43 0 prologue_end             ; chunk_delta_h.py:43:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.96:
.LBB0_0:
.Ltmp1:
	.loc	1 111 62 is_stmt 1              ; chunk_delta_h.py:111:62
	s_lshl_b32 s38, s16, 5
	.loc	1 111 80 is_stmt 0              ; chunk_delta_h.py:111:80
	s_ashr_i32 s39, s38, 31
	.loc	1 105 29 is_stmt 1              ; chunk_delta_h.py:105:29
	s_lshl_b32 s33, s17, 14
	.loc	1 112 24                        ; chunk_delta_h.py:112:24
	v_and_b32_e32 v39, 0xf8, v0
	v_lshrrev_b32_e32 v172, 3, v0
	v_and_b32_e32 v38, 7, v0
	s_lshl_b64 s[48:49], s[38:39], 7
	s_mov_b64 s[40:41], s[2:3]
	v_or_b32_e32 v4, s38, v172
	v_mov_b32_e32 v5, s39
	v_lshlrev_b32_e32 v40, 3, v38
	v_lshlrev_b32_e32 v2, 4, v39
	s_cmp_gt_i32 s38, -1
	s_mov_b64 s[2:3], 0x80
	v_or_b32_e32 v152, v2, v40
	s_cselect_b64 s[50:51], -1, 0
	v_cmp_gt_i64_e32 vcc, s[2:3], v[4:5]
	v_mov_b32_e32 v85, 0
	v_or_b32_e32 v3, s33, v152
	s_and_b64 s[18:19], s[50:51], vcc
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	s_and_saveexec_b64 s[2:3], s[18:19]
	s_cbranch_execz .LBB0_2
; %bb.1:
	.loc	1 0 24 is_stmt 0                ; chunk_delta_h.py:0:24
	v_add_u32_e32 v5, s48, v3
	v_mov_b32_e32 v4, 0
	v_ashrrev_i64 v[4:5], 30, v[4:5]
	v_lshl_add_u64 v[4:5], s[14:15], 0, v[4:5]
	.loc	1 112 24                        ; chunk_delta_h.py:112:24
	global_load_dwordx4 v[4:7], v[4:5], off
.LBB0_2:
	.loc	1 0 24                          ; chunk_delta_h.py:0:24
	s_or_b64 exec, exec, s[2:3]
	.loc	1 112 24                        ; chunk_delta_h.py:112:24
	v_or_b32_e32 v1, s33, v40
	v_or3_b32 v12, v1, v2, 4
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	s_and_saveexec_b64 s[2:3], s[18:19]
	s_cbranch_execz .LBB0_4
; %bb.3:
	v_add_u32_e32 v9, s48, v12
	v_mov_b32_e32 v8, 0
	v_ashrrev_i64 v[8:9], 30, v[8:9]
	v_lshl_add_u64 v[8:9], s[14:15], 0, v[8:9]
	global_load_dwordx4 v[8:11], v[8:9], off
.LBB0_4:
	.loc	1 0 24                          ; chunk_delta_h.py:0:24
	s_or_b64 exec, exec, s[2:3]
	.loc	1 112 16                        ; chunk_delta_h.py:112:16
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v15, 0, v11
	v_lshlrev_b32_e32 v60, 1, v0
	v_bfe_i32 v11, v0, 4, 1
	.loc	1 112 24                        ; chunk_delta_h.py:112:24
	v_and_b32_e32 v142, 32, v0
	.loc	1 112 16                        ; chunk_delta_h.py:112:16
	v_add_f32_e32 v1, 0, v4
	v_add_f32_e32 v4, 0, v5
	v_add_f32_e32 v5, 0, v6
	v_add_f32_e32 v6, 0, v7
	v_add_f32_e32 v7, 0, v8
	v_add_f32_e32 v8, 0, v9
	v_add_f32_e32 v9, 0, v10
	v_lshlrev_b32_e32 v13, 4, v38
	v_and_b32_e32 v10, 0x180, v60
	v_and_b32_e32 v154, 8, v0
	v_and_b32_e32 v11, 0x204, v11
	v_lshlrev_b32_e32 v14, 8, v154
	v_lshrrev_b32_e32 v16, 2, v142
	v_or3_b32 v10, v10, v11, v13
	v_accvgpr_write_b32 a49, v14
	v_or3_b32 v14, v10, v14, v16
	v_add_u32_e32 v10, 0, v14
	ds_write2st64_b32 v10, v1, v4 offset1:16
	v_xor_b32_e32 v1, 4, v14
	v_add_u32_e32 v11, 0, v1
	v_xor_b32_e32 v1, 64, v14
	v_accvgpr_write_b32 a48, v13
	v_add_u32_e32 v13, 0, v1
	v_xor_b32_e32 v1, 0x44, v14
	.loc	1 112 24                        ; chunk_delta_h.py:112:24
	v_and_b32_e32 v150, 0x80, v0
	.loc	1 112 16                        ; chunk_delta_h.py:112:16
	ds_write2st64_b32 v11, v5, v6 offset1:16
	v_add_u32_e32 v14, 0, v1
	v_and_b32_e32 v1, 52, v60
	v_bfe_i32 v6, v0, 2, 1
	s_movk_i32 s16, 0x440
	v_and_b32_e32 v156, 1, v0
	v_lshrrev_b32_e32 v48, 1, v150
	v_and_or_b32 v1, v6, s16, v1
	v_lshlrev_b32_e32 v4, 12, v156
	v_xor_b32_e32 v1, v1, v48
	v_or3_b32 v1, v1, v4, v16
	ds_write2st64_b32 v13, v7, v8 offset0:4 offset1:20
	ds_write2st64_b32 v14, v9, v15 offset0:4 offset1:20
	v_add_u32_e32 v15, 0, v1
	v_xor_b32_e32 v1, 4, v1
	v_accvgpr_write_b32 a41, v16
	v_add_u32_e32 v16, 0, v1
	v_add_u32_e32 v4, 0x800, v15
	v_add_u32_e32 v1, 0x800, v16
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b32 v[118:119], v15 offset1:32
	ds_read2_b32 v[116:117], v4 offset1:32
	ds_read2_b32 v[110:111], v15 offset0:64 offset1:96
	ds_read2_b32 v[108:109], v4 offset0:64 offset1:96
	ds_read2_b32 v[120:121], v16 offset0:128 offset1:160
	ds_read2_b32 v[122:123], v1 offset0:128 offset1:160
	ds_read2_b32 v[114:115], v16 offset0:192 offset1:224
	ds_read2_b32 v[112:113], v1 offset0:192 offset1:224
	v_and_b32_e32 v155, 16, v0
	v_and_b32_e32 v5, 4, v0
	v_mov_b32_e32 v2, 0
	v_cmp_eq_u32_e64 s[2:3], 0, v155
	v_cmp_eq_u32_e64 s[28:29], 0, v5
	.loc	1 117 28 is_stmt 1              ; chunk_delta_h.py:117:28
	s_or_b32 s49, s48, 64
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	s_and_saveexec_b64 s[20:21], s[18:19]
	s_cbranch_execz .LBB0_6
; %bb.5:
	.loc	1 0 28 is_stmt 0                ; chunk_delta_h.py:0:28
	v_add_u32_e32 v5, s49, v3
	v_mov_b32_e32 v4, 0
	v_ashrrev_i64 v[4:5], 30, v[4:5]
	v_lshl_add_u64 v[4:5], s[14:15], 0, v[4:5]
	.loc	1 117 28                        ; chunk_delta_h.py:117:28
	global_load_dwordx4 v[6:9], v[4:5], off
.LBB0_6:
	.loc	1 0 28                          ; chunk_delta_h.py:0:28
	s_or_b64 exec, exec, s[20:21]
	s_load_dword s52, s[0:1], 0x40
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 0
	.loc	1 117 28                        ; chunk_delta_h.py:117:28
	s_and_saveexec_b64 s[20:21], s[18:19]
	s_cbranch_execz .LBB0_8
; %bb.7:
	v_add_u32_e32 v3, s49, v12
	v_mov_b32_e32 v2, 0
	v_ashrrev_i64 v[2:3], 30, v[2:3]
	v_lshl_add_u64 v[2:3], s[14:15], 0, v[2:3]
	global_load_dwordx4 v[2:5], v[2:3], off
.LBB0_8:
	.loc	1 0 28                          ; chunk_delta_h.py:0:28
	s_or_b64 exec, exec, s[20:21]
	.loc	1 70 23 is_stmt 1               ; chunk_delta_h.py:70:23
	s_ashr_i32 s14, s17, 31
	s_lshr_b32 s14, s14, 29
	s_add_i32 s14, s17, s14
	.loc	1 117 20                        ; chunk_delta_h.py:117:20
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v1, 0, v6
	v_add_f32_e32 v6, 0, v7
	.loc	1 70 23                         ; chunk_delta_h.py:70:23
	s_ashr_i32 s16, s14, 3
	.loc	1 117 20                        ; chunk_delta_h.py:117:20
	v_add_f32_e32 v7, 0, v8
	v_add_f32_e32 v8, 0, v9
	v_add_f32_e32 v2, 0, v2
	v_add_f32_e32 v3, 0, v3
	v_add_f32_e32 v4, 0, v4
	v_add_f32_e32 v5, 0, v5
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write2st64_b32 v10, v1, v6 offset1:16
	ds_write2st64_b32 v11, v7, v8 offset1:16
	ds_write2st64_b32 v13, v2, v3 offset0:4 offset1:20
	ds_write2st64_b32 v14, v4, v5 offset0:4 offset1:20
	v_add_u32_e32 v1, 0x800, v15
	.loc	1 70 33                         ; chunk_delta_h.py:70:33
	s_and_b32 s14, s14, -8
	.loc	1 80 25                         ; chunk_delta_h.py:80:25
	s_mul_i32 s34, s52, s16
	.loc	1 117 20                        ; chunk_delta_h.py:117:20
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b32 v[132:133], v15 offset1:32
	ds_read2_b32 v[134:135], v1 offset1:32
	ds_read2_b32 v[126:127], v15 offset0:64 offset1:96
	ds_read2_b32 v[124:125], v1 offset0:64 offset1:96
	ds_read2_b32 v[136:137], v16 offset0:128 offset1:160
	v_add_u32_e32 v1, 0x800, v16
	.loc	1 70 33                         ; chunk_delta_h.py:70:33
	s_sub_i32 s44, s17, s14
	.loc	1 95 17                         ; chunk_delta_h.py:95:17
	s_lshl_b32 s35, s34, 3
	.loc	1 117 20                        ; chunk_delta_h.py:117:20
	ds_read2_b32 v[138:139], v1 offset0:128 offset1:160
	ds_read2_b32 v[130:131], v16 offset0:192 offset1:224
	ds_read2_b32 v[128:129], v1 offset0:192 offset1:224
	.loc	1 95 21                         ; chunk_delta_h.py:95:21
	s_add_i32 s57, s35, s44
.Ltmp2:
	.file	2 "/opt/venv/lib/python3.12/site-packages/triton/language" "standard.py"
	.loc	2 43 17                         ; standard.py:43:17 @[ chunk_delta_h.py:81:24 ]
	s_add_i32 s62, s52, 63
.Ltmp3:
	.loc	1 95 28                         ; chunk_delta_h.py:95:28
	s_lshl_b32 s54, s57, 7
	.loc	1 112 24                        ; chunk_delta_h.py:112:24
	v_and_b32_e32 v49, 15, v0
	v_and_b32_e32 v41, 0xf0, v0
	v_lshrrev_b32_e32 v144, 4, v0
	v_mov_b32_e32 v36, 0
	v_lshlrev_b32_e32 v145, 2, v49
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_cmp_gt_i32 s62, 63
	.loc	1 112 24                        ; chunk_delta_h.py:112:24
	s_mov_b32 s55, 0
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_cselect_b64 s[26:27], -1, 0
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_lshl_or_b32 v2, v41, 6, v145
	v_mov_b32_e32 v3, v36
	v_cmp_gt_i32_e32 vcc, s52, v144
	v_lshl_add_u64 v[52:53], v[2:3], 0, s[54:55]
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_and_b64 s[14:15], vcc, s[26:27]
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 0
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	s_and_saveexec_b64 s[20:21], s[14:15]
	s_cbranch_execz .LBB0_10
; %bb.9:
	.loc	1 0 22 is_stmt 0                ; chunk_delta_h.py:0:22
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v52
	v_ashrrev_i64 v[2:3], 30, v[2:3]
	v_lshl_add_u64 v[2:3], s[6:7], 0, v[2:3]
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	global_load_dwordx4 v[2:5], v[2:3], off
.LBB0_10:
	.loc	1 0 22                          ; chunk_delta_h.py:0:22
	s_or_b64 exec, exec, s[20:21]
	.loc	1 112 24 is_stmt 1              ; chunk_delta_h.py:112:24
	v_or_b32_e32 v146, 16, v144
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_lshl_or_b32 v6, v146, 10, v145
	v_mov_b32_e32 v7, v36
	v_lshl_add_u64 v[54:55], v[6:7], 0, s[54:55]
	v_cmp_gt_i32_e32 vcc, s52, v146
	v_mov_b32_e32 v37, v54
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_and_b64 s[20:21], vcc, s[26:27]
	v_ashrrev_i64 v[26:27], 30, v[36:37]
	v_mov_b32_e32 v6, v36
	v_mov_b32_e32 v8, v36
	v_mov_b32_e32 v9, v36
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	s_and_saveexec_b64 s[22:23], s[20:21]
	s_cbranch_execz .LBB0_12
; %bb.11:
	v_lshl_add_u64 v[6:7], s[6:7], 0, v[26:27]
	global_load_dwordx4 v[6:9], v[6:7], off
.LBB0_12:
	.loc	1 0 22 is_stmt 0                ; chunk_delta_h.py:0:22
	s_or_b64 exec, exec, s[22:23]
	.loc	1 112 24 is_stmt 1              ; chunk_delta_h.py:112:24
	v_or_b32_e32 v55, 32, v144
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_lshl_or_b32 v10, v55, 10, v145
	v_mov_b32_e32 v11, v36
	v_lshl_add_u64 v[56:57], v[10:11], 0, s[54:55]
	.loc	1 112 24                        ; chunk_delta_h.py:112:24
	v_mov_b32_e32 v46, 0
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_cmp_gt_i32_e32 vcc, s52, v55
	v_mov_b32_e32 v47, v56
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_and_b64 s[22:23], vcc, s[26:27]
	v_ashrrev_i64 v[28:29], 30, v[46:47]
	v_mov_b32_e32 v10, v46
	v_mov_b32_e32 v11, v46
	v_mov_b32_e32 v12, v46
	v_mov_b32_e32 v13, v46
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	s_and_saveexec_b64 s[24:25], s[22:23]
	s_cbranch_execz .LBB0_14
; %bb.13:
	v_lshl_add_u64 v[10:11], s[6:7], 0, v[28:29]
	global_load_dwordx4 v[10:13], v[10:11], off
.LBB0_14:
	.loc	1 0 22 is_stmt 0                ; chunk_delta_h.py:0:22
	s_or_b64 exec, exec, s[24:25]
	.loc	1 112 24 is_stmt 1              ; chunk_delta_h.py:112:24
	v_or_b32_e32 v181, 48, v144
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_lshl_or_b32 v14, v181, 10, v145
	v_mov_b32_e32 v15, v36
	v_lshl_add_u64 v[58:59], v[14:15], 0, s[54:55]
	v_cmp_gt_i32_e32 vcc, s52, v181
	v_mov_b32_e32 v140, v46
	v_mov_b32_e32 v141, v58
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_and_b64 s[24:25], vcc, s[26:27]
	v_ashrrev_i64 v[34:35], 30, v[140:141]
	v_mov_b32_e32 v14, v46
	v_mov_b32_e32 v15, v46
	v_mov_b32_e32 v16, v46
	v_mov_b32_e32 v17, v46
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	s_and_saveexec_b64 s[30:31], s[24:25]
	s_cbranch_execz .LBB0_16
; %bb.15:
	v_lshl_add_u64 v[14:15], s[6:7], 0, v[34:35]
	global_load_dwordx4 v[14:17], v[14:15], off
.LBB0_16:
	.loc	1 0 22 is_stmt 0                ; chunk_delta_h.py:0:22
	s_or_b64 exec, exec, s[30:31]
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	.loc	1 160 26 is_stmt 1              ; chunk_delta_h.py:160:26
	s_and_saveexec_b64 s[30:31], s[14:15]
	s_cbranch_execz .LBB0_18
; %bb.17:
	.loc	1 0 26 is_stmt 0                ; chunk_delta_h.py:0:26
	v_ashrrev_i32_e32 v53, 31, v52
	v_lshlrev_b64 v[20:21], 2, v[52:53]
	v_or_b32_e32 v20, 0x100, v20
	v_lshl_add_u64 v[20:21], s[6:7], 0, v[20:21]
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	global_load_dwordx4 v[22:25], v[20:21], off
.LBB0_18:
	.loc	1 0 26                          ; chunk_delta_h.py:0:26
	s_or_b64 exec, exec, s[30:31]
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execz .LBB0_20
; %bb.19:
	v_or_b32_e32 v26, 0x100, v26
	v_lshl_add_u64 v[18:19], s[6:7], 0, v[26:27]
	global_load_dwordx4 v[18:21], v[18:19], off
.LBB0_20:
	.loc	1 0 26                          ; chunk_delta_h.py:0:26
	s_or_b64 exec, exec, s[14:15]
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	s_and_saveexec_b64 s[14:15], s[22:23]
	s_cbranch_execz .LBB0_22
; %bb.21:
	v_or_b32_e32 v28, 0x100, v28
	v_lshl_add_u64 v[28:29], s[6:7], 0, v[28:29]
	global_load_dwordx4 v[30:33], v[28:29], off
.LBB0_22:
	.loc	1 0 26                          ; chunk_delta_h.py:0:26
	s_or_b64 exec, exec, s[14:15]
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	s_and_saveexec_b64 s[14:15], s[24:25]
	s_cbranch_execz .LBB0_24
; %bb.23:
	v_or_b32_e32 v34, 0x100, v34
	v_lshl_add_u64 v[26:27], s[6:7], 0, v[34:35]
	global_load_dwordx4 v[26:29], v[26:27], off
.LBB0_24:
	.loc	1 0 26                          ; chunk_delta_h.py:0:26
	s_or_b64 exec, exec, s[14:15]
	.loc	1 112 24 is_stmt 1              ; chunk_delta_h.py:112:24
	v_lshlrev_b32_e32 v35, 2, v38
	v_or_b32_e32 v50, s38, v35
	v_mov_b32_e32 v51, s39
	s_mov_b64 s[14:15], 0x80
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	v_cmp_gt_i64_e64 s[14:15], s[14:15], v[50:51]
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_cmp_gt_i32_e32 vcc, s52, v172
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	s_and_b64 s[58:59], s[50:51], s[14:15]
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_lshlrev_b32_e32 v34, 10, v172
	.loc	1 112 24                        ; chunk_delta_h.py:112:24
	v_mov_b32_e32 v107, 0
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	v_or_b32_e32 v106, v34, v35
	s_and_b64 s[14:15], s[58:59], vcc
	v_lshl_add_u64 v[176:177], v[106:107], 0, s[54:55]
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_and_b64 s[20:21], s[26:27], s[14:15]
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v225, 0
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execz .LBB0_26
; %bb.25:
	.loc	1 0 22 is_stmt 0                ; chunk_delta_h.py:0:22
	v_add_u32_e32 v42, s38, v176
	v_ashrrev_i32_e32 v43, 31, v42
	v_lshl_add_u64 v[42:43], v[42:43], 2, s[4:5]
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	global_load_dwordx4 v[222:225], v[42:43], off
.LBB0_26:
	.loc	1 0 22                          ; chunk_delta_h.py:0:22
	s_or_b64 exec, exec, s[14:15]
	.loc	1 112 24 is_stmt 1              ; chunk_delta_h.py:112:24
	v_or_b32_e32 v182, 32, v172
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_cmp_gt_i32_e32 vcc, s52, v182
	v_lshlrev_b32_e32 v38, 10, v182
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	v_or_b32_e32 v106, v35, v38
	s_and_b64 s[14:15], s[58:59], vcc
	v_lshl_add_u64 v[170:171], v[106:107], 0, s[54:55]
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_and_b64 s[20:21], s[26:27], s[14:15]
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v235, 0
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execz .LBB0_28
; %bb.27:
	v_add_u32_e32 v42, s38, v170
	v_ashrrev_i32_e32 v43, 31, v42
	v_lshl_add_u64 v[42:43], v[42:43], 2, s[4:5]
	global_load_dwordx4 v[232:235], v[42:43], off
.LBB0_28:
	.loc	1 0 22 is_stmt 0                ; chunk_delta_h.py:0:22
	s_or_b64 exec, exec, s[14:15]
	.loc	1 188 31 is_stmt 1              ; chunk_delta_h.py:188:31
	v_cndmask_b32_e64 v1, 0, 1, s[26:27]
	v_cmp_ne_u32_e64 s[22:23], 1, v1
	v_mov_b32_e32 v103, 0
	s_andn2_b64 vcc, exec, s[26:27]
	v_mov_b32_e32 v188, 0
	s_cbranch_vccnz .LBB0_30
; %bb.29:
	.loc	1 0 31 is_stmt 0                ; chunk_delta_h.py:0:31
	s_min_u32 s14, s52, 64
	s_lshl_b32 s14, s14, 3
	s_add_i32 s14, s44, s14
	s_add_i32 s14, s14, s35
	s_add_i32 s14, s14, -8
	s_ashr_i32 s15, s14, 31
	s_lshl_b64 s[14:15], s[14:15], 2
	s_add_u32 s14, s10, s14
	s_addc_u32 s15, s11, s15
	v_mov_b32_e32 v1, 0
	.loc	1 188 31                        ; chunk_delta_h.py:188:31
	global_load_dword v188, v1, s[14:15]
.LBB0_30:
	.loc	1 112 24 is_stmt 1              ; chunk_delta_h.py:112:24
	v_and_b32_e32 v57, 31, v0
	v_lshrrev_b32_e32 v148, 2, v150
	v_or_b32_e32 v59, v148, v57
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_cmp_gt_i32_e64 s[24:25], s52, v59
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_and_b64 s[20:21], s[24:25], s[26:27]
	v_mov_b32_e32 v226, 0
	.loc	1 192 26                        ; chunk_delta_h.py:192:26
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execz .LBB0_32
; %bb.31:
	.loc	1 0 26 is_stmt 0                ; chunk_delta_h.py:0:26
	v_lshlrev_b32_e32 v43, 3, v59
	s_mov_b32 s56, 0
	v_mov_b32_e32 v42, 0
	v_lshl_add_u64 v[42:43], s[56:57], 0, v[42:43]
	v_ashrrev_i64 v[42:43], 30, v[42:43]
	v_lshl_add_u64 v[42:43], s[10:11], 0, v[42:43]
	.loc	1 192 26                        ; chunk_delta_h.py:192:26
	global_load_dword v226, v[42:43], off
.LBB0_32:
	.loc	1 0 26                          ; chunk_delta_h.py:0:26
	s_or_b64 exec, exec, s[14:15]
	.loc	1 96 30 is_stmt 1               ; chunk_delta_h.py:96:30
	s_bfe_u32 s14, s44, 0x10007
	s_add_i32 s14, s44, s14
	s_bfe_i32 s14, s14, 0x80000
	s_sext_i32_i16 s14, s14
	.loc	1 96 42 is_stmt 0               ; chunk_delta_h.py:96:42
	s_lshl_b32 s14, s14, 6
	s_mov_b32 s56, 0
	s_lshl_b32 s15, s34, 9
	s_and_b32 s14, s14, 0xffffff80
	.loc	1 112 24 is_stmt 1              ; chunk_delta_h.py:112:24
	v_lshrrev_b32_e32 v158, 2, v39
	.loc	1 96 42                         ; chunk_delta_h.py:96:42
	s_add_i32 s14, s15, s14
	.loc	1 112 24                        ; chunk_delta_h.py:112:24
	v_or_b32_e32 v159, 1, v158
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_lshl_or_b32 v84, v39, 7, v40
	s_mov_b32 s15, s56
	v_lshl_or_b32 v42, v159, 9, v40
	v_mov_b32_e32 v43, v85
	v_lshl_add_u64 v[98:99], v[84:85], 0, s[14:15]
	v_cmp_gt_i32_e32 vcc, s52, v158
	v_lshl_add_u64 v[100:101], v[42:43], 0, s[14:15]
	v_cmp_gt_i32_e64 s[14:15], s52, v159
	v_lshlrev_b32_e32 v149, 1, v98
	v_bfrev_b32_e32 v1, 1
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_and_b64 vcc, vcc, s[26:27]
	s_mov_b32 s43, 0x27000
	s_mov_b32 s42, 0x7ffffffe
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	s_and_b32 s41, s41, 0xffff
	v_cndmask_b32_e32 v35, v1, v149, vcc
	v_lshlrev_b32_e32 v255, 1, v100
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_and_b64 s[14:15], s[14:15], s[26:27]
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_cndmask_b32_e64 v40, v1, v255, s[14:15]
	buffer_load_dwordx4 v[42:45], v35, s[40:43], 0 offen
	buffer_load_dwordx4 v[64:67], v40, s[40:43], 0 offen
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	v_or_b32_e32 v35, 0x80, v149
	v_cndmask_b32_e32 v35, v1, v35, vcc
	v_or_b32_e32 v40, 0x80, v255
	v_cndmask_b32_e64 v1, v1, v40, s[14:15]
	buffer_load_dwordx4 v[68:71], v35, s[40:43], 0 offen
	buffer_load_dwordx4 v[72:75], v1, s[40:43], 0 offen
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_lshlrev_b32_e32 v157, 4, v0
	v_lshrrev_b32_e32 v1, 1, v41
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_and_b32_e32 v35, 6, v0
	v_lshrrev_b32_e32 v39, 1, v39
	v_mov_b32_e32 v40, 0x440
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_xor_b32_e32 v1, v157, v1
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_lshlrev_b32_e32 v41, 2, v35
	.loc	1 112 16                        ; chunk_delta_h.py:112:16
	v_cmp_eq_u32_e32 vcc, 0, v156
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_add_u32_e32 v173, 0, v1
	v_xor_b32_e32 v1, 8, v1
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_cndmask_b32_e64 v40, v40, 0, vcc
	v_xor_b32_e32 v39, v41, v39
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_add_u32_e32 v174, 0, v1
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_xor_b32_e32 v1, v39, v40
	s_mov_b32 s31, 0x5040100
	v_lshl_or_b32 v1, v35, 10, v1
	s_mov_b32 s36, 0x7060302
	v_add_u32_e32 v175, 0, v1
	v_xor_b32_e32 v35, 8, v1
	v_xor_b32_e32 v39, 16, v1
	v_xor_b32_e32 v40, 24, v1
	v_xor_b32_e32 v41, 32, v1
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(4)
	ds_write2st64_b64 v173, v[2:3], v[6:7] offset1:8
	ds_write2st64_b64 v173, v[10:11], v[14:15] offset0:16 offset1:24
	ds_write2st64_b64 v174, v[4:5], v[8:9] offset1:8
	ds_write2st64_b64 v174, v[12:13], v[16:17] offset0:16 offset1:24
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	ds_write2st64_b64 v173, v[22:23], v[18:19] offset0:32 offset1:40
	ds_write2st64_b64 v173, v[30:31], v[26:27] offset0:48 offset1:56
	ds_write2st64_b64 v174, v[24:25], v[20:21] offset0:32 offset1:40
	ds_write2st64_b64 v174, v[32:33], v[28:29] offset0:48 offset1:56
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_add_u32_e32 v178, 0, v35
	v_add_u32_e32 v179, 0, v39
	v_add_u32_e32 v180, 0, v40
	v_add_u32_e32 v183, 0, v41
	.loc	1 112 24                        ; chunk_delta_h.py:112:24
	v_lshrrev_b32_e32 v51, 3, v142
.Ltmp4:
	.loc	2 43 30                         ; standard.py:43:30 @[ chunk_delta_h.py:81:24 ]
	s_ashr_i32 s30, s62, 31
.Ltmp5:
	.loc	1 112 24                        ; chunk_delta_h.py:112:24
	v_mov_b32_e32 v105, s39
	s_mov_b64 s[14:15], 0x80
	v_or_b32_e32 v104, s38, v51
.Ltmp6:
	.loc	2 43 30                         ; standard.py:43:30 @[ chunk_delta_h.py:81:24 ]
	s_lshr_b32 s30, s30, 26
	v_and_b32_e32 v151, 2, v0
.Ltmp7:
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	v_cmp_gt_i64_e32 vcc, s[14:15], v[104:105]
.Ltmp8:
	.loc	2 43 30                         ; standard.py:43:30 @[ chunk_delta_h.py:81:24 ]
	s_add_i32 s14, s62, s30
	s_ashr_i32 s55, s14, 6
.Ltmp9:
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_and_b64 s[60:61], vcc, s[50:51]
	.loc	1 112 24                        ; chunk_delta_h.py:112:24
	v_cmp_eq_u32_e64 s[20:21], 0, v142
	v_cmp_eq_u32_e64 s[34:35], 0, v150
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_cmpk_gt_i32 s62, 0x7f
	v_accvgpr_write_b32 a40, v48
	v_accvgpr_write_b32 a51, v51
	v_lshrrev_b32_e32 v12, 5, v0
	v_lshlrev_b32_e32 v39, 3, v49
	v_lshlrev_b32_e32 v8, 6, v150
	v_accvgpr_write_b32 a42, v49
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	s_waitcnt vmcnt(2)
	v_perm_b32 v2, v64, v42, s31
	v_perm_b32 v3, v64, v42, s36
	v_perm_b32 v4, v65, v43, s31
	v_perm_b32 v5, v65, v43, s36
	ds_write_b32 v175, v2 offset:32768
	ds_write_b32 v178, v3 offset:32896
	ds_write_b32 v179, v4 offset:33024
	ds_write_b32 v180, v5 offset:33152
	v_perm_b32 v2, v66, v44, s31
	ds_write_b32 v183, v2 offset:33280
	v_xor_b32_e32 v2, 40, v1
	v_add_u32_e32 v185, 0, v2
	v_perm_b32 v2, v66, v44, s36
	ds_write_b32 v185, v2 offset:33408
	v_xor_b32_e32 v2, 48, v1
	v_add_u32_e32 v186, 0, v2
	v_perm_b32 v2, v67, v45, s31
	ds_write_b32 v186, v2 offset:33536
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	s_waitcnt vmcnt(0)
	v_perm_b32 v2, v72, v68, s31
	ds_write_b32 v175, v2 offset:40960
	v_perm_b32 v2, v72, v68, s36
	ds_write_b32 v178, v2 offset:41088
	v_perm_b32 v2, v73, v69, s31
	ds_write_b32 v179, v2 offset:41216
	v_perm_b32 v2, v73, v69, s36
	ds_write_b32 v180, v2 offset:41344
	v_perm_b32 v2, v74, v70, s31
	v_xor_b32_e32 v1, 56, v1
	ds_write_b32 v183, v2 offset:41472
	v_perm_b32 v2, v74, v70, s36
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_add_u32_e32 v187, 0, v1
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	ds_write_b32 v185, v2 offset:41600
	v_perm_b32 v2, v75, v71, s31
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_perm_b32 v1, v67, v45, s36
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	ds_write_b32 v186, v2 offset:41728
	v_perm_b32 v2, v75, v71, s36
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_add_u32_e32 v3, 0x80, v187
	ds_write2st64_b32 v3, v1, v2 offset0:131 offset1:163
	v_mov_b32_e32 v2, 0x240
	v_cndmask_b32_e64 v2, v2, 0, s[28:29]
	v_lshlrev_b32_e32 v1, 2, v151
	v_and_or_b32 v2, v60, 48, v2
	v_lshl_or_b32 v1, v156, 10, v1
	v_xor_b32_e32 v2, v2, v48
	v_or3_b32 v99, v2, v1, v51
	v_lshlrev_b32_e32 v4, 8, v57
	v_lshlrev_b32_e32 v5, 2, v57
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_cbranch_scc1 .LBB0_34
; %bb.33:                               ; %._crit_edge525
	.loc	1 134 23                        ; chunk_delta_h.py:134:23
	v_mov_b32_e32 v1, 0x108
	v_cndmask_b32_e64 v1, v1, 0, s[2:3]
	v_and_or_b32 v1, v12, 2, v1
	v_accvgpr_write_b32 a52, v1
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_lshlrev_b32_e32 v35, 3, v49
	v_accvgpr_read_b32 v1, a41
	v_xor_b32_e32 v171, v35, v1
	v_or3_b32 v166, v4, v8, v171
	v_xor_b32_e32 v1, 64, v166
	v_accvgpr_write_b32 a55, v1
	v_or_b32_e32 v10, 0x80, v1
	v_xor_b32_e32 v1, 0x60, v166
	v_accvgpr_write_b32 a54, v1
	v_or_b32_e32 v14, 0x80, v1
	v_xor_b32_e32 v1, 0x70, v166
	v_accvgpr_write_b32 a56, v1
	v_or_b32_e32 v15, 0x80, v1
	.loc	1 155 26                        ; chunk_delta_h.py:155:26
	v_mov_b32_e32 v1, 0x420
	v_cndmask_b32_e64 v1, v1, 0, s[20:21]
	v_xor_b32_e32 v1, v1, v5
	v_or_b32_e32 v101, v171, v4
	v_or_b32_e32 v168, v1, v150
	v_xor_b32_e32 v1, 16, v101
	v_accvgpr_write_b32 a57, v1
	v_xor_b32_e32 v1, 32, v101
	v_accvgpr_write_b32 a58, v1
	v_xor_b32_e32 v1, 48, v101
	v_accvgpr_write_b32 a59, v1
	v_xor_b32_e32 v1, 64, v101
	v_accvgpr_write_b32 a60, v1
	v_xor_b32_e32 v1, 0x50, v101
	v_accvgpr_write_b32 a61, v1
	v_xor_b32_e32 v1, 0x60, v101
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_xor_b32_e32 v169, 16, v166
	v_xor_b32_e32 v230, 32, v166
	v_xor_b32_e32 v167, 48, v166
	v_xor_b32_e32 v53, 0x50, v166
	v_accvgpr_write_b32 a62, v1
	.loc	1 155 26                        ; chunk_delta_h.py:155:26
	v_xor_b32_e32 v1, 0x70, v101
	.loc	1 134 23                        ; chunk_delta_h.py:134:23
	v_xor_b32_e32 v9, 0x108, v99
	s_and_b32 s29, s13, 0xffff
	s_mov_b32 s28, s12
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_or_b32_e32 v2, 0x80, v166
	v_or_b32_e32 v3, 0x80, v169
	v_or_b32_e32 v6, 0x80, v230
	v_or_b32_e32 v7, 0x80, v167
	v_or_b32_e32 v11, 0x80, v53
	.loc	1 155 26                        ; chunk_delta_h.py:155:26
	v_xor_b32_e32 v40, 0x108, v168
	v_xor_b32_e32 v41, 0x210, v168
	v_xor_b32_e32 v42, 0x318, v168
	v_xor_b32_e32 v43, 0x840, v168
	v_xor_b32_e32 v44, 0x948, v168
	v_xor_b32_e32 v45, 0xa50, v168
	v_xor_b32_e32 v48, 0xb58, v168
	v_accvgpr_write_b32 a63, v1
	v_mov_b32_e32 v64, v118
	v_mov_b32_e32 v65, v110
	v_mov_b32_e32 v66, v116
	v_mov_b32_e32 v67, v108
	v_mov_b32_e32 v68, v120
	v_mov_b32_e32 v69, v114
	v_mov_b32_e32 v70, v122
	v_mov_b32_e32 v71, v112
	v_mov_b32_e32 v72, v119
	v_mov_b32_e32 v73, v111
	v_mov_b32_e32 v74, v117
	v_mov_b32_e32 v75, v109
	v_mov_b32_e32 v76, v121
	v_mov_b32_e32 v77, v115
	v_mov_b32_e32 v78, v123
	v_mov_b32_e32 v79, v113
	v_mov_b32_e32 v80, v132
	v_mov_b32_e32 v81, v126
	v_mov_b32_e32 v82, v134
	v_mov_b32_e32 v83, v124
	v_mov_b32_e32 v86, v136
	v_mov_b32_e32 v87, v130
	v_mov_b32_e32 v88, v138
	v_mov_b32_e32 v89, v128
	v_mov_b32_e32 v90, v133
	v_mov_b32_e32 v91, v127
	v_mov_b32_e32 v92, v135
	v_mov_b32_e32 v93, v125
	v_mov_b32_e32 v94, v137
	v_mov_b32_e32 v95, v131
	v_mov_b32_e32 v96, v139
	v_mov_b32_e32 v97, v129
	s_mov_b64 s[30:31], 0
	v_mov_b32_e32 v194, v226
	s_mov_b64 s[14:15], s[42:43]
	s_branch .LBB0_35
.LBB0_34:
	.loc	1 0 26 is_stmt 0                ; chunk_delta_h.py:0:26
	s_mov_b64 s[30:31], -1
                                        ; implicit-def: $vgpr96_vgpr97
                                        ; implicit-def: $vgpr94_vgpr95
                                        ; implicit-def: $vgpr92_vgpr93
                                        ; implicit-def: $vgpr90_vgpr91
                                        ; implicit-def: $vgpr88_vgpr89
                                        ; implicit-def: $vgpr86_vgpr87
                                        ; implicit-def: $vgpr82_vgpr83
                                        ; implicit-def: $vgpr80_vgpr81
                                        ; implicit-def: $vgpr78_vgpr79
                                        ; implicit-def: $vgpr76_vgpr77
                                        ; implicit-def: $vgpr74_vgpr75
                                        ; implicit-def: $vgpr72_vgpr73
                                        ; implicit-def: $vgpr70_vgpr71
                                        ; implicit-def: $vgpr68_vgpr69
                                        ; implicit-def: $vgpr66_vgpr67
                                        ; implicit-def: $vgpr64_vgpr65
                                        ; implicit-def: $vgpr194
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $agpr52
                                        ; implicit-def: $sgpr28_sgpr29
                                        ; implicit-def: $vgpr35
                                        ; implicit-def: $vgpr171
                                        ; implicit-def: $vgpr166
                                        ; implicit-def: $vgpr2
                                        ; implicit-def: $vgpr169
                                        ; implicit-def: $vgpr3
                                        ; implicit-def: $vgpr230
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr167
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $agpr55
                                        ; implicit-def: $vgpr10
                                        ; implicit-def: $vgpr53
                                        ; implicit-def: $vgpr11
                                        ; implicit-def: $agpr54
                                        ; implicit-def: $vgpr14
                                        ; implicit-def: $agpr56
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr168
                                        ; implicit-def: $vgpr40
                                        ; implicit-def: $vgpr41
                                        ; implicit-def: $vgpr42
                                        ; implicit-def: $vgpr43
                                        ; implicit-def: $vgpr44
                                        ; implicit-def: $vgpr45
                                        ; implicit-def: $vgpr48
                                        ; implicit-def: $vgpr101
                                        ; implicit-def: $agpr57
                                        ; implicit-def: $agpr58
                                        ; implicit-def: $agpr59
                                        ; implicit-def: $agpr60
                                        ; implicit-def: $agpr61
                                        ; implicit-def: $agpr62
                                        ; implicit-def: $agpr63
.LBB0_35:                               ; %Flow1322
	s_load_dwordx2 s[36:37], s[0:1], 0x38
	s_and_b64 s[46:47], s[24:25], s[60:61]
	s_mul_i32 s65, s55, s16
	s_lshl_b32 s64, s44, 14
	s_andn2_b64 vcc, exec, s[30:31]
	v_add_u32_e32 v160, 0, v99
	s_cbranch_vccnz .LBB0_86
; %bb.36:                               ; %.lr.ph
	v_xor_b32_e32 v1, 8, v99
	v_add_u32_e32 v189, 0, v1
	v_mov_b32_e32 v1, 0x108
	v_cndmask_b32_e64 v1, v1, 0, s[2:3]
	v_and_or_b32 v2, v12, 2, v1
	v_accvgpr_read_b32 v1, a51
	v_accvgpr_write_b32 a52, v2
	v_or3_b32 v1, v2, v1, v150
	v_accvgpr_read_b32 v2, a48
	v_accvgpr_read_b32 v3, a49
	v_or3_b32 v1, v1, v3, v2
	v_xor_b32_e32 v2, 8, v1
	v_add_u32_e32 v190, 0, v1
	v_add_u32_e32 v191, 0, v2
	v_xor_b32_e32 v2, 64, v1
	v_xor_b32_e32 v1, 0x48, v1
	v_add_u32_e32 v193, 0, v1
	.loc	1 134 31 is_stmt 1              ; chunk_delta_h.py:134:31
	v_mov_b32_e32 v1, 1
	v_and_b32_sdwa v3, v118, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	s_movk_i32 s1, 0x7fff
	v_add_u32_e32 v192, 0, v2
	v_and_b32_sdwa v2, v119, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v3, v118, v3, s1
	v_add3_u32 v2, v119, v2, s1
	v_lshrrev_b32_e32 v3, 16, v3
	v_mov_b32_e32 v6, 0x7fff
	v_cmp_o_f32_e32 vcc, v118, v118
	v_lshrrev_b32_e32 v2, 16, v2
	s_mov_b32 s0, 0x5040100
	v_cndmask_b32_e32 v3, v6, v3, vcc
	v_cmp_o_f32_e32 vcc, v119, v119
	v_and_b32_sdwa v7, v116, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v7, v116, v7, s1
	v_cndmask_b32_e32 v2, v6, v2, vcc
	v_perm_b32 v2, v2, v3, s0
	v_and_b32_sdwa v3, v117, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v3, v117, v3, s1
	v_lshrrev_b32_e32 v7, 16, v7
	v_cmp_o_f32_e32 vcc, v116, v116
	v_lshrrev_b32_e32 v3, 16, v3
	v_and_b32_sdwa v9, v120, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e32 v7, v6, v7, vcc
	v_cmp_o_f32_e32 vcc, v117, v117
	v_add3_u32 v9, v120, v9, s1
	v_lshrrev_b32_e32 v9, 16, v9
	v_cndmask_b32_e32 v3, v6, v3, vcc
	v_perm_b32 v3, v3, v7, s0
	v_and_b32_sdwa v7, v121, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v7, v121, v7, s1
	v_cmp_o_f32_e32 vcc, v120, v120
	v_lshrrev_b32_e32 v7, 16, v7
	v_and_b32_sdwa v10, v122, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e32 v9, v6, v9, vcc
	v_cmp_o_f32_e32 vcc, v121, v121
	v_add3_u32 v10, v122, v10, s1
	v_lshrrev_b32_e32 v10, 16, v10
	v_cndmask_b32_e32 v7, v6, v7, vcc
	v_perm_b32 v7, v7, v9, s0
	v_and_b32_sdwa v9, v123, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v9, v123, v9, s1
	v_cmp_o_f32_e32 vcc, v122, v122
	v_lshrrev_b32_e32 v9, 16, v9
	v_and_b32_sdwa v11, v110, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e32 v10, v6, v10, vcc
	v_cmp_o_f32_e32 vcc, v123, v123
	v_add3_u32 v11, v110, v11, s1
	v_lshrrev_b32_e32 v11, 16, v11
	v_cndmask_b32_e32 v9, v6, v9, vcc
	v_perm_b32 v9, v9, v10, s0
	v_and_b32_sdwa v10, v111, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v10, v111, v10, s1
	v_cmp_o_f32_e32 vcc, v110, v110
	v_lshrrev_b32_e32 v10, 16, v10
	v_and_b32_sdwa v12, v108, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e32 v11, v6, v11, vcc
	v_cmp_o_f32_e32 vcc, v111, v111
	v_add3_u32 v12, v108, v12, s1
	v_lshrrev_b32_e32 v12, 16, v12
	v_cndmask_b32_e32 v10, v6, v10, vcc
	v_perm_b32 v10, v10, v11, s0
	v_and_b32_sdwa v11, v109, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v11, v109, v11, s1
	v_cmp_o_f32_e32 vcc, v108, v108
	v_lshrrev_b32_e32 v11, 16, v11
	v_and_b32_sdwa v13, v114, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e32 v12, v6, v12, vcc
	v_cmp_o_f32_e32 vcc, v109, v109
	v_add3_u32 v13, v114, v13, s1
	v_lshrrev_b32_e32 v13, 16, v13
	v_cndmask_b32_e32 v11, v6, v11, vcc
	v_perm_b32 v11, v11, v12, s0
	v_and_b32_sdwa v12, v115, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v12, v115, v12, s1
	v_cmp_o_f32_e32 vcc, v114, v114
	v_lshrrev_b32_e32 v12, 16, v12
	v_and_b32_sdwa v14, v112, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e32 v13, v6, v13, vcc
	v_cmp_o_f32_e32 vcc, v115, v115
	v_add3_u32 v14, v112, v14, s1
	v_lshrrev_b32_e32 v14, 16, v14
	v_cndmask_b32_e32 v12, v6, v12, vcc
	v_perm_b32 v12, v12, v13, s0
	v_and_b32_sdwa v13, v113, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v13, v113, v13, s1
	v_cmp_o_f32_e32 vcc, v112, v112
	v_lshrrev_b32_e32 v13, 16, v13
	.loc	1 132 16                        ; chunk_delta_h.py:132:16
	s_lshl_b32 s28, s65, 17
	.loc	1 134 31                        ; chunk_delta_h.py:134:31
	v_cndmask_b32_e32 v14, v6, v14, vcc
	v_cmp_o_f32_e32 vcc, v113, v113
	.loc	1 132 16                        ; chunk_delta_h.py:132:16
	s_add_i32 s28, s28, s64
	.loc	1 134 23                        ; chunk_delta_h.py:134:23
	v_or_b32_e32 v18, s28, v152
	.loc	1 134 31 is_stmt 0              ; chunk_delta_h.py:134:31
	v_cndmask_b32_e32 v13, v6, v13, vcc
	v_perm_b32 v13, v13, v14, s0
	.loc	1 134 23                        ; chunk_delta_h.py:134:23
	v_add_u32_e32 v14, 0xc000, v160
	ds_write2_b32 v14, v2, v10 offset1:32
	v_add_u32_e32 v2, 0xc800, v160
	ds_write2_b32 v2, v3, v11 offset1:32
	v_add_u32_e32 v3, 0xc000, v189
	ds_write2_b32 v3, v7, v12 offset0:64 offset1:96
	v_add_u32_e32 v7, 0xc800, v189
	ds_write2_b32 v7, v9, v13 offset0:64 offset1:96
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_u16 v9, v190 offset:49152
	ds_read_u16 v10, v190 offset:50176
	ds_read_u16 v11, v191 offset:49152
	ds_read_u16 v15, v191 offset:50176
	ds_read_u16 v12, v192 offset:49664
	ds_read_u16 v16, v192 offset:50688
	ds_read_u16 v13, v193 offset:49664
	ds_read_u16 v17, v193 offset:50688
	s_waitcnt lgkmcnt(4)
	v_perm_b32 v11, v15, v11, s0
	v_perm_b32 v10, v10, v9, s0
	v_add_lshl_u32 v9, v18, s48, 1
	v_bfrev_b32_e32 v15, 1
	s_and_b32 s13, s13, 0xffff
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v13, v17, v13, s0
	v_perm_b32 v12, v16, v12, s0
	v_cndmask_b32_e64 v9, v15, v9, s[18:19]
	buffer_store_dwordx4 v[10:13], v9, s[12:15], 0 offen
	.loc	1 139 35 is_stmt 1              ; chunk_delta_h.py:139:35
	v_and_b32_sdwa v9, v133, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v9, v133, v9, s1
	v_and_b32_sdwa v10, v132, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v10, v132, v10, s1
	v_lshrrev_b32_e32 v10, 16, v10
	v_cmp_o_f32_e32 vcc, v132, v132
	v_lshrrev_b32_e32 v9, 16, v9
	v_and_b32_sdwa v11, v134, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e32 v10, v6, v10, vcc
	v_cmp_o_f32_e32 vcc, v133, v133
	v_add3_u32 v11, v134, v11, s1
	v_lshrrev_b32_e32 v11, 16, v11
	v_cndmask_b32_e32 v9, v6, v9, vcc
	v_perm_b32 v9, v9, v10, s0
	v_and_b32_sdwa v10, v135, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v10, v135, v10, s1
	v_cmp_o_f32_e32 vcc, v134, v134
	v_lshrrev_b32_e32 v10, 16, v10
	v_and_b32_sdwa v12, v136, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e32 v11, v6, v11, vcc
	v_cmp_o_f32_e32 vcc, v135, v135
	v_add3_u32 v12, v136, v12, s1
	v_lshrrev_b32_e32 v12, 16, v12
	v_cndmask_b32_e32 v10, v6, v10, vcc
	v_perm_b32 v10, v10, v11, s0
	v_and_b32_sdwa v11, v137, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v11, v137, v11, s1
	v_cmp_o_f32_e32 vcc, v136, v136
	v_lshrrev_b32_e32 v11, 16, v11
	v_and_b32_sdwa v13, v138, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e32 v12, v6, v12, vcc
	v_cmp_o_f32_e32 vcc, v137, v137
	v_add3_u32 v13, v138, v13, s1
	v_lshrrev_b32_e32 v13, 16, v13
	v_cndmask_b32_e32 v11, v6, v11, vcc
	v_perm_b32 v11, v11, v12, s0
	v_and_b32_sdwa v12, v139, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v12, v139, v12, s1
	v_cmp_o_f32_e32 vcc, v138, v138
	v_lshrrev_b32_e32 v12, 16, v12
	v_and_b32_sdwa v16, v126, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e32 v13, v6, v13, vcc
	v_cmp_o_f32_e32 vcc, v139, v139
	v_add3_u32 v16, v126, v16, s1
	v_lshrrev_b32_e32 v16, 16, v16
	v_cndmask_b32_e32 v12, v6, v12, vcc
	v_perm_b32 v12, v12, v13, s0
	v_and_b32_sdwa v13, v127, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v13, v127, v13, s1
	v_cmp_o_f32_e32 vcc, v126, v126
	v_lshrrev_b32_e32 v13, 16, v13
	v_and_b32_sdwa v17, v124, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e32 v16, v6, v16, vcc
	v_cmp_o_f32_e32 vcc, v127, v127
	v_add3_u32 v17, v124, v17, s1
	v_lshrrev_b32_e32 v17, 16, v17
	v_cndmask_b32_e32 v13, v6, v13, vcc
	v_perm_b32 v13, v13, v16, s0
	v_and_b32_sdwa v16, v125, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v16, v125, v16, s1
	v_cmp_o_f32_e32 vcc, v124, v124
	v_lshrrev_b32_e32 v16, 16, v16
	v_and_b32_sdwa v19, v130, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e32 v17, v6, v17, vcc
	v_cmp_o_f32_e32 vcc, v125, v125
	v_add3_u32 v19, v130, v19, s1
	v_lshrrev_b32_e32 v19, 16, v19
	v_cndmask_b32_e32 v16, v6, v16, vcc
	v_perm_b32 v16, v16, v17, s0
	v_and_b32_sdwa v17, v131, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v17, v131, v17, s1
	v_cmp_o_f32_e32 vcc, v130, v130
	v_lshrrev_b32_e32 v17, 16, v17
	.loc	1 139 27 is_stmt 0              ; chunk_delta_h.py:139:27
	s_waitcnt lgkmcnt(0)
	.loc	1 139 35                        ; chunk_delta_h.py:139:35
	v_cndmask_b32_e32 v19, v6, v19, vcc
	v_cmp_o_f32_e32 vcc, v131, v131
	.loc	1 139 27                        ; chunk_delta_h.py:139:27
	s_barrier
	.loc	1 152 63 is_stmt 1              ; chunk_delta_h.py:152:63
	s_ashr_i32 s53, s52, 31
	.loc	1 139 35                        ; chunk_delta_h.py:139:35
	v_cndmask_b32_e32 v17, v6, v17, vcc
	v_perm_b32 v17, v17, v19, s0
	v_and_b32_sdwa v19, v129, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_and_b32_sdwa v1, v128, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v1, v128, v1, s1
	v_add3_u32 v19, v129, v19, s1
	v_lshrrev_b32_e32 v1, 16, v1
	v_cmp_o_f32_e32 vcc, v128, v128
	v_lshrrev_b32_e32 v19, 16, v19
	v_mov_b32_e32 v22, 0
	v_cndmask_b32_e32 v1, v6, v1, vcc
	v_cmp_o_f32_e32 vcc, v129, v129
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v24, 0
	v_cndmask_b32_e32 v6, v6, v19, vcc
	v_perm_b32 v1, v6, v1, s0
	.loc	1 139 27 is_stmt 0              ; chunk_delta_h.py:139:27
	ds_write2_b32 v14, v9, v13 offset1:32
	ds_write2_b32 v2, v10, v16 offset1:32
	ds_write2_b32 v3, v11, v17 offset0:64 offset1:96
	ds_write2_b32 v7, v12, v1 offset0:64 offset1:96
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_u16 v1, v191 offset:49152
	ds_read_u16 v2, v191 offset:50176
	ds_read_u16 v3, v192 offset:49664
	ds_read_u16 v6, v193 offset:49664
	ds_read_u16 v7, v193 offset:50688
	ds_read_u16 v9, v192 offset:50688
	ds_read_u16 v10, v190 offset:49152
	ds_read_u16 v14, v190 offset:50176
	s_waitcnt lgkmcnt(6)
	v_perm_b32 v11, v2, v1, s0
	v_add_lshl_u32 v1, v18, s49, 1
	s_waitcnt lgkmcnt(2)
	v_perm_b32 v12, v9, v3, s0
	.loc	1 154 22 is_stmt 1              ; chunk_delta_h.py:154:22
	v_or_b32_e32 v2, 64, v144
	v_mov_b32_e32 v3, v36
	.loc	1 139 27                        ; chunk_delta_h.py:139:27
	v_perm_b32 v13, v7, v6, s0
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v10, v14, v10, s0
	v_cndmask_b32_e64 v1, v15, v1, s[18:19]
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_cmp_gt_u64_e32 vcc, s[52:53], v[2:3]
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v25, 0
	.loc	1 139 27                        ; chunk_delta_h.py:139:27
	buffer_store_dwordx4 v[10:13], v1, s[12:15], 0 offen
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_38
; %bb.37:
	.loc	1 0 22 is_stmt 0                ; chunk_delta_h.py:0:22
	v_add_u32_e32 v3, 0x10000, v52
	v_mov_b32_e32 v2, 0
	v_ashrrev_i64 v[2:3], 30, v[2:3]
	v_lshl_add_u64 v[2:3], s[6:7], 0, v[2:3]
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	global_load_dwordx4 v[22:25], v[2:3], off
.LBB0_38:
	.loc	1 0 22                          ; chunk_delta_h.py:0:22
	s_or_b64 exec, exec, s[0:1]
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_or_b32_e32 v2, 64, v146
	v_mov_b32_e32 v3, v36
	v_cmp_gt_u64_e64 s[0:1], s[52:53], v[2:3]
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	s_and_saveexec_b64 s[28:29], s[0:1]
	s_cbranch_execz .LBB0_40
; %bb.39:
	v_add_u32_e32 v3, 0x10000, v37
	v_mov_b32_e32 v2, v36
	v_ashrrev_i64 v[2:3], 30, v[2:3]
	v_lshl_add_u64 v[2:3], s[6:7], 0, v[2:3]
	global_load_dwordx4 v[18:21], v[2:3], off
.LBB0_40:
	.loc	1 0 22                          ; chunk_delta_h.py:0:22
	s_or_b64 exec, exec, s[28:29]
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_or_b32_e32 v2, 64, v55
	v_mov_b32_e32 v3, v46
	v_cmp_gt_u64_e64 s[28:29], s[52:53], v[2:3]
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	s_and_saveexec_b64 s[30:31], s[28:29]
	s_cbranch_execz .LBB0_42
; %bb.41:
	v_add_u32_e32 v3, 0x10000, v47
	v_mov_b32_e32 v2, v46
	v_ashrrev_i64 v[2:3], 30, v[2:3]
	v_lshl_add_u64 v[2:3], s[6:7], 0, v[2:3]
	global_load_dwordx4 v[30:33], v[2:3], off
.LBB0_42:
	.loc	1 0 22                          ; chunk_delta_h.py:0:22
	s_or_b64 exec, exec, s[30:31]
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_or_b32_e32 v2, 64, v181
	v_mov_b32_e32 v3, v46
	v_cmp_gt_u64_e64 s[30:31], s[52:53], v[2:3]
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	s_and_saveexec_b64 s[44:45], s[30:31]
	s_cbranch_execz .LBB0_44
; %bb.43:
	v_add_u32_e32 v3, 0x10000, v141
	v_mov_b32_e32 v2, v140
	v_ashrrev_i64 v[2:3], 30, v[2:3]
	v_lshl_add_u64 v[2:3], s[6:7], 0, v[2:3]
	global_load_dwordx4 v[26:29], v[2:3], off
.LBB0_44:
	.loc	1 0 22                          ; chunk_delta_h.py:0:22
	s_or_b64 exec, exec, s[44:45]
	v_accvgpr_read_b32 v1, a41
	v_xor_b32_e32 v171, v39, v1
	v_mov_b32_e32 v1, 0x420
	v_cndmask_b32_e64 v1, v1, 0, s[20:21]
	v_xor_b32_e32 v1, v1, v5
	v_or3_b32 v166, v4, v8, v171
	v_or_b32_e32 v168, v1, v150
	v_or_b32_e32 v101, v171, v4
	v_xor_b32_e32 v169, 16, v166
	v_xor_b32_e32 v230, 32, v166
	v_xor_b32_e32 v167, 48, v166
	v_xor_b32_e32 v38, 64, v166
	v_xor_b32_e32 v53, 0x50, v166
	v_xor_b32_e32 v62, 0x60, v166
	v_xor_b32_e32 v63, 0x70, v166
	v_xor_b32_e32 v1, 8, v168
	v_xor_b32_e32 v34, 16, v168
	v_xor_b32_e32 v35, 24, v168
	v_xor_b32_e32 v48, 64, v168
	v_xor_b32_e32 v49, 0x48, v168
	v_xor_b32_e32 v51, 0x50, v168
	v_xor_b32_e32 v61, 0x58, v168
	.loc	1 155 26 is_stmt 1              ; chunk_delta_h.py:155:26
	v_add_u32_e32 v211, 0, v101
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_add_u32_e32 v195, 0, v166
	v_add_u32_e32 v196, 0, v169
	v_add_u32_e32 v197, 0, v230
	v_add_u32_e32 v198, 0, v167
	v_add_u32_e32 v199, 0, v38
	v_add_u32_e32 v200, 0, v53
	v_add_u32_e32 v201, 0, v62
	v_add_u32_e32 v202, 0, v63
	.loc	1 155 26                        ; chunk_delta_h.py:155:26
	v_add_u32_e32 v203, 0, v168
	v_add_u32_e32 v204, 0, v1
	v_add_u32_e32 v205, 0, v34
	v_add_u32_e32 v206, 0, v35
	v_add_u32_e32 v207, 0, v48
	v_add_u32_e32 v208, 0, v49
	v_add_u32_e32 v209, 0, v51
	v_add_u32_e32 v210, 0, v61
	v_add_u32_e32 v1, 0xc000, v211
	v_accvgpr_write_b32 a53, v39
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	ds_read2_b64 v[2:5], v195 offset1:16
	ds_read2_b64 v[6:9], v196 offset1:16
	ds_read2_b64 v[10:13], v197 offset1:16
	ds_read2_b64 v[14:17], v198 offset1:16
	v_accvgpr_write_b32 a55, v38
	ds_read2_b64 v[38:41], v199 offset1:16
	ds_read2_b64 v[42:45], v200 offset1:16
	ds_read2_b64 v[64:67], v201 offset1:16
	ds_read2_b64 v[68:71], v202 offset1:16
	.loc	1 155 26                        ; chunk_delta_h.py:155:26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write2st64_b32 v203, v118, v110 offset0:192 offset1:208
	ds_write2st64_b32 v204, v116, v108 offset0:193 offset1:209
	ds_write2st64_b32 v205, v120, v114 offset0:194 offset1:210
	ds_write2st64_b32 v206, v122, v112 offset0:195 offset1:211
	ds_write2st64_b32 v207, v119, v111 offset0:200 offset1:216
	ds_write2st64_b32 v208, v117, v109 offset0:201 offset1:217
	ds_write2st64_b32 v209, v121, v115 offset0:202 offset1:218
	ds_write2st64_b32 v210, v123, v113 offset0:203 offset1:219
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b64 v[72:75], v1 offset1:16
	v_xor_b32_e32 v1, 16, v101
	v_add_u32_e32 v184, 0, v1
	v_accvgpr_write_b32 a57, v1
	v_add_u32_e32 v1, 0xc000, v184
	ds_read2_b64 v[76:79], v1 offset1:16
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[72:73], v[2:3], 0
	v_xor_b32_e32 v1, 32, v101
	v_add_u32_e32 v153, 0, v1
	v_accvgpr_write_b32 a58, v1
	v_add_u32_e32 v1, 0xc000, v153
	ds_read2_b64 v[80:83], v1 offset1:16
	v_xor_b32_e32 v1, 48, v101
	v_add_u32_e32 v51, 0, v1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[76:77], v[6:7], a[0:15]
	v_accvgpr_write_b32 a59, v1
	v_add_u32_e32 v1, 0xc000, v51
	ds_read2_b64 v[86:89], v1 offset1:16
	v_xor_b32_e32 v1, 64, v101
	v_accvgpr_write_b32 a60, v1
	v_add_u32_e32 v1, 0, v1
	v_add_u32_e32 v2, 0xc000, v1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[80:81], v[10:11], a[0:15]
	ds_read2_b64 v[90:93], v2 offset1:16
	v_xor_b32_e32 v2, 0x50, v101
	v_add_u32_e32 v143, 0, v2
	v_accvgpr_write_b32 a61, v2
	v_add_u32_e32 v2, 0xc000, v143
	ds_read2_b64 v[94:97], v2 offset1:16
	v_xor_b32_e32 v2, 0x60, v101
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[86:87], v[14:15], a[0:15]
	v_add_u32_e32 v147, 0, v2
	v_accvgpr_write_b32 a62, v2
	v_add_u32_e32 v2, 0xc000, v147
	ds_read2_b64 v[236:239], v2 offset1:16
	v_xor_b32_e32 v2, 0x70, v101
	v_add_u32_e32 v161, 0, v2
	v_accvgpr_write_b32 a63, v2
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[90:91], v[38:39], a[0:15]
	v_add_u32_e32 v2, 0xc000, v161
	ds_read2_b64 v[240:243], v2 offset1:16
	v_accvgpr_write_b32 a54, v62
	v_accvgpr_write_b32 a56, v63
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[94:95], v[42:43], a[0:15]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[236:237], v[64:65], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[240:241], v[68:69], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[74:75], v[4:5], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[78:79], v[8:9], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[82:83], v[12:13], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[88:89], v[16:17], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[92:93], v[40:41], a[0:15]
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mfma_f32_32x32x4_xf32 a[0:15], v[96:97], v[44:45], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[238:239], v[66:67], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[242:243], v[70:71], a[0:15]
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_46
; %bb.45:
	.loc	1 0 26 is_stmt 0                ; chunk_delta_h.py:0:26
	v_add_u32_e32 v3, 0x10040, v52
	v_mov_b32_e32 v2, 0
	v_ashrrev_i64 v[2:3], 30, v[2:3]
	v_lshl_add_u64 v[2:3], s[6:7], 0, v[2:3]
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	global_load_dwordx4 v[38:41], v[2:3], off
.LBB0_46:
	.loc	1 0 26                          ; chunk_delta_h.py:0:26
	s_or_b64 exec, exec, s[44:45]
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	s_and_saveexec_b64 s[44:45], s[0:1]
	s_cbranch_execz .LBB0_48
; %bb.47:
	v_add_u32_e32 v37, 0x10040, v37
	v_ashrrev_i64 v[2:3], 30, v[36:37]
	v_lshl_add_u64 v[2:3], s[6:7], 0, v[2:3]
	global_load_dwordx4 v[42:45], v[2:3], off
.LBB0_48:
	.loc	1 0 26                          ; chunk_delta_h.py:0:26
	s_or_b64 exec, exec, s[44:45]
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	s_and_saveexec_b64 s[0:1], s[28:29]
	s_cbranch_execz .LBB0_50
; %bb.49:
	v_add_u32_e32 v47, 0x10040, v47
	v_ashrrev_i64 v[2:3], 30, v[46:47]
	v_lshl_add_u64 v[2:3], s[6:7], 0, v[2:3]
	global_load_dwordx4 v[34:37], v[2:3], off
.LBB0_50:
	.loc	1 0 26                          ; chunk_delta_h.py:0:26
	s_or_b64 exec, exec, s[0:1]
	v_accvgpr_write_b32 a32, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	s_and_saveexec_b64 s[0:1], s[30:31]
	s_cbranch_execz .LBB0_52
; %bb.51:
	v_add_u32_e32 v141, 0x10040, v141
	v_ashrrev_i64 v[2:3], 30, v[140:141]
	v_lshl_add_u64 v[2:3], s[6:7], 0, v[2:3]
	global_load_dwordx4 v[46:49], v[2:3], off
.LBB0_52:
	.loc	1 0 26                          ; chunk_delta_h.py:0:26
	s_or_b64 exec, exec, s[0:1]
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	v_add_u32_e32 v61, 0x4000, v199
	ds_read2_b64 v[64:67], v61 offset1:16
	v_add_u32_e32 v61, 0x4000, v200
	ds_read2_b64 v[68:71], v61 offset1:16
	v_add_u32_e32 v61, 0x4000, v201
	ds_read2_b64 v[72:75], v61 offset1:16
	v_add_u32_e32 v61, 0x4000, v202
	v_add_u32_e32 v2, 0x4000, v195
	v_add_u32_e32 v6, 0x4000, v196
	v_add_u32_e32 v10, 0x4000, v197
	v_add_u32_e32 v14, 0x4000, v198
	ds_read2_b64 v[76:79], v61 offset1:16
	.loc	1 161 31 is_stmt 1              ; chunk_delta_h.py:161:31
	v_add_u32_e32 v61, 0xc000, v211
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	ds_read2_b64 v[2:5], v2 offset1:16
	ds_read2_b64 v[6:9], v6 offset1:16
	ds_read2_b64 v[10:13], v10 offset1:16
	ds_read2_b64 v[14:17], v14 offset1:16
	.loc	1 161 31                        ; chunk_delta_h.py:161:31
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write2st64_b32 v203, v132, v126 offset0:192 offset1:208
	ds_write2st64_b32 v204, v134, v124 offset0:193 offset1:209
	ds_write2st64_b32 v205, v136, v130 offset0:194 offset1:210
	ds_write2st64_b32 v206, v138, v128 offset0:195 offset1:211
	ds_write2st64_b32 v207, v133, v127 offset0:200 offset1:216
	ds_write2st64_b32 v208, v135, v125 offset0:201 offset1:217
	ds_write2st64_b32 v209, v137, v131 offset0:202 offset1:218
	ds_write2st64_b32 v210, v139, v129 offset0:203 offset1:219
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b64 v[80:83], v61 offset1:16
	v_add_u32_e32 v61, 0xc000, v184
	ds_read2_b64 v[86:89], v61 offset1:16
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[80:81], v[2:3], a[0:15]
	v_add_u32_e32 v2, 0xc000, v153
	ds_read2_b64 v[90:93], v2 offset1:16
	v_add_u32_e32 v2, 0xc000, v51
	ds_read2_b64 v[94:97], v2 offset1:16
	v_add_u32_e32 v2, 0xc000, v1
	ds_read2_b64 v[236:239], v2 offset1:16
	v_add_u32_e32 v2, 0xc000, v143
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[86:87], v[6:7], a[0:15]
	ds_read2_b64 v[240:243], v2 offset1:16
	v_add_u32_e32 v2, 0xc000, v147
	ds_read2_b64 v[244:247], v2 offset1:16
	v_add_u32_e32 v2, 0xc000, v161
	ds_read2_b64 v[248:251], v2 offset1:16
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_or_b32_e32 v84, 64, v172
	v_cmp_gt_u64_e32 vcc, s[52:53], v[84:85]
	.loc	1 161 31                        ; chunk_delta_h.py:161:31
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[90:91], v[10:11], a[0:15]
	s_and_b32 s45, s9, 0xffff
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	s_add_u32 s28, s38, 0x10000
	s_and_b64 s[30:31], s[58:59], vcc
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a35, 0
	.loc	1 161 31                        ; chunk_delta_h.py:161:31
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[94:95], v[14:15], a[0:15]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[236:237], v[64:65], a[0:15]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[240:241], v[68:69], a[0:15]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[244:245], v[72:73], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[248:249], v[76:77], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[82:83], v[4:5], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[88:89], v[8:9], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[92:93], v[12:13], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[96:97], v[16:17], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[238:239], v[66:67], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[242:243], v[70:71], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[246:247], v[74:75], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[250:251], v[78:79], a[0:15]
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	s_and_saveexec_b64 s[0:1], s[30:31]
	s_cbranch_execz .LBB0_54
; %bb.53:
	.loc	1 0 22 is_stmt 0                ; chunk_delta_h.py:0:22
	v_add_u32_e32 v3, s28, v176
	v_mov_b32_e32 v2, 0
	v_ashrrev_i64 v[2:3], 30, v[2:3]
	v_lshl_add_u64 v[2:3], s[4:5], 0, v[2:3]
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	global_load_dwordx4 a[32:35], v[2:3], off
.LBB0_54:
	.loc	1 0 22                          ; chunk_delta_h.py:0:22
	s_or_b64 exec, exec, s[0:1]
	.loc	1 154 22 is_stmt 1              ; chunk_delta_h.py:154:22
	v_or_b32_e32 v106, 64, v182
	.loc	1 0 0 is_stmt 0                 ; chunk_delta_h.py:0
	s_nop 6
	v_accvgpr_read_b32 v17, a15
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_cmp_gt_u64_e32 vcc, s[52:53], v[106:107]
	v_mov_b32_e32 v65, 0
	.loc	1 0 0                           ; chunk_delta_h.py:0
	v_accvgpr_read_b32 v16, a14
	v_accvgpr_read_b32 v15, a13
	v_accvgpr_read_b32 v14, a12
	v_accvgpr_read_b32 v13, a11
	v_accvgpr_read_b32 v12, a10
	v_accvgpr_read_b32 v11, a9
	v_accvgpr_read_b32 v10, a8
	v_accvgpr_read_b32 v9, a7
	v_accvgpr_read_b32 v8, a6
	v_accvgpr_read_b32 v7, a5
	v_accvgpr_read_b32 v6, a4
	v_accvgpr_read_b32 v5, a3
	v_accvgpr_read_b32 v4, a2
	v_accvgpr_read_b32 v3, a1
	v_accvgpr_read_b32 v2, a0
	.loc	1 177 22 is_stmt 1              ; chunk_delta_h.py:177:22
	s_and_b64 s[30:31], s[58:59], vcc
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a39, 0
	s_and_saveexec_b64 s[0:1], s[30:31]
	s_cbranch_execz .LBB0_56
; %bb.55:
	v_add_u32_e32 v63, s28, v170
	v_mov_b32_e32 v62, 0
	v_ashrrev_i64 v[62:63], 30, v[62:63]
	v_lshl_add_u64 v[62:63], s[4:5], 0, v[62:63]
	global_load_dwordx4 a[36:39], v[62:63], off
.LBB0_56:
	.loc	1 0 22 is_stmt 0                ; chunk_delta_h.py:0:22
	s_or_b64 exec, exec, s[0:1]
	v_and_b32_e32 v61, 0x660, v157
	v_and_b32_e32 v62, 0xe0, v0
	v_lshlrev_b32_e32 v63, 1, v154
	v_lshlrev_b32_e32 v64, 8, v155
	v_xor_b32_e32 v61, v61, v62
	v_lshl_add_u32 v62, v156, 11, 0
	v_add3_u32 v62, v62, v63, v64
	v_lshlrev_b32_e32 v63, 7, v0
	v_lshlrev_b32_e32 v67, 6, v142
	v_and_b32_e32 v63, 0x600, v63
	v_lshlrev_b32_e32 v64, 3, v0
	v_lshlrev_b32_e32 v66, 4, v156
	v_lshl_or_b32 v67, v151, 11, v67
	v_and_b32_e32 v64, 0xe0, v64
	v_lshlrev_b32_e32 v68, 1, v150
	v_or3_b32 v63, v63, v66, v67
	v_or3_b32 v63, v63, v68, v64
	v_and_b32_e32 v66, 64, v0
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	v_add_u32_e32 v221, v62, v61
	.loc	1 112 24 is_stmt 1              ; chunk_delta_h.py:112:24
	v_or_b32_e32 v105, 8, v104
	v_or_b32_e32 v177, 16, v104
	v_or_b32_e32 v220, 24, v104
	v_xor_b32_e32 v64, 32, v63
	v_cmp_eq_u32_e32 vcc, 0, v66
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_lshlrev_b32_e32 v66, 10, v59
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v221, v[222:225] offset:49152
	ds_write_b128 v221, v[232:235] offset:49408
	v_add_u32_e32 v222, 0, v63
	v_add_u32_e32 v82, v220, v66
	v_add_u32_e32 v83, v177, v66
	v_add_u32_e32 v86, v105, v66
	v_add_u32_e32 v87, v66, v104
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v223, 0, v64
	ds_read_b128 v[66:69], v222 offset:49152
	ds_read_b128 v[70:73], v223 offset:49152
	v_xor_b32_e32 v74, 64, v63
	v_xor_b32_e32 v75, 0x60, v63
	v_add_u32_e32 v224, 0, v74
	.loc	1 177 52 is_stmt 0              ; chunk_delta_h.py:177:52
	s_waitcnt lgkmcnt(1)
	v_pk_add_f32 v[4:5], v[68:69], v[4:5] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 183 26 is_stmt 1              ; chunk_delta_h.py:183:26
	v_add_lshl_u32 v61, s54, v87, 2
	v_bfrev_b32_e32 v68, 1
	s_and_b64 s[0:1], vcc, s[46:47]
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	v_add_u32_e32 v225, 0, v75
	ds_read_b128 v[74:77], v224 offset:49152
	ds_read_b128 v[78:81], v225 offset:49152
	.loc	1 177 52 is_stmt 0              ; chunk_delta_h.py:177:52
	v_pk_add_f32 v[2:3], v[66:67], v[2:3] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 183 26 is_stmt 1              ; chunk_delta_h.py:183:26
	v_cndmask_b32_e64 v61, v68, v61, s[0:1]
	s_mov_b32 s44, s8
	s_mov_b32 s46, s14
	s_mov_b32 s47, s15
	buffer_store_dwordx4 v[2:5], v61, s[44:47], 0 offen
	v_add_lshl_u32 v61, s54, v86, 2
	.loc	1 177 52                        ; chunk_delta_h.py:177:52
	s_waitcnt lgkmcnt(2)
	v_pk_add_f32 v[6:7], v[70:71], v[6:7] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[8:9], v[72:73], v[8:9] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 183 26                        ; chunk_delta_h.py:183:26
	v_cndmask_b32_e64 v61, v68, v61, s[0:1]
	buffer_store_dwordx4 v[6:9], v61, s[44:47], 0 offen
	v_add_lshl_u32 v61, s54, v83, 2
	.loc	1 177 52                        ; chunk_delta_h.py:177:52
	s_waitcnt lgkmcnt(1)
	v_pk_add_f32 v[10:11], v[74:75], v[10:11] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[12:13], v[76:77], v[12:13] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 183 26                        ; chunk_delta_h.py:183:26
	v_cndmask_b32_e64 v61, v68, v61, s[0:1]
	buffer_store_dwordx4 v[10:13], v61, s[44:47], 0 offen
	v_add_lshl_u32 v61, s54, v82, 2
	v_cndmask_b32_e64 v61, v68, v61, s[0:1]
	.loc	1 185 39                        ; chunk_delta_h.py:185:39
	s_min_u32 s0, s52, 0x80
	s_add_i32 s66, s57, -8
	.loc	1 188 56                        ; chunk_delta_h.py:188:56
	s_lshl_b32 s0, s0, 3
	.loc	1 188 60 is_stmt 0              ; chunk_delta_h.py:188:60
	s_add_i32 s0, s66, s0
	.loc	1 188 31                        ; chunk_delta_h.py:188:31
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	.loc	1 177 52 is_stmt 1              ; chunk_delta_h.py:177:52
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[14:15], v[78:79], v[14:15] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[16:17], v[80:81], v[16:17] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 188 31                        ; chunk_delta_h.py:188:31
	s_add_u32 s0, s10, s0
	.loc	1 183 26                        ; chunk_delta_h.py:183:26
	buffer_store_dwordx4 v[14:17], v61, s[44:47], 0 offen
	.loc	1 188 31                        ; chunk_delta_h.py:188:31
	s_addc_u32 s1, s11, s1
	global_load_dword v227, v65, s[0:1]
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_or_b32_e32 v102, 64, v59
	v_accvgpr_write_b32 a50, v157
	v_accvgpr_write_b32 a45, v154
	v_accvgpr_write_b32 a46, v155
	v_accvgpr_write_b32 a47, v156
	v_accvgpr_write_b32 a43, v150
	v_accvgpr_write_b32 a44, v151
	s_mov_b32 s56, 64
	v_cmp_gt_u64_e64 s[0:1], s[52:53], v[102:103]
	v_mov_b32_e32 v194, 0
	.loc	1 192 26                        ; chunk_delta_h.py:192:26
	s_and_saveexec_b64 s[28:29], s[0:1]
	s_cbranch_execz .LBB0_58
; %bb.57:
	.loc	1 0 26 is_stmt 0                ; chunk_delta_h.py:0:26
	v_lshl_add_u32 v61, v59, 3, s57
	v_add_u32_e32 v62, 0x200, v61
	v_ashrrev_i32_e32 v63, 31, v62
	v_lshl_add_u64 v[62:63], v[62:63], 2, s[10:11]
	.loc	1 192 26                        ; chunk_delta_h.py:192:26
	global_load_dword v194, v[62:63], off
.LBB0_58:
	.loc	1 0 26                          ; chunk_delta_h.py:0:26
	s_or_b64 exec, exec, s[28:29]
	v_mov_b32_e32 v62, 0x1010
	v_mov_b32_e32 v61, 0x808
	v_cndmask_b32_e64 v62, v62, 0, s[34:35]
	v_accvgpr_read_b32 v63, a41
	v_cndmask_b32_e64 v61, v61, 0, s[2:3]
	v_or_b32_e32 v62, v62, v63
	v_xor_b32_e32 v61, v61, v62
	v_accvgpr_read_b32 v62, a53
	v_xor_b32_e32 v61, v61, v62
	v_accvgpr_read_b32 v62, a42
	v_mov_b32_e32 v63, 0x220
	v_lshl_or_b32 v61, v62, 7, v61
	v_lshlrev_b32_e32 v62, 1, v57
	v_cndmask_b32_e64 v63, v63, 0, s[20:21]
	v_xor_b32_e32 v62, v63, v62
	v_accvgpr_read_b32 v63, a40
	v_or_b32_e32 v74, v62, v63
	.loc	1 193 53 is_stmt 1              ; chunk_delta_h.py:193:53
	v_sub_f32_e32 v63, v188, v226
	.loc	1 193 42 is_stmt 0              ; chunk_delta_h.py:193:42
	v_mul_f32_e32 v83, 0x3fb8aa3b, v63
	s_mov_b32 s67, 0xc2fc0000
	v_mov_b32_e32 v226, 0x42800000
	v_cmp_gt_f32_e64 s[28:29], s67, v83
	v_not_b32_e32 v228, 63
	s_movk_i32 s68, 0x7fff
	v_cndmask_b32_e64 v83, 0, v226, s[28:29]
	v_fmac_f32_e32 v83, 0x3fb8aa3b, v63
	v_exp_f32_e32 v83, v83
	v_cndmask_b32_e64 v86, 0, v228, s[28:29]
	.loc	1 235 21 is_stmt 1              ; chunk_delta_h.py:235:21
	v_mov_b32_e32 v229, 0x7fff
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_or_b32_e32 v64, 64, v158
	.loc	1 193 42                        ; chunk_delta_h.py:193:42
	v_ldexp_f32 v83, v83, v86
	.loc	1 193 59 is_stmt 0              ; chunk_delta_h.py:193:59
	v_cndmask_b32_e64 v83, 0, v83, s[24:25]
	.loc	1 193 24                        ; chunk_delta_h.py:193:24
	v_mul_f32_e32 v2, v83, v2
	v_mul_f32_e32 v3, v83, v3
	v_mul_f32_e32 v4, v83, v4
	v_mul_f32_e32 v5, v83, v5
	v_mul_f32_e32 v6, v83, v6
	v_mul_f32_e32 v7, v83, v7
	v_mul_f32_e32 v8, v83, v8
	v_mul_f32_e32 v9, v83, v9
	v_mul_f32_e32 v10, v83, v10
	v_mul_f32_e32 v11, v83, v11
	v_mul_f32_e32 v12, v83, v12
	v_mul_f32_e32 v13, v83, v13
	v_mul_f32_e32 v14, v83, v14
	v_mul_f32_e32 v15, v83, v15
	v_mul_f32_e32 v16, v83, v16
	v_mul_f32_e32 v17, v83, v17
	.loc	1 235 21 is_stmt 1              ; chunk_delta_h.py:235:21
	v_bfe_u32 v83, v2, 16, 1
	v_add3_u32 v83, v2, v83, s68
	v_cmp_o_f32_e64 s[24:25], v2, v2
	v_bfe_u32 v2, v3, 16, 1
	v_lshrrev_b32_e32 v83, 16, v83
	v_add3_u32 v2, v3, v2, s68
	v_cndmask_b32_e64 v83, v229, v83, s[24:25]
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[24:25], v3, v3
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_or_b32_e32 v62, 64, v159
	v_mov_b32_e32 v63, v65
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e64 v86, v229, v2, s[24:25]
	v_bfe_u32 v2, v4, 16, 1
	v_add3_u32 v2, v4, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[24:25], v4, v4
	v_xor_b32_e32 v70, 64, v61
	v_xor_b32_e32 v75, 8, v74
	v_cndmask_b32_e64 v87, v229, v2, s[24:25]
	v_bfe_u32 v2, v5, 16, 1
	v_add3_u32 v2, v5, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[24:25], v5, v5
	v_xor_b32_e32 v76, 16, v74
	v_xor_b32_e32 v77, 24, v74
	v_cndmask_b32_e64 v88, v229, v2, s[24:25]
	v_bfe_u32 v2, v6, 16, 1
	v_add3_u32 v2, v6, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[24:25], v6, v6
	v_xor_b32_e32 v78, 64, v74
	v_xor_b32_e32 v79, 0x48, v74
	v_cndmask_b32_e64 v89, v229, v2, s[24:25]
	v_bfe_u32 v2, v7, 16, 1
	v_add3_u32 v2, v7, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[24:25], v7, v7
	v_xor_b32_e32 v80, 0x50, v74
	v_xor_b32_e32 v81, 0x58, v74
	v_cndmask_b32_e64 v90, v229, v2, s[24:25]
	v_bfe_u32 v2, v8, 16, 1
	v_add3_u32 v2, v8, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[24:25], v8, v8
	v_lshl_or_b32 v82, v57, 7, v171
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_add_u32_e32 v3, 0x10000, v255
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e64 v91, v229, v2, s[24:25]
	v_bfe_u32 v2, v9, 16, 1
	v_add3_u32 v2, v9, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[24:25], v9, v9
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_cmp_gt_u64_e64 s[28:29], s[52:53], v[62:63]
	v_xor_b32_e32 v66, 16, v61
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e64 v92, v229, v2, s[24:25]
	v_bfe_u32 v2, v10, 16, 1
	v_add3_u32 v2, v10, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[24:25], v10, v10
	v_xor_b32_e32 v67, 32, v61
	v_xor_b32_e32 v69, 48, v61
	v_cndmask_b32_e64 v93, v229, v2, s[24:25]
	v_bfe_u32 v2, v11, 16, 1
	v_add3_u32 v2, v11, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[24:25], v11, v11
	v_xor_b32_e32 v71, 0x50, v61
	v_xor_b32_e32 v72, 0x60, v61
	v_cndmask_b32_e64 v94, v229, v2, s[24:25]
	v_bfe_u32 v2, v12, 16, 1
	v_add3_u32 v2, v12, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[24:25], v12, v12
	v_xor_b32_e32 v73, 0x70, v61
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_cndmask_b32_e64 v6, v68, v3, s[28:29]
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e64 v95, v229, v2, s[24:25]
	v_bfe_u32 v2, v13, 16, 1
	v_add3_u32 v2, v13, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[24:25], v13, v13
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_add_u32_e32 v231, 0, v61
	v_add_u32_e32 v235, 0, v70
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e64 v96, v229, v2, s[24:25]
	v_bfe_u32 v2, v14, 16, 1
	v_add3_u32 v2, v14, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[24:25], v14, v14
	v_add_u32_e32 v239, 0, v74
	v_add_u32_e32 v240, 0, v75
	v_cndmask_b32_e64 v97, v229, v2, s[24:25]
	v_bfe_u32 v2, v15, 16, 1
	v_add3_u32 v2, v15, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[24:25], v15, v15
	v_add_u32_e32 v241, 0, v76
	v_add_u32_e32 v242, 0, v77
	v_cndmask_b32_e64 v103, v229, v2, s[24:25]
	v_bfe_u32 v2, v16, 16, 1
	v_add3_u32 v2, v16, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[24:25], v16, v16
	v_add_u32_e32 v243, 0, v78
	v_add_u32_e32 v244, 0, v79
	v_cndmask_b32_e64 v140, v229, v2, s[24:25]
	v_bfe_u32 v2, v17, 16, 1
	v_add3_u32 v2, v17, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[24:25], v17, v17
	v_add_u32_e32 v245, 0, v80
	v_add_u32_e32 v246, 0, v81
	v_cndmask_b32_e64 v141, v229, v2, s[24:25]
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_add_u32_e32 v2, 0x10000, v149
	v_cmp_gt_u64_e64 s[24:25], s[52:53], v[64:65]
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_add_u32_e32 v247, 0, v82
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_add_u32_e32 v232, 0, v66
	v_cndmask_b32_e64 v2, v68, v2, s[24:25]
	buffer_load_dwordx4 v[2:5], v2, s[40:43], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[6:9], v6, s[40:43], 0 offen
	v_add_u32_e32 v233, 0, v67
	v_add_u32_e32 v234, 0, v69
	ds_read_b64 v[10:11], v231 offset:32768
	ds_read_b64 v[12:13], v232 offset:32768
	ds_read_b64 v[14:15], v233 offset:32768
	ds_read_b64 v[16:17], v234 offset:32768
	v_add_u32_e32 v236, 0, v71
	v_add_u32_e32 v237, 0, v72
	v_add_u32_e32 v238, 0, v73
	ds_read_b64 v[62:63], v235 offset:32768
	ds_read_b64 v[64:65], v236 offset:32768
	ds_read_b64 v[66:67], v237 offset:32768
	ds_read_b64 v[70:71], v238 offset:32768
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b16 v239, v83 offset:49152
	ds_write_b16 v239, v93 offset:51200
	ds_write_b16 v240, v86 offset:49280
	ds_write_b16 v240, v94 offset:51328
	ds_write_b16 v241, v87 offset:49408
	ds_write_b16 v241, v95 offset:51456
	ds_write_b16 v242, v88 offset:49536
	ds_write_b16 v242, v96 offset:51584
	ds_write_b16 v243, v89 offset:50176
	ds_write_b16 v243, v97 offset:52224
	ds_write_b16 v244, v90 offset:50304
	ds_write_b16 v244, v103 offset:52352
	ds_write_b16 v245, v91 offset:50432
	ds_write_b16 v245, v140 offset:52480
	ds_write_b16 v246, v92 offset:50560
	ds_write_b16 v246, v141 offset:52608
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[72:73], v247 offset:49152
	v_xor_b32_e32 v61, 16, v82
	v_add_u32_e32 v248, 0, v61
	.loc	1 241 37                        ; chunk_delta_h.py:241:37
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[72:73], v[10:11], 0
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	ds_read_b64 v[10:11], v248 offset:49152
	v_xor_b32_e32 v61, 32, v82
	v_add_u32_e32 v249, 0, v61
	v_xor_b32_e32 v61, 48, v82
	v_add_u32_e32 v250, 0, v61
	v_xor_b32_e32 v61, 64, v82
	v_add_u32_e32 v251, 0, v61
	.loc	1 241 37                        ; chunk_delta_h.py:241:37
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[10:11], v[12:13], a[0:15]
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	ds_read_b64 v[12:13], v249 offset:49152
	v_xor_b32_e32 v61, 0x50, v82
	v_add_u32_e32 v252, 0, v61
	v_xor_b32_e32 v61, 0x60, v82
	v_xor_b32_e32 v69, 0x70, v82
	v_add_u32_e32 v253, 0, v61
	v_add_u32_e32 v254, 0, v69
	.loc	1 241 37                        ; chunk_delta_h.py:241:37
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[12:13], v[14:15], a[0:15]
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	ds_read_b64 v[14:15], v250 offset:49152
	.loc	1 194 27                        ; chunk_delta_h.py:194:27
	v_mul_f32_e32 v61, 0x3fb8aa3b, v188
	v_cmp_gt_f32_e64 s[30:31], s67, v61
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	s_mov_b32 s69, 0x5040100
	s_mov_b32 s70, 0x7060302
	.loc	1 194 27                        ; chunk_delta_h.py:194:27
	v_cndmask_b32_e64 v61, 0, v226, s[30:31]
	v_fmac_f32_e32 v61, 0x3fb8aa3b, v188
	.loc	1 241 37                        ; chunk_delta_h.py:241:37
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[14:15], v[16:17], a[0:15]
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	ds_read_b64 v[16:17], v251 offset:49152
	.loc	1 194 27                        ; chunk_delta_h.py:194:27
	v_exp_f32_e32 v61, v61
	s_and_b64 s[30:31], s[30:31], exec
	s_cselect_b32 s30, 0xffffffc0, 0
	.loc	1 195 20                        ; chunk_delta_h.py:195:20
	v_mov_b32_e32 v76, v120
	.loc	1 194 27                        ; chunk_delta_h.py:194:27
	v_ldexp_f32 v96, v61, s30
	.loc	1 195 20                        ; chunk_delta_h.py:195:20
	v_mov_b32_e32 v77, v114
	.loc	1 241 37                        ; chunk_delta_h.py:241:37
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[16:17], v[62:63], a[0:15]
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	ds_read_b64 v[62:63], v252 offset:49152
	ds_read_b64 v[74:75], v253 offset:49152
	ds_read_b64 v[80:81], v254 offset:49152
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	ds_read_b64 v[78:79], v231 offset:40960
	ds_read_b64 v[82:83], v232 offset:40960
	ds_read_b64 v[86:87], v233 offset:40960
	ds_read_b64 v[88:89], v234 offset:40960
	.loc	1 195 20                        ; chunk_delta_h.py:195:20
	v_mov_b32_e32 v114, v121
	.loc	1 197 24                        ; chunk_delta_h.py:197:24
	v_mov_b32_e32 v90, v132
	v_mov_b32_e32 v91, v126
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[72:73], v[78:79], 0
	.loc	1 197 24                        ; chunk_delta_h.py:197:24
	v_mov_b32_e32 v92, v134
	v_mov_b32_e32 v93, v124
	v_mov_b32_e32 v94, v136
	v_mov_b32_e32 v95, v130
	v_mov_b32_e32 v126, v133
	v_mov_b32_e32 v124, v135
	v_mov_b32_e32 v130, v137
	.loc	1 241 37                        ; chunk_delta_h.py:241:37
	v_mfma_f32_32x32x8_bf16 a[0:15], v[62:63], v[64:65], a[0:15]
	.loc	1 195 20                        ; chunk_delta_h.py:195:20
	v_mov_b32_e32 v64, v118
	v_mov_b32_e32 v65, v110
	v_mov_b32_e32 v110, v119
	s_and_b64 s[46:47], s[0:1], s[60:61]
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_cmpk_lt_u32 s62, 0xc0
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[10:11], v[82:83], a[16:31]
	.loc	1 241 37                        ; chunk_delta_h.py:241:37
	v_mfma_f32_32x32x8_bf16 a[0:15], v[74:75], v[66:67], a[0:15]
	.loc	1 195 20                        ; chunk_delta_h.py:195:20
	v_mov_b32_e32 v66, v116
	v_mov_b32_e32 v67, v108
	v_mov_b32_e32 v108, v117
	.loc	1 197 24                        ; chunk_delta_h.py:197:24
	v_mov_b32_e32 v116, v138
	v_mov_b32_e32 v117, v128
	v_mov_b32_e32 v128, v139
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[12:13], v[86:87], a[16:31]
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	v_add_u32_e32 v12, 0x10080, v149
	v_cndmask_b32_e64 v12, v68, v12, s[24:25]
	.loc	1 241 37                        ; chunk_delta_h.py:241:37
	v_mfma_f32_32x32x8_bf16 a[0:15], v[80:81], v[70:71], a[0:15]
	.loc	1 195 20                        ; chunk_delta_h.py:195:20
	v_mov_b32_e32 v70, v122
	v_mov_b32_e32 v71, v112
	v_mov_b32_e32 v112, v123
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[14:15], v[88:89], a[16:31]
	.loc	1 241 16                        ; chunk_delta_h.py:241:16
	s_nop 5
	v_accvgpr_read_b32 v11, a8
	v_accvgpr_read_b32 v10, a0
	v_fma_f32 v64, v64, v96, v10
	v_fma_f32 v65, v65, v96, v11
	v_accvgpr_read_b32 v11, a9
	v_accvgpr_read_b32 v10, a1
	v_pk_fma_f32 v[66:67], v[66:67], v[96:97], v[10:11] op_sel_hi:[1,0,1]
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	ds_read_b64 v[10:11], v235 offset:40960
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[16:17], v[10:11], a[16:31]
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	v_add_u32_e32 v10, 0x10080, v255
	v_cndmask_b32_e64 v14, v68, v10, s[28:29]
	ds_read_b64 v[72:73], v236 offset:40960
	ds_read_b64 v[78:79], v237 offset:40960
	ds_read_b64 v[82:83], v238 offset:40960
	buffer_load_dwordx4 v[10:13], v12, s[40:43], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[14:17], v14, s[40:43], 0 offen
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	s_waitcnt vmcnt(9)
	ds_write2st64_b64 v173, v[22:23], v[18:19] offset1:8
	ds_write2st64_b64 v173, v[30:31], v[26:27] offset0:16 offset1:24
	ds_write2st64_b64 v174, v[24:25], v[20:21] offset1:8
	ds_write2st64_b64 v174, v[32:33], v[28:29] offset0:16 offset1:24
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	ds_write2st64_b64 v173, v[38:39], v[42:43] offset0:32 offset1:40
	ds_write2st64_b64 v173, v[34:35], v[46:47] offset0:48 offset1:56
	ds_write2st64_b64 v174, v[40:41], v[44:45] offset0:32 offset1:40
	ds_write2st64_b64 v174, v[36:37], v[48:49] offset0:48 offset1:56
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	s_waitcnt vmcnt(2)
	v_perm_b32 v18, v6, v2, s69
	v_perm_b32 v2, v6, v2, s70
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[62:63], v[72:73], a[16:31]
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	ds_write_b32 v175, v18 offset:32768
	ds_write_b32 v178, v2 offset:32896
	v_perm_b32 v2, v7, v3, s69
	.loc	1 241 16                        ; chunk_delta_h.py:241:16
	v_accvgpr_read_b32 v63, a12
	v_accvgpr_read_b32 v62, a4
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	ds_write_b32 v179, v2 offset:33024
	v_perm_b32 v2, v7, v3, s70
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[74:75], v[78:79], a[16:31]
	.loc	1 241 16                        ; chunk_delta_h.py:241:16
	v_accvgpr_read_b32 v69, a10
	v_accvgpr_read_b32 v68, a2
	v_fma_f32 v72, v110, v96, v62
	v_fma_f32 v73, v111, v96, v63
	v_accvgpr_read_b32 v63, a13
	v_accvgpr_read_b32 v62, a5
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	ds_write_b32 v180, v2 offset:33152
	v_perm_b32 v2, v8, v4, s69
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[80:81], v[82:83], a[16:31]
	.loc	1 241 16                        ; chunk_delta_h.py:241:16
	v_fma_f32 v68, v76, v96, v68
	v_fma_f32 v69, v77, v96, v69
	v_accvgpr_read_b32 v77, a11
	v_accvgpr_read_b32 v76, a3
	v_fma_f32 v74, v108, v96, v62
	v_fma_f32 v75, v109, v96, v63
	v_accvgpr_read_b32 v63, a14
	v_accvgpr_read_b32 v62, a6
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	ds_write_b32 v183, v2 offset:33280
	v_perm_b32 v2, v8, v4, s70
	.loc	1 241 16                        ; chunk_delta_h.py:241:16
	v_pk_fma_f32 v[70:71], v[70:71], v[96:97], v[76:77] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[76:77], v[114:115], v[96:97], v[62:63] op_sel_hi:[1,0,1]
	v_accvgpr_read_b32 v63, a15
	v_accvgpr_read_b32 v62, a7
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	ds_write_b32 v185, v2 offset:33408
	v_perm_b32 v2, v9, v5, s69
	.loc	1 241 16                        ; chunk_delta_h.py:241:16
	v_pk_fma_f32 v[78:79], v[112:113], v[96:97], v[62:63] op_sel_hi:[1,0,1]
	.loc	1 247 20                        ; chunk_delta_h.py:247:20
	v_accvgpr_read_b32 v63, a24
	v_accvgpr_read_b32 v62, a16
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	ds_write_b32 v186, v2 offset:33536
	v_perm_b32 v2, v9, v5, s70
	.loc	1 247 20                        ; chunk_delta_h.py:247:20
	v_pk_fma_f32 v[80:81], v[90:91], v[96:97], v[62:63] op_sel_hi:[1,0,1]
	v_accvgpr_read_b32 v63, a25
	v_accvgpr_read_b32 v62, a17
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	ds_write_b32 v187, v2 offset:33664
	.loc	1 247 20                        ; chunk_delta_h.py:247:20
	v_pk_fma_f32 v[82:83], v[92:93], v[96:97], v[62:63] op_sel_hi:[1,0,1]
	v_accvgpr_read_b32 v63, a26
	v_accvgpr_read_b32 v62, a18
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 247 20                        ; chunk_delta_h.py:247:20
	v_pk_fma_f32 v[86:87], v[94:95], v[96:97], v[62:63] op_sel_hi:[1,0,1]
	v_accvgpr_read_b32 v63, a27
	v_accvgpr_read_b32 v62, a19
	v_pk_fma_f32 v[88:89], v[116:117], v[96:97], v[62:63] op_sel_hi:[1,0,1]
	v_accvgpr_read_b32 v63, a28
	v_accvgpr_read_b32 v62, a20
	v_pk_fma_f32 v[90:91], v[126:127], v[96:97], v[62:63] op_sel_hi:[1,0,1]
	v_accvgpr_read_b32 v63, a29
	v_accvgpr_read_b32 v62, a21
	v_pk_fma_f32 v[92:93], v[124:125], v[96:97], v[62:63] op_sel_hi:[1,0,1]
	v_accvgpr_read_b32 v63, a30
	v_accvgpr_read_b32 v62, a22
	v_pk_fma_f32 v[94:95], v[130:131], v[96:97], v[62:63] op_sel_hi:[1,0,1]
	v_accvgpr_read_b32 v63, a31
	v_accvgpr_read_b32 v62, a23
	v_pk_fma_f32 v[96:97], v[128:129], v[96:97], v[62:63] op_sel_hi:[1,0,1]
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	s_waitcnt vmcnt(0)
	v_perm_b32 v2, v14, v10, s69
	ds_write_b32 v175, v2 offset:40960
	v_perm_b32 v2, v14, v10, s70
	ds_write_b32 v178, v2 offset:41088
	v_perm_b32 v2, v15, v11, s69
	ds_write_b32 v179, v2 offset:41216
	v_perm_b32 v2, v15, v11, s70
	ds_write_b32 v180, v2 offset:41344
	v_perm_b32 v2, v16, v12, s69
	ds_write_b32 v183, v2 offset:41472
	v_perm_b32 v2, v16, v12, s70
	ds_write_b32 v185, v2 offset:41600
	v_perm_b32 v2, v17, v13, s69
	ds_write_b32 v186, v2 offset:41728
	v_perm_b32 v2, v17, v13, s70
	ds_write_b32 v187, v2 offset:41856
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_cbranch_scc1 .LBB0_85
; %bb.59:                               ; %.peel.next
	s_lshl_b32 s0, s52, 3
	s_add_i32 s0, s0, -8
	s_mul_i32 s0, s16, s0
	s_add_i32 s17, s17, s0
	v_and_b32_e32 v2, 0x100, v60
	v_add_u32_e32 v2, s17, v2
	v_lshlrev_b32_e32 v3, 3, v57
	s_movk_i32 s0, 0x400
	v_add3_u32 v108, v2, v3, s0
	s_lshl_b32 s0, s55, 17
	s_add_i32 s0, s0, 0xfffe0000
	s_mul_i32 s0, s16, s0
	s_add_i32 s0, s0, s33
	v_add_u32_e32 v116, v148, v57
	s_add_i32 s71, s55, -2
	s_add_i32 s72, s0, 0x20000
	s_movk_i32 s56, 0x80
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v117, 1
	v_bfrev_b32_e32 v118, 1
.LBB0_60:                               ; =>This Inner Loop Header: Depth=1
	.loc	1 134 31                        ; chunk_delta_h.py:134:31
	v_and_b32_sdwa v3, v64, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_and_b32_sdwa v2, v72, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v3, v64, v3, s68
	v_add3_u32 v2, v72, v2, s68
	v_lshrrev_b32_e32 v3, 16, v3
	v_cmp_o_f32_e64 s[0:1], v64, v64
	v_lshrrev_b32_e32 v2, 16, v2
	v_and_b32_sdwa v4, v66, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e64 v3, v229, v3, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v72, v72
	v_add3_u32 v4, v66, v4, s68
	v_lshrrev_b32_e32 v4, 16, v4
	v_cndmask_b32_e64 v2, v229, v2, s[0:1]
	v_perm_b32 v2, v2, v3, s69
	v_and_b32_sdwa v3, v74, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v3, v74, v3, s68
	v_cmp_o_f32_e64 s[0:1], v66, v66
	v_lshrrev_b32_e32 v3, 16, v3
	v_and_b32_sdwa v5, v68, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e64 v4, v229, v4, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v74, v74
	v_add3_u32 v5, v68, v5, s68
	v_lshrrev_b32_e32 v5, 16, v5
	v_cndmask_b32_e64 v3, v229, v3, s[0:1]
	v_perm_b32 v3, v3, v4, s69
	v_and_b32_sdwa v4, v76, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v4, v76, v4, s68
	v_cmp_o_f32_e64 s[0:1], v68, v68
	v_lshrrev_b32_e32 v4, 16, v4
	v_and_b32_sdwa v6, v70, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e64 v5, v229, v5, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v76, v76
	v_add3_u32 v6, v70, v6, s68
	v_lshrrev_b32_e32 v6, 16, v6
	v_cndmask_b32_e64 v4, v229, v4, s[0:1]
	v_perm_b32 v4, v4, v5, s69
	v_and_b32_sdwa v5, v78, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v5, v78, v5, s68
	v_cmp_o_f32_e64 s[0:1], v70, v70
	v_lshrrev_b32_e32 v5, 16, v5
	v_and_b32_sdwa v7, v65, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e64 v6, v229, v6, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v78, v78
	v_add3_u32 v7, v65, v7, s68
	v_lshrrev_b32_e32 v7, 16, v7
	v_cndmask_b32_e64 v5, v229, v5, s[0:1]
	v_perm_b32 v5, v5, v6, s69
	v_and_b32_sdwa v6, v73, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v6, v73, v6, s68
	v_cmp_o_f32_e64 s[0:1], v65, v65
	v_lshrrev_b32_e32 v6, 16, v6
	v_and_b32_sdwa v8, v67, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e64 v7, v229, v7, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v73, v73
	v_add3_u32 v8, v67, v8, s68
	v_lshrrev_b32_e32 v8, 16, v8
	v_cndmask_b32_e64 v6, v229, v6, s[0:1]
	v_perm_b32 v6, v6, v7, s69
	v_and_b32_sdwa v7, v75, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v7, v75, v7, s68
	v_cmp_o_f32_e64 s[0:1], v67, v67
	v_lshrrev_b32_e32 v7, 16, v7
	v_and_b32_sdwa v9, v69, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e64 v8, v229, v8, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v75, v75
	v_add3_u32 v9, v69, v9, s68
	v_lshrrev_b32_e32 v9, 16, v9
	v_cndmask_b32_e64 v7, v229, v7, s[0:1]
	v_perm_b32 v7, v7, v8, s69
	v_and_b32_sdwa v8, v77, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v8, v77, v8, s68
	v_cmp_o_f32_e64 s[0:1], v69, v69
	v_lshrrev_b32_e32 v8, 16, v8
	v_and_b32_sdwa v10, v71, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e64 v9, v229, v9, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v77, v77
	v_add3_u32 v10, v71, v10, s68
	v_lshrrev_b32_e32 v10, 16, v10
	v_cndmask_b32_e64 v8, v229, v8, s[0:1]
	v_perm_b32 v8, v8, v9, s69
	v_and_b32_sdwa v9, v79, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v9, v79, v9, s68
	v_cmp_o_f32_e64 s[0:1], v71, v71
	v_lshrrev_b32_e32 v9, 16, v9
	.loc	1 134 23 is_stmt 0              ; chunk_delta_h.py:134:23
	v_or_b32_e32 v14, s72, v152
	.loc	1 134 31                        ; chunk_delta_h.py:134:31
	v_cndmask_b32_e64 v10, v229, v10, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v79, v79
	.loc	1 139 35 is_stmt 1              ; chunk_delta_h.py:139:35
	v_and_b32_sdwa v15, v89, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v15, v89, v15, s68
	.loc	1 134 31                        ; chunk_delta_h.py:134:31
	v_cndmask_b32_e64 v9, v229, v9, s[0:1]
	v_perm_b32 v9, v9, v10, s69
	.loc	1 134 23 is_stmt 0              ; chunk_delta_h.py:134:23
	v_add_u32_e32 v10, 0xc000, v160
	ds_write2_b32 v10, v2, v6 offset1:32
	v_add_u32_e32 v6, 0xc800, v160
	ds_write2_b32 v6, v3, v7 offset1:32
	v_add_u32_e32 v7, 0xc000, v189
	ds_write2_b32 v7, v4, v8 offset0:64 offset1:96
	v_add_u32_e32 v8, 0xc800, v189
	ds_write2_b32 v8, v5, v9 offset0:64 offset1:96
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_u16 v2, v191 offset:49152
	ds_read_u16 v3, v191 offset:50176
	ds_read_u16 v4, v192 offset:49664
	ds_read_u16 v5, v193 offset:49664
	ds_read_u16 v9, v193 offset:50688
	ds_read_u16 v11, v192 offset:50688
	ds_read_u16 v12, v190 offset:49152
	ds_read_u16 v13, v190 offset:50176
	s_waitcnt lgkmcnt(6)
	v_perm_b32 v3, v3, v2, s69
	s_waitcnt lgkmcnt(3)
	v_perm_b32 v5, v9, v5, s69
	v_add_lshl_u32 v9, v14, s48, 1
	s_waitcnt lgkmcnt(2)
	v_perm_b32 v4, v11, v4, s69
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v2, v13, v12, s69
	v_cndmask_b32_e64 v9, v118, v9, s[18:19]
	buffer_store_dwordx4 v[2:5], v9, s[12:15], 0 offen
	.loc	1 139 35 is_stmt 1              ; chunk_delta_h.py:139:35
	v_cmp_o_f32_e64 s[0:1], v80, v80
	v_and_b32_sdwa v9, v88, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_and_b32_sdwa v3, v80, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_and_b32_sdwa v2, v90, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v3, v80, v3, s68
	v_add3_u32 v2, v90, v2, s68
	v_lshrrev_b32_e32 v3, 16, v3
	v_lshrrev_b32_e32 v2, 16, v2
	v_cndmask_b32_e64 v3, v229, v3, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v90, v90
	v_and_b32_sdwa v4, v82, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v4, v82, v4, s68
	v_cndmask_b32_e64 v2, v229, v2, s[0:1]
	v_perm_b32 v2, v2, v3, s69
	v_and_b32_sdwa v3, v92, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v3, v92, v3, s68
	v_lshrrev_b32_e32 v4, 16, v4
	v_cmp_o_f32_e64 s[0:1], v82, v82
	v_lshrrev_b32_e32 v3, 16, v3
	v_and_b32_sdwa v5, v86, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e64 v4, v229, v4, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v92, v92
	v_add3_u32 v5, v86, v5, s68
	v_lshrrev_b32_e32 v5, 16, v5
	v_cndmask_b32_e64 v3, v229, v3, s[0:1]
	v_perm_b32 v3, v3, v4, s69
	v_and_b32_sdwa v4, v94, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v4, v94, v4, s68
	v_cmp_o_f32_e64 s[0:1], v86, v86
	v_lshrrev_b32_e32 v4, 16, v4
	v_add3_u32 v9, v88, v9, s68
	v_cndmask_b32_e64 v5, v229, v5, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v94, v94
	v_lshrrev_b32_e32 v9, 16, v9
	v_and_b32_sdwa v11, v81, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e64 v4, v229, v4, s[0:1]
	v_perm_b32 v4, v4, v5, s69
	v_and_b32_sdwa v5, v96, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v5, v96, v5, s68
	v_cmp_o_f32_e64 s[0:1], v88, v88
	v_lshrrev_b32_e32 v5, 16, v5
	v_add3_u32 v11, v81, v11, s68
	v_cndmask_b32_e64 v9, v229, v9, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v96, v96
	v_lshrrev_b32_e32 v11, 16, v11
	v_and_b32_sdwa v12, v83, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e64 v5, v229, v5, s[0:1]
	v_perm_b32 v5, v5, v9, s69
	v_and_b32_sdwa v9, v91, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v9, v91, v9, s68
	v_cmp_o_f32_e64 s[0:1], v81, v81
	v_lshrrev_b32_e32 v9, 16, v9
	v_add3_u32 v12, v83, v12, s68
	v_cndmask_b32_e64 v11, v229, v11, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v91, v91
	v_lshrrev_b32_e32 v12, 16, v12
	v_and_b32_sdwa v13, v87, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e64 v9, v229, v9, s[0:1]
	v_perm_b32 v9, v9, v11, s69
	v_and_b32_sdwa v11, v93, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v11, v93, v11, s68
	v_cmp_o_f32_e64 s[0:1], v83, v83
	v_lshrrev_b32_e32 v11, 16, v11
	v_add3_u32 v13, v87, v13, s68
	v_cndmask_b32_e64 v12, v229, v12, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v93, v93
	v_lshrrev_b32_e32 v13, 16, v13
	v_lshrrev_b32_e32 v15, 16, v15
	v_cndmask_b32_e64 v11, v229, v11, s[0:1]
	v_perm_b32 v11, v11, v12, s69
	v_and_b32_sdwa v12, v95, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v12, v95, v12, s68
	v_cmp_o_f32_e64 s[0:1], v87, v87
	v_lshrrev_b32_e32 v12, 16, v12
	.loc	1 139 27 is_stmt 0              ; chunk_delta_h.py:139:27
	s_waitcnt lgkmcnt(0)
	.loc	1 139 35                        ; chunk_delta_h.py:139:35
	v_cndmask_b32_e64 v13, v229, v13, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v95, v95
	.loc	1 139 27                        ; chunk_delta_h.py:139:27
	s_barrier
	.loc	1 152 63 is_stmt 1              ; chunk_delta_h.py:152:63
	s_ashr_i32 s57, s56, 31
	.loc	1 139 35                        ; chunk_delta_h.py:139:35
	v_cndmask_b32_e64 v12, v229, v12, s[0:1]
	v_perm_b32 v12, v12, v13, s69
	v_and_b32_sdwa v13, v97, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v13, v97, v13, s68
	v_cmp_o_f32_e64 s[0:1], v89, v89
	v_lshrrev_b32_e32 v13, 16, v13
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	s_lshl_b64 s[16:17], s[56:57], 10
	.loc	1 139 35                        ; chunk_delta_h.py:139:35
	v_cndmask_b32_e64 v15, v229, v15, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v97, v97
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_cndmask_b32_e64 v13, v229, v13, s[0:1]
	v_perm_b32 v13, v13, v15, s69
	.loc	1 139 27 is_stmt 0              ; chunk_delta_h.py:139:27
	ds_write2_b32 v10, v2, v9 offset1:32
	ds_write2_b32 v6, v3, v11 offset1:32
	ds_write2_b32 v7, v4, v12 offset0:64 offset1:96
	ds_write2_b32 v8, v5, v13 offset0:64 offset1:96
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_u16 v2, v191 offset:49152
	ds_read_u16 v3, v191 offset:50176
	ds_read_u16 v4, v192 offset:49664
	ds_read_u16 v5, v193 offset:49664
	ds_read_u16 v6, v193 offset:50688
	ds_read_u16 v7, v192 offset:50688
	ds_read_u16 v8, v190 offset:49152
	ds_read_u16 v9, v190 offset:50176
	s_waitcnt lgkmcnt(6)
	v_perm_b32 v3, v3, v2, s69
	s_waitcnt lgkmcnt(3)
	v_perm_b32 v5, v6, v5, s69
	v_add_lshl_u32 v6, v14, s49, 1
	s_waitcnt lgkmcnt(2)
	v_perm_b32 v4, v7, v4, s69
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v2, v9, v8, s69
	v_cndmask_b32_e64 v6, v118, v6, s[18:19]
	buffer_store_dwordx4 v[2:5], v6, s[12:15], 0 offen
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	.loc	1 154 22 is_stmt 1              ; chunk_delta_h.py:154:22
	v_mov_b32_e32 v3, s57
	v_or_b32_e32 v2, s56, v144
	v_cmp_gt_i64_e64 s[0:1], s[52:53], v[2:3]
	s_and_saveexec_b64 s[24:25], s[0:1]
	s_cbranch_execz .LBB0_62
; %bb.61:                               ;   in Loop: Header=BB0_60 Depth=1
	.loc	1 0 22 is_stmt 0                ; chunk_delta_h.py:0:22
	v_add_u32_e32 v111, s16, v52
	v_ashrrev_i64 v[2:3], 30, v[110:111]
	v_lshl_add_u64 v[2:3], s[6:7], 0, v[2:3]
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	global_load_dwordx4 v[18:21], v[2:3], off
.LBB0_62:                               ;   in Loop: Header=BB0_60 Depth=1
	.loc	1 0 22                          ; chunk_delta_h.py:0:22
	s_or_b64 exec, exec, s[24:25]
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_mov_b32_e32 v3, s57
	v_or_b32_e32 v2, s56, v146
	v_cmp_gt_i64_e64 s[24:25], s[52:53], v[2:3]
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	s_and_saveexec_b64 s[28:29], s[24:25]
	s_cbranch_execz .LBB0_64
; %bb.63:                               ;   in Loop: Header=BB0_60 Depth=1
	v_add_u32_e32 v111, s16, v54
	v_ashrrev_i64 v[2:3], 30, v[110:111]
	v_lshl_add_u64 v[2:3], s[6:7], 0, v[2:3]
	global_load_dwordx4 v[26:29], v[2:3], off
.LBB0_64:                               ;   in Loop: Header=BB0_60 Depth=1
	.loc	1 0 22                          ; chunk_delta_h.py:0:22
	s_or_b64 exec, exec, s[28:29]
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_mov_b32_e32 v3, s57
	v_or_b32_e32 v2, s56, v55
	v_cmp_gt_i64_e64 s[28:29], s[52:53], v[2:3]
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	s_and_saveexec_b64 s[30:31], s[28:29]
	s_cbranch_execz .LBB0_66
; %bb.65:                               ;   in Loop: Header=BB0_60 Depth=1
	v_add_u32_e32 v111, s16, v56
	v_ashrrev_i64 v[2:3], 30, v[110:111]
	v_lshl_add_u64 v[2:3], s[6:7], 0, v[2:3]
	global_load_dwordx4 v[22:25], v[2:3], off
.LBB0_66:                               ;   in Loop: Header=BB0_60 Depth=1
	.loc	1 0 22                          ; chunk_delta_h.py:0:22
	s_or_b64 exec, exec, s[30:31]
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_mov_b32_e32 v3, s57
	v_or_b32_e32 v2, s56, v181
	v_cmp_gt_i64_e64 s[30:31], s[52:53], v[2:3]
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	s_and_saveexec_b64 s[62:63], s[30:31]
	s_cbranch_execz .LBB0_68
; %bb.67:                               ;   in Loop: Header=BB0_60 Depth=1
	v_add_u32_e32 v111, s16, v58
	v_ashrrev_i64 v[2:3], 30, v[110:111]
	v_lshl_add_u64 v[2:3], s[6:7], 0, v[2:3]
	global_load_dwordx4 v[34:37], v[2:3], off
.LBB0_68:                               ;   in Loop: Header=BB0_60 Depth=1
	.loc	1 0 22                          ; chunk_delta_h.py:0:22
	s_or_b64 exec, exec, s[62:63]
	.loc	1 155 26 is_stmt 1              ; chunk_delta_h.py:155:26
	v_add_u32_e32 v3, 0xc000, v211
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	ds_read2_b64 v[8:11], v195 offset1:16
	ds_read2_b64 v[12:15], v196 offset1:16
	ds_read2_b64 v[38:41], v197 offset1:16
	ds_read2_b64 v[42:45], v198 offset1:16
	ds_read2_b64 v[46:49], v199 offset1:16
	ds_read2_b64 v[112:115], v200 offset1:16
	ds_read2_b64 v[120:123], v201 offset1:16
	ds_read2_b64 v[124:127], v202 offset1:16
	.loc	1 155 26                        ; chunk_delta_h.py:155:26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write2st64_b32 v203, v64, v65 offset0:192 offset1:208
	ds_write2st64_b32 v204, v66, v67 offset0:193 offset1:209
	ds_write2st64_b32 v205, v68, v69 offset0:194 offset1:210
	ds_write2st64_b32 v206, v70, v71 offset0:195 offset1:211
	ds_write2st64_b32 v207, v72, v73 offset0:200 offset1:216
	ds_write2st64_b32 v208, v74, v75 offset0:201 offset1:217
	ds_write2st64_b32 v209, v76, v77 offset0:202 offset1:218
	ds_write2st64_b32 v210, v78, v79 offset0:203 offset1:219
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b64 v[128:131], v3 offset1:16
	v_add_u32_e32 v2, 0xc000, v184
	ds_read2_b64 v[132:135], v2 offset1:16
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[128:129], v[8:9], 0
	v_add_u32_e32 v4, 0xc000, v153
	ds_read2_b64 v[136:139], v4 offset1:16
	v_add_u32_e32 v5, 0xc000, v51
	ds_read2_b64 v[148:151], v5 offset1:16
	v_add_u32_e32 v6, 0xc000, v1
	ds_read2_b64 v[162:165], v6 offset1:16
	v_add_u32_e32 v7, 0xc000, v143
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[132:133], v[12:13], a[0:15]
	ds_read2_b64 v[60:63], v7 offset1:16
	v_add_u32_e32 v8, 0xc000, v147
	ds_read2_b64 v[154:157], v8 offset1:16
	v_add_u32_e32 v9, 0xc000, v161
	ds_read2_b64 v[212:215], v9 offset1:16
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	s_or_b32 s17, s16, 64
	v_mov_b32_e32 v31, 0
	.loc	1 155 26                        ; chunk_delta_h.py:155:26
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[136:137], v[38:39], a[0:15]
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[148:149], v[42:43], a[0:15]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[162:163], v[46:47], a[0:15]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[60:61], v[112:113], a[0:15]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[154:155], v[120:121], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[212:213], v[124:125], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[130:131], v[10:11], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[134:135], v[14:15], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[138:139], v[40:41], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[150:151], v[44:45], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[164:165], v[48:49], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[62:63], v[114:115], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[156:157], v[122:123], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[214:215], v[126:127], a[0:15]
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	s_and_saveexec_b64 s[62:63], s[0:1]
	s_cbranch_execz .LBB0_70
; %bb.69:                               ;   in Loop: Header=BB0_60 Depth=1
	.loc	1 0 26 is_stmt 0                ; chunk_delta_h.py:0:26
	v_add_u32_e32 v111, s17, v52
	v_ashrrev_i64 v[10:11], 30, v[110:111]
	v_lshl_add_u64 v[10:11], s[6:7], 0, v[10:11]
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	global_load_dwordx4 v[30:33], v[10:11], off
.LBB0_70:                               ;   in Loop: Header=BB0_60 Depth=1
	.loc	1 0 26                          ; chunk_delta_h.py:0:26
	s_or_b64 exec, exec, s[62:63]
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	s_and_saveexec_b64 s[0:1], s[24:25]
	s_cbranch_execz .LBB0_72
; %bb.71:                               ;   in Loop: Header=BB0_60 Depth=1
	v_add_u32_e32 v111, s17, v54
	v_ashrrev_i64 v[10:11], 30, v[110:111]
	v_lshl_add_u64 v[10:11], s[6:7], 0, v[10:11]
	global_load_dwordx4 v[42:45], v[10:11], off
.LBB0_72:                               ;   in Loop: Header=BB0_60 Depth=1
	.loc	1 0 26                          ; chunk_delta_h.py:0:26
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	s_and_saveexec_b64 s[0:1], s[28:29]
	s_cbranch_execz .LBB0_74
; %bb.73:                               ;   in Loop: Header=BB0_60 Depth=1
	v_add_u32_e32 v111, s17, v56
	v_ashrrev_i64 v[10:11], 30, v[110:111]
	v_lshl_add_u64 v[10:11], s[6:7], 0, v[10:11]
	global_load_dwordx4 v[38:41], v[10:11], off
.LBB0_74:                               ;   in Loop: Header=BB0_60 Depth=1
	.loc	1 0 26                          ; chunk_delta_h.py:0:26
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	s_and_saveexec_b64 s[0:1], s[30:31]
	s_cbranch_execz .LBB0_76
; %bb.75:                               ;   in Loop: Header=BB0_60 Depth=1
	v_add_u32_e32 v111, s17, v58
	v_ashrrev_i64 v[10:11], 30, v[110:111]
	v_lshl_add_u64 v[10:11], s[6:7], 0, v[10:11]
	global_load_dwordx4 v[46:49], v[10:11], off
.LBB0_76:                               ;   in Loop: Header=BB0_60 Depth=1
	.loc	1 0 26                          ; chunk_delta_h.py:0:26
	s_or_b64 exec, exec, s[0:1]
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	v_add_u32_e32 v84, 0x4000, v198
	ds_read2_b64 v[112:115], v84 offset1:16
	v_add_u32_e32 v84, 0x4000, v199
	ds_read2_b64 v[122:125], v84 offset1:16
	v_add_u32_e32 v84, 0x4000, v200
	ds_read2_b64 v[126:129], v84 offset1:16
	v_add_u32_e32 v84, 0x4000, v201
	v_add_u32_e32 v10, 0x4000, v195
	v_add_u32_e32 v14, 0x4000, v196
	v_add_u32_e32 v60, 0x4000, v197
	ds_read2_b64 v[130:133], v84 offset1:16
	v_add_u32_e32 v84, 0x4000, v202
	ds_read2_b64 v[10:13], v10 offset1:16
	ds_read2_b64 v[14:17], v14 offset1:16
	ds_read2_b64 v[60:63], v60 offset1:16
	ds_read2_b64 v[134:137], v84 offset1:16
	.loc	1 161 31 is_stmt 1              ; chunk_delta_h.py:161:31
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write2st64_b32 v203, v80, v81 offset0:192 offset1:208
	ds_write2st64_b32 v204, v82, v83 offset0:193 offset1:209
	ds_write2st64_b32 v205, v86, v87 offset0:194 offset1:210
	ds_write2st64_b32 v206, v88, v89 offset0:195 offset1:211
	ds_write2st64_b32 v207, v90, v91 offset0:200 offset1:216
	ds_write2st64_b32 v208, v92, v93 offset0:201 offset1:217
	ds_write2st64_b32 v209, v94, v95 offset0:202 offset1:218
	ds_write2st64_b32 v210, v96, v97 offset0:203 offset1:219
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b64 v[148:151], v3 offset1:16
	ds_read2_b64 v[154:157], v2 offset1:16
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[148:149], v[10:11], a[0:15]
	ds_read2_b64 v[162:165], v4 offset1:16
	ds_read2_b64 v[2:5], v5 offset1:16
	ds_read2_b64 v[212:215], v6 offset1:16
	ds_read2_b64 v[216:219], v7 offset1:16
	ds_read2_b64 v[138:141], v8 offset1:16
	ds_read2_b64 v[6:9], v9 offset1:16
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_mov_b32_e32 v85, s57
	.loc	1 161 31                        ; chunk_delta_h.py:161:31
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[154:155], v[14:15], a[0:15]
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_or_b32_e32 v84, s56, v172
	v_cmp_gt_i64_e64 s[0:1], s[52:53], v[84:85]
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	s_add_u32 s16, s16, s38
	s_and_b64 s[24:25], s[58:59], s[0:1]
	v_mov_b32_e32 v121, 0
	.loc	1 161 31                        ; chunk_delta_h.py:161:31
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[162:163], v[60:61], a[0:15]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[2:3], v[112:113], a[0:15]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[212:213], v[122:123], a[0:15]
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[216:217], v[126:127], a[0:15]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[138:139], v[130:131], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[6:7], v[134:135], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[150:151], v[12:13], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[156:157], v[16:17], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[164:165], v[62:63], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[4:5], v[114:115], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[214:215], v[124:125], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[218:219], v[128:129], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[140:141], v[132:133], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[8:9], v[136:137], a[0:15]
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	s_and_saveexec_b64 s[0:1], s[24:25]
	s_cbranch_execz .LBB0_78
; %bb.77:                               ;   in Loop: Header=BB0_60 Depth=1
	.loc	1 0 22 is_stmt 0                ; chunk_delta_h.py:0:22
	v_add_u32_e32 v111, s16, v176
	v_ashrrev_i64 v[2:3], 30, v[110:111]
	v_lshl_add_u64 v[2:3], s[4:5], 0, v[2:3]
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	global_load_dwordx4 v[120:123], v[2:3], off
.LBB0_78:                               ;   in Loop: Header=BB0_60 Depth=1
	.loc	1 0 22                          ; chunk_delta_h.py:0:22
	s_or_b64 exec, exec, s[0:1]
	.loc	1 154 22 is_stmt 1              ; chunk_delta_h.py:154:22
	v_mov_b32_e32 v107, s57
	v_or_b32_e32 v106, s56, v182
	v_cmp_gt_i64_e64 s[0:1], s[52:53], v[106:107]
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	s_and_b64 s[24:25], s[58:59], s[0:1]
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	s_and_saveexec_b64 s[0:1], s[24:25]
	s_cbranch_execz .LBB0_80
; %bb.79:                               ;   in Loop: Header=BB0_60 Depth=1
	v_add_u32_e32 v111, s16, v170
	v_ashrrev_i64 v[2:3], 30, v[110:111]
	v_lshl_add_u64 v[2:3], s[4:5], 0, v[2:3]
	global_load_dwordx4 v[124:127], v[2:3], off
.LBB0_80:                               ;   in Loop: Header=BB0_60 Depth=1
	.loc	1 0 22 is_stmt 0                ; chunk_delta_h.py:0:22
	s_or_b64 exec, exec, s[0:1]
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v221, a[32:35] offset:49152
	ds_write_b128 v221, a[36:39] offset:49408
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[60:63], v222 offset:49152
	.loc	1 0 0                           ; chunk_delta_h.py:0
	v_accvgpr_read_b32 v17, a15
	.loc	1 154 22 is_stmt 1              ; chunk_delta_h.py:154:22
	v_lshlrev_b32_e32 v102, 10, v102
	.loc	1 0 0 is_stmt 0                 ; chunk_delta_h.py:0
	v_accvgpr_read_b32 v3, a1
	v_accvgpr_read_b32 v2, a0
	.loc	1 177 22 is_stmt 1              ; chunk_delta_h.py:177:22
	v_add_u32_e32 v103, v220, v102
	v_add_u32_e32 v111, v177, v102
	v_add_u32_e32 v119, v105, v102
	v_add_u32_e32 v102, v102, v104
	.loc	1 0 0 is_stmt 0                 ; chunk_delta_h.py:0
	v_accvgpr_read_b32 v5, a3
	v_accvgpr_read_b32 v4, a2
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	ds_read_b128 v[112:115], v223 offset:49152
	ds_read_b128 v[128:131], v224 offset:49152
	ds_read_b128 v[132:135], v225 offset:49152
	.loc	1 177 52                        ; chunk_delta_h.py:177:52
	s_waitcnt lgkmcnt(3)
	v_pk_add_f32 v[2:3], v[60:61], v[2:3] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 183 26 is_stmt 1              ; chunk_delta_h.py:183:26
	v_add_lshl_u32 v60, s54, v102, 2
	s_and_b64 s[0:1], vcc, s[46:47]
	.loc	1 177 52                        ; chunk_delta_h.py:177:52
	v_pk_add_f32 v[4:5], v[62:63], v[4:5] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 183 26                        ; chunk_delta_h.py:183:26
	v_cndmask_b32_e64 v60, v118, v60, s[0:1]
	s_mov_b32 s44, s8
	s_mov_b32 s46, s14
	s_mov_b32 s47, s15
	.loc	1 0 0 is_stmt 0                 ; chunk_delta_h.py:0
	v_accvgpr_read_b32 v9, a7
	v_accvgpr_read_b32 v8, a6
	v_accvgpr_read_b32 v7, a5
	v_accvgpr_read_b32 v6, a4
	.loc	1 183 26                        ; chunk_delta_h.py:183:26
	buffer_store_dwordx4 v[2:5], v60, s[44:47], 0 offen
	v_add_lshl_u32 v60, s54, v119, 2
	.loc	1 177 52 is_stmt 1              ; chunk_delta_h.py:177:52
	s_waitcnt lgkmcnt(2)
	v_pk_add_f32 v[6:7], v[112:113], v[6:7] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[8:9], v[114:115], v[8:9] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 183 26                        ; chunk_delta_h.py:183:26
	v_cndmask_b32_e64 v60, v118, v60, s[0:1]
	.loc	1 0 0 is_stmt 0                 ; chunk_delta_h.py:0
	v_accvgpr_read_b32 v13, a11
	v_accvgpr_read_b32 v12, a10
	v_accvgpr_read_b32 v11, a9
	v_accvgpr_read_b32 v10, a8
	.loc	1 183 26                        ; chunk_delta_h.py:183:26
	buffer_store_dwordx4 v[6:9], v60, s[44:47], 0 offen
	v_add_lshl_u32 v60, s54, v111, 2
	.loc	1 177 52 is_stmt 1              ; chunk_delta_h.py:177:52
	s_waitcnt lgkmcnt(1)
	v_pk_add_f32 v[10:11], v[128:129], v[10:11] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[12:13], v[130:131], v[12:13] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 183 26                        ; chunk_delta_h.py:183:26
	v_cndmask_b32_e64 v60, v118, v60, s[0:1]
	.loc	1 0 0 is_stmt 0                 ; chunk_delta_h.py:0
	v_accvgpr_read_b32 v16, a14
	v_accvgpr_read_b32 v15, a13
	v_accvgpr_read_b32 v14, a12
	.loc	1 183 26                        ; chunk_delta_h.py:183:26
	buffer_store_dwordx4 v[10:13], v60, s[44:47], 0 offen
	v_add_lshl_u32 v60, s54, v103, 2
	.loc	1 177 52 is_stmt 1              ; chunk_delta_h.py:177:52
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[14:15], v[132:133], v[14:15] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[16:17], v[134:135], v[16:17] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 183 26                        ; chunk_delta_h.py:183:26
	v_cndmask_b32_e64 v60, v118, v60, s[0:1]
	buffer_store_dwordx4 v[14:17], v60, s[44:47], 0 offen
	.loc	1 185 39                        ; chunk_delta_h.py:185:39
	s_add_i32 s44, s56, 64
	s_min_i32 s0, s44, s52
	.loc	1 188 56                        ; chunk_delta_h.py:188:56
	s_lshl_b32 s0, s0, 3
	.loc	1 188 60 is_stmt 0              ; chunk_delta_h.py:188:60
	s_add_i32 s0, s66, s0
	.loc	1 188 31                        ; chunk_delta_h.py:188:31
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s10, s0
	s_addc_u32 s1, s11, s1
	global_load_dword v111, v110, s[0:1]
	.loc	1 154 22 is_stmt 1              ; chunk_delta_h.py:154:22
	v_mov_b32_e32 v103, s57
	v_or_b32_e32 v102, s56, v59
	v_cmp_gt_i64_e64 s[0:1], s[52:53], v[102:103]
	.loc	1 192 26                        ; chunk_delta_h.py:192:26
	s_and_saveexec_b64 s[16:17], s[0:1]
	s_cbranch_execz .LBB0_82
; %bb.81:                               ;   in Loop: Header=BB0_60 Depth=1
	.loc	1 0 26 is_stmt 0                ; chunk_delta_h.py:0:26
	v_ashrrev_i32_e32 v109, 31, v108
	v_lshl_add_u64 v[60:61], v[108:109], 2, s[10:11]
	.loc	1 192 26                        ; chunk_delta_h.py:192:26
	global_load_dword v109, v[60:61], off
.LBB0_82:                               ;   in Loop: Header=BB0_60 Depth=1
	.loc	1 0 26                          ; chunk_delta_h.py:0:26
	s_or_b64 exec, exec, s[16:17]
	.loc	1 193 53 is_stmt 1              ; chunk_delta_h.py:193:53
	v_sub_f32_e32 v62, v227, v194
	.loc	1 193 42 is_stmt 0              ; chunk_delta_h.py:193:42
	v_mul_f32_e32 v63, 0x3fb8aa3b, v62
	v_cmp_gt_f32_e64 s[16:17], s67, v63
	.loc	1 187 50 is_stmt 1              ; chunk_delta_h.py:187:50
	v_add_u32_e32 v60, s56, v116
	v_subrev_u32_e32 v103, 64, v60
	.loc	1 193 42                        ; chunk_delta_h.py:193:42
	v_cndmask_b32_e64 v63, 0, v226, s[16:17]
	v_fmac_f32_e32 v63, 0x3fb8aa3b, v62
	v_exp_f32_e32 v112, v63
	v_cndmask_b32_e64 v113, 0, v228, s[16:17]
	.loc	1 187 50                        ; chunk_delta_h.py:187:50
	v_cmp_gt_i32_e64 s[16:17], s52, v103
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_mov_b32_e32 v61, s57
	.loc	1 193 42                        ; chunk_delta_h.py:193:42
	v_ldexp_f32 v112, v112, v113
	.loc	1 193 59 is_stmt 0              ; chunk_delta_h.py:193:59
	v_cndmask_b32_e64 v103, 0, v112, s[16:17]
	.loc	1 193 24                        ; chunk_delta_h.py:193:24
	v_mul_f32_e32 v2, v103, v2
	v_mul_f32_e32 v3, v103, v3
	v_mul_f32_e32 v4, v103, v4
	v_mul_f32_e32 v5, v103, v5
	v_mul_f32_e32 v6, v103, v6
	v_mul_f32_e32 v7, v103, v7
	v_mul_f32_e32 v8, v103, v8
	v_mul_f32_e32 v9, v103, v9
	v_mul_f32_e32 v10, v103, v10
	v_mul_f32_e32 v11, v103, v11
	v_mul_f32_e32 v12, v103, v12
	v_mul_f32_e32 v13, v103, v13
	v_mul_f32_e32 v14, v103, v14
	v_mul_f32_e32 v15, v103, v15
	v_mul_f32_e32 v16, v103, v16
	v_mul_f32_e32 v17, v103, v17
	.loc	1 194 27 is_stmt 1              ; chunk_delta_h.py:194:27
	v_mul_f32_e32 v103, 0x3fb8aa3b, v227
	v_cmp_gt_f32_e64 s[30:31], s67, v103
	s_and_b64 s[16:17], s[30:31], exec
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_bfe_u32 v103, v2, 16, 1
	v_add3_u32 v103, v2, v103, s68
	v_cmp_o_f32_e64 s[16:17], v2, v2
	v_bfe_u32 v2, v3, 16, 1
	v_lshrrev_b32_e32 v103, 16, v103
	v_add3_u32 v2, v3, v2, s68
	v_cndmask_b32_e64 v103, v229, v103, s[16:17]
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[16:17], v3, v3
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_or_b32_e32 v60, s56, v158
	v_mov_b32_e32 v63, s57
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e64 v112, v229, v2, s[16:17]
	v_bfe_u32 v2, v4, 16, 1
	v_add3_u32 v2, v4, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[16:17], v4, v4
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_or_b32_e32 v62, s56, v159
	.loc	1 194 27                        ; chunk_delta_h.py:194:27
	s_cselect_b32 s46, 0xffffffc0, 0
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e64 v113, v229, v2, s[16:17]
	v_bfe_u32 v2, v5, 16, 1
	v_add3_u32 v2, v5, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[16:17], v5, v5
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_cmp_gt_i64_e64 s[24:25], s[52:53], v[60:61]
	v_cmp_gt_i64_e64 s[28:29], s[52:53], v[62:63]
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e64 v114, v229, v2, s[16:17]
	v_bfe_u32 v2, v6, 16, 1
	v_add3_u32 v2, v6, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[16:17], v6, v6
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_add_i32 s71, s71, -1
	s_add_i32 s72, s72, 0x20000
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e64 v115, v229, v2, s[16:17]
	v_bfe_u32 v2, v7, 16, 1
	v_add3_u32 v2, v7, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[16:17], v7, v7
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	v_add_u32_e32 v108, 0x200, v108
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	s_nop 0
	v_cndmask_b32_e64 v119, v229, v2, s[16:17]
	v_bfe_u32 v2, v8, 16, 1
	v_add3_u32 v2, v8, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[16:17], v8, v8
	s_nop 1
	v_cndmask_b32_e64 v132, v229, v2, s[16:17]
	v_bfe_u32 v2, v9, 16, 1
	v_add3_u32 v2, v9, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[16:17], v9, v9
	s_nop 1
	v_cndmask_b32_e64 v133, v229, v2, s[16:17]
	v_bfe_u32 v2, v10, 16, 1
	v_add3_u32 v2, v10, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[16:17], v10, v10
	s_nop 1
	v_cndmask_b32_e64 v134, v229, v2, s[16:17]
	v_bfe_u32 v2, v11, 16, 1
	v_add3_u32 v2, v11, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[16:17], v11, v11
	s_nop 1
	v_cndmask_b32_e64 v135, v229, v2, s[16:17]
	v_bfe_u32 v2, v12, 16, 1
	v_add3_u32 v2, v12, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[16:17], v12, v12
	s_nop 1
	v_cndmask_b32_e64 v136, v229, v2, s[16:17]
	v_bfe_u32 v2, v13, 16, 1
	v_add3_u32 v2, v13, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[16:17], v13, v13
	s_nop 1
	v_cndmask_b32_e64 v137, v229, v2, s[16:17]
	v_bfe_u32 v2, v14, 16, 1
	v_add3_u32 v2, v14, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[16:17], v14, v14
	s_nop 1
	v_cndmask_b32_e64 v138, v229, v2, s[16:17]
	v_bfe_u32 v2, v15, 16, 1
	v_add3_u32 v2, v15, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[16:17], v15, v15
	s_nop 1
	v_cndmask_b32_e64 v139, v229, v2, s[16:17]
	v_bfe_u32 v2, v16, 16, 1
	v_add3_u32 v2, v16, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[16:17], v16, v16
	s_nop 1
	v_cndmask_b32_e64 v140, v229, v2, s[16:17]
	v_bfe_u32 v2, v17, 16, 1
	v_add3_u32 v2, v17, v2, s68
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[16:17], v17, v17
	s_nop 1
	v_cndmask_b32_e64 v141, v229, v2, s[16:17]
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	s_lshl_b32 s16, s56, 9
	v_add_lshl_u32 v2, s16, v98, 1
	v_add_lshl_u32 v3, s16, v100, 1
	v_cndmask_b32_e64 v2, v118, v2, s[24:25]
	v_cndmask_b32_e64 v6, v118, v3, s[28:29]
	buffer_load_dwordx4 v[2:5], v2, s[40:43], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[6:9], v6, s[40:43], 0 offen
	ds_read_b64 v[10:11], v231 offset:32768
	ds_read_b64 v[12:13], v232 offset:32768
	ds_read_b64 v[14:15], v233 offset:32768
	ds_read_b64 v[16:17], v234 offset:32768
	ds_read_b64 v[60:61], v235 offset:32768
	ds_read_b64 v[62:63], v236 offset:32768
	ds_read_b64 v[128:129], v237 offset:32768
	ds_read_b64 v[130:131], v238 offset:32768
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b16 v239, v103 offset:49152
	ds_write_b16 v239, v134 offset:51200
	ds_write_b16 v240, v112 offset:49280
	ds_write_b16 v240, v135 offset:51328
	ds_write_b16 v241, v113 offset:49408
	ds_write_b16 v241, v136 offset:51456
	ds_write_b16 v242, v114 offset:49536
	ds_write_b16 v242, v137 offset:51584
	ds_write_b16 v243, v115 offset:50176
	ds_write_b16 v243, v138 offset:52224
	ds_write_b16 v244, v119 offset:50304
	ds_write_b16 v244, v139 offset:52352
	ds_write_b16 v245, v132 offset:50432
	ds_write_b16 v245, v140 offset:52480
	ds_write_b16 v246, v133 offset:50560
	ds_write_b16 v246, v141 offset:52608
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[132:133], v247 offset:49152
	ds_read_b64 v[134:135], v248 offset:49152
	ds_read_b64 v[114:115], v249 offset:49152
	ds_read_b64 v[112:113], v250 offset:49152
	.loc	1 241 37                        ; chunk_delta_h.py:241:37
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[132:133], v[10:11], 0
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	ds_read_b64 v[10:11], v254 offset:49152
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	s_or_b32 s16, s16, 64
	.loc	1 241 37                        ; chunk_delta_h.py:241:37
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[134:135], v[12:13], a[16:31]
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	ds_read_b64 v[12:13], v253 offset:49152
	.loc	1 241 37                        ; chunk_delta_h.py:241:37
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[114:115], v[14:15], a[16:31]
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	ds_read_b64 v[14:15], v252 offset:49152
	.loc	1 241 37                        ; chunk_delta_h.py:241:37
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[112:113], v[16:17], a[16:31]
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	ds_read_b64 v[16:17], v251 offset:49152
	.loc	1 241 37                        ; chunk_delta_h.py:241:37
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[16:17], v[60:61], a[16:31]
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	ds_read_b64 v[60:61], v231 offset:40960
	.loc	1 241 37                        ; chunk_delta_h.py:241:37
	v_mfma_f32_32x32x8_bf16 a[16:31], v[14:15], v[62:63], a[16:31]
	.loc	1 194 27                        ; chunk_delta_h.py:194:27
	v_cndmask_b32_e64 v62, 0, v226, s[30:31]
	v_fmac_f32_e32 v62, 0x3fb8aa3b, v227
	v_exp_f32_e32 v103, v62
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	ds_read_b64 v[62:63], v234 offset:40960
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[132:133], v[60:61], 0
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	ds_read_b64 v[60:61], v232 offset:40960
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[134:135], v[60:61], a[0:15]
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	ds_read_b64 v[60:61], v233 offset:40960
	.loc	1 241 37                        ; chunk_delta_h.py:241:37
	v_mfma_f32_32x32x8_bf16 a[16:31], v[12:13], v[128:129], a[16:31]
	.loc	1 194 27                        ; chunk_delta_h.py:194:27
	v_ldexp_f32 v128, v103, s46
	s_and_b64 s[46:47], s[0:1], s[60:61]
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_cmp_lg_u32 s71, 0
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[114:115], v[60:61], a[0:15]
	.loc	1 241 37                        ; chunk_delta_h.py:241:37
	v_mfma_f32_32x32x8_bf16 a[16:31], v[10:11], v[130:131], a[16:31]
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	v_mfma_f32_32x32x8_bf16 a[0:15], v[112:113], v[62:63], a[0:15]
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	v_add_lshl_u32 v63, s16, v100, 1
	v_cndmask_b32_e64 v103, v118, v63, s[28:29]
	.loc	1 241 16                        ; chunk_delta_h.py:241:16
	s_nop 7
	v_accvgpr_read_b32 v61, a24
	v_accvgpr_read_b32 v60, a16
	v_fma_f32 v64, v64, v128, v60
	v_fma_f32 v65, v65, v128, v61
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	v_add_lshl_u32 v60, s16, v98, 1
	v_cndmask_b32_e64 v62, v118, v60, s[24:25]
	ds_read_b64 v[60:61], v235 offset:40960
	ds_read_b64 v[132:133], v236 offset:40960
	ds_read_b64 v[134:135], v237 offset:40960
	ds_read_b64 v[136:137], v238 offset:40960
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[16:17], v[60:61], a[0:15]
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	buffer_load_dwordx4 v[60:63], v62, s[40:43], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[112:115], v103, s[40:43], 0 offen
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	s_waitcnt vmcnt(9)
	ds_write2st64_b64 v173, v[18:19], v[26:27] offset1:8
	ds_write2st64_b64 v173, v[22:23], v[34:35] offset0:16 offset1:24
	ds_write2st64_b64 v174, v[20:21], v[28:29] offset1:8
	ds_write2st64_b64 v174, v[24:25], v[36:37] offset0:16 offset1:24
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	ds_write2st64_b64 v173, v[30:31], v[42:43] offset0:32 offset1:40
	ds_write2st64_b64 v173, v[38:39], v[46:47] offset0:48 offset1:56
	ds_write2st64_b64 v174, v[32:33], v[44:45] offset0:32 offset1:40
	ds_write2st64_b64 v174, v[40:41], v[48:49] offset0:48 offset1:56
	.loc	1 241 16                        ; chunk_delta_h.py:241:16
	v_accvgpr_read_b32 v17, a26
	v_accvgpr_read_b32 v16, a18
	v_accvgpr_read_b32 v131, a25
	v_accvgpr_read_b32 v130, a17
	v_pk_fma_f32 v[68:69], v[68:69], v[128:129], v[16:17] op_sel_hi:[1,0,1]
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[14:15], v[132:133], a[0:15]
	.loc	1 241 16                        ; chunk_delta_h.py:241:16
	v_accvgpr_read_b32 v15, a28
	v_accvgpr_read_b32 v14, a20
	v_accvgpr_read_b32 v17, a27
	v_accvgpr_read_b32 v16, a19
	v_fma_f32 v72, v72, v128, v14
	v_fma_f32 v73, v73, v128, v15
	v_accvgpr_read_b32 v15, a29
	v_accvgpr_read_b32 v14, a21
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[12:13], v[134:135], a[0:15]
	.loc	1 241 16                        ; chunk_delta_h.py:241:16
	v_accvgpr_read_b32 v13, a30
	v_accvgpr_read_b32 v12, a22
	v_fma_f32 v76, v76, v128, v12
	v_fma_f32 v77, v77, v128, v13
	v_accvgpr_read_b32 v13, a31
	v_accvgpr_read_b32 v12, a23
	v_pk_fma_f32 v[66:67], v[66:67], v[128:129], v[130:131] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[70:71], v[70:71], v[128:129], v[16:17] op_sel_hi:[1,0,1]
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[10:11], v[136:137], a[0:15]
	.loc	1 241 16                        ; chunk_delta_h.py:241:16
	v_fma_f32 v74, v74, v128, v14
	v_fma_f32 v75, v75, v128, v15
	v_fma_f32 v78, v78, v128, v12
	v_fma_f32 v79, v79, v128, v13
	.loc	1 247 20                        ; chunk_delta_h.py:247:20
	s_nop 6
	v_accvgpr_read_b32 v11, a8
	v_accvgpr_read_b32 v10, a0
	v_pk_fma_f32 v[80:81], v[80:81], v[128:129], v[10:11] op_sel_hi:[1,0,1]
	v_accvgpr_read_b32 v11, a9
	v_accvgpr_read_b32 v10, a1
	v_pk_fma_f32 v[82:83], v[82:83], v[128:129], v[10:11] op_sel_hi:[1,0,1]
	v_accvgpr_read_b32 v11, a10
	v_accvgpr_read_b32 v10, a2
	v_pk_fma_f32 v[86:87], v[86:87], v[128:129], v[10:11] op_sel_hi:[1,0,1]
	v_accvgpr_read_b32 v11, a11
	v_accvgpr_read_b32 v10, a3
	v_pk_fma_f32 v[88:89], v[88:89], v[128:129], v[10:11] op_sel_hi:[1,0,1]
	v_accvgpr_read_b32 v11, a12
	v_accvgpr_read_b32 v10, a4
	v_pk_fma_f32 v[90:91], v[90:91], v[128:129], v[10:11] op_sel_hi:[1,0,1]
	v_accvgpr_read_b32 v11, a13
	v_accvgpr_read_b32 v10, a5
	v_pk_fma_f32 v[92:93], v[92:93], v[128:129], v[10:11] op_sel_hi:[1,0,1]
	v_accvgpr_read_b32 v11, a14
	v_accvgpr_read_b32 v10, a6
	v_pk_fma_f32 v[94:95], v[94:95], v[128:129], v[10:11] op_sel_hi:[1,0,1]
	v_accvgpr_read_b32 v11, a15
	v_accvgpr_read_b32 v10, a7
	v_pk_fma_f32 v[96:97], v[96:97], v[128:129], v[10:11] op_sel_hi:[1,0,1]
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	s_waitcnt vmcnt(2)
	v_perm_b32 v10, v6, v2, s69
	v_perm_b32 v2, v6, v2, s70
	ds_write_b32 v175, v10 offset:32768
	ds_write_b32 v178, v2 offset:32896
	v_perm_b32 v2, v7, v3, s69
	ds_write_b32 v179, v2 offset:33024
	v_perm_b32 v2, v7, v3, s70
	ds_write_b32 v180, v2 offset:33152
	v_perm_b32 v2, v8, v4, s69
	ds_write_b32 v183, v2 offset:33280
	v_perm_b32 v2, v8, v4, s70
	ds_write_b32 v185, v2 offset:33408
	v_perm_b32 v2, v9, v5, s69
	ds_write_b32 v186, v2 offset:33536
	v_perm_b32 v2, v9, v5, s70
	ds_write_b32 v187, v2 offset:33664
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	v_perm_b32 v2, v112, v60, s69
	ds_write_b32 v175, v2 offset:40960
	v_perm_b32 v2, v112, v60, s70
	ds_write_b32 v178, v2 offset:41088
	v_perm_b32 v2, v113, v61, s69
	ds_write_b32 v179, v2 offset:41216
	v_perm_b32 v2, v113, v61, s70
	ds_write_b32 v180, v2 offset:41344
	v_perm_b32 v2, v114, v62, s69
	ds_write_b32 v183, v2 offset:41472
	v_perm_b32 v2, v114, v62, s70
	ds_write_b32 v185, v2 offset:41600
	v_perm_b32 v2, v115, v63, s69
	ds_write_b32 v186, v2 offset:41728
	v_perm_b32 v2, v115, v63, s70
	ds_write_b32 v187, v2 offset:41856
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_cbranch_scc0 .LBB0_84
; %bb.83:                               ;   in Loop: Header=BB0_60 Depth=1
	.loc	1 0 21 is_stmt 0                ; chunk_delta_h.py:0:21
	s_mov_b32 s56, s44
	v_mov_b32_e32 v194, v109
	v_mov_b32_e32 v227, v111
	v_accvgpr_write_b32 a39, v127
	v_accvgpr_write_b32 a38, v126
	v_accvgpr_write_b32 a37, v125
	v_accvgpr_write_b32 a36, v124
	v_accvgpr_write_b32 a35, v123
	v_accvgpr_write_b32 a34, v122
	v_accvgpr_write_b32 a33, v121
	v_accvgpr_write_b32 a32, v120
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_branch .LBB0_60
.LBB0_84:                               ; %Flow
	.loc	1 0 21                          ; chunk_delta_h.py:0:21
	v_accvgpr_write_b32 a35, v123
	v_accvgpr_write_b32 a34, v122
	v_accvgpr_write_b32 a33, v121
	v_accvgpr_write_b32 a32, v120
	v_mov_b32_e32 v227, v111
	v_accvgpr_write_b32 a39, v127
	v_accvgpr_write_b32 a38, v126
	v_accvgpr_write_b32 a37, v125
	v_accvgpr_write_b32 a36, v124
	v_mov_b32_e32 v194, v109
.LBB0_85:                               ; %._crit_edge
	v_accvgpr_read_b32 v1, a55
	v_or_b32_e32 v10, 0x80, v1
	v_accvgpr_read_b32 v1, a54
	v_or_b32_e32 v14, 0x80, v1
	v_accvgpr_read_b32 v1, a56
	v_lshlrev_b64 v[34:35], 10, v[84:85]
	v_xor_b32_e32 v9, 0x108, v99
	v_or_b32_e32 v2, 0x80, v166
	v_or_b32_e32 v3, 0x80, v169
	v_or_b32_e32 v6, 0x80, v230
	v_or_b32_e32 v7, 0x80, v167
	v_or_b32_e32 v11, 0x80, v53
	v_or_b32_e32 v15, 0x80, v1
	v_xor_b32_e32 v40, 0x108, v168
	v_xor_b32_e32 v41, 0x210, v168
	v_xor_b32_e32 v42, 0x318, v168
	v_xor_b32_e32 v43, 0x840, v168
	v_xor_b32_e32 v44, 0x948, v168
	v_xor_b32_e32 v45, 0xa50, v168
	v_xor_b32_e32 v48, 0xb58, v168
	v_lshlrev_b64 v[38:39], 10, v[106:107]
	v_mov_b32_e32 v188, v227
	v_accvgpr_read_b32 v235, a39
	v_accvgpr_read_b32 v234, a38
	v_accvgpr_read_b32 v233, a37
	v_accvgpr_read_b32 v232, a36
	v_accvgpr_read_b32 v225, a35
	v_accvgpr_read_b32 v224, a34
	v_accvgpr_read_b32 v223, a33
	v_accvgpr_read_b32 v222, a32
	s_mov_b64 s[28:29], s[12:13]
	v_accvgpr_read_b32 v35, a53
	v_accvgpr_read_b32 v150, a43
	v_accvgpr_read_b32 v151, a44
	v_accvgpr_read_b32 v154, a45
	v_accvgpr_read_b32 v155, a46
	v_accvgpr_read_b32 v156, a47
	v_accvgpr_read_b32 v157, a50
.LBB0_86:
	.loc	1 134 31 is_stmt 1              ; chunk_delta_h.py:134:31
	v_mov_b32_e32 v1, 1
	v_and_b32_sdwa v5, v64, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	s_movk_i32 s5, 0x7fff
	v_and_b32_sdwa v4, v72, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v5, v64, v5, s5
	v_add3_u32 v4, v72, v4, s5
	v_lshrrev_b32_e32 v5, 16, v5
	v_mov_b32_e32 v8, 0x7fff
	v_cmp_o_f32_e32 vcc, v64, v64
	v_lshrrev_b32_e32 v4, 16, v4
	s_mov_b32 s4, 0x5040100
	v_cndmask_b32_e32 v5, v8, v5, vcc
	v_cmp_o_f32_e32 vcc, v72, v72
	v_and_b32_sdwa v12, v66, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v12, v66, v12, s5
	v_cndmask_b32_e32 v4, v8, v4, vcc
	v_perm_b32 v4, v4, v5, s4
	v_and_b32_sdwa v5, v74, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v5, v74, v5, s5
	v_lshrrev_b32_e32 v12, 16, v12
	v_cmp_o_f32_e32 vcc, v66, v66
	v_lshrrev_b32_e32 v5, 16, v5
	v_and_b32_sdwa v13, v68, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e32 v12, v8, v12, vcc
	v_cmp_o_f32_e32 vcc, v74, v74
	v_add3_u32 v13, v68, v13, s5
	v_lshrrev_b32_e32 v13, 16, v13
	v_cndmask_b32_e32 v5, v8, v5, vcc
	v_perm_b32 v5, v5, v12, s4
	v_and_b32_sdwa v12, v76, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v12, v76, v12, s5
	v_cmp_o_f32_e32 vcc, v68, v68
	v_lshrrev_b32_e32 v12, 16, v12
	v_and_b32_sdwa v16, v70, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e32 v13, v8, v13, vcc
	v_cmp_o_f32_e32 vcc, v76, v76
	v_add3_u32 v16, v70, v16, s5
	v_lshrrev_b32_e32 v16, 16, v16
	v_cndmask_b32_e32 v12, v8, v12, vcc
	v_perm_b32 v12, v12, v13, s4
	v_and_b32_sdwa v13, v78, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v13, v78, v13, s5
	v_cmp_o_f32_e32 vcc, v70, v70
	v_lshrrev_b32_e32 v13, 16, v13
	v_and_b32_sdwa v17, v65, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e32 v16, v8, v16, vcc
	v_cmp_o_f32_e32 vcc, v78, v78
	v_add3_u32 v17, v65, v17, s5
	v_lshrrev_b32_e32 v17, 16, v17
	v_cndmask_b32_e32 v13, v8, v13, vcc
	v_perm_b32 v13, v13, v16, s4
	v_and_b32_sdwa v16, v73, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v16, v73, v16, s5
	v_cmp_o_f32_e32 vcc, v65, v65
	v_lshrrev_b32_e32 v16, 16, v16
	v_and_b32_sdwa v18, v67, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e32 v17, v8, v17, vcc
	v_cmp_o_f32_e32 vcc, v73, v73
	v_add3_u32 v18, v67, v18, s5
	v_lshrrev_b32_e32 v18, 16, v18
	v_cndmask_b32_e32 v16, v8, v16, vcc
	v_perm_b32 v16, v16, v17, s4
	v_and_b32_sdwa v17, v75, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v17, v75, v17, s5
	v_cmp_o_f32_e32 vcc, v67, v67
	v_lshrrev_b32_e32 v17, 16, v17
	v_and_b32_sdwa v19, v69, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e32 v18, v8, v18, vcc
	v_cmp_o_f32_e32 vcc, v75, v75
	v_add3_u32 v19, v69, v19, s5
	v_lshrrev_b32_e32 v19, 16, v19
	v_cndmask_b32_e32 v17, v8, v17, vcc
	v_perm_b32 v17, v17, v18, s4
	v_and_b32_sdwa v18, v77, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v18, v77, v18, s5
	v_cmp_o_f32_e32 vcc, v69, v69
	v_lshrrev_b32_e32 v18, 16, v18
	v_and_b32_sdwa v20, v71, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e32 v19, v8, v19, vcc
	v_cmp_o_f32_e32 vcc, v77, v77
	v_add3_u32 v20, v71, v20, s5
	v_lshrrev_b32_e32 v20, 16, v20
	v_cndmask_b32_e32 v18, v8, v18, vcc
	v_perm_b32 v18, v18, v19, s4
	v_and_b32_sdwa v19, v79, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v19, v79, v19, s5
	v_cmp_o_f32_e32 vcc, v71, v71
	v_lshrrev_b32_e32 v19, 16, v19
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_add_i32 s55, s55, -1
	.loc	1 134 31                        ; chunk_delta_h.py:134:31
	v_cndmask_b32_e32 v20, v8, v20, vcc
	v_cmp_o_f32_e32 vcc, v79, v79
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_max_i32 s0, s55, 0
	.loc	1 132 16                        ; chunk_delta_h.py:132:16
	s_add_i32 s0, s0, s65
	.loc	1 134 31                        ; chunk_delta_h.py:134:31
	v_cndmask_b32_e32 v19, v8, v19, vcc
	v_perm_b32 v19, v19, v20, s4
	.loc	1 134 23 is_stmt 0              ; chunk_delta_h.py:134:23
	v_add_u32_e32 v20, 0xc000, v160
	ds_write2_b32 v20, v4, v16 offset1:32
	v_add_u32_e32 v4, 0xc800, v160
	ds_write2_b32 v4, v5, v17 offset1:32
	v_add_u32_e32 v5, 0, v9
	v_add_u32_e32 v9, 0xc000, v5
	v_add_u32_e32 v5, 0xc800, v5
	ds_write2_b32 v9, v12, v18 offset1:32
	ds_write2_b32 v5, v13, v19 offset1:32
	v_accvgpr_read_b32 v12, a51
	v_accvgpr_read_b32 v13, a52
	v_or3_b32 v12, v13, v150, v12
	v_accvgpr_read_b32 v13, a48
	v_accvgpr_read_b32 v16, a49
	v_or3_b32 v12, v12, v16, v13
	v_add_u32_e32 v13, 0, v12
	v_xad_u32 v21, v12, 8, 0
	v_xor_b32_e32 v16, 0x240, v12
	v_xor_b32_e32 v12, 0x248, v12
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v22, 0, v16
	v_add_u32_e32 v12, 0, v12
	ds_read_u16 v16, v13 offset:49152
	ds_read_u16 v23, v13 offset:50176
	ds_read_u16 v17, v21 offset:49152
	ds_read_u16 v24, v21 offset:50176
	ds_read_u16 v18, v22 offset:49152
	ds_read_u16 v25, v22 offset:50176
	ds_read_u16 v19, v12 offset:49152
	ds_read_u16 v26, v12 offset:50176
	.loc	1 132 16 is_stmt 1              ; chunk_delta_h.py:132:16
	s_lshl_b32 s0, s0, 17
	s_add_i32 s0, s0, s64
	.loc	1 134 23                        ; chunk_delta_h.py:134:23
	v_or_b32_e32 v27, s0, v152
	s_waitcnt lgkmcnt(4)
	v_perm_b32 v17, v24, v17, s4
	v_perm_b32 v16, v23, v16, s4
	v_add_lshl_u32 v23, v27, s48, 1
	v_bfrev_b32_e32 v24, 1
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	s_and_b64 vcc, s[18:19], s[26:27]
	.loc	1 134 23                        ; chunk_delta_h.py:134:23
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v19, v26, v19, s4
	v_perm_b32 v18, v25, v18, s4
	v_cndmask_b32_e32 v23, v24, v23, vcc
	s_mov_b32 s30, s14
	s_mov_b32 s31, s15
	buffer_store_dwordx4 v[16:19], v23, s[28:31], 0 offen
	.loc	1 139 35                        ; chunk_delta_h.py:139:35
	v_cmp_o_f32_e64 s[0:1], v80, v80
	v_and_b32_sdwa v23, v88, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_and_b32_sdwa v17, v80, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_and_b32_sdwa v16, v90, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v17, v80, v17, s5
	v_add3_u32 v16, v90, v16, s5
	v_lshrrev_b32_e32 v17, 16, v17
	v_lshrrev_b32_e32 v16, 16, v16
	v_cndmask_b32_e64 v17, v8, v17, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v90, v90
	v_and_b32_sdwa v18, v82, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v18, v82, v18, s5
	v_cndmask_b32_e64 v16, v8, v16, s[0:1]
	v_perm_b32 v16, v16, v17, s4
	v_and_b32_sdwa v17, v92, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v17, v92, v17, s5
	v_lshrrev_b32_e32 v18, 16, v18
	v_cmp_o_f32_e64 s[0:1], v82, v82
	v_lshrrev_b32_e32 v17, 16, v17
	v_and_b32_sdwa v19, v86, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e64 v18, v8, v18, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v92, v92
	v_add3_u32 v19, v86, v19, s5
	v_lshrrev_b32_e32 v19, 16, v19
	v_cndmask_b32_e64 v17, v8, v17, s[0:1]
	v_perm_b32 v17, v17, v18, s4
	v_and_b32_sdwa v18, v94, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v18, v94, v18, s5
	v_cmp_o_f32_e64 s[0:1], v86, v86
	v_lshrrev_b32_e32 v18, 16, v18
	v_add3_u32 v23, v88, v23, s5
	v_cndmask_b32_e64 v19, v8, v19, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v94, v94
	v_lshrrev_b32_e32 v23, 16, v23
	v_and_b32_sdwa v25, v81, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e64 v18, v8, v18, s[0:1]
	v_perm_b32 v18, v18, v19, s4
	v_and_b32_sdwa v19, v96, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v19, v96, v19, s5
	v_cmp_o_f32_e64 s[0:1], v88, v88
	v_lshrrev_b32_e32 v19, 16, v19
	v_add3_u32 v25, v81, v25, s5
	v_cndmask_b32_e64 v23, v8, v23, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v96, v96
	v_lshrrev_b32_e32 v25, 16, v25
	v_and_b32_sdwa v26, v83, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e64 v19, v8, v19, s[0:1]
	v_perm_b32 v19, v19, v23, s4
	v_and_b32_sdwa v23, v91, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v23, v91, v23, s5
	v_cmp_o_f32_e64 s[0:1], v81, v81
	v_lshrrev_b32_e32 v23, 16, v23
	v_add3_u32 v26, v83, v26, s5
	v_cndmask_b32_e64 v25, v8, v25, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v91, v91
	v_lshrrev_b32_e32 v26, 16, v26
	v_and_b32_sdwa v28, v87, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e64 v23, v8, v23, s[0:1]
	v_perm_b32 v23, v23, v25, s4
	v_and_b32_sdwa v25, v93, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v25, v93, v25, s5
	v_cmp_o_f32_e64 s[0:1], v83, v83
	v_lshrrev_b32_e32 v25, 16, v25
	v_add3_u32 v28, v87, v28, s5
	v_cndmask_b32_e64 v26, v8, v26, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v93, v93
	v_lshrrev_b32_e32 v28, 16, v28
	.loc	1 139 27 is_stmt 0              ; chunk_delta_h.py:139:27
	s_waitcnt lgkmcnt(0)
	.loc	1 139 35                        ; chunk_delta_h.py:139:35
	v_cndmask_b32_e64 v25, v8, v25, s[0:1]
	v_perm_b32 v25, v25, v26, s4
	v_and_b32_sdwa v26, v95, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v26, v95, v26, s5
	v_cmp_o_f32_e64 s[0:1], v87, v87
	v_lshrrev_b32_e32 v26, 16, v26
	.loc	1 139 27                        ; chunk_delta_h.py:139:27
	s_barrier
	.loc	1 139 35                        ; chunk_delta_h.py:139:35
	v_cndmask_b32_e64 v28, v8, v28, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v95, v95
	.loc	1 154 22 is_stmt 1              ; chunk_delta_h.py:154:22
	v_add_u32_e32 v36, 0, v166
	v_add_u32_e32 v49, 0, v230
	.loc	1 139 35                        ; chunk_delta_h.py:139:35
	v_cndmask_b32_e64 v26, v8, v26, s[0:1]
	v_perm_b32 v26, v26, v28, s4
	v_and_b32_sdwa v28, v97, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_and_b32_sdwa v1, v89, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_add3_u32 v1, v89, v1, s5
	v_add3_u32 v28, v97, v28, s5
	v_lshrrev_b32_e32 v1, 16, v1
	v_cmp_o_f32_e64 s[0:1], v89, v89
	v_lshrrev_b32_e32 v28, 16, v28
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_add_u32_e32 v37, 0, v2
	.loc	1 139 35                        ; chunk_delta_h.py:139:35
	v_cndmask_b32_e64 v1, v8, v1, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v97, v97
	.loc	1 154 22                        ; chunk_delta_h.py:154:22
	v_add_u32_e32 v46, 0, v169
	v_add_u32_e32 v47, 0, v3
	.loc	1 139 35                        ; chunk_delta_h.py:139:35
	v_cndmask_b32_e64 v8, v8, v28, s[0:1]
	v_perm_b32 v1, v8, v1, s4
	.loc	1 139 27 is_stmt 0              ; chunk_delta_h.py:139:27
	ds_write2_b32 v20, v16, v23 offset1:32
	ds_write2_b32 v4, v17, v25 offset1:32
	ds_write2_b32 v9, v18, v26 offset1:32
	ds_write2_b32 v5, v19, v1 offset1:32
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_u16 v1, v21 offset:49152
	ds_read_u16 v4, v21 offset:50176
	ds_read_u16 v5, v22 offset:49152
	ds_read_u16 v8, v12 offset:49152
	ds_read_u16 v9, v12 offset:50176
	ds_read_u16 v12, v22 offset:50176
	ds_read_u16 v16, v13 offset:49152
	ds_read_u16 v13, v13 offset:50176
	s_waitcnt lgkmcnt(6)
	v_perm_b32 v17, v4, v1, s4
	v_add_lshl_u32 v1, v27, s49, 1
	s_waitcnt lgkmcnt(3)
	v_perm_b32 v19, v9, v8, s4
	s_waitcnt lgkmcnt(2)
	v_perm_b32 v18, v12, v5, s4
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v16, v13, v16, s4
	v_cndmask_b32_e32 v1, v24, v1, vcc
	buffer_store_dwordx4 v[16:19], v1, s[28:31], 0 offen
	v_accvgpr_read_b32 v1, a55
	.loc	1 154 22 is_stmt 1              ; chunk_delta_h.py:154:22
	v_add_u32_e32 v100, 0, v1
	v_accvgpr_read_b32 v1, a54
	v_add_u32_e32 v105, 0, v1
	v_accvgpr_read_b32 v1, a56
	v_add_u32_e32 v107, 0, v1
	v_accvgpr_read_b32 v1, a63
	ds_read_b64 v[20:21], v36
	ds_read_b64 v[2:3], v37
	ds_read_b64 v[18:19], v46
	ds_read_b64 v[4:5], v47
	v_add_u32_e32 v84, 0, v6
	v_add_u32_e32 v98, 0, v167
	v_add_u32_e32 v99, 0, v7
	ds_read_b64 v[24:25], v49
	ds_read_b64 v[6:7], v84
	ds_read_b64 v[22:23], v98
	ds_read_b64 v[8:9], v99
	v_add_u32_e32 v102, 0, v10
	v_add_u32_e32 v103, 0, v53
	v_add_u32_e32 v104, 0, v11
	ds_read_b64 v[28:29], v100
	ds_read_b64 v[10:11], v102
	ds_read_b64 v[26:27], v103
	ds_read_b64 v[12:13], v104
	v_add_u32_e32 v106, 0, v14
	v_add_u32_e32 v108, 0, v15
	ds_read_b64 v[32:33], v105
	ds_read_b64 v[14:15], v106
	ds_read_b64 v[30:31], v107
	ds_read_b64 v[16:17], v108
	v_add_u32_e32 v39, 0, v1
	v_accvgpr_read_b32 v1, a62
	v_add_u32_e32 v56, 0, v1
	v_accvgpr_read_b32 v1, a61
	v_add_u32_e32 v58, 0, v1
	v_accvgpr_read_b32 v1, a60
	v_add_u32_e32 v60, 0, v1
	v_accvgpr_read_b32 v1, a59
	v_add_u32_e32 v62, 0, v1
	v_accvgpr_read_b32 v1, a58
	v_add_u32_e32 v85, 0, v1
	v_accvgpr_read_b32 v1, a57
	.loc	1 155 26                        ; chunk_delta_h.py:155:26
	v_add_u32_e32 v109, 0, v168
	v_add_u32_e32 v110, 0, v40
	v_add_u32_e32 v111, 0, v41
	v_add_u32_e32 v113, 0, v42
	v_add_u32_e32 v114, 0, v43
	v_add_u32_e32 v115, 0, v44
	v_add_u32_e32 v116, 0, v45
	v_add_u32_e32 v117, 0, v48
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a0, 0
	.loc	1 155 26                        ; chunk_delta_h.py:155:26
	s_and_b64 vcc, exec, s[22:23]
	v_add_u32_e32 v112, 0, v1
	v_add_u32_e32 v101, 0, v101
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write2st64_b32 v109, v64, v65 offset0:192 offset1:208
	ds_write2st64_b32 v110, v66, v67 offset0:192 offset1:208
	ds_write2st64_b32 v111, v68, v69 offset0:192 offset1:208
	ds_write2st64_b32 v113, v70, v71 offset0:192 offset1:208
	ds_write2st64_b32 v114, v72, v73 offset0:192 offset1:208
	ds_write2st64_b32 v115, v74, v75 offset0:192 offset1:208
	ds_write2st64_b32 v116, v76, v77 offset0:192 offset1:208
	ds_write2st64_b32 v117, v78, v79 offset0:192 offset1:208
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB0_88
; %bb.87:
	v_add_u32_e32 v1, 0xc000, v101
	ds_read2_b64 v[40:43], v1 offset1:16
	v_add_u32_e32 v1, 0xc000, v112
	ds_read2_b64 v[52:55], v1 offset1:16
	v_add_u32_e32 v1, 0xc000, v85
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[40:41], v[20:21], 0
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[52:53], v[18:19], a[0:15]
	ds_read2_b64 v[18:21], v1 offset1:16
	v_add_u32_e32 v1, 0xc000, v62
	ds_read2_b64 v[118:121], v1 offset1:16
	v_add_u32_e32 v1, 0xc000, v60
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[18:19], v[24:25], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[118:119], v[22:23], a[0:15]
	ds_read2_b64 v[22:25], v1 offset1:16
	v_add_u32_e32 v1, 0xc000, v58
	ds_read2_b64 v[122:125], v1 offset1:16
	v_add_u32_e32 v1, 0xc000, v56
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[22:23], v[28:29], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[122:123], v[26:27], a[0:15]
	ds_read2_b64 v[26:29], v1 offset1:16
	v_add_u32_e32 v1, 0xc000, v39
	ds_read2_b64 v[126:129], v1 offset1:16
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[26:27], v[32:33], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[126:127], v[30:31], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[42:43], v[2:3], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[54:55], v[4:5], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[20:21], v[6:7], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[120:121], v[8:9], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[24:25], v[10:11], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[124:125], v[12:13], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[28:29], v[14:15], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[128:129], v[16:17], a[0:15]
.LBB0_88:
	.loc	1 160 26                        ; chunk_delta_h.py:160:26
	ds_read_b64 v[54:55], v36 offset:16384
	ds_read_b64 v[18:19], v37 offset:16384
	ds_read_b64 v[52:53], v46 offset:16384
	ds_read_b64 v[20:21], v47 offset:16384
	ds_read_b64 v[48:49], v49 offset:16384
	ds_read_b64 v[22:23], v84 offset:16384
	ds_read_b64 v[46:47], v98 offset:16384
	ds_read_b64 v[24:25], v99 offset:16384
	ds_read_b64 v[44:45], v100 offset:16384
	ds_read_b64 v[26:27], v102 offset:16384
	ds_read_b64 v[42:43], v103 offset:16384
	ds_read_b64 v[28:29], v104 offset:16384
	ds_read_b64 v[40:41], v105 offset:16384
	ds_read_b64 v[30:31], v106 offset:16384
	ds_read_b64 v[36:37], v107 offset:16384
	ds_read_b64 v[32:33], v108 offset:16384
	v_accvgpr_read_b32 v17, a15
	v_accvgpr_read_b32 v16, a14
	v_accvgpr_read_b32 v15, a13
	v_accvgpr_read_b32 v14, a12
	v_accvgpr_read_b32 v13, a11
	v_accvgpr_read_b32 v12, a10
	v_accvgpr_read_b32 v11, a9
	v_accvgpr_read_b32 v10, a8
	v_accvgpr_read_b32 v9, a7
	v_accvgpr_read_b32 v8, a6
	v_accvgpr_read_b32 v7, a5
	v_accvgpr_read_b32 v6, a4
	v_accvgpr_read_b32 v5, a3
	v_accvgpr_read_b32 v4, a2
	v_accvgpr_read_b32 v3, a1
	v_accvgpr_read_b32 v2, a0
	.loc	1 161 31                        ; chunk_delta_h.py:161:31
	s_and_b64 vcc, exec, s[22:23]
	v_accvgpr_read_b32 v61, a42
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write2st64_b32 v109, v80, v81 offset0:192 offset1:208
	ds_write2st64_b32 v110, v82, v83 offset0:192 offset1:208
	ds_write2st64_b32 v111, v86, v87 offset0:192 offset1:208
	ds_write2st64_b32 v113, v88, v89 offset0:192 offset1:208
	ds_write2st64_b32 v114, v90, v91 offset0:192 offset1:208
	ds_write2st64_b32 v115, v92, v93 offset0:192 offset1:208
	ds_write2st64_b32 v116, v94, v95 offset0:192 offset1:208
	ds_write2st64_b32 v117, v96, v97 offset0:192 offset1:208
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB0_90
; %bb.89:
	v_add_u32_e32 v1, 0xc000, v101
	ds_read2_b64 v[98:101], v1 offset1:16
	v_add_u32_e32 v1, 0xc000, v112
	v_accvgpr_write_b32 a0, v2
	v_accvgpr_write_b32 a1, v3
	v_accvgpr_write_b32 a2, v4
	v_accvgpr_write_b32 a3, v5
	v_accvgpr_write_b32 a4, v6
	v_accvgpr_write_b32 a5, v7
	v_accvgpr_write_b32 a6, v8
	v_accvgpr_write_b32 a7, v9
	v_accvgpr_write_b32 a8, v10
	v_accvgpr_write_b32 a9, v11
	v_accvgpr_write_b32 a10, v12
	v_accvgpr_write_b32 a11, v13
	v_accvgpr_write_b32 a12, v14
	v_accvgpr_write_b32 a13, v15
	v_accvgpr_write_b32 a14, v16
	v_accvgpr_write_b32 a15, v17
	ds_read2_b64 v[2:5], v1 offset1:16
	v_add_u32_e32 v1, 0xc000, v85
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[98:99], v[54:55], a[0:15]
	ds_read2_b64 v[6:9], v1 offset1:16
	v_add_u32_e32 v1, 0xc000, v62
	ds_read2_b64 v[10:13], v1 offset1:16
	v_add_u32_e32 v1, 0xc000, v60
	ds_read2_b64 v[14:17], v1 offset1:16
	v_add_u32_e32 v1, 0xc000, v58
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[2:3], v[52:53], a[0:15]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[6:7], v[48:49], a[0:15]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[10:11], v[46:47], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[14:15], v[44:45], a[0:15]
	ds_read2_b64 v[44:47], v1 offset1:16
	v_add_u32_e32 v1, 0xc000, v56
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[44:45], v[42:43], a[0:15]
	ds_read2_b64 v[42:45], v1 offset1:16
	v_add_u32_e32 v1, 0xc000, v39
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[42:43], v[40:41], a[0:15]
	ds_read2_b64 v[40:43], v1 offset1:16
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x4_xf32 a[0:15], v[40:41], v[36:37], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[100:101], v[18:19], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[4:5], v[20:21], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[8:9], v[22:23], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[12:13], v[24:25], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[16:17], v[26:27], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[46:47], v[28:29], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[44:45], v[30:31], a[0:15]
	v_mfma_f32_32x32x4_xf32 a[0:15], v[42:43], v[32:33], a[0:15]
	s_nop 10
	v_accvgpr_read_b32 v17, a15
	v_accvgpr_read_b32 v16, a14
	v_accvgpr_read_b32 v15, a13
	v_accvgpr_read_b32 v14, a12
	v_accvgpr_read_b32 v13, a11
	v_accvgpr_read_b32 v12, a10
	v_accvgpr_read_b32 v11, a9
	v_accvgpr_read_b32 v10, a8
	v_accvgpr_read_b32 v9, a7
	v_accvgpr_read_b32 v8, a6
	v_accvgpr_read_b32 v7, a5
	v_accvgpr_read_b32 v6, a4
	v_accvgpr_read_b32 v5, a3
	v_accvgpr_read_b32 v4, a2
	v_accvgpr_read_b32 v3, a1
	v_accvgpr_read_b32 v2, a0
.LBB0_90:
	.loc	1 177 22                        ; chunk_delta_h.py:177:22
	v_and_b32_e32 v18, 0x660, v157
	v_and_b32_e32 v32, 0xe0, v0
	v_lshlrev_b32_e32 v19, 1, v154
	v_xor_b32_e32 v18, v18, v32
	v_lshlrev_b32_e32 v36, 11, v156
	v_lshlrev_b32_e32 v33, 8, v155
	v_add3_u32 v18, 0, v19, v18
	v_add3_u32 v39, v18, v36, v33
	v_lshlrev_b32_e32 v18, 7, v0
	v_lshlrev_b32_e32 v21, 6, v142
	v_and_b32_e32 v18, 0x600, v18
	v_and_b32_e32 v37, 28, v0
	v_lshlrev_b32_e32 v20, 4, v156
	v_lshl_or_b32 v21, v151, 11, v21
	v_lshlrev_b32_e32 v19, 3, v37
	v_lshlrev_b32_e32 v22, 1, v150
	v_or3_b32 v18, v18, v20, v21
	v_or3_b32 v22, v18, v22, v19
	v_add_u32_e32 v48, 0, v22
	v_xad_u32 v49, v22, 32, 0
	v_xad_u32 v51, v22, 64, 0
	v_xor_b32_e32 v22, 0x60, v22
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v39, v[222:225] offset:49152
	ds_write_b128 v39, v[232:235] offset:49408
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[18:21], v48 offset:49152
	ds_read_b128 v[26:29], v49 offset:49152
	v_add_u32_e32 v52, 0, v22
	ds_read_b128 v[40:43], v51 offset:49152
	ds_read_b128 v[44:47], v52 offset:49152
	.loc	1 112 24                        ; chunk_delta_h.py:112:24
	v_or_b32_e32 v1, 1, v50
	v_or_b32_e32 v30, 2, v50
	v_or_b32_e32 v31, 3, v50
	.loc	1 177 52                        ; chunk_delta_h.py:177:52
	s_waitcnt lgkmcnt(3)
	v_pk_add_f32 v[22:23], v[18:19], v[2:3] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[24:25], v[20:21], v[4:5] neg_lo:[0,1] neg_hi:[0,1]
	s_waitcnt lgkmcnt(2)
	v_pk_add_f32 v[18:19], v[26:27], v[6:7] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[20:21], v[28:29], v[8:9] neg_lo:[0,1] neg_hi:[0,1]
	s_waitcnt lgkmcnt(1)
	v_pk_add_f32 v[6:7], v[40:41], v[10:11] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[8:9], v[42:43], v[12:13] neg_lo:[0,1] neg_hi:[0,1]
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[2:3], v[44:45], v[14:15] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 183 26                        ; chunk_delta_h.py:183:26
	v_add3_u32 v10, v34, v50, s54
	v_add3_u32 v11, v34, v1, s54
	v_add3_u32 v12, v34, v30, s54
	v_add3_u32 v13, v34, v31, s54
	v_add3_u32 v15, v38, v1, s54
	v_and_b32_e32 v0, 64, v0
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_mov_b32_e32 v1, 0x1010
	.loc	1 177 52                        ; chunk_delta_h.py:177:52
	v_pk_add_f32 v[4:5], v[46:47], v[16:17] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 183 26                        ; chunk_delta_h.py:183:26
	v_add3_u32 v14, v38, v50, s54
	v_add3_u32 v16, v38, v30, s54
	v_add3_u32 v17, v38, v31, s54
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v39, v[10:13] offset:49152
	ds_write_b128 v39, v[14:17] offset:49408
	v_cmp_eq_u32_e32 vcc, 0, v0
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_mov_b32_e32 v0, 0x808
	v_cndmask_b32_e64 v1, v1, 0, s[34:35]
	v_accvgpr_read_b32 v10, a41
	v_cndmask_b32_e64 v0, v0, 0, s[2:3]
	v_or_b32_e32 v1, v1, v10
	v_xor_b32_e32 v0, v0, v1
	v_xor_b32_e32 v0, v0, v35
	v_lshl_or_b32 v10, v61, 7, v0
	.loc	1 183 26                        ; chunk_delta_h.py:183:26
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_b64 s[0:1], vcc, s[46:47]
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_add_u32_e32 v34, 0, v10
	.loc	1 183 26                        ; chunk_delta_h.py:183:26
	ds_read_b32 v11, v48 offset:49152
	ds_read_b32 v12, v49 offset:49152
	ds_read_b32 v13, v51 offset:49152
	ds_read_b32 v14, v52 offset:49152
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	ds_read_b64 v[0:1], v34 offset:32768
	.loc	1 183 26                        ; chunk_delta_h.py:183:26
	s_waitcnt lgkmcnt(4)
	v_lshlrev_b32_e32 v11, 2, v11
	v_bfrev_b32_e32 v15, 1
	s_and_b64 vcc, s[26:27], s[0:1]
	s_and_b32 s9, s9, 0xffff
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, 0x7ffffffe
	v_cndmask_b32_e32 v11, v15, v11, vcc
	buffer_store_dwordx4 v[22:25], v11, s[8:11], 0 offen
	s_waitcnt lgkmcnt(3)
	v_lshlrev_b32_e32 v11, 2, v12
	v_cndmask_b32_e32 v11, v15, v11, vcc
	buffer_store_dwordx4 v[18:21], v11, s[8:11], 0 offen
	s_waitcnt lgkmcnt(2)
	v_lshlrev_b32_e32 v11, 2, v13
	v_cndmask_b32_e32 v11, v15, v11, vcc
	.loc	1 193 53                        ; chunk_delta_h.py:193:53
	v_sub_f32_e32 v12, v188, v194
	.loc	1 183 26                        ; chunk_delta_h.py:183:26
	buffer_store_dwordx4 v[6:9], v11, s[8:11], 0 offen
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v11, 2, v14
	.loc	1 193 42                        ; chunk_delta_h.py:193:42
	v_mul_f32_e32 v13, 0x3fb8aa3b, v12
	s_mov_b32 s0, 0xc2fc0000
	.loc	1 183 26                        ; chunk_delta_h.py:183:26
	v_cndmask_b32_e32 v11, v15, v11, vcc
	.loc	1 193 42                        ; chunk_delta_h.py:193:42
	v_mov_b32_e32 v14, 0x42800000
	v_cmp_gt_f32_e32 vcc, s0, v13
	.loc	1 183 26                        ; chunk_delta_h.py:183:26
	buffer_store_dwordx4 v[2:5], v11, s[8:11], 0 offen
	.loc	1 187 30                        ; chunk_delta_h.py:187:30
	v_or_b32_e32 v11, s56, v59
	.loc	1 193 42                        ; chunk_delta_h.py:193:42
	v_cndmask_b32_e32 v13, 0, v14, vcc
	v_fmac_f32_e32 v13, 0x3fb8aa3b, v12
	v_exp_f32_e32 v12, v13
	v_not_b32_e32 v13, 63
	v_cndmask_b32_e32 v13, 0, v13, vcc
	.loc	1 187 50                        ; chunk_delta_h.py:187:50
	v_cmp_gt_i32_e32 vcc, s52, v11
	.loc	1 193 42                        ; chunk_delta_h.py:193:42
	v_ldexp_f32 v12, v12, v13
	s_movk_i32 s0, 0x7fff
	.loc	1 193 59 is_stmt 0              ; chunk_delta_h.py:193:59
	v_cndmask_b32_e32 v11, 0, v12, vcc
	.loc	1 193 24                        ; chunk_delta_h.py:193:24
	v_mul_f32_e32 v12, v11, v22
	v_mul_f32_e32 v13, v11, v23
	v_mul_f32_e32 v14, v11, v24
	v_mul_f32_e32 v15, v11, v25
	v_mul_f32_e32 v16, v11, v18
	v_mul_f32_e32 v17, v11, v19
	v_mul_f32_e32 v18, v11, v20
	v_mul_f32_e32 v19, v11, v21
	v_mul_f32_e32 v6, v11, v6
	v_mul_f32_e32 v7, v11, v7
	v_mul_f32_e32 v8, v11, v8
	v_mul_f32_e32 v9, v11, v9
	v_mul_f32_e32 v2, v11, v2
	v_mul_f32_e32 v3, v11, v3
	v_mul_f32_e32 v4, v11, v4
	v_mul_f32_e32 v5, v11, v5
	.loc	1 235 21 is_stmt 1              ; chunk_delta_h.py:235:21
	v_bfe_u32 v11, v12, 16, 1
	v_add3_u32 v11, v12, v11, s0
	v_lshrrev_b32_e32 v11, 16, v11
	v_mov_b32_e32 v20, 0x7fff
	v_cmp_o_f32_e32 vcc, v12, v12
	v_mov_b32_e32 v46, 0x220
	v_lshlrev_b32_e32 v45, 1, v57
	v_cndmask_b32_e32 v21, v20, v11, vcc
	v_bfe_u32 v11, v13, 16, 1
	v_add3_u32 v11, v13, v11, s0
	v_lshrrev_b32_e32 v11, 16, v11
	v_cmp_o_f32_e32 vcc, v13, v13
	v_cndmask_b32_e64 v46, v46, 0, s[20:21]
	v_xor_b32_e32 v45, v46, v45
	v_cndmask_b32_e32 v22, v20, v11, vcc
	v_bfe_u32 v11, v14, 16, 1
	v_add3_u32 v11, v14, v11, s0
	v_lshrrev_b32_e32 v11, 16, v11
	v_cmp_o_f32_e32 vcc, v14, v14
	v_accvgpr_read_b32 v48, a40
	v_or_b32_e32 v45, v45, v48
	v_cndmask_b32_e32 v23, v20, v11, vcc
	v_bfe_u32 v11, v15, 16, 1
	v_add3_u32 v11, v15, v11, s0
	v_lshrrev_b32_e32 v11, 16, v11
	v_cmp_o_f32_e32 vcc, v15, v15
	v_add_u32_e32 v46, 0, v45
	.loc	1 194 27                        ; chunk_delta_h.py:194:27
	v_accvgpr_write_b32 a15, 0
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e32 v24, v20, v11, vcc
	v_bfe_u32 v11, v16, 16, 1
	v_add3_u32 v11, v16, v11, s0
	v_lshrrev_b32_e32 v11, 16, v11
	v_cmp_o_f32_e32 vcc, v16, v16
	.loc	1 194 27                        ; chunk_delta_h.py:194:27
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a14, 0
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e32 v16, v20, v11, vcc
	v_bfe_u32 v11, v17, 16, 1
	v_add3_u32 v11, v17, v11, s0
	v_lshrrev_b32_e32 v11, 16, v11
	v_cmp_o_f32_e32 vcc, v17, v17
	.loc	1 194 27                        ; chunk_delta_h.py:194:27
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a13, 0
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e32 v17, v20, v11, vcc
	v_bfe_u32 v11, v18, 16, 1
	v_add3_u32 v11, v18, v11, s0
	v_lshrrev_b32_e32 v11, 16, v11
	v_cmp_o_f32_e32 vcc, v18, v18
	.loc	1 194 27                        ; chunk_delta_h.py:194:27
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a12, 0
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e32 v18, v20, v11, vcc
	v_bfe_u32 v11, v19, 16, 1
	v_add3_u32 v11, v19, v11, s0
	v_lshrrev_b32_e32 v11, 16, v11
	v_cmp_o_f32_e32 vcc, v19, v19
	.loc	1 194 27                        ; chunk_delta_h.py:194:27
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a11, 0
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e32 v19, v20, v11, vcc
	v_bfe_u32 v11, v6, 16, 1
	v_add3_u32 v11, v6, v11, s0
	v_cmp_o_f32_e32 vcc, v6, v6
	v_bfe_u32 v6, v7, 16, 1
	v_lshrrev_b32_e32 v11, 16, v11
	v_add3_u32 v6, v7, v6, s0
	v_cndmask_b32_e32 v25, v20, v11, vcc
	v_lshrrev_b32_e32 v6, 16, v6
	v_cmp_o_f32_e32 vcc, v7, v7
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_xor_b32_e32 v11, 0x50, v10
	v_add_u32_e32 v42, 0, v11
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e32 v26, v20, v6, vcc
	v_bfe_u32 v6, v8, 16, 1
	v_add3_u32 v6, v8, v6, s0
	v_lshrrev_b32_e32 v6, 16, v6
	v_cmp_o_f32_e32 vcc, v8, v8
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_xor_b32_e32 v11, 0x60, v10
	v_add_u32_e32 v43, 0, v11
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e32 v27, v20, v6, vcc
	v_bfe_u32 v6, v9, 16, 1
	v_add3_u32 v6, v9, v6, s0
	v_lshrrev_b32_e32 v6, 16, v6
	v_cmp_o_f32_e32 vcc, v9, v9
	.loc	1 194 27                        ; chunk_delta_h.py:194:27
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a10, 0
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e32 v28, v20, v6, vcc
	v_bfe_u32 v6, v2, 16, 1
	v_add3_u32 v6, v2, v6, s0
	v_cmp_o_f32_e32 vcc, v2, v2
	v_bfe_u32 v2, v3, 16, 1
	v_lshrrev_b32_e32 v6, 16, v6
	v_add3_u32 v2, v3, v2, s0
	v_cndmask_b32_e32 v29, v20, v6, vcc
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e32 vcc, v3, v3
	.loc	1 194 27                        ; chunk_delta_h.py:194:27
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a9, 0
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e32 v30, v20, v2, vcc
	v_bfe_u32 v2, v4, 16, 1
	v_add3_u32 v2, v4, v2, s0
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e32 vcc, v4, v4
	.loc	1 194 27                        ; chunk_delta_h.py:194:27
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a8, 0
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e32 v31, v20, v2, vcc
	v_bfe_u32 v2, v5, 16, 1
	v_add3_u32 v2, v5, v2, s0
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e32 vcc, v5, v5
	.loc	1 194 27                        ; chunk_delta_h.py:194:27
	v_accvgpr_write_b32 a0, 0
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	v_accvgpr_write_b32 a31, 0
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	v_cndmask_b32_e32 v20, v20, v2, vcc
	.loc	1 240 22                        ; chunk_delta_h.py:240:22
	v_xor_b32_e32 v2, 16, v10
	v_add_u32_e32 v38, 0, v2
	v_xor_b32_e32 v2, 32, v10
	v_add_u32_e32 v39, 0, v2
	v_xor_b32_e32 v2, 48, v10
	v_add_u32_e32 v40, 0, v2
	v_xor_b32_e32 v2, 64, v10
	v_xor_b32_e32 v10, 0x70, v10
	v_add_u32_e32 v41, 0, v2
	ds_read_b64 v[8:9], v38 offset:32768
	ds_read_b64 v[6:7], v39 offset:32768
	ds_read_b64 v[4:5], v40 offset:32768
	ds_read_b64 v[2:3], v41 offset:32768
	v_add_u32_e32 v44, 0, v10
	ds_read_b64 v[14:15], v42 offset:32768
	ds_read_b64 v[12:13], v43 offset:32768
	ds_read_b64 v[10:11], v44 offset:32768
	.loc	1 235 21                        ; chunk_delta_h.py:235:21
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b16 v46, v21 offset:49152
	ds_write_b16 v46, v25 offset:51200
	v_xad_u32 v21, v45, 8, 0
	ds_write_b16 v21, v22 offset:49280
	ds_write_b16 v21, v26 offset:51328
	v_xad_u32 v21, v45, 16, 0
	ds_write_b16 v21, v23 offset:49408
	ds_write_b16 v21, v27 offset:51456
	v_xad_u32 v21, v45, 24, 0
	ds_write_b16 v21, v24 offset:49536
	ds_write_b16 v21, v28 offset:51584
	v_xad_u32 v21, v45, 64, 0
	ds_write_b16 v21, v16 offset:50176
	ds_write_b16 v21, v29 offset:52224
	v_xor_b32_e32 v16, 0x48, v45
	v_add_u32_e32 v16, 0, v16
	ds_write_b16 v16, v17 offset:50304
	ds_write_b16 v16, v30 offset:52352
	v_xor_b32_e32 v16, 0x50, v45
	v_add_u32_e32 v16, 0, v16
	ds_write_b16 v16, v18 offset:50432
	ds_write_b16 v16, v31 offset:52480
	v_xor_b32_e32 v16, 0x58, v45
	v_add_u32_e32 v16, 0, v16
	ds_write_b16 v16, v19 offset:50560
	ds_write_b16 v16, v20 offset:52608
	v_lshl_or_b32 v20, v57, 7, v171
	v_add_u32_e32 v16, 0, v20
	v_xad_u32 v17, v20, 16, 0
	v_xad_u32 v18, v20, 32, 0
	v_xad_u32 v21, v20, 48, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[26:27], v16 offset:49152
	ds_read_b64 v[22:23], v17 offset:49152
	ds_read_b64 v[18:19], v18 offset:49152
	ds_read_b64 v[16:17], v21 offset:49152
	v_xad_u32 v21, v20, 64, 0
	v_xor_b32_e32 v24, 0x50, v20
	v_xor_b32_e32 v25, 0x60, v20
	v_xor_b32_e32 v20, 0x70, v20
	v_add_u32_e32 v24, 0, v24
	v_add_u32_e32 v25, 0, v25
	v_add_u32_e32 v20, 0, v20
	ds_read_b64 v[30:31], v21 offset:49152
	ds_read_b64 v[28:29], v24 offset:49152
	ds_read_b64 v[24:25], v25 offset:49152
	ds_read_b64 v[20:21], v20 offset:49152
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	v_accvgpr_write_b32 a23, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a24, 0
	.loc	1 241 37                        ; chunk_delta_h.py:241:37
	s_and_b64 vcc, exec, s[22:23]
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	v_accvgpr_write_b32 a16, 0
	.loc	1 241 37                        ; chunk_delta_h.py:241:37
	s_cbranch_vccnz .LBB0_92
; %bb.91:
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[26:27], v[0:1], 0
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[22:23], v[8:9], a[16:31]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[18:19], v[6:7], a[16:31]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[16:17], v[4:5], a[16:31]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[30:31], v[2:3], a[16:31]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[28:29], v[14:15], a[16:31]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[24:25], v[12:13], a[16:31]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 a[16:31], v[20:21], v[10:11], a[16:31]
.LBB0_92:
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_nop 10
	v_accvgpr_read_b32 v1, a31
	v_accvgpr_read_b32 v0, a23
	v_accvgpr_read_b32 v3, a30
	v_accvgpr_read_b32 v2, a22
	v_accvgpr_read_b32 v5, a29
	v_accvgpr_read_b32 v4, a21
	v_accvgpr_read_b32 v7, a28
	v_accvgpr_read_b32 v6, a20
	v_accvgpr_read_b32 v9, a27
	v_accvgpr_read_b32 v8, a19
	v_accvgpr_read_b32 v11, a26
	v_accvgpr_read_b32 v10, a18
	v_accvgpr_read_b32 v13, a25
	v_accvgpr_read_b32 v12, a17
	v_accvgpr_read_b32 v15, a24
	s_and_b64 vcc, exec, s[22:23]
	v_accvgpr_read_b32 v14, a16
	s_cbranch_vccnz .LBB0_94
; %bb.93:
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	ds_read_b64 v[46:47], v34 offset:40960
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[26:27], v[46:47], 0
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	ds_read_b64 v[26:27], v38 offset:40960
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[22:23], v[26:27], a[0:15]
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	ds_read_b64 v[22:23], v39 offset:40960
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[18:19], v[22:23], a[0:15]
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	ds_read_b64 v[18:19], v40 offset:40960
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[16:17], v[18:19], a[0:15]
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	ds_read_b64 v[16:17], v41 offset:40960
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[30:31], v[16:17], a[0:15]
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	ds_read_b64 v[16:17], v42 offset:40960
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[28:29], v[16:17], a[0:15]
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	ds_read_b64 v[16:17], v43 offset:40960
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[24:25], v[16:17], a[0:15]
	.loc	1 246 26                        ; chunk_delta_h.py:246:26
	ds_read_b64 v[16:17], v44 offset:40960
	.loc	1 247 41                        ; chunk_delta_h.py:247:41
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 a[0:15], v[20:21], v[16:17], a[0:15]
.LBB0_94:
	.loc	1 194 27                        ; chunk_delta_h.py:194:27
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v30, 0x3fb8aa3b, v188
	s_mov_b32 s0, 0xc2fc0000
	v_mov_b32_e32 v34, 0x42800000
	v_cmp_gt_f32_e32 vcc, s0, v30
	s_and_b64 s[0:1], vcc, exec
	s_cselect_b32 s0, 0xffffffc0, 0
	v_cndmask_b32_e32 v30, 0, v34, vcc
	v_fmac_f32_e32 v30, 0x3fb8aa3b, v188
	v_exp_f32_e32 v34, v30
	.loc	1 112 24                        ; chunk_delta_h.py:112:24
	v_mov_b32_e32 v39, s39
	v_or_b32_e32 v38, s38, v144
	.loc	1 194 27                        ; chunk_delta_h.py:194:27
	v_accvgpr_read_b32 v17, a15
	v_ldexp_f32 v34, v34, s0
	s_mov_b64 s[0:1], 0x80
	.loc	1 112 24                        ; chunk_delta_h.py:112:24
	v_cmp_gt_i64_e32 vcc, s[0:1], v[38:39]
	.loc	1 263 23                        ; chunk_delta_h.py:263:23
	v_mov_b32_e32 v39, 0x440
	.loc	1 134 23                        ; chunk_delta_h.py:134:23
	v_cmp_eq_u32_e64 s[0:1], 0, v151
	.loc	1 241 16                        ; chunk_delta_h.py:241:16
	v_pk_fma_f32 v[14:15], v[64:65], v[34:35], v[14:15] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[10:11], v[68:69], v[34:35], v[10:11] op_sel_hi:[1,0,1]
	.loc	1 263 23                        ; chunk_delta_h.py:263:23
	v_cndmask_b32_e64 v39, v39, 0, s[0:1]
	v_lshl_or_b32 v37, v37, 1, v39
	v_xor_b32_e32 v37, v37, v48
	v_lshl_add_u32 v39, v142, 3, 0
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	v_cndmask_b32_e64 v15, v65, v15, s[26:27]
	v_cndmask_b32_e64 v14, v64, v14, s[26:27]
	.loc	1 241 16                        ; chunk_delta_h.py:241:16
	v_pk_fma_f32 v[12:13], v[66:67], v[34:35], v[12:13] op_sel_hi:[1,0,1]
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	v_cndmask_b32_e64 v11, v69, v11, s[26:27]
	v_cndmask_b32_e64 v10, v68, v10, s[26:27]
	.loc	1 241 16                        ; chunk_delta_h.py:241:16
	v_pk_fma_f32 v[8:9], v[70:71], v[34:35], v[8:9] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[0:1], v[78:79], v[34:35], v[0:1] op_sel_hi:[1,0,1]
	.loc	1 263 23                        ; chunk_delta_h.py:263:23
	v_add3_u32 v36, v39, v37, v36
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	v_cndmask_b32_e64 v13, v67, v13, s[26:27]
	v_cndmask_b32_e64 v12, v66, v12, s[26:27]
	v_cndmask_b32_e64 v9, v71, v9, s[26:27]
	v_cndmask_b32_e64 v8, v70, v8, s[26:27]
	.loc	1 241 16                        ; chunk_delta_h.py:241:16
	v_pk_fma_f32 v[6:7], v[72:73], v[34:35], v[6:7] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[4:5], v[74:75], v[34:35], v[4:5] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[2:3], v[76:77], v[34:35], v[2:3] op_sel_hi:[1,0,1]
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	v_cndmask_b32_e64 v0, v78, v0, s[26:27]
	.loc	1 263 23                        ; chunk_delta_h.py:263:23
	ds_write2_b64 v36, v[14:15], v[10:11] offset1:16
	v_add_u32_e32 v10, 0x1000, v36
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	v_cndmask_b32_e64 v7, v73, v7, s[26:27]
	v_cndmask_b32_e64 v6, v72, v6, s[26:27]
	v_cndmask_b32_e64 v5, v75, v5, s[26:27]
	v_cndmask_b32_e64 v4, v74, v4, s[26:27]
	v_cndmask_b32_e64 v3, v77, v3, s[26:27]
	v_cndmask_b32_e64 v2, v76, v2, s[26:27]
	v_cndmask_b32_e64 v1, v79, v1, s[26:27]
	.loc	1 263 23                        ; chunk_delta_h.py:263:23
	ds_write2_b64 v10, v[12:13], v[8:9] offset1:16
	ds_write2_b64 v36, v[6:7], v[2:3] offset0:64 offset1:80
	ds_write2_b64 v10, v[4:5], v[0:1] offset0:64 offset1:80
	v_lshlrev_b32_e32 v0, 2, v32
	v_or3_b32 v0, v35, v0, v33
	v_add_u32_e32 v4, 0, v0
	v_xor_b32_e32 v0, 0x440, v0
	v_add_u32_e32 v5, 0, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2st64_b64 v[12:15], v4 offset1:4
	ds_read2st64_b64 v[40:43], v5 offset1:4
	.loc	1 194 27                        ; chunk_delta_h.py:194:27
	v_accvgpr_read_b32 v16, a7
	v_accvgpr_read_b32 v19, a14
	v_accvgpr_read_b32 v18, a6
	v_accvgpr_read_b32 v21, a13
	v_accvgpr_read_b32 v20, a5
	v_accvgpr_read_b32 v23, a12
	v_accvgpr_read_b32 v22, a4
	v_accvgpr_read_b32 v25, a11
	v_accvgpr_read_b32 v24, a3
	v_accvgpr_read_b32 v27, a10
	v_accvgpr_read_b32 v26, a2
	v_accvgpr_read_b32 v29, a9
	v_accvgpr_read_b32 v28, a1
	v_accvgpr_read_b32 v31, a8
	v_accvgpr_read_b32 v30, a0
	.loc	1 247 20                        ; chunk_delta_h.py:247:20
	v_pk_fma_f32 v[30:31], v[80:81], v[34:35], v[30:31] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[28:29], v[82:83], v[34:35], v[28:29] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[26:27], v[86:87], v[34:35], v[26:27] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[24:25], v[88:89], v[34:35], v[24:25] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[22:23], v[90:91], v[34:35], v[22:23] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[20:21], v[92:93], v[34:35], v[20:21] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[18:19], v[94:95], v[34:35], v[18:19] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[16:17], v[96:97], v[34:35], v[16:17] op_sel_hi:[1,0,1]
	.loc	1 263 23                        ; chunk_delta_h.py:263:23
	v_lshlrev_b32_e32 v34, 7, v144
	v_or3_b32 v34, v34, v145, s33
	v_lshlrev_b32_e32 v38, 7, v146
	v_add_lshl_u32 v6, v34, s48, 2
	v_bfrev_b32_e32 v7, 1
	.loc	1 112 24                        ; chunk_delta_h.py:112:24
	s_and_b64 vcc, s[50:51], vcc
	.loc	1 263 23                        ; chunk_delta_h.py:263:23
	v_or3_b32 v38, v38, v145, s33
	s_and_b32 s37, s37, 0xffff
	s_mov_b32 s39, 0x27000
	s_mov_b32 s38, 0x7ffffffe
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v0, v12
	v_mov_b32_e32 v1, v14
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v2, v40
	v_mov_b32_e32 v3, v42
	v_cndmask_b32_e32 v6, v7, v6, vcc
	buffer_store_dwordx4 v[0:3], v6, s[36:39], 0 offen
	v_add_lshl_u32 v6, v38, s48, 2
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	v_cndmask_b32_e64 v31, v81, v31, s[26:27]
	v_cndmask_b32_e64 v30, v80, v30, s[26:27]
	v_cndmask_b32_e64 v27, v87, v27, s[26:27]
	v_cndmask_b32_e64 v26, v86, v26, s[26:27]
	.loc	1 263 23                        ; chunk_delta_h.py:263:23
	v_mov_b32_e32 v0, v13
	v_mov_b32_e32 v1, v15
	v_mov_b32_e32 v2, v41
	v_mov_b32_e32 v3, v43
	v_cndmask_b32_e32 v6, v7, v6, vcc
	.loc	1 130 21                        ; chunk_delta_h.py:130:21
	v_cndmask_b32_e64 v29, v83, v29, s[26:27]
	v_cndmask_b32_e64 v28, v82, v28, s[26:27]
	v_cndmask_b32_e64 v25, v89, v25, s[26:27]
	v_cndmask_b32_e64 v24, v88, v24, s[26:27]
	v_cndmask_b32_e64 v23, v91, v23, s[26:27]
	v_cndmask_b32_e64 v22, v90, v22, s[26:27]
	v_cndmask_b32_e64 v21, v93, v21, s[26:27]
	v_cndmask_b32_e64 v20, v92, v20, s[26:27]
	v_cndmask_b32_e64 v19, v95, v19, s[26:27]
	v_cndmask_b32_e64 v18, v94, v18, s[26:27]
	v_cndmask_b32_e64 v17, v97, v17, s[26:27]
	v_cndmask_b32_e64 v16, v96, v16, s[26:27]
	.loc	1 263 23                        ; chunk_delta_h.py:263:23
	buffer_store_dwordx4 v[0:3], v6, s[36:39], 0 offen
	.loc	1 268 27                        ; chunk_delta_h.py:268:27
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write2_b64 v36, v[30:31], v[26:27] offset1:16
	ds_write2_b64 v10, v[28:29], v[24:25] offset1:16
	ds_write2_b64 v36, v[22:23], v[18:19] offset0:64 offset1:80
	ds_write2_b64 v10, v[20:21], v[16:17] offset0:64 offset1:80
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2st64_b64 v[8:11], v4 offset1:4
	ds_read2st64_b64 v[12:15], v5 offset1:4
	v_add_lshl_u32 v4, v34, s49, 2
	v_cndmask_b32_e32 v4, v7, v4, vcc
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v0, v8
	v_mov_b32_e32 v1, v10
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v2, v12
	v_mov_b32_e32 v3, v14
	buffer_store_dwordx4 v[0:3], v4, s[36:39], 0 offen
	v_add_lshl_u32 v4, v38, s49, 2
	v_cndmask_b32_e32 v4, v7, v4, vcc
	v_mov_b32_e32 v0, v9
	v_mov_b32_e32 v1, v11
	v_mov_b32_e32 v2, v13
	v_mov_b32_e32 v3, v15
	buffer_store_dwordx4 v[0:3], v4, s[36:39], 0 offen
	.loc	1 261 4                         ; chunk_delta_h.py:261:4
	s_endpgm
.Ltmp10:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel qwen_gdn_bt64_gfx942_asm_v0
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 88
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 14
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 320
		.amdhsa_next_free_sgpr 73
		.amdhsa_accum_offset 256
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	qwen_gdn_bt64_gfx942_asm_v0, .Lfunc_end0-qwen_gdn_bt64_gfx942_asm_v0
	.cfi_endproc
                                        ; -- End function
	.set qwen_gdn_bt64_gfx942_asm_v0.num_vgpr, 256
	.set qwen_gdn_bt64_gfx942_asm_v0.num_agpr, 64
	.set qwen_gdn_bt64_gfx942_asm_v0.numbered_sgpr, 73
	.set qwen_gdn_bt64_gfx942_asm_v0.num_named_barrier, 0
	.set qwen_gdn_bt64_gfx942_asm_v0.private_seg_size, 0
	.set qwen_gdn_bt64_gfx942_asm_v0.uses_vcc, 1
	.set qwen_gdn_bt64_gfx942_asm_v0.uses_flat_scratch, 0
	.set qwen_gdn_bt64_gfx942_asm_v0.has_dyn_sized_stack, 0
	.set qwen_gdn_bt64_gfx942_asm_v0.has_recursion, 0
	.set qwen_gdn_bt64_gfx942_asm_v0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 23996
; TotalNumSgprs: 79
; NumVgprs: 256
; NumAgprs: 64
; TotalNumVgprs: 320
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 9
; VGPRBlocks: 39
; NumSGPRsForWavesPerEU: 79
; NumVGPRsForWavesPerEU: 320
; AccumOffset: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 63
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	14                              ; DW_FORM_strp
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	32                              ; DW_AT_inline
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x44 DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.byte	2                               ; Abbrev [2] 0x2a:0x6 DW_TAG_subprogram
	.long	.Linfo_string3                  ; DW_AT_name
	.byte	1                               ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x30:0x1e DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	81                              ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"chunk_delta_h.py"              ; string offset=7
.Linfo_string2:
	.asciz	"/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fla/ops" ; string offset=24
.Linfo_string3:
	.asciz	"qwen_gdn_bt64_gfx942_asm_v0" ; string offset=98
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     64
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 88
    .max_flat_workgroup_size: 256
    .name:           qwen_gdn_bt64_gfx942_asm_v0
    .private_segment_fixed_size: 0
    .sgpr_count:     79
    .sgpr_spill_count: 0
    .symbol:         qwen_gdn_bt64_gfx942_asm_v0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     320
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
