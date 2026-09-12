	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6

	.section .text.qwen_gfx942_asm_smoke,"ax",@progbits
	.globl qwen_gfx942_asm_smoke
	.p2align 8
	.type qwen_gfx942_asm_smoke,@function
qwen_gfx942_asm_smoke:
	// Kernarg layout, matching SmokeArgs in smoke_harness.cpp:
	//   +0  input  : uint32_t*
	//   +8  output : uint32_t*
	//   +16 addend : uint32_t
	// v0 is the workitem id and s2 is workgroup_id_x.
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	s_load_dword s8, s[0:1], 0x10
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, s2
	v_lshlrev_b32_e32 v1, 7, v3
	v_add_u32_e32 v1, v1, v0
	v_lshlrev_b32_e32 v1, 2, v1
	global_load_dword v2, v1, s[4:5]
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v4, s8
	v_add_u32_e32 v2, v2, v4
	global_store_dword v1, v2, s[6:7]
	s_endpgm
.Lqwen_gfx942_asm_smoke_end:
	.size qwen_gfx942_asm_smoke, .Lqwen_gfx942_asm_smoke_end-qwen_gfx942_asm_smoke

	.section .rodata,"a",@progbits
	.p2align 6, 0x0
	.amdhsa_kernel qwen_gfx942_asm_smoke
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 9
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
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

	.set qwen_gfx942_asm_smoke.num_vgpr, 5
	.set qwen_gfx942_asm_smoke.num_agpr, 0
	.set qwen_gfx942_asm_smoke.numbered_sgpr, 9
	.set qwen_gfx942_asm_smoke.num_named_barrier, 0
	.set qwen_gfx942_asm_smoke.private_seg_size, 0
	.set qwen_gfx942_asm_smoke.uses_vcc, 1
	.set qwen_gfx942_asm_smoke.uses_flat_scratch, 0
	.set qwen_gfx942_asm_smoke.has_dyn_sized_stack, 0
	.set qwen_gfx942_asm_smoke.has_recursion, 0
	.set qwen_gfx942_asm_smoke.has_indirect_call, 0

	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space: global
        .offset:        0
        .size:          8
        .value_kind:    global_buffer
      - .address_space: global
        .offset:        8
        .size:          8
        .value_kind:    global_buffer
      - .offset:        16
        .size:          4
        .value_kind:    by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .language: OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 128
    .name: qwen_gfx942_asm_smoke
    .private_segment_fixed_size: 0
    .sgpr_count: 9
    .sgpr_spill_count: 0
    .symbol: qwen_gfx942_asm_smoke.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 5
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target: amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
