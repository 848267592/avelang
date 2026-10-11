"""A69 BW16 persistent K2 rebuild.

The source keeps the A68 mathematical body and changes the structural contract
only: one device-side dynamic chunk loop, live FP32 H fragments, and four-wave
Kr ownership.  The packet views are intentionally explicit and use the pinned
BW16 source layouts recorded in the A69 ownership ledger.
"""

from __future__ import annotations

import torch
import avelang
import avelang.language as al


HEADS = 12
HD = 128
C = 32
V = 128
BW = 16
WG = 256
LANES = 64
TRACE_PHASES = 12


@avelang.jit
def _a70_bw16_k2_persistent(
    kd_ptr: al.Pointer(al.bf16),
    qd_ptr: al.Pointer(al.bf16),
    kr_ptr: al.Pointer(al.bf16),
    gt_ptr: al.Pointer(al.f32),
    inv_mqk_ptr: al.Pointer(al.f16),
    v_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.f32),
    initial_state_ptr: al.Pointer(al.f32),
    nt_ptr: al.Pointer(al.i32),
    out_ptr: al.Pointer(al.bf16),
    final_state_ptr: al.Pointer(al.f32),
    debug_ptr: al.Pointer(al.f32),
    T: al.constexpr,
    NT: al.constexpr,
    DEBUG: al.constexpr,
):
    kd = al.make_tensor(kd_ptr, al.bf16, al.make_layout((HEADS * NT, C, HD), (C * HD, HD, 1)))
    qd = al.make_tensor(qd_ptr, al.bf16, al.make_layout((HEADS * NT, C, HD), (C * HD, HD, 1)))
    kr = al.make_tensor(kr_ptr, al.bf16, al.make_layout((HEADS * NT, C, HD), (C * HD, HD, 1)))
    gt = al.make_tensor(gt_ptr, al.f32, al.make_layout((HEADS * NT, HD), (HD, 1)))
    inv_mqk = al.make_tensor(
        inv_mqk_ptr,
        al.f16,
        al.make_layout((HEADS * NT, 2, C, C), (2 * C * C, C * C, C, 1)),
    )
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((T, HEADS, V), (HEADS * V, V, 1)))
    beta = al.make_tensor(beta_ptr, al.f32, al.make_layout((T, HEADS), (HEADS, 1)))
    initial_state = al.make_tensor(
        initial_state_ptr,
        al.f32,
        al.make_layout((HEADS, HD, V), (HD * V, V, 1)),
    )
    nt_tensor = al.make_tensor(nt_ptr, al.i32, al.make_layout((1, 1), (1, 1)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((T, HEADS, V), (HEADS * V, V, 1)))
    final_state = al.make_tensor(
        final_state_ptr,
        al.f32,
        al.make_layout((HEADS, HD, V), (HD * V, V, 1)),
    )
    debug = al.make_tensor(
        debug_ptr,
        al.f32,
        al.make_layout((HEADS * TRACE_PHASES * HD * V,), (1,)),
    )

    # 30,720 bytes: the nine C67/AITER slots.  Each alias is temporal and is
    # fenced by the surrounding CTA barrier before reuse.
    pool = al.make_shared((7680,), al.u32)
    kd1 = al.view(al.subview(pool, (0,), (1024,), (1,)), al.bf16, al.make_layout((2048,), (1,)))
    kd2 = al.view(al.subview(pool, (1024,), (1024,), (1,)), al.bf16, al.make_layout((2048,), (1,)))
    qd1 = al.view(al.subview(pool, (2048,), (1024,), (1,)), al.bf16, al.make_layout((2048,), (1,)))
    qd2 = al.view(al.subview(pool, (3072,), (1024,), (1,)), al.bf16, al.make_layout((2048,), (1,)))
    kr1_t = al.view(al.subview(pool, (4096,), (1024,), (1,)), al.bf16, al.make_layout((2048,), (1,)))
    kr2_t = al.view(al.subview(pool, (5120,), (1024,), (1,)), al.bf16, al.make_layout((2048,), (1,)))
    inv = al.view(al.subview(pool, (6144,), (512,), (1,)), al.f16, al.make_layout((1024,), (1,)))
    mqk = al.view(al.subview(pool, (6656,), (512,), (1,)), al.f16, al.make_layout((1024,), (1,)))
    overlay_bf = al.view(al.subview(pool, (7168,), (512,), (1,)), al.bf16, al.make_layout((1024,), (1,)))
    overlay_f16 = al.view(al.subview(pool, (7168,), (512,), (1,)), al.f16, al.make_layout((1024,), (1,)))

    tid = al.thread_id(0)
    wave = tid >> 6
    lane = tid & 63
    pid_x = al.block_id(0)
    head = al.block_id(1)
    w0 = pid_x * BW

    # #mma [4,1] H ownership.  This is the only state representation that
    # crosses the dynamic loop; H is never reloaded from or written to HBM.
    h1 = al.full((4,), 0.0, al.f32)
    h2 = al.full((4,), 0.0, al.f32)
    # Pinned C67 #mma carrier: row=16*wave+(lane&15),
    # col=4*(lane>>4)+frag.  This is distinct from the #mma1 paired-wave
    # carrier used by the 32x16 products below.
    h_row = wave * 16 + (lane & 15)
    h_col_base = (lane >> 4) * 4
    for acc_i in al.range(4):
        h1[acc_i] = initial_state[head, h_row, w0 + h_col_base + acc_i]
        h2[acc_i] = initial_state[head, 64 + h_row, w0 + h_col_base + acc_i]

    nt_limit = nt_tensor[0, 0]
    chunk_idx = 0
    while chunk_idx < nt_limit:
        ws_idx = head * NT + chunk_idx
        tok_base = chunk_idx * C

        # Kd/Qd source packet ownership: blocked1, four 8-row bands.
        kd_row = wave * 8 + lane // 8
        kd_k_base = (lane & 7) * 8
        kd_write_byte = (tid << 4) ^ (tid & 120)
        for q in al.range(4):
            kd1[(kd_write_byte >> 1) + q] = kd[ws_idx, kd_row, kd_k_base + q]
            kd1[((kd_write_byte ^ 8) >> 1) + q] = kd[ws_idx, kd_row, kd_k_base + 4 + q]
            kd2[(kd_write_byte >> 1) + q] = kd[ws_idx, kd_row, 64 + kd_k_base + q]
            kd2[((kd_write_byte ^ 8) >> 1) + q] = kd[ws_idx, kd_row, 64 + kd_k_base + 4 + q]
        al.syncthreads()

        # Exact C67 #shared3 producer: H fragment f=0..3 is written at
        # H_write xor {0,136,272,408} bytes.
        h_write_byte = (wave << 5) ^ (((lane & 48) << 5) | (lane << 1))
        overlay_bf[h_write_byte >> 1] = al.convert(h1[0], al.bf16)
        overlay_bf[(h_write_byte ^ 136) >> 1] = al.convert(h1[1], al.bf16)
        overlay_bf[(h_write_byte ^ 272) >> 1] = al.convert(h1[2], al.bf16)
        overlay_bf[(h_write_byte ^ 408) >> 1] = al.convert(h1[3], al.bf16)
        al.syncthreads()

        # Kd@H1.  All four waves issue the MFMA.  The paired #mma1 result
        # carrier is row=16*(wave>>1)+(lane&15), col=4*(lane>>4)+frag.
        acc_kd1 = al.full((4,), 0.0, al.f32)
        a_shared_byte = (((wave << 6) & 128) << 4) | ((lane & 15) << 7) | (((lane & 15) << 3) ^ ((lane & 48) >> 1))
        h_read_byte = ((lane & 15) << 7) | (((lane & 15) << 3) ^ ((lane & 48) >> 1))
        for k_step in al.range(4):
            a_frag = al.full((4,), 0, al.bf16)
            b_frag = al.full((4,), 0, al.bf16)
            for q in al.range(4):
                a_frag[q] = kd1[((a_shared_byte ^ (k_step << 5)) >> 1) + q]
                b_frag[q] = overlay_bf[((h_read_byte ^ (k_step << 5)) >> 1) + q]
            al.amdgpu.s_nop(1)
            if k_step == 0:
                acc_kd1 = al.amdgpu.mfma_16x16x16_bf16_f32_vgpr_zero(b_frag, a_frag)
            else:
                acc_kd1 = al.amdgpu.mfma_16x16x16_bf16_f32_vgpr(b_frag, a_frag, acc_kd1)
        acc_kd1 = al.amdgpu.mfma_hazard_drain(acc_kd1, 6)
        al.syncthreads()

        # Kd@H2 uses the same overlay slot after the H1 consumer has retired.
        overlay_bf[h_write_byte >> 1] = al.convert(h2[0], al.bf16)
        overlay_bf[(h_write_byte ^ 136) >> 1] = al.convert(h2[1], al.bf16)
        overlay_bf[(h_write_byte ^ 272) >> 1] = al.convert(h2[2], al.bf16)
        overlay_bf[(h_write_byte ^ 408) >> 1] = al.convert(h2[3], al.bf16)
        al.syncthreads()
        acc_kd2 = al.full((4,), 0.0, al.f32)
        for k_step in al.range(4):
            a_frag = al.full((4,), 0, al.bf16)
            b_frag = al.full((4,), 0, al.bf16)
            for q in al.range(4):
                a_frag[q] = kd2[((a_shared_byte ^ (k_step << 5)) >> 1) + q]
                b_frag[q] = overlay_bf[((h_read_byte ^ (k_step << 5)) >> 1) + q]
            al.amdgpu.s_nop(1)
            if k_step == 0:
                acc_kd2 = al.amdgpu.mfma_16x16x16_bf16_f32_vgpr_zero(b_frag, a_frag)
            else:
                acc_kd2 = al.amdgpu.mfma_16x16x16_bf16_f32_vgpr(b_frag, a_frag, acc_kd2)
        acc_kd2 = al.amdgpu.mfma_hazard_drain(acc_kd2, 6)
        al.syncthreads()

        u_pre = al.full((4,), 0.0, al.f32)
        if (wave & 1) == 0:
            out_col_base = (lane >> 4) * 4
            row = (wave >> 1) * 16 + (lane & 15)
            t = tok_base + row
            for acc_i in al.range(4):
                pred = acc_kd1[acc_i] + acc_kd2[acc_i]
                value = al.convert(0.0, al.f32)
                beta_sigmoid = al.convert(0.0, al.f32)
                if t < T:
                    value = al.convert(v[t, head, w0 + out_col_base + acc_i], al.f32)
                    beta_value = beta[t, head]
                    one_f32 = al.convert(1.0, al.f32)
                    beta_sigmoid = one_f32 / (one_f32 + al.exp(-beta_value))
                u_pre[acc_i] = (value - pred) * beta_sigmoid
                if DEBUG:
                    debug[((head * TRACE_PHASES + 0) * HD + row) * V + w0 + out_col_base + acc_i] = pred
                    debug[((head * TRACE_PHASES + 1) * HD + row) * V + w0 + out_col_base + acc_i] = u_pre[acc_i]

        # INV packet and u_pre FP16 overlay.
        inv_row = wave * 8 + lane // 8
        inv_col_base = (lane & 7) * 4
        inv_write_byte = (tid << 3) ^ ((tid >> 1) & 56)
        for q in al.range(4):
            inv[(inv_write_byte >> 1) + q] = inv_mqk[ws_idx, 0, inv_row, inv_col_base + q]
        u0_byte = ((((wave << 6) & 128) >> 2) | ((lane & 48) << 4) | ((lane & 15) << 1)) ^ (lane & 48)
        u2_byte = ((((tid << 2) & 56) ^ ((lane & 48) >> 1)) | ((lane & 15) << 6))
        if (wave & 1) == 0:
            overlay_f16[u0_byte >> 1] = al.convert(u_pre[0], al.f16)
            overlay_f16[(u0_byte + 64) >> 1] = al.convert(u_pre[1], al.f16)
            overlay_f16[(u0_byte ^ 136) >> 1] = al.convert(u_pre[2], al.f16)
            overlay_f16[((u0_byte ^ 136) + 64) >> 1] = al.convert(u_pre[3], al.f16)
        al.syncthreads()

        acc_u = al.full((4,), 0.0, al.f32)
        a_fp16_byte = (((wave << 6) & 128) << 3) | ((lane & 15) << 6) | (((tid << 2) & 56) ^ ((lane & 48) >> 1))
        for k_step in al.range(2):
            a_frag = al.full((4,), 0, al.f16)
            b_frag = al.full((4,), 0, al.f16)
            for q in al.range(4):
                a_frag[q] = inv[((a_fp16_byte ^ (k_step << 5)) >> 1) + q]
                b_frag[q] = overlay_f16[((u2_byte ^ (k_step << 5)) >> 1) + q]
            al.amdgpu.s_nop(1)
            acc_u = al.amdgpu.mfma_16x16x16_f16_f32_vgpr(b_frag, a_frag, acc_u)
            if DEBUG and k_step == 0:
                for q in al.range(4):
                    debug[(head * TRACE_PHASES + 8) * HD * V + tid * 4 + q] = al.convert(a_frag[q], al.f32)
                    debug[(head * TRACE_PHASES + 9) * HD * V + tid * 4 + q] = al.convert(b_frag[q], al.f32)
        acc_u = al.amdgpu.mfma_hazard_drain(acc_u, 6)
        if DEBUG:
            for q in al.range(4):
                debug[(head * TRACE_PHASES + 11) * HD * V + tid * 4 + q] = acc_u[q]
        if DEBUG and (wave & 1) == 0:
            row = (wave >> 1) * 16 + (lane & 15)
            col_base = (lane >> 4) * 4
            for acc_i in al.range(4):
                debug[((head * TRACE_PHASES + 2) * HD + row) * V + w0 + col_base + acc_i] = acc_u[acc_i]
        al.syncthreads()

        # Post-INV U FP16 overlay is the only consumer view for Mqk.
        if (wave & 1) == 0:
            overlay_f16[u0_byte >> 1] = al.convert(acc_u[0], al.f16)
            overlay_f16[(u0_byte + 64) >> 1] = al.convert(acc_u[1], al.f16)
            overlay_f16[(u0_byte ^ 136) >> 1] = al.convert(acc_u[2], al.f16)
            overlay_f16[((u0_byte ^ 136) + 64) >> 1] = al.convert(acc_u[3], al.f16)
        al.syncthreads()

        for q in al.range(4):
            mqk[(inv_write_byte >> 1) + q] = inv_mqk[ws_idx, 1, inv_row, inv_col_base + q]
        al.syncthreads()

        acc_mqk = al.full((4,), 0.0, al.f32)
        for k_step in al.range(2):
            a_frag = al.full((4,), 0, al.f16)
            b_frag = al.full((4,), 0, al.f16)
            for q in al.range(4):
                a_frag[q] = mqk[((a_fp16_byte ^ (k_step << 5)) >> 1) + q]
                b_frag[q] = overlay_f16[((u2_byte ^ (k_step << 5)) >> 1) + q]
            al.amdgpu.s_nop(1)
            acc_mqk = al.amdgpu.mfma_16x16x16_f16_f32_vgpr(b_frag, a_frag, acc_mqk)
        acc_mqk = al.amdgpu.mfma_hazard_drain(acc_mqk, 6)
        if DEBUG and (wave & 1) == 0:
            row = (wave >> 1) * 16 + (lane & 15)
            col_base = (lane >> 4) * 4
            for acc_i in al.range(4):
                debug[((head * TRACE_PHASES + 4) * HD + row) * V + w0 + col_base + acc_i] = acc_mqk[acc_i]
        al.syncthreads()

        # Qd@H1/H2 and final output accumulator.
        qd_row = wave * 8 + lane // 8
        qd_k_base = (lane & 7) * 8
        qd_write_byte = kd_write_byte
        for q in al.range(4):
            qd1[(qd_write_byte >> 1) + q] = qd[ws_idx, qd_row, qd_k_base + q]
            qd1[((qd_write_byte ^ 8) >> 1) + q] = qd[ws_idx, qd_row, qd_k_base + 4 + q]
        al.syncthreads()
        overlay_bf[h_write_byte >> 1] = al.convert(h1[0], al.bf16)
        overlay_bf[(h_write_byte ^ 136) >> 1] = al.convert(h1[1], al.bf16)
        overlay_bf[(h_write_byte ^ 272) >> 1] = al.convert(h1[2], al.bf16)
        overlay_bf[(h_write_byte ^ 408) >> 1] = al.convert(h1[3], al.bf16)
        al.syncthreads()
        acc_qd1 = al.full((4,), 0.0, al.f32)
        for k_step in al.range(4):
            a_frag = al.full((4,), 0, al.bf16)
            b_frag = al.full((4,), 0, al.bf16)
            for q in al.range(4):
                a_frag[q] = qd1[((a_shared_byte ^ (k_step << 5)) >> 1) + q]
                b_frag[q] = overlay_bf[((h_read_byte ^ (k_step << 5)) >> 1) + q]
            al.amdgpu.s_nop(1)
            if k_step == 0:
                acc_qd1 = al.amdgpu.mfma_16x16x16_bf16_f32_vgpr_zero(b_frag, a_frag)
            else:
                acc_qd1 = al.amdgpu.mfma_16x16x16_bf16_f32_vgpr(b_frag, a_frag, acc_qd1)
        acc_qd1 = al.amdgpu.mfma_hazard_drain(acc_qd1, 6)
        al.syncthreads()

        for q in al.range(4):
            qd2[(qd_write_byte >> 1) + q] = qd[ws_idx, qd_row, 64 + qd_k_base + q]
            qd2[((qd_write_byte ^ 8) >> 1) + q] = qd[ws_idx, qd_row, 64 + qd_k_base + 4 + q]
        al.syncthreads()
        overlay_bf[h_write_byte >> 1] = al.convert(h2[0], al.bf16)
        overlay_bf[(h_write_byte ^ 136) >> 1] = al.convert(h2[1], al.bf16)
        overlay_bf[(h_write_byte ^ 272) >> 1] = al.convert(h2[2], al.bf16)
        overlay_bf[(h_write_byte ^ 408) >> 1] = al.convert(h2[3], al.bf16)
        al.syncthreads()
        acc_qd2 = al.full((4,), 0.0, al.f32)
        for k_step in al.range(4):
            a_frag = al.full((4,), 0, al.bf16)
            b_frag = al.full((4,), 0, al.bf16)
            for q in al.range(4):
                a_frag[q] = qd2[((a_shared_byte ^ (k_step << 5)) >> 1) + q]
                b_frag[q] = overlay_bf[((h_read_byte ^ (k_step << 5)) >> 1) + q]
            al.amdgpu.s_nop(1)
            if k_step == 0:
                acc_qd2 = al.amdgpu.mfma_16x16x16_bf16_f32_vgpr_zero(b_frag, a_frag)
            else:
                acc_qd2 = al.amdgpu.mfma_16x16x16_bf16_f32_vgpr(b_frag, a_frag, acc_qd2)
        acc_qd2 = al.amdgpu.mfma_hazard_drain(acc_qd2, 6)
        al.syncthreads()

        if (wave & 1) == 0:
            out_col_base = (lane >> 4) * 4
            row = (wave >> 1) * 16 + (lane & 15)
            t = tok_base + row
            for acc_i in al.range(4):
                output_value = acc_qd1[acc_i] + acc_qd2[acc_i] + acc_mqk[acc_i]
                if t < T:
                    out[t, head, w0 + out_col_base + acc_i] = al.convert(output_value, al.bf16)
                if DEBUG:
                    debug[((head * TRACE_PHASES + 3) * HD + row) * V + w0 + out_col_base + acc_i] = acc_qd1[acc_i] + acc_qd2[acc_i]
                    debug[((head * TRACE_PHASES + 5) * HD + row) * V + w0 + out_col_base + acc_i] = output_value
        al.syncthreads()

        # U BF16 overlay for the recurrent Kr update.
        if (wave & 1) == 0:
            overlay_bf[u0_byte >> 1] = al.convert(acc_u[0], al.bf16)
            overlay_bf[(u0_byte + 64) >> 1] = al.convert(acc_u[1], al.bf16)
            overlay_bf[(u0_byte ^ 136) >> 1] = al.convert(acc_u[2], al.bf16)
            overlay_bf[((u0_byte ^ 136) + 64) >> 1] = al.convert(acc_u[3], al.bf16)
        al.syncthreads()

        kr_row_base = (lane & 7) * 8
        kr_col = wave * 8 + lane // 8
        kr_write_base = tid << 3
        for q in al.range(8):
            k_row = kr_row_base + q
            kr1_t[kr_write_base + q] = kr[ws_idx, kr_col, k_row]
            kr2_t[kr_write_base + q] = kr[ws_idx, kr_col, 64 + k_row]
        al.syncthreads()

        acc_kr1 = al.full((4,), 0.0, al.f32)
        acc_kr2 = al.full((4,), 0.0, al.f32)
        kr_read_base = (wave << 5) | ((lane & 48) << 5) | ((lane & 15) << 1)
        for k_step in al.range(2):
            a1 = al.full((4,), 0, al.bf16)
            a2 = al.full((4,), 0, al.bf16)
            b = al.full((4,), 0, al.bf16)
            kr_offset = k_step * 2048
            a1[0] = kr1_t[(kr_read_base | (kr_offset + 0)) >> 1]
            a1[1] = kr1_t[(kr_read_base | (kr_offset + 128)) >> 1]
            a1[2] = kr1_t[(kr_read_base | (kr_offset + 256)) >> 1]
            a1[3] = kr1_t[(kr_read_base | (kr_offset + 384)) >> 1]
            a2[0] = kr2_t[(kr_read_base | (kr_offset + 0)) >> 1]
            a2[1] = kr2_t[(kr_read_base | (kr_offset + 128)) >> 1]
            a2[2] = kr2_t[(kr_read_base | (kr_offset + 256)) >> 1]
            a2[3] = kr2_t[(kr_read_base | (kr_offset + 384)) >> 1]
            b[0] = overlay_bf[((u2_byte ^ (k_step << 5)) >> 1) + 0]
            b[1] = overlay_bf[((u2_byte ^ (k_step << 5)) >> 1) + 1]
            b[2] = overlay_bf[((u2_byte ^ (k_step << 5)) >> 1) + 2]
            b[3] = overlay_bf[((u2_byte ^ (k_step << 5)) >> 1) + 3]
            # Keep the packed ordinary-V payload in the same permute dataflow
            # as the pinned Kr sequence.  The scalar LDS addresses above are
            # the B70 exact Kr0|offset map; perm is payload-preserving here.
            kr_word1 = al.convert(al.bitcast(a1[0], al.u16), al.u32) | (al.convert(al.bitcast(a1[1], al.u16), al.u32) << 16)
            kr_word2 = al.convert(al.bitcast(a1[2], al.u16), al.u32) | (al.convert(al.bitcast(a1[3], al.u16), al.u32) << 16)
            kr_word1 = al.amdgpu.perm(kr_word1, kr_word1, 0x03020100)
            kr_word2 = al.amdgpu.perm(kr_word2, kr_word2, 0x03020100)
            a1[0] = al.bitcast(al.convert(kr_word1 & 0xFFFF, al.u16), al.bf16)
            a1[1] = al.bitcast(al.convert(kr_word1 >> 16, al.u16), al.bf16)
            a1[2] = al.bitcast(al.convert(kr_word2 & 0xFFFF, al.u16), al.bf16)
            a1[3] = al.bitcast(al.convert(kr_word2 >> 16, al.u16), al.bf16)
            al.amdgpu.s_nop(1)
            if k_step == 0:
                acc_kr1 = al.amdgpu.mfma_16x16x16_bf16_f32_vgpr_zero(b, a1)
                acc_kr2 = al.amdgpu.mfma_16x16x16_bf16_f32_vgpr_zero(b, a2)
            else:
                acc_kr1 = al.amdgpu.mfma_16x16x16_bf16_f32_vgpr(b, a1, acc_kr1)
                acc_kr2 = al.amdgpu.mfma_16x16x16_bf16_f32_vgpr(b, a2, acc_kr2)
        acc_kr1 = al.amdgpu.mfma_hazard_drain(acc_kr1, 6)
        acc_kr2 = al.amdgpu.mfma_hazard_drain(acc_kr2, 6)

        for acc_i in al.range(4):
            h1[acc_i] = h1[acc_i] * gt[ws_idx, h_row] + acc_kr1[acc_i]
            h2[acc_i] = h2[acc_i] * gt[ws_idx, 64 + h_row] + acc_kr2[acc_i]
            if DEBUG:
                debug[((head * TRACE_PHASES + 6) * HD + h_row) * V + w0 + h_col_base + acc_i] = h1[acc_i]
                debug[((head * TRACE_PHASES + 7) * HD + h_row) * V + w0 + h_col_base + acc_i] = h2[acc_i]
        al.syncthreads()
        chunk_idx += 1

    for acc_i in al.range(4):
        final_state[head, h_row, w0 + h_col_base + acc_i] = h1[acc_i]
        final_state[head, 64 + h_row, w0 + h_col_base + acc_i] = h2[acc_i]


def a70_bw16_k2(*, workspace, v, beta, initial_state, debug: bool = False):
    """Launch one persistent `(8,12,1)` A69 K2 grid."""
    T = int(v.shape[1])
    nt = (T + C - 1) // C
    out = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    nt_scalar = torch.tensor([[nt]], device=v.device, dtype=torch.int32)
    debug_data = torch.zeros((HEADS * TRACE_PHASES * HD * V,), device=v.device, dtype=torch.float32)
    _a70_bw16_k2_persistent[lambda: ((V // BW, HEADS, 1), (WG, 1, 1))](
        workspace.kd,
        workspace.qd,
        workspace.kr,
        workspace.gt,
        workspace.inv_mqk,
        v,
        beta,
        initial_state,
        nt_scalar,
        out,
        final_state,
        debug_data,
        T,
        nt,
        debug,
        num_warps=4,
    )
    return out, final_state, debug_data if debug else None

