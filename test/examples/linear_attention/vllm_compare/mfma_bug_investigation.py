import sys
import torch
import avelang
import avelang.language as al
from prototype_qwen_gdn_mfma_delta_staged import delta_state_staged_with_dummy

BT, BV = 16, 16

def check_result(name, actual, expected):
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max().item()
    print(f"test name: {name}")
    print("expected: max_abs == 0.0")
    print(f"actual max_abs: {max_abs}")
    print("per tile max_abs: ", end="")
    tiles = []
    for base in range(0, 128, 16):
        tile_abs = diff[:, base : base + 16].max().item()
        tiles.append(f"{base}:{base+16}={tile_abs:.9g}")
    print(", ".join(tiles))
    passed = max_abs < 1e-3
    print(f"pass/fail: {'PASS' if passed else 'FAIL'}")
    return passed

# ================= Experiment 1: all_shared_top =================
@avelang.jit
def ext1_kernel(v_ptr: al.Pointer(al.bf16), k_ptr: al.Pointer(al.bf16), init_ptr: al.Pointer(al.f32), out_ptr: al.Pointer(al.f32), total_k: al.constexpr, scale: al.constexpr):
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((BT, BV), (BV, 1)))
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((BT, total_k), (total_k, 1)))
    init = al.make_tensor(init_ptr, al.f32, al.make_layout((BV, total_k), (total_k, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((BV, total_k), (total_k, 1)))
    lane = al.thread_id(0); lane_col = lane & 15; lane_group = lane >> 4

    state = al.make_shared((BV, 128), al.f32)
    h0 = al.make_shared((BV, 64), al.bf16)
    h1 = al.make_shared((BV, 64), al.bf16)
    w0 = al.make_shared((BT, 64), al.bf16)
    w1 = al.make_shared((BT, 64), al.bf16)
    v_t2 = al.make_shared((BV, BT), al.bf16)
    k_t2 = al.make_shared((128, BT), al.bf16)

    h0_v = al.view(h0, al.i32, al.make_layout((BV, 8, 4), (32, 4, 1)))
    h1_v = al.view(h1, al.i32, al.make_layout((BV, 8, 4), (32, 4, 1)))
    w0_v = al.view(w0, al.i32, al.make_layout((BT, 8, 4), (32, 4, 1)))
    w1_v = al.view(w1, al.i32, al.make_layout((BT, 8, 4), (32, 4, 1)))
    v_v2 = al.view(v_t2, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    k_v2 = al.view(k_t2, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))

    for rep in al.range(16):
        idx = lane + rep * 64; vv = idx // 64; kk = idx - vv * 64
        h0[vv, kk] = al.convert(init[vv, kk], al.bf16)
        h1[vv, kk] = al.convert(init[vv, kk + 64], al.bf16)
        w0[vv, kk] = k[vv, kk]
        w1[vv, kk] = k[vv, kk + 64]

    for rep in al.range(32):
        idx = lane + rep * 64; row = idx // 128; col = idx - row * 128
        state[row, col] = al.convert(init[row, col] * scale, al.f32)
    al.syncthreads()

    pred_acc = al.full((4,), 0.0, al.f32)
    for batch in al.range(2):
        kv = lane_group + batch * 4
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(w0_v[lane_col, kv], al.Tensor((2,4,1), al.bf16))[0], al.view(h0_v[lane_col, kv], al.Tensor((2,4,1), al.bf16))[0], pred_acc)
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(w0_v[lane_col, kv], al.Tensor((2,4,1), al.bf16))[1], al.view(h0_v[lane_col, kv], al.Tensor((2,4,1), al.bf16))[1], pred_acc)
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(w1_v[lane_col, kv], al.Tensor((2,4,1), al.bf16))[0], al.view(h1_v[lane_col, kv], al.Tensor((2,4,1), al.bf16))[0], pred_acc)
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(w1_v[lane_col, kv], al.Tensor((2,4,1), al.bf16))[1], al.view(h1_v[lane_col, kv], al.Tensor((2,4,1), al.bf16))[1], pred_acc)
    al.syncthreads()

    for rep in al.range(32):
        idx = lane + rep * 64; row = idx // 16; tok = idx - row * 16
        k_t2[row, tok] = k[tok, row]
        if row < 16: v_t2[row, tok] = v[tok, row]
    al.syncthreads()

    for tile in al.range(8):
        base = tile * 16; acc2 = al.full((4,), 0.0, al.f32)
        if lane_group == 0:
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(v_v2[lane_col, 0], al.Tensor((2,4,1), al.bf16))[0], al.view(k_v2[base + lane_col, 0], al.Tensor((2,4,1), al.bf16))[0], acc2)
        if lane_group == 1:
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(v_v2[lane_col, 0], al.Tensor((2,4,1), al.bf16))[1], al.view(k_v2[base + lane_col, 0], al.Tensor((2,4,1), al.bf16))[1], acc2)
        if lane_group == 2:
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(v_v2[lane_col, 1], al.Tensor((2,4,1), al.bf16))[0], al.view(k_v2[base + lane_col, 1], al.Tensor((2,4,1), al.bf16))[0], acc2)
        if lane_group == 3:
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(v_v2[lane_col, 1], al.Tensor((2,4,1), al.bf16))[1], al.view(k_v2[base + lane_col, 1], al.Tensor((2,4,1), al.bf16))[1], acc2)
        for r in al.range(4):
            state[lane_group * 4 + r, base + lane_col] = state[lane_group * 4 + r, base + lane_col] + acc2[r]
        al.syncthreads()

    for rep in al.range(32):
        idx = lane + rep * 64; row = idx // 128; col = idx - row * 128
        out[row, col] = state[row, col]

# ================= Experiment 2: no_dead_store =================
@avelang.jit
def ext2_kernel(v_ptr: al.Pointer(al.bf16), k_ptr: al.Pointer(al.bf16), init_ptr: al.Pointer(al.f32), out_ptr: al.Pointer(al.f32), total_k: al.constexpr, scale: al.constexpr):
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((BT, BV), (BV, 1)))
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((BT, total_k), (total_k, 1)))
    init = al.make_tensor(init_ptr, al.f32, al.make_layout((BV, total_k), (total_k, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((BV, total_k), (total_k, 1)))
    lane = al.thread_id(0); lane_col = lane & 15; lane_group = lane >> 4

    state = al.make_shared((BV, 128), al.f32)
    h0 = al.make_shared((BV, 64), al.bf16)
    w0 = al.make_shared((BT, 64), al.bf16)
    
    h0_v = al.view(h0, al.i32, al.make_layout((BV, 8, 4), (32, 4, 1)))
    w0_v = al.view(w0, al.i32, al.make_layout((BT, 8, 4), (32, 4, 1)))

    for rep in al.range(8):
        idx = lane + rep * 64; vv = idx // 64; kk = idx - vv * 64
        h0[vv, kk] = al.convert(init[vv, kk], al.bf16)
        w0[vv, kk] = k[vv, kk]

    for rep in al.range(32):
        idx = lane + rep * 64; row = idx // 128; col = idx - row * 128
        state[row, col] = al.convert(init[row, col] * scale, al.f32)
    al.syncthreads()

    pred_acc = al.full((4,), 0.0, al.f32)
    for batch in al.range(2):
        kv = lane_group + batch * 4
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(w0_v[lane_col, kv], al.Tensor((2,4,1), al.bf16))[0], al.view(h0_v[lane_col, kv], al.Tensor((2,4,1), al.bf16))[0], pred_acc)
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(w0_v[lane_col, kv], al.Tensor((2,4,1), al.bf16))[1], al.view(h0_v[lane_col, kv], al.Tensor((2,4,1), al.bf16))[1], pred_acc)
    al.syncthreads()

    v_t2 = al.make_shared((BV, BT), al.bf16)
    k_t2 = al.make_shared((128, BT), al.bf16)
    v_v2 = al.view(v_t2, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    k_v2 = al.view(k_t2, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))

    for rep in al.range(32):
        idx = lane + rep * 64; row = idx // 16; tok = idx - row * 16
        k_t2[row, tok] = k[tok, row]
        if row < 16: v_t2[row, tok] = v[tok, row]
    al.syncthreads()

    for tile in al.range(8):
        base = tile * 16; acc2 = al.full((4,), 0.0, al.f32)
        if lane_group == 0:
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(v_v2[lane_col, 0], al.Tensor((2,4,1), al.bf16))[0], al.view(k_v2[base + lane_col, 0], al.Tensor((2,4,1), al.bf16))[0], acc2)
        if lane_group == 1:
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(v_v2[lane_col, 0], al.Tensor((2,4,1), al.bf16))[1], al.view(k_v2[base + lane_col, 0], al.Tensor((2,4,1), al.bf16))[1], acc2)
        if lane_group == 2:
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(v_v2[lane_col, 1], al.Tensor((2,4,1), al.bf16))[0], al.view(k_v2[base + lane_col, 1], al.Tensor((2,4,1), al.bf16))[0], acc2)
        if lane_group == 3:
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(v_v2[lane_col, 1], al.Tensor((2,4,1), al.bf16))[1], al.view(k_v2[base + lane_col, 1], al.Tensor((2,4,1), al.bf16))[1], acc2)
        for r in al.range(4):
            state[lane_group * 4 + r, base + lane_col] = state[lane_group * 4 + r, base + lane_col] + acc2[r]
        al.syncthreads()

    for rep in al.range(32):
        idx = lane + rep * 64; row = idx // 128; col = idx - row * 128
        out[row, col] = state[row, col]

# ================= Experiment 4: padded_shared =================
@avelang.jit
def ext4_kernel(v_ptr: al.Pointer(al.bf16), k_ptr: al.Pointer(al.bf16), init_ptr: al.Pointer(al.f32), out_ptr: al.Pointer(al.f32), total_k: al.constexpr, scale: al.constexpr):
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((BT, BV), (BV, 1)))
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((BT, total_k), (total_k, 1)))
    init = al.make_tensor(init_ptr, al.f32, al.make_layout((BV, total_k), (total_k, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((BV, total_k), (total_k, 1)))
    lane = al.thread_id(0); lane_col = lane & 15; lane_group = lane >> 4

    state = al.make_shared((BV, 160), al.f32)
    h0 = al.make_shared((BV, 80), al.bf16)
    w0 = al.make_shared((BT, 80), al.bf16)
    v_t2 = al.make_shared((BV, 32), al.bf16)
    k_t2 = al.make_shared((160, BT), al.bf16)

    h0_v = al.view(h0, al.i32, al.make_layout((BV, 8, 4), (40, 4, 1)))
    w0_v = al.view(w0, al.i32, al.make_layout((BT, 8, 4), (40, 4, 1)))
    v_v2 = al.view(v_t2, al.i32, al.make_layout((BV, 2, 4), (16, 4, 1)))
    k_v2 = al.view(k_t2, al.i32, al.make_layout((128, 2, 4), (16, 4, 1)))

    for rep in al.range(8):
        idx = lane + rep * 64; vv = idx // 64; kk = idx - vv * 64
        h0[vv, kk] = al.convert(init[vv, kk], al.bf16)
        w0[vv, kk] = k[vv, kk]

    for rep in al.range(32):
        idx = lane + rep * 64; row = idx // 128; col = idx - row * 128
        state[row, col] = al.convert(init[row, col] * scale, al.f32)
    al.syncthreads()

    pred_acc = al.full((4,), 0.0, al.f32)
    for batch in al.range(2):
        kv = lane_group + batch * 4
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(w0_v[lane_col, kv], al.Tensor((2,4,1), al.bf16))[0], al.view(h0_v[lane_col, kv], al.Tensor((2,4,1), al.bf16))[0], pred_acc)
    al.syncthreads()

    for rep in al.range(32):
        idx = lane + rep * 64; row = idx // 16; tok = idx - row * 16
        k_t2[row, tok] = k[tok, row]
        if row < 16: v_t2[row, tok] = v[tok, row]
    al.syncthreads()

    for tile in al.range(8):
        base = tile * 16; acc2 = al.full((4,), 0.0, al.f32)
        if lane_group == 0:
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(v_v2[lane_col, 0], al.Tensor((2,4,1), al.bf16))[0], al.view(k_v2[base + lane_col, 0], al.Tensor((2,4,1), al.bf16))[0], acc2)
        if lane_group == 1:
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(v_v2[lane_col, 0], al.Tensor((2,4,1), al.bf16))[1], al.view(k_v2[base + lane_col, 0], al.Tensor((2,4,1), al.bf16))[1], acc2)
        if lane_group == 2:
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(v_v2[lane_col, 1], al.Tensor((2,4,1), al.bf16))[0], al.view(k_v2[base + lane_col, 1], al.Tensor((2,4,1), al.bf16))[0], acc2)
        if lane_group == 3:
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(v_v2[lane_col, 1], al.Tensor((2,4,1), al.bf16))[1], al.view(k_v2[base + lane_col, 1], al.Tensor((2,4,1), al.bf16))[1], acc2)
        for r in al.range(4):
            state[lane_group * 4 + r, base + lane_col] = state[lane_group * 4 + r, base + lane_col] + acc2[r]
        al.syncthreads()

    for rep in al.range(32):
        idx = lane + rep * 64; row = idx // 128; col = idx - row * 128
        out[row, col] = state[row, col]

# ================= Experiment 5: noop_no_mfma =================
@avelang.jit
def ext5_kernel(v_ptr: al.Pointer(al.bf16), k_ptr: al.Pointer(al.bf16), init_ptr: al.Pointer(al.f32), out_ptr: al.Pointer(al.f32), total_k: al.constexpr, scale: al.constexpr):
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((BT, BV), (BV, 1)))
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((BT, total_k), (total_k, 1)))
    init = al.make_tensor(init_ptr, al.f32, al.make_layout((BV, total_k), (total_k, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((BV, total_k), (total_k, 1)))
    lane = al.thread_id(0); lane_col = lane & 15; lane_group = lane >> 4

    state = al.make_shared((BV, 128), al.f32)
    h0 = al.make_shared((BV, 64), al.bf16)
    w0 = al.make_shared((BT, 64), al.bf16)
    v_t2 = al.make_shared((BV, BT), al.bf16)
    k_t2 = al.make_shared((128, BT), al.bf16)

    h0_v = al.view(h0, al.i32, al.make_layout((BV, 8, 4), (32, 4, 1)))
    w0_v = al.view(w0, al.i32, al.make_layout((BT, 8, 4), (32, 4, 1)))
    v_v2 = al.view(v_t2, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    k_v2 = al.view(k_t2, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))

    for rep in al.range(8):
        idx = lane + rep * 64; vv = idx // 64; kk = idx - vv * 64
        h0[vv, kk] = al.convert(init[vv, kk], al.bf16)
        w0[vv, kk] = k[vv, kk]

    for rep in al.range(32):
        idx = lane + rep * 64; row = idx // 128; col = idx - row * 128
        state[row, col] = al.convert(init[row, col] * scale, al.f32)
    al.syncthreads()

    pred_acc = al.full((4,), 0.0, al.f32)
    for batch in al.range(2):
        kv = lane_group + batch * 4
        tmp_a = al.view(w0_v[lane_col, kv], al.Tensor((2,4,1), al.bf16))[0]
        tmp_b = al.view(h0_v[lane_col, kv], al.Tensor((2,4,1), al.bf16))[0]
        # NOOP MFMA -> scalar acc
        pred_acc[0] = pred_acc[0] + al.convert(tmp_a[0], al.f32) + al.convert(tmp_b[0], al.f32)
    al.syncthreads()

    for rep in al.range(32):
        idx = lane + rep * 64; row = idx // 16; tok = idx - row * 16
        k_t2[row, tok] = k[tok, row]
        if row < 16: v_t2[row, tok] = v[tok, row]
    al.syncthreads()

    for tile in al.range(8):
        base = tile * 16; acc2 = al.full((4,), 0.0, al.f32)
        if lane_group == 0:
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(v_v2[lane_col, 0], al.Tensor((2,4,1), al.bf16))[0], al.view(k_v2[base + lane_col, 0], al.Tensor((2,4,1), al.bf16))[0], acc2)
        if lane_group == 1:
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(v_v2[lane_col, 0], al.Tensor((2,4,1), al.bf16))[1], al.view(k_v2[base + lane_col, 0], al.Tensor((2,4,1), al.bf16))[1], acc2)
        if lane_group == 2:
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(v_v2[lane_col, 1], al.Tensor((2,4,1), al.bf16))[0], al.view(k_v2[base + lane_col, 1], al.Tensor((2,4,1), al.bf16))[0], acc2)
        if lane_group == 3:
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(al.view(v_v2[lane_col, 1], al.Tensor((2,4,1), al.bf16))[1], al.view(k_v2[base + lane_col, 1], al.Tensor((2,4,1), al.bf16))[1], acc2)
        for r in al.range(4):
            state[lane_group * 4 + r, base + lane_col] = state[lane_group * 4 + r, base + lane_col] + acc2[r]
        al.syncthreads()

    for rep in al.range(32):
        idx = lane + rep * 64; row = idx // 128; col = idx - row * 128
        out[row, col] = state[row, col] + pred_acc[0] * al.convert(0.0, al.f32)

def main():
    gen = torch.Generator(device='cuda').manual_seed(20260614)
    v = torch.randn((16,16),device='cuda',dtype=torch.bfloat16,generator=gen)
    k = torch.randn((16,128),device='cuda',dtype=torch.bfloat16,generator=gen)
    init = torch.randn((16,128),device='cuda',dtype=torch.float32,generator=gen)
    
    print("Computing expected dummy footprint...")
    out_dummy = torch.empty_like(init, dtype=torch.float32)
    out_dummy = delta_state_staged_with_dummy(v, k, init)
    
    print("--- Exp 1: all_shared_declared_at_top ---")
    res1 = run_and_check("all_shared_declared_at_top", ext1_kernel, v, k, init, 1.0, out_dummy)
    print("结论：排除或保留哪个假设 => " + ("保留 dynamic shared memory lifetime 问题" if not res1 else "排除 shared memory dynamic declare"))
    
    print("--- Exp 2: no_dead_store_before_pred ---")
    res2 = run_and_check("no_dead_store_before_pred", ext2_kernel, v, k, init, 1.0, out_dummy)
    print("结论：排除或保留哪个假设 => " + ("保留 shared liveness/dead store issue" if not res2 else "排除 dead store overlap"))
    
    print("--- Exp 4: padded_unique_shared_buffers ---")
    res4 = run_and_check("padded_unique_shared_buffers", ext4_kernel, v, k, init, 1.0, out_dummy)
    print("结论：排除或保留哪个假设 => " + ("保留 contiguous LDS layout/bank conflict" if not res4 else "排除 bank conflict/contiguous overlap"))
    
    print("--- Exp 5: no_op_pred_no_mfma ---")
    res5 = run_and_check("no_op_pred_no_mfma", ext5_kernel, v, k, init, 1.0, out_dummy)
    print("结论：排除或保留哪个假设 => " + ("如果是PASS, 强烈保留 C: Avelang MFMA lowering 指令触发了问题" if res5 else "问题出在 staging LDS 本身"))

def run_and_check(name, kernel, v, k, init, scale, expected):
    out = torch.empty_like(init, dtype=torch.float32)
    kernel[lambda: ((1, 1, 1), (64, 1, 1))](v.contiguous(), k.contiguous(), init.contiguous(), out, k.shape[1], float(scale))
    diff = (out.float() - expected.float()).abs()
    max_abs = diff.max().item()
    print(f"test name: {name}")
    print("expected: max_abs == 0.0")
    print(f"actual max_abs: {max_abs}")
    print("per tile max_abs: ", end="")
    tiles = []
    for base in range(0, 128, 16):
        tile_abs = diff[:, base : base + 16].max().item()
        tiles.append(f"{base}:{base+16}={tile_abs:.9g}")
    print(", ".join(tiles))
    passed = max_abs < 1e-3
    print(f"pass/fail: {'PASS' if passed else 'FAIL'}")
    return passed

if __name__ == '__main__':
    main()
