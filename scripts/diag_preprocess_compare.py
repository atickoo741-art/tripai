"""Compare preprocess kernel output vs Python reference step-by-step."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import peqat_cuda

torch.manual_seed(42)

M, K, N = 1, 4096, 4096

x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
smooth = (torch.randn(K, dtype=torch.bfloat16, device="cuda").abs() + 0.5).contiguous()
clip = (torch.randn(K, dtype=torch.bfloat16, device="cuda").abs() + 0.5).contiguous()

# ---- Kernel preprocess ----
x_packed, s_x = peqat_cuda.preprocess_per_token_forward(x, smooth, clip)

# ---- Python reference preprocess ----
x_f32 = x.float()
smooth_f32 = smooth.float()
clip_f32 = clip.float()

x_scaled = x_f32 / smooth_f32
x_clamped = x_scaled.clamp(-clip_f32, clip_f32)

x_abs_max = x_clamped.abs().amax(dim=1)
s_x_ref = (x_abs_max / 7.0).clamp(min=1e-8)

x_q_ref = (x_clamped / s_x_ref[:, None]).round().clamp(-8, 7).to(torch.int8)

# Check s_x
s_x_diff = (s_x - s_x_ref).abs().max().item()
print(f"s_x max diff: {s_x_diff:.10f}")
print(f"  kernel s_x: {s_x[0].item():.10f}")
print(f"  ref    s_x: {s_x_ref[0].item():.10f}")

if s_x_diff > 1e-6:
    print("  *** s_x MISMATCH! ***")
else:
    print("  s_x OK")

# Repack 4→8
x_packed_8 = peqat_cuda.repack_4_to_8_forward(x_packed)

# Manual repack of reference
K8 = K // 8
x_packed_4_ref = torch.zeros(M, K // 4, dtype=torch.int32, device="cuda")
for k in range(0, K, 4):
    grp = k // 4
    word = 0
    for i in range(4):
        v = int(x_q_ref[0, k + i].item()) & 0xF
        word |= v << (i * 4)
    x_packed_4_ref[0, grp] = word

x_packed_8_ref = torch.zeros(M, K // 8, dtype=torch.int32, device="cuda")
for i in range(K8):
    lo = x_packed_4_ref[0, i * 2].item() & 0xFFFF
    hi = x_packed_4_ref[0, i * 2 + 1].item() & 0xFFFF
    val = (hi << 16) | lo
    if val >= 0x80000000:
        val -= 0x100000000
    x_packed_8_ref[0, i] = val

# Compare packed values
packed_diff = (x_packed_8 != x_packed_8_ref).sum().item()
print(f"\nx_packed_8 mismatched words: {packed_diff} / {K8}")

if packed_diff > 0:
    # Find first mismatch
    for i in range(K8):
        if x_packed_8[0, i].item() != x_packed_8_ref[0, i].item():
            kv = i * 8
            k0 = kv // 4
            k1 = kv // 4 + 1
            print(f"  First mismatch at word {i} (K={kv}-{kv+7}):")
            print(f"    kernel: 0x{x_packed_8[0, i].item():08x}")
            print(f"    ref:    0x{x_packed_8_ref[0, i].item():08x}")
            # Unpack to see nibbles
            for nib in range(8):
                kv_nib = kv + nib
                k_val = x_packed_8[0, i].item()
                r_val = x_packed_8_ref[0, i].item()
                k_nib = (k_val >> (nib * 4)) & 0xF
                r_nib = (r_val >> (nib * 4)) & 0xF
                if k_nib != r_nib:
                    print(f"    nibble {nib} (K={kv_nib}): kernel=0x{k_nib:x} ref=0x{r_nib:x}")
                    print(f"      x_clamped_kernel={x_clamped[0, kv_nib].item():.6f}")
                    print(f"      x_clamped_ref={x_clamped[0, kv_nib].item():.6f}")
                    print(f"      x_q_ref[{kv_nib}]={x_q_ref[0, kv_nib].item()}")
                    # Recompute
                    scaled = x_clamped[0, kv_nib].item() / s_x_ref[0].item()
                    q_py = round(scaled)
                    print(f"      scaled={scaled:.6f} round={q_py} clamp={max(-8, min(7, q_py))}")
            break
else:
    print("  x_packed_8 matches perfectly!")
    # Now test MMA
    w = torch.randint(-8, 7, (N, K), dtype=torch.int8, device="cuda")
    w_scale = torch.randn(N, dtype=torch.float32, device="cuda").abs() + 0.1

    # Pack weights
    packed_u8 = (w[:, ::2].int() & 0xF).to(torch.uint8) | (((w[:, 1::2].int() & 0xF) << 4).to(torch.uint8))
    lo = (packed_u8 & 0xF).to(torch.int8)
    hi = ((packed_u8 >> 4) & 0xF).to(torch.int8)
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)
    w_q_ref = torch.empty(N, K, dtype=torch.int8, device="cuda")
    w_q_ref[:, ::2] = lo
    w_q_ref[:, 1::2] = hi
    w_q_ref = w_q_ref[:, :K]

    w_i32 = w_q_ref.to(torch.int32) & 0xF
    K8_w = K // 8
    w_groups = w_i32[:, :K8_w * 8].reshape(N, K8_w, 8)
    w_packed_tc = (w_groups[:, :, 0] | (w_groups[:, :, 1] << 4) | (w_groups[:, :, 2] << 8)
                   | (w_groups[:, :, 3] << 12) | (w_groups[:, :, 4] << 16)
                   | (w_groups[:, :, 5] << 20) | (w_groups[:, :, 6] << 24)
                   | (w_groups[:, :, 7] << 28))

    # Reference matmul
    x_dq = x_q_ref.float() * s_x_ref[:, None]
    w_dq = w_q_ref.float() * w_scale[:, None]
    y_ref = x_dq @ w_dq.T

    # Kernel MMA
    if M < 16:
        pad = 16 - M
        x_8_padded = torch.cat([x_packed_8, torch.zeros(pad, K8, dtype=torch.int32, device="cuda")], dim=0)
        s_x_padded = torch.cat([s_x, torch.ones(pad, device="cuda")], dim=0)
    else:
        x_8_padded = x_packed_8
        s_x_padded = s_x

    y_tc = peqat_cuda.fused_int4_tc_per_token_forward(
        x_8_padded, w_packed_tc, w_scale.float(), s_x_padded, N, K
    )
    y_tc = y_tc[:M, :N].contiguous()

    # Compare
    diff = (y_ref.bfloat16().float() - y_tc.float()).abs()
    max_err = diff.max().item()
    cos = torch.nn.CosineSimilarity(dim=-1)
    sim = cos(y_ref.float().flatten(), y_tc.float().flatten()).item()

    print(f"\n  MMA result: cos_sim={sim:.6f} max_err={max_err:.4f}")

    if max_err > 1.0:
        # Find worst elements
        max_idx = diff.argmax().item()
        r = max_idx // N
        c = max_idx % N
        print(f"  Worst diff at y[{r},{c}]: kernel={y_tc[0,c].item():.4f} ref={y_ref[0,c].item():.4f}")
        # Check which K elements contribute to this output
        acc_kernel = y_tc[0, c].item() / (s_x[0].item() * w_scale[c].item())
        acc_ref = y_ref[0, c].item() / (s_x_ref[0].item() * w_scale[c].item())
        print(f"  Unscaled acc: kernel={acc_kernel:.1f} ref={acc_ref:.1f}")
        # Check per-tile (K=64) contributions
        for kt in range(K // 64):
            k_start = kt * 64
            ref_sum = 0
            for k in range(k_start, k_start + 64):
                ref_sum += x_q_ref[0, k].item() * w_q_ref[c, k].item()
            print(f"    K tile {kt} ({k_start}-{k_start+63}): ref_sum={ref_sum}")
