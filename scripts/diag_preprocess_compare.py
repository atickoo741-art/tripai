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

# Use KERNEL's s_x for reference to test bit-exactness
s_x_ref = s_x  # force match

# Quantize using kernel's s_x
x_q_ref = (x_clamped / s_x_ref[:, None]).round().clamp(-8, 7).to(torch.int8)

# Manual pack 4-per-int32 matching kernel format
x_packed_4_ref = torch.zeros(M, K // 4, dtype=torch.int32, device="cuda")
for k in range(0, K, 4):
    grp = k // 4
    word = 0
    for i in range(4):
        v = int(x_q_ref[0, k + i].item()) & 0xF
        word |= v << (i * 4)
    x_packed_4_ref[0, grp] = word

# Repack 4→8
K8 = K // 8
x_packed_8_ref = torch.zeros(M, K8, dtype=torch.int32, device="cuda")
for i in range(K8):
    lo = x_packed_4_ref[0, i * 2].item() & 0xFFFF
    hi = x_packed_4_ref[0, i * 2 + 1].item() & 0xFFFF
    val = (hi << 16) | lo
    if val >= 0x80000000:
        val -= 0x100000000
    x_packed_8_ref[0, i] = val

# Compare packed values
packed_diff = (x_packed != x_packed_4_ref).sum().item()
packed_8_diff = (peqat_cuda.repack_4_to_8_forward(x_packed) != x_packed_8_ref).sum().item()
print(f"x_packed diff (4-per-int32): {packed_diff} / {K//4}")
print(f"x_packed_8 diff (8-per-int32): {packed_8_diff} / {K8}")

if packed_diff > 0:
    print("\n*** Mismatch even with same s_x — kernel quantization differs from Python! ***")
    # Find first mismatch in raw 4-per-int32
    for i in range(K // 4):
        if x_packed[0, i].item() != x_packed_4_ref[0, i].item():
            k0 = i * 4
            print(f"  First 4-per-int32 mismatch at word {i} (K={k0}-{k0+3}):")
            print(f"    kernel: 0x{x_packed[0, i].item():08x}")
            print(f"    ref:    0x{x_packed_4_ref[0, i].item():08x}")
            for nib in range(4):
                kv = k0 + nib
                k_nib = (x_packed[0, i].item() >> (nib * 4)) & 0xF
                r_nib = (x_packed_4_ref[0, i].item() >> (nib * 4)) & 0xF
                if k_nib != r_nib:
                    xc = x_clamped[0, kv].item()
                    sx = s_x_ref[0].item()
                    scaled = xc / sx
                    print(f"    nibble {nib} (K={kv}): kernel=0x{k_nib:x} ref=0x{r_nib:x}")
                    print(f"      x_clamped={xc:.10f} s_x={sx:.10f} scaled={scaled:.10f}")
                    print(f"      rintf would give: {(int)(scaled + 0.5) if scaled >= 0 else (int)(scaled - 0.5)}")
            break
elif packed_8_diff > 0:
    print("Raw 4-per-int32 matches, but 8-per-int32 differs (repack bug?)")
else:
    print("\n*** ALL preprocess outputs match perfectly with matching s_x! ***")
    # Now test MMA with matching data
    w = torch.randint(-8, 7, (N, K), dtype=torch.int8, device="cuda")
    w_scale = torch.randn(N, dtype=torch.float32, device="cuda").abs() + 0.1

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
    x_packed_8 = peqat_cuda.repack_4_to_8_forward(x_packed)
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

    diff = (y_ref.bfloat16().float() - y_tc.float()).abs()
    max_err = diff.max().item()
    cos = torch.nn.CosineSimilarity(dim=-1)
    sim = cos(y_ref.float().flatten(), y_tc.float().flatten()).item()

    print(f"\nMMA result with matching s_x: cos_sim={sim:.6f} max_err={max_err:.4f}")
    if max_err < 1.0:
        print("*** MMA PASSES with matching s_x! ***")
