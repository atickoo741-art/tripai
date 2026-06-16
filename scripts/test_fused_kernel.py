"""Fused kernel correctness test — tiny shapes to fit low-GPU-memory scenarios."""
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csrc"))


def pack_weights(w_q):
    rows, cols = w_q.shape
    if cols % 2 != 0:
        w_q = torch.cat([w_q, torch.zeros(rows, 1, dtype=w_q.dtype, device=w_q.device)], dim=1)
        cols = cols + 1
    w_q_flat = w_q.flatten().int()
    w_q_clamped = torch.clamp(w_q_flat, -8, 7).reshape(rows, cols)
    low = (w_q_clamped[:, ::2] & 0xF).to(torch.uint8)
    high = ((w_q_clamped[:, 1::2] & 0xF) << 4).to(torch.uint8)
    return (low | high).contiguous()


def reference_linear(x_dq, w_q, w_scale):
    w_dq = w_q.float() * w_scale.unsqueeze(1).float()
    return x_dq.float() @ w_dq.T


def test_synthetic():
    torch.manual_seed(42)
    # Tiny shapes — each <1MB GPU
    shapes = [
        (128, 256),
        (256, 128),
        (64, 64),
        (1, 256),     # decode-like M=1
        (8, 128),
    ]

    import peqat_cuda

    for D, K in shapes:
        w_f32 = torch.randn(D, K, device="cuda", dtype=torch.float32)
        qmax = 7
        w_scale = (w_f32.abs().amax(dim=1) / qmax).clamp(min=1e-8)
        w_q = torch.clamp(torch.round(w_f32 / w_scale.unsqueeze(1)), -8, 7).to(torch.int8)
        packed = pack_weights(w_q)

        x_dq = torch.randn(4, K, device="cuda", dtype=torch.float32)
        y_ref = reference_linear(x_dq, w_q, w_scale)

        y_fused = peqat_cuda.fused_linear_forward(
            x_dq, packed, w_scale.to(torch.float16), D, K
        )

        max_err = (y_ref - y_fused).abs().max().item()
        allclose = torch.allclose(y_ref, y_fused, atol=1e-4, rtol=1e-3)

        if not allclose:
            print(f"  FAIL [{D:>4}, {K:>4}] max_err={max_err:.2e}")
            return False
        print(f"  OK  [{D:>4}, {K:>4}] max_err={max_err:.2e}")

    return True


def test_fused_vs_old():
    """Fused kernel must match existing unpack+cuBLAS path."""
    import peqat_cuda

    torch.manual_seed(123)
    shapes = [(128, 256), (256, 256), (64, 128), (1, 256)]

    for D, K in shapes:
        w_f32 = torch.randn(D, K, device="cuda", dtype=torch.float32)
        qmax = 7
        w_scale = (w_f32.abs().amax(dim=1) / qmax).clamp(min=1e-8)
        w_q = torch.clamp(torch.round(w_f32 / w_scale.unsqueeze(1)), -8, 7).to(torch.int8)
        packed = pack_weights(w_q)
        w_scale_f16 = w_scale.to(torch.float16)

        x_dq = torch.randn(4, K, device="cuda", dtype=torch.float32)

        y_old = peqat_cuda.linear_forward(x_dq, packed, w_scale_f16, D, K)
        y_fused = peqat_cuda.fused_linear_forward(x_dq, packed, w_scale_f16, D, K)

        max_err = (y_old - y_fused).abs().max().item()
        ok = torch.allclose(y_old, y_fused, atol=1e-4, rtol=1e-3)

        if not ok:
            print(f"  FAIL vs-old [{D:>4}, {K:>4}] max_err={max_err:.2e}")
            return False
        print(f"  OK  vs-old [{D:>4}, {K:>4}] max_err={max_err:.2e}")

    return True


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1

    free, total = torch.cuda.mem_get_info()
    print(f"GPU memory: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")

    try:
        import peqat_cuda
    except ImportError:
        print("peqat_cuda not built. Run: cd csrc && python3 setup.py build_ext --inplace")
        return 1

    print("1. Synthetic shapes vs reference:")
    ok = test_synthetic()

    print("\n2. Fused vs old unpack+cuBLAS path:")
    ok = test_fused_vs_old() and ok

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
