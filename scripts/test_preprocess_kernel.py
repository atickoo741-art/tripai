import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csrc"))
from peqat_deploy.quant.reference import peqat_linear_reference


def preprocess_ref(x: torch.Tensor, smooth: torch.Tensor, clip: torch.Tensor):
    qmax = 7
    x_s = x / smooth.unsqueeze(0)
    x_c = torch.clamp(x_s, -clip, clip)
    s_x = clip / qmax
    x_q = torch.clamp(torch.round(x_c / s_x), -8, 7).to(torch.int8)
    x_dq = x_q.float() * s_x
    return x_q, x_dq, s_x


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 1

    try:
        import peqat_cuda
    except ImportError:
        print("FAIL: peqat_cuda extension not built. Run:")
        print("  cd csrc && python setup.py build_ext --inplace")
        return 1

    torch.manual_seed(42)
    N, K = 4, 4096
    x = torch.randn(N, K, device="cuda", dtype=torch.float32)
    smooth = (torch.randn(K, device="cuda", dtype=torch.float32).abs() + 0.5)
    clip = (torch.randn(K, device="cuda", dtype=torch.float32).abs() * 3.0 + 0.5)

    x_q_ref, x_dq_ref, x_scale_ref = preprocess_ref(x, smooth, clip)
    x_q_cu, x_dq_cu, x_scale_cu = peqat_cuda.preprocess_forward(x, smooth, clip)

    atol = 1e-5
    q_match = torch.all(x_q_ref == x_q_cu)
    dq_max_err = (x_dq_ref - x_dq_cu).abs().max().item()
    scale_max_err = (x_scale_ref - x_scale_cu).abs().max().item()
    dq_allclose = torch.allclose(x_dq_ref, x_dq_cu, atol=atol)

    print(f"x_q exact match:              {q_match}")
    print(f"x_dq allclose (atol={atol}):  {dq_allclose}")
    print(f"x_dq max absolute error:      {dq_max_err:.2e}")
    print(f"x_scale max absolute error:   {scale_max_err:.2e}")

    if dq_allclose and q_match:
        print("\nPASS: preprocess kernel matches reference")
    else:
        mismatch = (x_q_ref != x_q_cu).sum().item()
        print(f"\nPARTIAL: x_q mismatches: {mismatch}/{N*K} elements")
        if not dq_allclose:
            print(f"         x_dq exceeds tolerance (atol={atol})")
        print("FAIL: preprocess kernel does NOT match reference")

    return 0


if __name__ == "__main__":
    sys.exit(main())
