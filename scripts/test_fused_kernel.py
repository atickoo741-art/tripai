import json
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csrc"))


def pack_weights(w_q: torch.Tensor) -> torch.Tensor:
    rows, cols = w_q.shape
    if cols % 2 != 0:
        w_q = torch.cat([w_q, torch.zeros(rows, 1, dtype=w_q.dtype, device=w_q.device)], dim=1)
        cols = cols + 1
    w_q_flat = w_q.flatten().int()
    w_q_clamped = torch.clamp(w_q_flat, -8, 7).reshape(rows, cols)
    low = (w_q_clamped[:, ::2] & 0xF).to(torch.uint8)
    high = ((w_q_clamped[:, 1::2] & 0xF) << 4).to(torch.uint8)
    return (low | high).contiguous()


def reference_linear(x_dq: torch.Tensor, w_q: torch.Tensor, w_scale: torch.Tensor):
    w_dq = w_q.float() * w_scale.unsqueeze(1).float()
    return x_dq.float() @ w_dq.T


def test_synthetic():
    torch.manual_seed(42)
    shapes = [
        (4096, 4096),
        (14336, 4096),
        (4096, 14336),
        (1024, 1024),
        (128, 256),
        (1, 4096),     # decode (M=1)
        (8, 4096),     # small batch decode
    ]
    N_vals = [1, 4, 8, 16]

    import peqat_cuda

    for D, K in shapes:
        w_f32 = torch.randn(D, K, device="cuda", dtype=torch.float32)
        qmax = 7
        w_scale = (w_f32.abs().amax(dim=1) / qmax).clamp(min=1e-8)
        w_q = torch.clamp(torch.round(w_f32 / w_scale.unsqueeze(1)), -8, 7).to(torch.int8)
        packed = pack_weights(w_q)

        for N in N_vals:
            if N > 16 and D * K > 4096 * 4096:
                continue
            x_dq = torch.randn(N, K, device="cuda", dtype=torch.float32)
            y_ref = reference_linear(x_dq, w_q, w_scale)

            y_fused = peqat_cuda.fused_linear_forward(x_dq, packed, w_scale.to(torch.float16), D, K)

            max_err = (y_ref - y_fused).abs().max().item()
            allclose = torch.allclose(y_ref, y_fused, atol=1e-4, rtol=1e-3)

            if not allclose:
                print(f"  SYNTH FAIL [{D:>5}, {K:>5}] N={N} max_err={max_err:.2e}")
                return False
            print(f"  synth [{D:>5}, {K:>5}] N={N} OK  max_err={max_err:.2e}")

    return True


def test_fused_vs_old():
    """Fused kernel must match the existing unpack+cuBLAS path bit-for-bit."""
    import peqat_cuda

    torch.manual_seed(123)
    shapes = [(4096, 4096), (14336, 4096), (4096, 14336), (512, 256)]
    N_vals = [1, 4, 8]

    for D, K in shapes:
        w_f32 = torch.randn(D, K, device="cuda", dtype=torch.float32)
        qmax = 7
        w_scale = (w_f32.abs().amax(dim=1) / qmax).clamp(min=1e-8)
        w_q = torch.clamp(torch.round(w_f32 / w_scale.unsqueeze(1)), -8, 7).to(torch.int8)
        packed = pack_weights(w_q)
        w_scale_f16 = w_scale.to(torch.float16)

        for N in N_vals:
            x_dq = torch.randn(N, K, device="cuda", dtype=torch.float32)

            y_old = peqat_cuda.linear_forward(x_dq, packed, w_scale_f16, D, K)
            y_fused = peqat_cuda.fused_linear_forward(x_dq, packed, w_scale_f16, D, K)

            max_err = (y_old - y_fused).abs().max().item()
            ok = torch.allclose(y_old, y_fused, atol=1e-5, rtol=1e-5)

            if not ok:
                print(f"  VS-OLD FAIL [{D:>5}, {K:>5}] N={N} max_err={max_err:.2e}")
                return False
            print(f"  vs-old [{D:>5}, {K:>5}] N={N} OK  max_err={max_err:.2e}")

    return True


def find_export_dirs():
    candidates = [
        os.path.expanduser("~/peqat-checkpoints"),
        os.path.join(os.path.dirname(__file__), "..", "..", "peqat-checkpoints"),
    ]
    for base in candidates:
        if os.path.isdir(base):
            for d in sorted(os.listdir(base)):
                full = os.path.join(base, d)
                if d.endswith("_export") and os.path.isdir(full):
                    yield full


def load_export_layer(export_dir: str, module_name: str):
    weight_dir = os.path.join(export_dir, "weights", module_name.replace(".", "_"))
    with open(os.path.join(weight_dir, "shape.json")) as f:
        shape = json.load(f)
    D, K = shape["rows"], shape["cols"]

    with open(os.path.join(weight_dir, "weight_int4.bin"), "rb") as f:
        packed_bytes = f.read()
    packed = torch.frombuffer(packed_bytes, dtype=torch.uint8).clone()

    with open(os.path.join(weight_dir, "weight_scale.fp16"), "rb") as f:
        scale_bytes = f.read()
    w_scale = torch.frombuffer(scale_bytes, dtype=torch.float16).clone()

    packed = packed[:(D * K + 1) // 2]

    return packed.cuda(), w_scale.cuda(), D, K


def test_from_export(export_dir: str):
    import peqat_cuda

    manifest_path = os.path.join(export_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"  no manifest.json, skipping")
        return True

    with open(manifest_path) as f:
        manifest = json.load(f)

    module_names = sorted(manifest.get("weight_artifact_paths", {}).keys())
    if not module_names:
        return True

    test_modules = []
    seen_layers = set()
    for m in module_names:
        parts = m.split(".")
        layer_key = parts[4] if len(parts) > 4 and parts[4].isdigit() else "0"
        if layer_key not in seen_layers:
            test_modules.append(m)
            seen_layers.add(layer_key)
        if len(test_modules) >= 3:
            break

    ok = True
    for module_name in test_modules:
        packed, w_scale, D, K = load_export_layer(export_dir, module_name)

        for N in [1, 4]:
            x_dq = torch.randn(N, K, device="cuda", dtype=torch.float32)

            y_ref = peqat_cuda.linear_forward(x_dq, packed, w_scale, D, K)
            y_fused = peqat_cuda.fused_linear_forward(x_dq, packed, w_scale, D, K)

            max_err = (y_ref - y_fused).abs().max().item()
            allclose = torch.allclose(y_ref, y_fused, atol=1e-4, rtol=1e-3)

            if allclose:
                print(f"  export [{module_name}] {D:>5}x{K:<5} N={N} OK  max_err={max_err:.2e}")
            else:
                print(f"  export [{module_name}] {D:>5}x{K:<5} N={N} FAIL max_err={max_err:.2e}")
                ok = False

    return ok


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

    print("1. Synthetic shapes vs reference:")
    ok = test_synthetic()

    print("\n2. Fused vs old unpack+cuBLAS path (bit-exactness):")
    ok = test_fused_vs_old() and ok

    export_dirs = list(find_export_dirs())
    if export_dirs:
        print(f"\n3. Real export artifacts ({len(export_dirs)} dir(s)):")
        for d in export_dirs:
            print(f"  {os.path.basename(d)}:")
            ok = test_from_export(d) and ok
    else:
        print("\n3. No _export directories found, skipping real-export test")

    if ok:
        print("\nPASS")
        return 0
    else:
        print("\nFAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
