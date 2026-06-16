import json
import struct
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
    """Self-contained test with synthetic data."""
    torch.manual_seed(42)
    shapes = [
        (4096, 4096),
        (14336, 4096),
        (4096, 14336),
        (1024, 1024),
    ]
    N = 4

    for D, K in shapes:
        w_f32 = torch.randn(D, K, device="cuda", dtype=torch.float32)
        qmax = 7
        w_scale = (w_f32.abs().amax(dim=1) / qmax).clamp(min=1e-8)
        w_q = torch.clamp(torch.round(w_f32 / w_scale.unsqueeze(1)), -8, 7).to(torch.int8)
        packed = pack_weights(w_q)

        x_dq = torch.randn(N, K, device="cuda", dtype=torch.float32)
        y_ref = reference_linear(x_dq, w_q, w_scale)

        import peqat_cuda
        y_cu = peqat_cuda.linear_forward(x_dq, packed, w_scale.to(torch.float16), D, K)

        max_err = (y_ref - y_cu).abs().max().item()
        allclose = torch.allclose(y_ref, y_cu, atol=1e-4, rtol=1e-3)

        if not allclose:
            print(f"SYNTHETIC FAIL [{D:>5}, {K:>5}] max_err={max_err:.2e}")
            return False
        print(f"  synthetic [{D:>5}, {K:>5}] OK  max_err={max_err:.2e}")
    return True


def find_export_dirs():
    """Find export directories in known locations."""
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
    """Load one layer's packed artifacts from an export directory."""
    weight_dir = os.path.join(export_dir, "weights", module_name.replace(".", "_"))
    if not os.path.isdir(weight_dir):
        raise FileNotFoundError(f"Weight dir not found: {weight_dir}")

    with open(os.path.join(weight_dir, "shape.json")) as f:
        shape = json.load(f)
    D, K = shape["rows"], shape["cols"]

    with open(os.path.join(weight_dir, "weight_int4.bin"), "rb") as f:
        packed_bytes = f.read()
    packed = torch.frombuffer(packed_bytes, dtype=torch.uint8).clone()

    with open(os.path.join(weight_dir, "weight_scale.fp16"), "rb") as f:
        scale_bytes = f.read()
    w_scale = torch.frombuffer(scale_bytes, dtype=torch.float16).clone()

    # Truncate padding: actual packed size is (D * K + 1) // 2
    packed = packed[:(D * K + 1) // 2]

    return packed.cuda(), w_scale.cuda(), D, K


def test_from_export(export_dir: str):
    """Test against actual exported artifacts."""
    import peqat_cuda

    manifest_path = os.path.join(export_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"  no manifest.json, skipping real export test")
        return True

    with open(manifest_path) as f:
        manifest = json.load(f)

    module_names = sorted(manifest.get("weight_artifact_paths", {}).keys())
    if not module_names:
        print(f"  no weight_artifact_paths in manifest")
        return True

    # Pick one module from each of up to 3 different layers
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

    N = 4
    ok = True
    for module_name in test_modules:
        packed, w_scale, D, K = load_export_layer(export_dir, module_name)

        x_dq = torch.randn(N, K, device="cuda", dtype=torch.float32)

        # Build reference: dequantize from packed
        low = (packed & 0xF).cpu()
        high = ((packed >> 4) & 0xF).cpu()
        interleaved = torch.stack([low, high], dim=1).flatten()
        w_q = interleaved[:D * K].reshape(D, K).to(torch.uint8)
        w_q = torch.where(w_q > 7, w_q - 16, w_q).to(torch.int8)
        y_ref = reference_linear(x_dq.cpu(), w_q, w_scale.cpu())

        y_cu = peqat_cuda.linear_forward(x_dq, packed, w_scale, D, K)

        max_err = (y_ref.cuda() - y_cu).abs().max().item()
        allclose = torch.allclose(y_ref.cuda(), y_cu, atol=1e-4, rtol=1e-3)

        if allclose:
            print(f"  export [{module_name}] {D:>5}x{K:<5} OK  max_err={max_err:.2e}")
        else:
            print(f"  export [{module_name}] {D:>5}x{K:<5} FAIL max_err={max_err:.2e}")
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

    print("Synthetic test:")
    ok = test_synthetic()

    export_dirs = list(find_export_dirs())
    if export_dirs:
        print(f"\nReal export test ({len(export_dirs)} dir(s) found):")
        for d in export_dirs:
            print(f"  {os.path.basename(d)}:")
            ok = test_from_export(d) and ok
    else:
        print("\nNo _export directories found, skipping real-export test")

    if ok:
        print("\nPASS")
        return 0
    else:
        print("\nFAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
