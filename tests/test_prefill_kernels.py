"""Phase 0 correctness suite for the prefill (and decode) kernels.

Needs a CUDA GPU and a peqat_cuda built from THIS repo's csrc/ (the import
path is printed in the pytest header by tests/conftest.py). Everything skips
cleanly without them.

What "exact" means here:
- Preprocess: the int4 codes and s_x must equal an fp32 torch reference
  bit for bit. The kernel and the reference do the same IEEE fp32 ops
  (x / smooth, clamp, max, / 7, x / s_x, round-half-even).
- GEMM / GEMV: the reference computes the INT32 accumulators with an fp32
  matmul. Every partial sum is an integer below K * 64 <= 704,512 < 2^24, so
  fp32 holds it exactly whatever the summation order (TF32 is switched off
  below). The epilogue is then replayed with the kernel's op order,
  (acc * s_x) * w_scale in fp32 and one round-to-nearest to bf16, so the
  bf16 outputs must be identical.

Run: pytest -q tests/test_prefill_kernels.py
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)
peqat_cuda = pytest.importorskip("peqat_cuda")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# The reference GEMM must be exact fp32, never TF32.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

M_LIST = [2, 7, 31, 128, 511, 512, 2048]
NK_LIST = [(4096, 4096), (12288, 4096), (11008, 4096), (4096, 11008)]
K_LIST = [4096, 11008]

DEV = "cuda"


# --------------------------------------------------------------------------
# Layout helpers: 8 consecutive K values per int32, q0 in bits 0-3.
# This is what preprocess_per_token_forward emits and what
# loader.CUDAQuantizedLinear._repack_for_tc builds for the weights.
# --------------------------------------------------------------------------

def pack8(codes):
    """int8 [R, K] in [-8, 7] -> int32 [R, K/8]."""
    R, K = codes.shape
    assert K % 8 == 0
    nib = (codes.to(torch.int64) & 0xF).reshape(R, K // 8, 8)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64, device=codes.device)
    words = (nib << shifts).sum(dim=-1)  # distinct bit fields, so sum == OR
    words = torch.where(words >= 2**31, words - 2**32, words)
    return words.to(torch.int32).contiguous()


def unpack8(words):
    """int32 [R, K/8] -> int8 [R, K]."""
    R, W = words.shape
    w = words.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(0, 32, 4, dtype=torch.int64, device=words.device)
    nib = (w.unsqueeze(-1) >> shifts) & 0xF
    nib = torch.where(nib > 7, nib - 16, nib)
    return nib.reshape(R, W * 8).to(torch.int8)


def pack_low_high(codes):
    """int8 [N, K] -> flat uint8 [N*K/2], even column in the low nibble
    (the weight_int4.bin layout that loader.load_export reads)."""
    lo = codes[:, 0::2].to(torch.int16) & 0xF
    hi = (codes[:, 1::2].to(torch.int16) & 0xF) << 4
    return (lo | hi).to(torch.uint8).reshape(-1).contiguous()


def gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def rand_codes(rows, cols, g):
    return torch.randint(-8, 8, (rows, cols), generator=g, device=DEV, dtype=torch.int8)


# --------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------

def ref_preprocess(x, smooth, clip):
    """fp32 replay of preprocess_per_token_kernel."""
    xs = x.float() / smooth.float()
    c = clip.float()
    xc = torch.minimum(torch.maximum(xs, -c), c)
    s_x = (xc.abs().amax(dim=1) / 7.0).clamp_min(1e-8)
    q = torch.round(xc / s_x[:, None]).clamp(-8, 7).to(torch.int8)
    return q, s_x


def ref_gemm(xq, wq, s_x, w_scale):
    """Exact INT32 accumulators, then the kernel's epilogue op order."""
    acc = xq.float() @ wq.float().t()
    return ((acc * s_x[:, None]) * w_scale[None, :]).to(torch.bfloat16)


def assert_bf16_equal(y, y_ref, what):
    assert y.shape == y_ref.shape, f"{what}: shape {tuple(y.shape)} != {tuple(y_ref.shape)}"
    if torch.equal(y, y_ref):
        return
    diff = y.float() - y_ref.float()
    bad = (y != y_ref)
    idx = bad.nonzero()[0].tolist()
    raise AssertionError(
        f"{what}: {bad.sum().item()} of {y.numel()} outputs differ; "
        f"max |diff| {diff.abs().max().item():.6g}; first at {idx}: "
        f"got {y[tuple(idx)].item()} want {y_ref[tuple(idx)].item()}"
    )


def make_activation_params(K, g):
    smooth = (torch.rand(K, generator=g, device=DEV) + 0.1).bfloat16()
    clip = (torch.rand(K, generator=g, device=DEV) * 5.0 + 0.01).bfloat16()
    return smooth, clip


# --------------------------------------------------------------------------
# Layout sanity
# --------------------------------------------------------------------------

def test_pack8_roundtrip():
    g = gen(1)
    codes = rand_codes(33, 4096, g)
    assert torch.equal(unpack8(pack8(codes)), codes)


@pytest.mark.parametrize("N,K", [(4096, 4096), (4096, 11008)])
def test_pack8_matches_loader_repack(N, K):
    loader = pytest.importorskip("peqat_deploy.runtime.loader")
    g = gen(2 + N + K)
    wq = rand_codes(N, K, g)
    repacked = loader.CUDAQuantizedLinear._repack_for_tc(pack_low_high(wq), N, K)
    assert torch.equal(repacked.to(torch.int32), pack8(wq))


# --------------------------------------------------------------------------
# Per-token preprocess
# --------------------------------------------------------------------------

@pytest.mark.parametrize("K", K_LIST)
@pytest.mark.parametrize("M", M_LIST)
def test_preprocess_per_token_exact(M, K):
    g = gen(100 + M * 7 + K)
    x = (torch.randn(M, K, generator=g, device=DEV) * 3.0).bfloat16()
    smooth, clip = make_activation_params(K, g)

    packed, s_x = peqat_cuda.preprocess_per_token_forward(x, smooth, clip)
    assert packed.shape == (M, K // 8) and packed.dtype == torch.int32
    assert s_x.shape == (M,) and s_x.dtype == torch.float32

    q_ref, s_ref = ref_preprocess(x, smooth, clip)
    assert torch.equal(s_x, s_ref), f"s_x max diff {(s_x - s_ref).abs().max().item():.3e}"
    q = unpack8(packed)
    mism = (q != q_ref).sum().item()
    assert mism == 0, f"{mism} of {q.numel()} int4 codes differ"


def test_preprocess_per_token_edge_rows():
    K = 4096
    g = gen(3)
    smooth, clip = make_activation_params(K, g)
    rows = [
        torch.zeros(K, device=DEV),                      # s_x hits the 1e-8 floor
        torch.ones(K, device=DEV),
        -torch.rand(K, generator=g, device=DEV) * 10.0,  # all negative
        torch.randn(K, generator=g, device=DEV) * 100.0,  # mostly clipped
        torch.randn(K, generator=g, device=DEV) * 1e-3,
    ]
    spike = torch.randn(K, generator=g, device=DEV) * 1e-2
    spike[123] = 1e4                                     # one outlier
    rows.append(spike)
    x = torch.stack(rows).bfloat16()

    packed, s_x = peqat_cuda.preprocess_per_token_forward(x, smooth, clip)
    q_ref, s_ref = ref_preprocess(x, smooth, clip)
    assert torch.equal(s_x, s_ref)
    assert torch.equal(unpack8(packed), q_ref)


# --------------------------------------------------------------------------
# INT4 tensor-core GEMM (prefill) and GEMV (decode)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("N,K", NK_LIST)
@pytest.mark.parametrize("M", M_LIST)
def test_int4_tc_gemm_exact(M, N, K):
    """Raw M: the kernel bounds-checks M itself, so no padding here."""
    g = gen(200 + M * 13 + N + K)
    xq = rand_codes(M, K, g)
    wq = rand_codes(N, K, g)
    s_x = torch.rand(M, generator=g, device=DEV) * 0.1 + 1e-3
    w_scale = torch.rand(N, generator=g, device=DEV) * 0.01 + 1e-4

    y = peqat_cuda.fused_int4_tc_per_token_forward(pack8(xq), pack8(wq), w_scale, s_x, N, K)
    assert_bf16_equal(y, ref_gemm(xq, wq, s_x, w_scale), f"TC GEMM M={M} N={N} K={K}")


@pytest.mark.parametrize("N,K", NK_LIST)
def test_int4_gemv_exact(N, K):
    g = gen(300 + N + K)
    xq = rand_codes(1, K, g)
    wq = rand_codes(N, K, g)
    s_x = torch.rand(1, generator=g, device=DEV) * 0.1 + 1e-3
    w_scale = torch.rand(N, generator=g, device=DEV) * 0.01 + 1e-4

    y = peqat_cuda.fused_int4_gemv_per_token_forward(pack8(xq), pack8(wq), w_scale, s_x, N, K)
    assert_bf16_equal(y, ref_gemm(xq, wq, s_x, w_scale), f"GEMV N={N} K={K}")


# --------------------------------------------------------------------------
# Whole quantized linear, as the runtime calls it
# --------------------------------------------------------------------------

@pytest.mark.parametrize("N,K", [(4096, 4096), (4096, 11008)])
@pytest.mark.parametrize("M", M_LIST)
def test_loader_linear_exact(M, N, K):
    loader = pytest.importorskip("peqat_deploy.runtime.loader")
    g = gen(400 + M * 17 + N + K)
    wq = rand_codes(N, K, g)
    w_scale_fp16 = (torch.rand(N, generator=g, device=DEV) * 0.01 + 1e-4).half()
    smooth, clip = make_activation_params(K, g)
    x = (torch.randn(M, K, generator=g, device=DEV) * 3.0).bfloat16()

    lin = loader.CUDAQuantizedLinear(
        packed_w=pack_low_high(wq), w_scale=w_scale_fp16,
        smooth=smooth, clip=clip, out_features=N, in_features=K,
    )
    y = lin(x)

    q_ref, s_ref = ref_preprocess(x, smooth, clip)
    # Today's loader rounds the fp16 weight scale through bf16 on the prefill
    # path (loader.py:87/94, then .float() at :97). If Phase 3 switches to an
    # fp32 scale precomputed from fp16, change this line to
    # w_scale_fp16.float() and say so in the commit.
    w_scale_used = w_scale_fp16.bfloat16().float()
    assert_bf16_equal(y, ref_gemm(q_ref, wq, s_ref, w_scale_used), f"loader linear M={M} N={N} K={K}")


# --------------------------------------------------------------------------
# Hadamard kernels (bug B; see tests/test_hadamard_kernel_emulation.py)
# --------------------------------------------------------------------------

BUG_B = pytest.mark.xfail(
    strict=True,
    reason="bug B: 256 threads do one butterfly each per level; K > 512 is incomplete",
)


def exact_hadamard(x):
    from peqat_deploy.quant.fold_hadamard import fast_hadamard
    return fast_hadamard(x.float().clone())


@pytest.mark.parametrize(
    "K", [256, 512, pytest.param(1024, marks=BUG_B), pytest.param(4096, marks=BUG_B)]
)
def test_hadamard_butterfly_matches_fold(K):
    """Training kernel vs the exact transform export folds into the weights.
    Tolerance covers the kernel's bf16 rounding at every level."""
    g = gen(500 + K)
    x = torch.randn(4, K, generator=g, device=DEV).bfloat16()
    y = peqat_cuda.hadamard_butterfly_forward(x).float()
    ref = exact_hadamard(x)
    rel = ((y - ref).norm(dim=1) / ref.norm(dim=1)).max().item()
    assert rel < 3e-2, f"K={K}: relative error {rel:.3e}"


@pytest.mark.parametrize(
    "K", [512, pytest.param(4096, marks=BUG_B), pytest.param(11008, marks=BUG_B)]
)
def test_preprocess_hadamard_rotation(K):
    """Dequantized output must sit within one quant step of the consistent
    transform: H @ x for power-of-2 K, identity for K=11008 (export leaves
    those weights unrotated and training's HadamardFunction skips them)."""
    g = gen(600 + K)
    x = torch.randn(8, K, generator=g, device=DEV).bfloat16()
    packed, s_x = peqat_cuda.preprocess_hadamard_forward(x)
    deq = unpack8(packed).float() * s_x[:, None]
    target = exact_hadamard(x) if (K & (K - 1)) == 0 else x.float()
    steps = ((deq - target).abs() / s_x[:, None]).max().item()
    assert steps <= 1.0, f"K={K}: off by up to {steps:.2f} quantization steps"
