"""CPU emulation of the CUDA Hadamard butterfly (bug B). Needs no GPU and no torch.

csrc/peqat_hadamard_preprocess.cu:44-61 and csrc/hadamard_butterfly.cu:32-49 are
launched with 256 threads, and each thread performs ONE butterfly per level. A
full Walsh-Hadamard transform needs K/2 butterflies per level, so for K > 512
the kernels compute a partial transform that is neither H @ x nor orthogonal.

This file replays the kernels' exact index arithmetic in plain Python (fp64,
without the kernels' per-level bf16 rounding) and compares it with the
transform that export folds into the weights (peqat_deploy/quant/fold_hadamard.py
fast_hadamard). Within one level every thread touches distinct indices, so a
sequential replay of the threads equals the parallel kernel.

The K > 512 cases are expected to FAIL today; they are marked xfail(strict)
so the suite stays green until the kernels are fixed, and then forces this
file to be updated.

Run:  pytest -q tests/test_hadamard_kernel_emulation.py
  or: python tests/test_hadamard_kernel_emulation.py
"""

import math
import random

try:
    import pytest
except ImportError:  # allow running as a plain script
    pytest = None

THREADS = 256  # launch_preprocess_hadamard / launch_hadamard_butterfly


def kernel_butterfly(x, threads=THREADS):
    """Replay of the kernels' butterfly loop and sqrt(K) normalisation."""
    s = list(x)
    K = len(s)
    stride = 1
    while stride < K:
        pairs = K // (stride * 2)
        elems_per_pair = stride
        for tid in range(threads):
            my_pair = tid // elems_per_pair
            if my_pair < pairs:
                base = my_pair * stride * 2
                offset = tid % elems_per_pair
                i0 = base + offset
                i1 = base + stride + offset
                a0, a1 = s[i0], s[i1]
                s[i0], s[i1] = a0 + a1, a0 - a1
        stride <<= 1
    inv = 1.0 / math.sqrt(K)
    return [v * inv for v in s]


def exact_hadamard(x):
    """Same algorithm as fold_hadamard.fast_hadamard (Sylvester order, / sqrt(K))."""
    s = list(x)
    K = len(s)
    h = 1
    while h < K:
        for base in range(0, K, 2 * h):
            for j in range(h):
                a, b = s[base + j], s[base + h + j]
                s[base + j], s[base + h + j] = a + b, a - b
        h *= 2
    inv = 1.0 / math.sqrt(K)
    return [v * inv for v in s]


def _norm(v):
    return math.sqrt(sum(t * t for t in v))


def _sample(K, seed):
    rng = random.Random(seed)
    return [rng.gauss(0.0, 1.0) for _ in range(K)]


def check_power_of_two(K):
    x = _sample(K, K)
    got, want = kernel_butterfly(x), exact_hadamard(x)
    max_err = max(abs(a - b) for a, b in zip(got, want))
    assert max_err < 1e-9, (
        f"K={K}: kernel butterfly differs from H @ x by up to {max_err:.3e}; "
        f"output norm / input norm = {_norm(got) / _norm(x):.4f} (H is orthogonal, so 1.0)"
    )


def check_non_power_of_two(K):
    # fold_hadamard skips non-power-of-2 in_features and training's
    # HadamardFunction returns x unchanged, so the consistent activation
    # transform here is the identity.
    x = _sample(K, K)
    got = kernel_butterfly(x)
    max_err = max(abs(a - b) for a, b in zip(got, x))
    assert max_err < 1e-9, (
        f"K={K}: weights are left unrotated, but the kernel changes x by up to "
        f"{max_err:.3e}; output norm / input norm = {_norm(got) / _norm(x):.4f}"
    )


BUG_B = "bug B: 256 threads do one butterfly each per level; K > 512 is incomplete"

if pytest is not None:

    @pytest.mark.parametrize(
        "K",
        [
            64,
            256,
            512,
            pytest.param(1024, marks=pytest.mark.xfail(strict=True, reason=BUG_B)),
            pytest.param(4096, marks=pytest.mark.xfail(strict=True, reason=BUG_B)),
        ],
    )
    def test_butterfly_matches_exact_hadamard(K):
        check_power_of_two(K)

    @pytest.mark.xfail(strict=True, reason=BUG_B + "; K=11008 is not a power of 2")
    def test_non_power_of_two_is_identity():
        check_non_power_of_two(11008)

    def test_exact_reference_is_orthogonal():
        x = _sample(4096, 7)
        assert abs(_norm(exact_hadamard(x)) / _norm(x) - 1.0) < 1e-12


if __name__ == "__main__":
    for K in (64, 256, 512, 1024, 4096):
        try:
            check_power_of_two(K)
            print(f"[PASS] K={K}: kernel butterfly == H @ x")
        except AssertionError as e:
            print(f"[FAIL] {e}")
    try:
        check_non_power_of_two(11008)
        print("[PASS] K=11008: identity")
    except AssertionError as e:
        print(f"[FAIL] {e}")
