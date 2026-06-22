# Hadamard PE-QAT: Pure INT4 Tensor Core Deployment via Rotated QAT

## Motivation

Current PE-QAT has two problems sharing a root cause: **outlier channels.**

1. **PPL regression (11 vs 7.5):** `PerTokenActSTEQuant` detaches the per-token scale, removing the LSQ quantization gradient. The clip only learns from boundary signals, never from in-range quantization error. Per-token scale works poorly because individual outlier channels dominate `max||` and starve all other channels of bits.

2. **Prefill is 2× slower than FP16 (190ms vs 91ms):** Per-token scale requires a separate preprocess kernel (max reduction across all K, then quantize + pack) that writes to DRAM, then the TC kernel reads it back. Two DRAM round-trips per layer.

**Walsh-Hadamard rotation** diffuses outlier channels uniformly across all K dimensions. After rotation, no channel dominates the magnitude distribution, so per-token dynamic quantization works without channel-specific clipping — and without the PPL regression. SpinQuant already proves this: rotation-only PTQ achieves PPL ~6 on Llama-2-7B.

## Approach

### Walsh-Hadamard Transform

A fixed orthogonal matrix H (size K×K) computed via butterfly, no learned parameters:

```
H @ H^T = I          (when normalized)
H(x) = x @ H^T       (rotate activations along K)
H(W) = W @ H^T       (rotate weights along K identically)

y = H(x) @ H(W)^T = x @ H^T @ H @ W^T = x @ W^T    (identity in FP)
```

**Critical:** H is symmetric, so `H^T = H`. The normalization factor is `√K`:

```
(H/√K) @ (H/√K) = (1/K) · K · I = I    ✓
```

### Training (`fake_quant.py`)

**New:** `fast_hadamard(x)` — in-place Walsh-Hadamard butterfly with √K normalization.

```python
def fast_hadamard(x):
    """Walsh-Hadamard transform. H @ x along last dim. Normalized: H @ H^T = I."""
    K = x.shape[-1]
    h = 1
    while h < K:
        x = x.reshape(-1, h * 2)
        a, b = x[:, :h].clone(), x[:, h:].clone()
        x[:, :h] = a + b
        x[:, h:] = a - b
        h *= 2
    return x / math.sqrt(K)
```

**Removed:** `QATW4A4LinearCore` — no `activation_clip`, no `smooth_factor`.

**Simplified:** `PerTokenActSTEQuant` — no clip params, no boundary gradient, no LSQ term. Pure STE.

```
Forward:
  s_x = |x|.amax(dim=-1, keepdim=True) / 7    (detached)
  x_q = clamp(round(x / s_x), -8, 7)
  x_dq = x_q * s_x

Backward:
  return grad_output    # full STE, no mask — no clip params, no boundary
```

**Rewritten:** `PEQATLoRAModule.forward`

```python
def forward(self, x):
    # 1. Hadamard rotate input along K
    x_rot = fast_hadamard(x)

    # 2. Per-token quant on rotated activations (no clip/smooth)
    quant_input = PerTokenActSTEQuant.apply(x_rot, self.num_bits)

    # 3. Hadamard rotate merged weight along K
    #    IMPORTANT: fast_hadamard(W) operates on in_features (last dim)
    #    math: y = H(x) @ H(W)^T = x @ H @ H^T @ W^T = x @ W^T
    merged = self.get_merged_weight()        # [out, in]
    W_rot = fast_hadamard(merged)            # same H, same direction as input

    # 4. Weight quant (unchanged: per-row symmetric INT4)
    quant_weight = WeightSTEQuant.apply(W_rot, self.num_bits)

    # 5. Linear
    bias = self.qat_core.original_linear.bias
    return F.linear(quant_input, quant_weight, bias)
```

`WeightSTEQuant` unchanged (per-row symmetric, detached amax scale).

### Training (`peqat_training.py`)

- Remove all smooth/clip config init/freeze logic
- Remove `unified_calibration()` call
- Single optimizer group: LoRA only (no 10× LR multipliers for clip/smooth)
- `calibrations.py` untouched (not called)

### Config (`config.yaml`)

```yaml
model:
  num_bits_weight: 4
  num_bits_activation: 4

lora:
  rank: 32
  alpha: 64
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  learning_rate: 1e-4
  weight_decay: 0.01
  optimizer:
    lora_lr_multiplier: 1.0
```

Removed: `activation_clip`, `smooth_factor`, `calibration` section, optimizer multipliers.

### Deployment — Export Pipeline

**New:** `peqat_deploy/quant/fold_hadamard.py` — replaces `fold_smoothing.py`.

Before per-row INT4 quantization, apply H to every merged weight:

```python
def fold_hadamard(merged_weights):
    """In-place: W' = H @ W along in_features dimension.
       Same H used during training. No scaling, no data dependence."""
    for name, w in merged_weights.items():
        w.data = fast_hadamard(w)   # same function as training
```

This is called in `export_peqat.py` at the same pipeline position as `fold_smoothing` was (after LoRA merge, before quantize_weights).

**Removed from export:**
- `extract_quant_params.py` — no quant params to extract
- `fold_smoothing.py` — replaced by `fold_hadamard.py`
- No `activation/` directory in exports (no smooth/clip tensors)

### Deployment — CUDA: Preprocess (`csrc/peqat_preprocess_per_token.cu`)

Add Hadamard butterfly before per-token max reduction:

```
preprocess_per_token_kernel(x, x_packed, s_x, M, K):
    row = blockIdx.x
    tid = threadIdx.x

    // Phase 0: Hadamard butterfly (new)
    shared_buf = x[row, :] in bf16
    for stride in [1, 2, 4, ..., K/2]:
        if tid < K/2:
            idx = (tid // stride) * (stride * 2) + (tid % stride)
            a = shared_buf[idx]
            b = shared_buf[idx + stride]
            shared_buf[idx] = a + b
            shared_buf[idx + stride] = a - b
        __syncthreads()

    // Normalize by sqrt(K)
    shared_buf[tid] /= sqrt(K)
    __syncthreads()

    // Phase 1: Per-token max reduction (existing)
    local_max = 0
    for k in range(tid, K, 256):
        local_max = max(local_max, |shared_buf[k]|)
    warp_reduce(local_max)
    // Write s_x[row] = block_max / 7

    // Phase 2: Quantize + pack (existing)
```

**Total butterfly cost:** Σ over `h ∈ {1,2,...,K/2}` of `K/2 × 2` = `K·log₂(K)` = 4096·12 = ~50K add/sub per token. This is the same or cheaper than the current preprocess (smooth divide + clip clamp + absmax is ~5 ops per element = 20K ops per token, plus extra global reads for smooth[K] and clip[K] kernels). The key difference: no smooth/clip global reads.

No smooth/clip kernel parameters.

### Deployment — Runtime (`peqat_deploy/runtime/loader.py`)

`CUDAQuantizedLinear` loses `smooth` and `clip` buffers:

```python
class CUDAQuantizedLinear(nn.Module):
    def __init__(self, packed_w, w_scale, out_features, in_features):
        self.register_buffer("packed_w", packed_w)
        self.register_buffer("w_scale", w_scale)

    def forward(self, x):
        x_packed, s_x = peqat_cuda.preprocess_per_token_forward(x)  # no smooth/clip
        return peqat_cuda.fused_int4_tc_per_token_forward(
            x_packed, self.w_packed_tc, self.w_scale, s_x,
            self.out_features, self.in_features
        )
```

### What Doesn't Change

| Component | Status |
|-----------|--------|
| `WeightSTEQuant` (per-row symmetric INT4) | Unchanged |
| INT4 TC GEMM kernel (`fused_int4_tc_per_token_kernel`) | Unchanged |
| GEMV decode kernel (`fused_int4_gemv_kernel`) | Unchanged |
| Weight packing format (`weight_int4.bin`, `weight_scale.fp16`) | Unchanged |
| Weight repack `int32[N, K/8]` for TC | Unchanged |
| `calibrations.py` | Unchanged (unused) |
| `scripts/export_peqat.py` (add `fold_hadamard` step) | Minor addition |
| `scripts/validate_export.py` | Unchanged |
| `scripts/run_benchmarks.py` | Unchanged |

## Expected Accuracy

| Method | Activation quant | Learned params | PPL (wikitext2) |
|---|---|---|---|
| FP16 | None | None | ~5.47 |
| SpinQuant (PTQ) | Static per-group | None | ~6.0 |
| Current per-token (no H) | Per-token dynamic | 2.3M clip/smooth | ~11 |
| Original PE-QAT per-channel | Per-channel | 2.3M clip/smooth | ~7.5 |
| **H + per-token (this work)** | **Per-token dynamic** | **LoRA only (0.7M)** | **~6-7** |
| H + per-token + learnable W scales | Per-token dynamic | LoRA + W row scales | ~5.5-6 |

SpinQuant gets PPL ~6 with **no fine-tuning** — rotation + PTQ alone. Adding LoRA QAT on top of the rotation should *at minimum* match that (LoRA adapts weights to the quantization noise). The ~6-7 estimate is conservative; the right target is competitive with or better than SpinQuant.

## Expected Latency

| Phase | Current (2 kernels) | Fused (1 kernel) |
|---|---|---|
| Kernel launches/layer | 2 (preprocess + TC) | 1 |
| DRAM reads/layer | 1× activation + 0.5× x_packed | 1× activation |
| DRAM writes/layer | 0.5× x_packed | 0 |
| Total DRAM traffic/layer | 2.5× activation size | 1× activation size |
| Estimated prefill (seq 128) | ~190ms | ~130-140ms |

The remaining gap to FP16 (~91ms) comes from weight dequantization in the TC kernel — INT4 weights must be unpacked and scaled. This is intrinsic to INT4 deployment and independent of the preprocess changes.

## Implementation Order

1. `fast_hadamard()` in `fake_quant.py`
2. Rewrite `PEQATLoRAModule.forward` to use H (correctly: `fast_hadamard(W)`, not `.T.T`)
3. Simplify `PerTokenActSTEQuant` (no clip, no boundary gradient)
4. Remove `QATW4A4LinearCore`
5. Clean up `peqat_training.py` (remove smooth/clip logic)
6. Clean up `config.yaml`
7. Add `peqat_deploy/quant/fold_hadamard.py`
8. Update `scripts/export_peqat.py` to call `fold_hadamard`
9. Train 1 epoch on wikitext2, measure PPL
10. Add H to `csrc/peqat_preprocess_per_token.cu`
11. Simplify `peqat_deploy/runtime/loader.py`
12. Run benchmark comparison (prefill + decode)
13. Accuracy eval on reasoning tasks (PIQA, HellaSwag, ARC, WinoGrande)

## Open Questions

1. **Learnable weight scales:** `WeightSTEQuant` still uses detached row amax. Adding learned per-row scales (2× LoRA param count) is a simple next step if PPL isn't competitive.

2. **Residual clip:** If the rotation isn't perfectly diffusing all outliers (some projections may have structured sparsity), adding a learned per-group clip (G=64, one per TC tile) costs 32 params/projection and is trivially TC-compatible. This would be a fallback path if PPL > 7.

3. **First/last layer:** The embedding and LM head don't have smooth/clip in the current design, so no rotation needed there. Verify the embedding → first-layer interaction is numerically clean.

4. **RMSNorm interaction:** Since rotation commutes with RMSNorm up to the scale factor, verify that `RMSNorm(H(x)) ≈ H(RMSNorm(x))`. Theory says yes for a uniformly distributed vector; measure empirically.
