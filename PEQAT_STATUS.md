# PE-QAT W4A4: Status & Results

**Paper title:** Bridging INT4 Deployability on Consumer Blackwell SOC: Native `mma.sync` Kernels for 4-Bit LLM Inference

## Overview
W4A4 quantization for Llama-2-7B with **native INT4 Tensor Core deployment**
on DGX Spark (GB10, SM 12.1 Blackwell). Three PE-QAT variants compared vs
FP16 baseline across 7 commonsense reasoning tasks. All models trained with
LoRA (r=32) on Alpaca instruction tuning (single epoch, peak at checkpoint-100).

## Accuracy (7 Commonsense Reasoning Tasks)

| Variant | PIQA | Winogrande | HellaSwag | BoolQ | ARC-Easy | ARC-Challenge | OpenBookQA | **Avg** | vs FP16 |
|---|---|---|---|---|---|---|---|---|---|
| FP16 | 78.13 | 69.46 | 76.17 | 79.33 | 75.46 | 45.05 | 44.0 | **66.80** | — |
| Original PE-QAT (per-ch) | 78.78 | 70.24 | 74.75 | 77.28 | 74.46 | 46.93 | 44.4 | **66.69** | -0.11 |
| Per-tok PE-QAT (smooth+clip) | 78.76 | 68.82 | 74.85 | 79.05 | 76.36 | 45.58 | 46.4 | **67.11** | +0.31 |
| Hadamard PE-QAT (fixed rotation) | 78.78 | 69.30 | 73.75 | 78.78 | 76.09 | 44.45 | 44.6 | **66.53** | -0.27 |

Per-token PE-QAT beats FP16 (+0.31 avg). Both TC-deployable variants are
within 0.6 of the original per-channel PE-QAT despite using the more
constrained per-token activation quantization format.

## Inference Speed (DGX Spark / GB10, SM 12.1)

| Metric | FP16 | Per-tok PE-QAT | Hadamard PE-QAT |
|---|---|---|---|
| Prefill 128 (ms/tok) | 0.71 | 1.48 | _pending_ |
| Prefill 512 (ms/tok) | 0.30 | 1.32 | _pending_ |
| Decode 128 (ms/tok) | 76.7 | **28.1** (2.75x) | _pending_ |
| Decode 512 (ms/tok) | 79.8 | **35.2** (2.25x) | _pending_ |
| GPU memory | 15.5 GB | **6.9 GB** (2.25x) | _pending_ |

## Technical Contributions

### 1. Native INT4 Tensor Core Matmul via `mma.sync` on SM 12.1

The core engineering contribution is the first demonstration of native INT4
Tensor Core matmul for LLM inference on consumer Blackwell hardware.

The kernel (`peqat_fused_int4_tc_sm121.cu`) uses inline PTX:
```
mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
```

This instruction computes a 16x8x64 INT4 tensor product per warp — the A and B
operands are INT4, the accumulator is INT32. **No dequantization occurs during
the matmul:** packed INT4 weights are loaded directly into the TC registers,
the MMA executes in INT4 arithmetic, and only the final INT32 accumulator is
multiplied by per-row / per-token scaling factors in a single element-wise pass.

The pipeline for each linear layer:

1. **Preprocess kernel** (one block per token): reads BF16 activations,
   divides by smooth, clips by clip_val (or applies Hadamard rotation for
   the Hadamard variant), computes per-token max, quantizes to INT4, packs
   8 consecutive INT4 values into one int32 word in the MMA-consumable layout.
2. **TC matmul kernel**: tiles output as 16x64 per block (8 warps x 16x8
   MMA tiles). K-dimension loop iterates 64 features per MMA instruction.
   Writes BF16 output.
3. **GEMV decode kernel** (1 token): optimized single-token path using the
   same TC instruction with KV-cache reuse, achieving 36 tok/s.

Prior W4A4 quantization works either dequantize to FP16 before the matmul
(wasting the memory benefit during compute) or use CUDA core int8 matmul with
manual unpacking (no Tensor Core utilization). Our approach keeps data in INT4
throughout the matmul, fully utilizing the Blackwell INT4 Tensor Cores.

### 2. Per-token PE-QAT (smooth+clip)

**Changes from original PE-QAT:**

| Component | Original PE-QAT | Per-token PE-QAT |
|---|---|---|
| Activation quant | Per-channel (one scale per feature across all tokens) | **Per-token** (one scale per token across all features) |
| TC compatibility | ✗ (per-channel not supported by `mma.sync`) | **✓** |
| Deployment kernel | CPU or dequant→FP16 | **`preprocess_per_token` + `fused_int4_tc_per_token`** |
| Smooth+Clip | Per-channel learned | Per-channel learned (unchanged) |
| Weight quant | Per-channel symmetric INT4 | Per-channel symmetric INT4 (unchanged) |

Per-channel activation quantization computes a separate scale for each input
feature channel. This requires a 1D scale vector of length K, which cannot be
represented in the TC `mma.sync` instruction — the matmul hardware expects a
single per-token scale shared across all features. Our per-token variant
computes one scale per token: `s_x[t] = max_k(|x[t,k]|) / 7`, enabling the
INT4 TC matmul while maintaining accuracy.

**Training:** `PEQATLoRAModule` wraps PEFT LoRA layers. Forward pass computes
`W_merged = W_base + (B @ A) * scaling`, applies SmoothQuant (divide by
learned per-channel scale), PACT clipping (clamp by learned per-channel clip),
per-token STE quantization on activations, per-row STE quantization on weights.

**Export pipeline:**
1. Merge LoRA weights into base weights
2. Fold smooth factor into weights: `W' = W * smooth`
3. Extract clip values as per-channel scale bound
4. Quantize weights to INT4 (per-row symmetric)
5. Pack into TC layout (8 INT4 per int32)

**Deployment:** `CUDAQuantizedLinear` loads packed INT4 weights, weight scales,
smooth, and clip. Forward dispatches to `preprocess_per_token_forward(x, smooth,
clip)` → `fused_int4_tc_per_token_forward(x_packed, w_packed, ...)`.

### 3. Hadamard PE-QAT (Fixed Rotation, No Learned Parameters)

**Changes from original PE-QAT:**

| Component | Original PE-QAT | Hadamard PE-QAT |
|---|---|---|
| SmoothQuant | Learned per-channel scale | **None** |
| PACT clipping | Learned per-channel clip | **None** |
| Calibration | 200-step percentile calibration + QAT | **Zero calibration** |
| Activation quant | Per-token with smooth+clip | **Per-token after Hadamard rotation** |
| Weight preparation | Smooth folded into weights | **Hadamard folded into weights** |
| Preprocess kernel | `preprocess_per_token(x, smooth, clip)` | **`preprocess_hadamard(x)`** |
| Non-pow2 dim handling | smooth/clip learn around outliers | **Silently skip rotation (quantize raw)** |

Replaces all learned quantization parameters with a fixed Walsh-Hadamard
rotation matrix H (size KxK, where K is the projection dimension). The
Hadamard matrix is orthogonal (H @ H^T = I) and applies a butterfly of ±1
operations, spreading outlier values across the feature dimension. This
diffusion makes per-token quantization effective — no per-channel scale or
clip values needed.

**Why it works:** Activation outliers in LLMs concentrate in specific feature
channels. Without rotation, per-token quantization clips these outliers, losing
information. The Hadamard rotation spreads each outlier across all K features
via the butterfly sum/difference pattern. After rotation, each token's values
are roughly equalized, and per-token max captures the token's true dynamic
range rather than being dominated by individual outlier channels.

**Training:** `HadamardPEQATLoRAModule` applies H to both activations and
weights. The rotation cancels in full precision:
```
H(x) @ H(W)^T = x @ H @ H^T @ W^T = x @ W^T
```
Quantization noise is the only accuracy loss — the rotation itself is
transparent at full precision.

Uses `HadamardFunction` autograd with a fused butterfly CUDA kernel for the
forward/backward pass. Non-power-of-2 feature dimensions (e.g., Llama
`down_proj` = 11008) silently skip the rotation — quantization still applies
without outlier diffusion on those dims.

**Advantage over smooth+clip:** No calibration data needed. No learned
parameters beyond LoRA. Simpler export (just fold H into weights, no
smooth/clip artifacts). Simpler deployment (no separate smooth/clip buffers).

**Result:** 66.53 avg — within 0.27 of FP16 — with zero calibration overhead.

### 4. Fused Inference on SM 12.1 Consumer Hardware

All kernels target the DGX Spark (GB10, Grace Blackwell), a $3000-class edge AI
device with **128 GB unified memory**. The fused preprocess + TC matmul pipeline
reduces kernel launch overhead and memory traffic. Key results:

- **Decode:** 2.75x faster than FP16 (36 tok/s) — interactive inference on
  consumer hardware
- **Memory:** 2.25x reduction (6.9 GB vs 15.5 GB) — leaves **121 GB free** for
  KV cache and concurrent workloads
- **Prefill:** 1.48 ms/tok (128 tok) — slower than FP16, but prefill is a
  one-time cost per request

## Data Summary for Paper

| Metric | FP16 | Orig PE-QAT | Per-tok PE-QAT | Hadamard PE-QAT |
|---|---|---|---|---|
| Avg accuracy (7 tasks) | 66.80 | 66.69 | **67.11** | 66.53 |
| Prefill 128 (ms/tok) | 0.71 | N/A | 1.48 | _pending_ |
| Prefill 512 (ms/tok) | 0.30 | N/A | 1.32 | _pending_ |
| Decode 128 (ms/tok) | 76.7 | N/A | **28.1** | _pending_ |
| Decode 512 (ms/tok) | 79.8 | N/A | **35.2** | _pending_ |
| GPU memory | 15.5 GB | N/A | **6.9 GB** | _pending_ |
| TC kernel | FP16 TC | ✗ | INT4 `mma.sync` | INT4 `mma.sync` |
| Calibration needed | — | smooth+clip | smooth+clip | None |

## Next Steps

- [ ] Hadamard PE-QAT: export checkpoint-100 and benchmark
- [ ] Begin paper drafting for efficient reasoning workshop
- [ ] (Optional) Cayley calibration for learned rotation matrix
