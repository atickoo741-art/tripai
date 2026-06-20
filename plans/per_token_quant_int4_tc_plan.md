# PE-QAT Per-Token Activation Quantization for INT4 Tensor Cores on SM 12.1

**Target:** NVIDIA DGX Spark (GB10 Grace Blackwell Superchip) — Compute Capability 12.1
**ISA:** `mma.sync.aligned.m16n8k64.row.col.s4.s4.s32` via inline PTX
**Compiler target:** `-arch=sm_121f`

---

## 1. Conflicting Plans to Discard

| Document | Conflict | Action |
|----------|----------|--------|
| `plans/fused_int4_gemm_design.md` | Assumes FP32/unpack WMMA, bf16 fragment types, `BM=128, BN=128, BK=32`. Different tile sizes, different MMA approach entirely. | **Discard** |
| `peqat_kernel_project_plan.md` | Preprocess kernel spec mandates per-channel `x_scale[K]` output. Non-goals section explicitly blocks per-token activation variants. | **Discard kernel spec** — keep benchmark harness, eval structure, and project scaffolding |
| `csrc/depr/peqat_gemm_int4_tc.cu` | Attempted Ampere SM80 `m16n8k32` with `experimental::precision::b4`. Wrong SM target, wrong tile size, wrong packing. Its own comment block proves per-channel activation scales cannot use INT4 MMA. | **Discard** — tile layout reference only |

**Keep:**
- Export pipeline (`peqat_deploy/`) — mostly unchanged
- Training `PEQATLoRAModule` structure — minimal changes
- `csrc/peqat_gemm_bf16_wmma.cu` — keep as bf16 fallback for non-SM121 hardware
- `csrc/peqat_preprocess.cu` — keep as reference; new preprocess supersedes it

---

## 2. Target Hardware

| Spec | Value |
|------|-------|
| **Device** | NVIDIA DGX Spark (GB10 Grace Blackwell Superchip) |
| **Compute Capability** | SM 12.1 |
| **CUDA Toolkit** | 13.x (required for SM 12.1 support) |
| **Compiler target** | `-arch=sm_121f` (enables `.s4` sub-byte MMA) |
| **ISA** | `mma.sync.aligned.m16n8k64.row.col.s4.s4.s32` |
| **Tensor Core precision** | INT4 (`.s4`) sub-byte integer, K=64 per MMA instruction |
| **Max warps/SM** | 48 (SM 12.0) |
| **Memory bandwidth** | 273 GB/s unified LPDDR5x (bottleneck) |
| **No `tcgen05`** | SM 12.1 uses `mma.sync` only. No `wgmma` required. No TMEM. No cluster-level sync (1x1x1 only). |

---

## 3. High-Level Dataflow

```
Training:   x → /smooth[k] → clamp(-clip[k], clip[k]) → s_x[t]=max|clamped|/7 → round/s_x[t] → clamp(-8,7)
                         per-channel smooth+clip              per-token scale          uniform INT4

Deployment (CUDA):
  Preprocess:  x → /smooth[k] → clamp(-clip[k], clip[k]) → s_x[t]=max|clamped|/7 → pack 4×INT4 into int32
  GEMM:        x_packed[int32[M, K/4]] @ W_packed[int32[N, K/4]] → int32[M, N] → * s_x[t] * w_scale[i] → bf16[M, N]
```

The per-channel `smooth[k]` + `clip[k]` are applied element-wise during preprocessing. The per-token `s_x[t]` (a scalar per token row) factors cleanly out of the K-sum, enabling pure INT4 MMA:

```
y[t][n] = sum_k x_q[t][k] * W'_q[n][k] * s_x[t] * w_scale[n]
        = s_x[t] * w_scale[n] * sum_k x_q[t][k] * W'_q[n][k]
        = s_x[t] * w_scale[n] * accum[t][n]
```

Both `s_x[t]` and `w_scale[n]` are constant across K — they apply post-TC.

---

## 4. Training Changes (`fake_quant.py`)

### 4.1 New `PerTokenActSTEQuant`

Replace the existing `ActSTEQuant` class (line 127) with:

**Forward:**
```
Input: x, clip_vals, num_bits=4
       x shape: [M, K] or [B, S, K] — already smoothed (x/smooth applied by PEQATLoRAModule)
       clip_vals: [K]

qmax = 7, qmin = -8
clip = |clip_vals|.clamp(min=1e-4)

x_clipped = clamp(x, -clip_broadcast, clip_broadcast)    # per-channel clip
s_x = |x_clipped|.amax(dim=-1, keepdim=True) / qmax       # per-token scale [M,1] or [B,S,1]
s_x = s_x.detach()                                         # stop gradient through scale
x_q = clamp(round(x_clipped / s_x), qmin, qmax)           # uniform INT4
x_dq = x_q * s_x                                           # dequantized
return x_dq
```

`s_x` is **detached** — the only gradient into `clip_vals` is the PACT boundary gradient from values outside `[-clip, clip]`. No quantization gradient flows through the scale.

**Backward:**
```
x, x_clipped, clip, scale = ctx.saved_tensors
qmax = ctx.qmax

# Broadcast clip for element-wise operations
if x.dim() == 3:                                    # [B, S, K]
    clip_b = clip.view(1, 1, -1)
elif x.dim() == 2:                                  # [M, K]
    clip_b = clip.view(1, -1)

# Input gradient (STE): only in-range values pass gradient through
mask_inside = (x.abs() <= clip_b).float()
grad_x = grad_output * mask_inside

# Clip gradient: PACT boundary gradient only
# Values exactly at the clipping boundary signal clip to expand
grad_clip_upper = (grad_output * (x > clip_b).float()).sum(dim=(0, 1 if x.dim() == 3 else 0))
grad_clip_lower = (grad_output * (x < -clip_b).float() * -1.0).sum(dim=(0, 1 if x.dim() == 3 else 0))
grad_alpha = grad_clip_upper + grad_clip_lower

return grad_x, grad_alpha, None, None
```

**The gradient accumulation dimension needs care:**
- For 3D input `[batch, seq_len, in_features]` (training via HF Trainer): sum over `dim=(0,1)` to produce `grad_alpha` shape `[K]`
- For 2D input `[batch, in_features]` (direct single-layer calls): sum over `dim=0`
- Both cases are handled above with the conditional `dim` parameter

The `use_quant_grad` flag is removed — irrelevant when scale is detached.

### 4.2 `PEQATLoRAModule.forward` — one line change

```python
# Line 355: swap function call only
quant_input = PerTokenActSTEQuant.apply(scaled_input, clip_vals, self.num_bits)
```

Everything else unchanged:
- `scaled_input = x / smooth_factor` (per-channel smooth, line 347)
- `scaled_weights = merged_weight * smooth_factor.unsqueeze(0)` (line 348)
- `WeightSTEQuant.apply(scaled_weights, self.num_bits)` (line 358)
- `F.linear(quant_input, quant_weight, bias)` (line 362)

### 4.3 Training Config

No new flags. The existing `use_quant_grad` config key becomes a no-op (leave for backward compatibility or remove from config parsing).

---

## 5. Export Pipeline Changes

### 5.1 Unchanged Steps

| Step | File | Reason |
|------|------|--------|
| Source discovery | `io/discover.py` | Same file discovery needs |
| Config loading | `config/loader.py` | Same config structure |
| Base model + adapter | `io/hf_loader.py`, `io/peft_loader.py` | Same HuggingFace loading |
| LoRA merge | `quant/merge_lora.py` | Same `W_eff = W + (α/r)BA` |
| Smooth fold | `quant/fold_smoothing.py` | Same `W' = W_eff * smooth` |
| Extract quant params | `quant/extract_quant_params.py` | Same smooth[K], clip[K] extraction |
| Weight quantize | `quant/quantize_weights.py` | Same per-row INT4 quantization |
| Weight pack | `quant/pack_int4.py` | Same uint8 low_high packing |
| Manifest | `quant/manifest.py` | Same metadata |
| Save artifacts | `io/save_artifacts.py` | Same directory tree |

### 5.2 Removed

**`quant/fold_act_scale.py`** — entire file deleted. No activation scale to fold (computed dynamically per-token at runtime). Remove `--fold-weights` flag from `scripts/export_peqat.py`.

### 5.3 Export Artifacts (unchanged layout)

```
activation/smooth.safetensors    # [in_features] bf16 per-channel
activation/clip.safetensors      # [in_features] bf16 per-channel
weights/*/weight_int4.bin        # uint8 packed INT4 (existing format)
weights/*/weight_scale.fp16      # [out_features] fp16 per-row
```

No folded weights. No per-channel `x_scale`. Every parameter needed for runtime is present.

---

## 6. CUDA Kernels for SM 12.1

### 6.1 Preprocess Kernel: `csrc/peqat_preprocess_per_token_sm121.cu`

**Purpose:** Convert bf16 activations to packed INT4 + per-token scales in one fused kernel.

**Interface:**
```
Input:  x[M, K] bf16,  smooth[K] bf16,  clip[K] bf16
Output: x_packed[M, K/4] int32,  s_x[M] f32
```

**Algorithm per token row `t`:**

```
x_s[k]    = x[t][k] / smooth[k]                         # per-channel smooth
x_c[k]    = clamp(x_s[k], -clip[k], clip[k])             # per-channel clip
maxv      = max(|x_c[k]|) over k ∈ [0, K)                # warp-level reduction
s_x[t]    = maxv / 7                                      # per-token scale (f32)

for k in range(0, K, 4):
    q0 = clamp(round(x_c[k+0] / s_x[t]), -8, 7) & 0xF
    q1 = clamp(round(x_c[k+1] / s_x[t]), -8, 7) & 0xF
    q2 = clamp(round(x_c[k+2] / s_x[t]), -8, 7) & 0xF
    q3 = clamp(round(x_c[k+3] / s_x[t]), -8, 7) & 0xF
    x_packed[t][k/4] = q0 | (q1<<4) | (q2<<8) | (q3<<12)   # int32
```

**Kernel launch config:**
- Grid: `[M, 1]` — one block per token row
- Block: 256 threads
- **Phase 1 (reduce):** each thread loads one `x/smooth`, computes `clamp`, warp-shuffle `absmax` reduction
- **Phase 2 (pack):** each thread quantizes 4 consecutive K values, writes one `int32`
- Write `s_x[t]` to `f32[M]` output buffer

### 6.2 Weight Repack (host-side, `runtime/loader.py`)

Convert existing `uint8[N, K_packed]` (low_high, 2 INT4 per byte) to TC-compatible `int32[N, K/4]` (4 INT4 per int32).

```
For each output row r ∈ [0, N):
  For each group g ∈ [0, K/4):
    word = (w8[r][4*g+0] & 0xF)
         | ((w8[r][4*g+1] & 0xF) << 4)
         | ((w8[r][4*g+2] & 0xF) << 8)
         | ((w8[r][4*g+3] & 0xF) << 12)
    w_packed[r][g] = word                          # [N, K/4] int32
```

Python `torch` operation during model load, executed once. The exact bit layout must match what the A-fragment (col-major) and B-fragment (row-major) of `mma.sync.aligned.m16n8k64.s4` expect. The packing permutation must be verified by inspecting compiled PTX on SM 12.1 or consulting CUTLASS SM120 templates.

**Optional hardware assist:** SM 12.1 supports `ldmatrix.b8x16.b4x16_p64` for FP4 format conversion during shared memory load. If a `.s4` variant of this `ldmatrix` instruction exists, use it in the fused kernel instead of software unpacking.

### 6.3 Fused INT4 TC Kernel: `csrc/peqat_fused_int4_tc_sm121.cu`

**Interface:**
```
Input:  x_packed[M, K/4] int32,  w_packed[N, K/4] int32,
        w_scale[N] bf16,  s_x[M] f32
Output: y[M, N] bf16
```

**Tile config:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `WMMA_M` | 16 | Native TC tile M |
| `WMMA_N` | 8 | Native TC tile N (m16n8k64) |
| `WMMA_K` | 64 | Native TC tile K (sub-byte INT4) |
| `BM` | 128 | Block tile M (8 warps × 16 rows/warp) |
| `BN` | 64 | Block tile N (8 column groups × 8 cols/group) |
| `BK` | 64 | Block tile K = one MMA instruction deep |

**Inline PTX instruction** (compiled with `-arch=sm_121f`):

```cuda
asm volatile(
    "mma.sync.aligned.m16n8k64.row.col.s4.s4.s32 "
    "{%0, %1}, {%2, %3, %4, %5}, {%6}, {%7, %8};\n"
    : "=r"(d[0]), "=r"(d[1])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
      "r"(b[0]),
      "r"(c[0]), "r"(c[1]));
```

Where:
- `a[0..3]`: 4 int32 values = 16 INT4 values (A fragment, col_major, 16×64)
- `b[0]`: 1 int32 = 2 or 4 INT4 values (B fragment, row_major, 64×8)
- `d[0..1]`: 2 int32 accumulator outputs per thread
- `c[0..1]`: 2 int32 accumulator inputs

**The exact fragment sizes are SM-version-specific and must be confirmed** by compiling a minimal test kernel and inspecting register usage (`--ptxas-options=-v`).

**Kernel structure:**

```
For each tile (bx, by) covering [BM × BN] output:
  Load A tile from x_packed[m_start..m_start+BM, :K/4] → s_a
  Load B tile from w_packed[n_start..n_start+BN, :K/4] → s_b
  Load s_x[m_start..m_start+BM] → s_sx

  For k_group from 0 to K/4 in steps of BK/4 = 16 int32 words per tile:
    # Each warp loads its A fragment (16×64 INT4) from s_a
    # Each warp loads its B fragment (64×8 INT4) from s_b
    # Each warp issues one mma.sync.m16n8k64.s4

    For each warp (1 of 8, covering 16 rows × 8 cols):
      Load A_fragment from s_a (col_major, 16×64)
      For n_chunk in range(0, BN, 8):         # 8 column groups
        Load B_fragment from s_b (row_major, 64×8)
        mma.sync.aligned.m16n8k64.row.col.s4.s4.s32
  End K loop

  # Epilogue: post-multiply by per-token s_x and per-row w_scale
  For each output element in accumulator:
    out_row = m_start + local_m
    out_col = n_start + local_n
    y[out_row, out_col] = accum[local_m, local_n]
                         * s_x[out_row]
                         * w_scale[out_col]
    → convert to bf16, store to global memory
```

**Post-multiply derivation:**
```
accum[t][n] = sum_{k} x_q[t][k] * W'_q[n][k]          (TC int32 output)

y[t][n] = sum_k x[t][k] * W[n][k]
        = sum_k (x_q[t][k] * s_x[t]) * (W'_q[n][k] * w_scale[n])  (dequantize both)
        = s_x[t] * w_scale[n] * sum_k x_q[t][k] * W'_q[n][k]
        = s_x[t] * w_scale[n] * accum[t][n]

No division by 7 is needed because:
  s_x[t] = max(|x_clipped|) / 7    →  x_dq[t][k] = x_q[t][k] * s_x[t]
  w_scale[n] = row_max / 7          →  W'_dq[n][k] = W'_q[n][k] * w_scale[n]
Both scales already include the /7 normalization.
```

### 6.4 Setup & Build (`csrc/setup.py`)

```python
CUDAExtension("peqat_cuda", [
    "bindings.cpp",
    "peqat_preprocess_per_token_sm121.cu",      # NEW
    "peqat_fused_int4_tc_sm121.cu",             # NEW
    "peqat_preprocess.cu",                      # keep as reference
    "peqat_gemm_bf16_wmma.cu",                  # keep as bf16 fallback
], extra_compile_args={"nvcc": ["-arch=sm_121f"]})
```

### 6.5 Python Bindings (`csrc/bindings.cpp`)

```cpp
// Per-token preprocess: x/smooth → clamp → per-token max/7 → pack 4×INT4
// Returns {x_packed int32[M, K/4], s_x float32[M]}
std::vector<torch::Tensor> preprocess_per_token_forward(
    torch::Tensor x,        // [M, K] bf16
    torch::Tensor smooth,   // [K] bf16
    torch::Tensor clip      // [K] bf16
);

// Fused INT4 TC GEMM with per-token + per-row scale epilogue
// Returns y[M, N] bf16
torch::Tensor fused_int4_tc_per_token_forward(
    torch::Tensor x_packed,   // [M, K/4] int32
    torch::Tensor w_packed,   // [N, K/4] int32
    torch::Tensor w_scale,    // [N] bf16
    torch::Tensor s_x,        // [M] f32
    int64_t rows,             // N
    int64_t cols              // K
);
```

---

## 7. Runtime Changes (`runtime/loader.py`)

### 7.1 `CUDAQuantizedLinear`

```python
class CUDAQuantizedLinear(nn.Module):
    def __init__(self, packed_w_uint8, w_scale, smooth, clip,
                 out_features, in_features):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features

        self.register_buffer("smooth", smooth.contiguous().bfloat16())    # [K]
        self.register_buffer("clip", clip.contiguous().bfloat16())        # [K]
        self.register_buffer("w_scale", w_scale.contiguous().bfloat16())  # [N]

        # Repack: uint8[N, K_packed] → int32[N, K/4] (once at init)
        w_packed = self._repack_for_tc(packed_w_uint8, out_features, in_features)
        self.register_buffer("w_packed", w_packed.contiguous())           # [N, K/4] int32

    def _repack_for_tc(self, packed_bytes, rows, cols):
        """Convert existing uint8 low_high packing → INT4 TC int32 packing."""
        # Unpack from uint8[N, K_packed] to int8[N, K]
        w_q = self._unpack_int4(packed_bytes, rows, cols)      # [N, K] int8

        # Repack 4 INT4 per int32 for TC
        w_i32 = w_q.to(torch.int32) & 0xF
        w_groups = w_i32.view(rows, -1, 4)                     # [N, K/4, 4]
        packed = (w_groups[:,:,0]
                | (w_groups[:,:,1] << 4)
                | (w_groups[:,:,2] << 8)
                | (w_groups[:,:,3] << 12))
        return packed                                           # [N, K/4] int32

    @staticmethod
    def _unpack_int4(packed_bytes, rows, cols):
        """Unpack uint8 low_high → int8."""
        import torch
        packed = torch.frombuffer(packed_bytes, dtype=torch.uint8).clone()
        K_packed = (cols + 1) // 2
        packed = packed[:rows * K_packed].reshape(rows, K_packed)
        lo = (packed & 0xF).to(torch.int8)
        hi = ((packed >> 4) & 0xF).to(torch.int8)
        lo = torch.where(lo > 7, lo - 16, lo)
        hi = torch.where(hi > 7, hi - 16, hi)
        w = torch.empty(rows, cols, dtype=torch.int8)
        # Protect against odd cols
        w[:, ::2] = lo[:, :(cols + 1) // 2]
        w[:, 1::2] = hi[:, :cols // 2]
        return w

    @torch.no_grad()
    def forward(self, x):
        ndim = x.dim()
        if ndim == 3:
            B, S, K = x.shape
            x_2d = x.reshape(B * S, K)           # [M, K]
        else:
            x_2d = x

        x_packed, s_x = peqat_cuda.preprocess_per_token_forward(
            x_2d, self.smooth, self.clip
        )                                        # [M, K/4] int32 + [M] f32

        y = peqat_cuda.fused_int4_tc_per_token_forward(
            x_packed, self.w_packed, self.w_scale, s_x,
            self.out_features, self.in_features
        )                                        # [M, N] bf16

        if ndim == 3:
            y = y.reshape(B, S, self.out_features)
        return y.to(x.dtype)
```

### 7.2 `build_cuda_model`

Remove all folded-weight parameters and conditional branches. Single deterministic path:

```python
def build_cuda_model(export_dir, base_model_name, device="cuda"):
    model, tokenizer = load_base_model(base_model_name, device)
    export = load_export(export_dir, device)

    replacements = {}
    for name, module in model.named_modules():
        safe_name = f"base_model.model.{name}"
        if safe_name not in export["smooth"]:
            continue
        layer = export["layers"][safe_name]
        replacements[name] = CUDAQuantizedLinear(
            packed_w_uint8=layer["packed_w_bytes"],   # raw bytes from weight_int4.bin
            w_scale=layer["w_scale"],
            smooth=export["smooth"][safe_name],
            clip=export["clip"][safe_name],
            out_features=layer["rows"],
            in_features=layer["cols"],
        )

    for name, new_module in replacements.items():
        parent_path, child_name = name.rsplit(".", 1)
        parent = model
        for part in parent_path.split("."):
            parent = getattr(parent, part)
        setattr(parent, child_name, new_module)

    return model.to(device), tokenizer
```

---

## 8. Implementation Order

| Phase | File(s) | What | Verification |
|-------|---------|------|-------------|
| **P1** | `fake_quant.py` | `PerTokenActSTEQuant` class | Forward numeric match against Python reference; gradient correctness via `torch.autograd.gradcheck` |
| **P2** | `fake_quant.py` | Swap call in `PEQATLoRAModule.forward` | Train 1 epoch, verify loss converges |
| **P3** | `scripts/export_peqat.py`, `quant/fold_act_scale.py` | Remove `--fold-weights`, delete fold file | Export runs clean, no folded weight artifacts |
| **P4** | `csrc/peqat_preprocess_per_token_sm121.cu`, `csrc/bindings.cpp` | Preprocess kernel | `max|preprocess_kernel(x,s,c) - torch_reference(x,s,c)| < 1e-5` over random inputs |
| **P5** | `runtime/loader.py` | `_repack_for_tc()` host-side utility | Round-trip: `original_packed → repack → unpack → original_packed` is identity |
| **P6** | `csrc/peqat_fused_int4_tc_sm121.cu`, `csrc/bindings.cpp` | Fused INT4 TC kernel | Single layer: `|y_kernel - y_torch_reference| < 1e-4` over random inputs |
| **P7** | `runtime/loader.py` | `CUDAQuantizedLinear.forward` + `build_cuda_model` | Model loads end-to-end, single forward pass matches reference |
| **P8** | `csrc/setup.py` | Update build to include new CUDA sources | `pip install -e .` succeeds, `peqat_cuda` module has all ops |
| **P9** | `scripts/run_benchmarks.py` | Benchmarks + accuracy eval | WikiText-2 PPL within 0.5 of training-time PPL |

---

## 9. Verification Gates

| Gate | Condition | Fail action |
|------|-----------|------------|
| Unit: PerTokenActSTEQuant | Forward output matches reference within `1e-5`. Backward gradient check passes. | Fix autograd function |
| Unit: preprocess kernel | `max|preprocess_kernel - reference| < 1e-5` | Fix kernel math |
| Unit: weight repack | `repack → unpack` round-trip reproduces original values | Fix packing permutation |
| Unit: single-layer TC | `max|fused_int4_tc - reference_linear| < 1e-4` per layer | Fix tile config, fragment layout, or post-scale |
| Integration: per-layer | All 224 target layers individually match reference within tolerance | Fix per-layer padding/alignment |
| Integration: full model PPL | WikiText-2 PPL within 0.5 of training-time per-token PPL | Debug per-layer numeric drift |
| Integration: end-to-end | Model generates coherent text on a test prompt | Fix edge-case shapes (odd K, M=1) |

---

## 10. Critical SM 12.1 Specifics

| Concern | Detail |
|---------|--------|
| **Compiler target** | `-arch=sm_121f` — the `f` suffix (family mode) is **required** for `.s4`/`.u4` sub-byte MMA. Standard `sm_121` will generate illegal instruction or fallback code. |
| **Fragment layout** | Must be verified empirically. Compile a minimal inline PTX test with `--ptxas-options=-v`, inspect register count and fragment layout. A-fragment likely 4×int32 per thread, B-fragment 1×int32 per thread, C-fragment 2×int32 per thread. Adjust packing permutation in `_repack_for_tc` if needed. |
| **Thread block clusters** | Only `1x1x1` supported on SM 12.1 — no cluster-level synchronization to worry about. |
| **TMEM** | Not available on SM 12.1 — all operands from registers or shared memory. |
| **No `tcgen05`** | SM 12.1 uses `mma.sync` only. Every BK=64 tile requires issuing `m16n8k64` MMA instructions. |
| **`ldmatrix.b4x16`** | Check if `ldmatrix.b8x16.b4x16_p64` or a `.s4` variant exists on SM 12.1 for faster weight tile shared memory load. If available, implement in Step 6 of the fused kernel. |
| **Register pressure** | 48 warps/SM limit on SM 12.0. The fused kernel must keep register count low (target ≤ 64 registers per thread) to maintain occupancy. Post-scale multiply in epilogue vs. inline is a tradeoff. |
| **Memory bound** | 273 GB/s LPDDR5x unified memory is the bottleneck. Kernel is memory-bound, not compute-bound. Activate reads (bf16 × 2 bytes per element) dominate bandwidth. The INT4 TC reduces compute latency but not memory pressure. Consider kernel fusion with preceding RMSNorm/QKV projection if memory traffic is the measured bottleneck. |

---

## 11. Non-Goals

- **SM 80/89 Ampere/Ada fallback** — not required. Target SM 12.1 exclusively.
- **Multi-GPU** — single DGX Spark only.
- **KV-cache quantization** — out of scope.
- **Folded weights** — replaced entirely by per-token dynamic scale.
- **`wgmma` or `tcgen05` paths** — not available on SM 12.1.
- **CUTLASS integration** — pure custom kernel. No CUTLASS dependency.
- **MXFP4** — this plan targets `.s4` (INT4), not microscaling FP4. Different PTX instruction family.
