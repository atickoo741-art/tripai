# PE-QAT Deployment — Complete Reference

## Project Goal

Implement a deployment path for PE-QAT that converts a trained PEFT+QAT checkpoint into packed INT4 inference artifacts, builds CUDA kernels for efficient inference, and benchmarks latency/throughput/memory. Target: COLM 2026 Efficient Reasoning Workshop.

## Repository Structure

```
~/Parameter-Efficient-QAT-for-W4A4/
├── fake_quant.py              ORIGINAL training code — PEQATLoRAModule, ActSTEQuant, WeightSTEQuant
├── peqat_training.py          ORIGINAL training script
├── config.yaml                ORIGINAL training config
├── peqat/                     (empty — training outputs go elsewhere)
├── baseline_lora_adapters/    ORIGINAL LoRA-only baselines
│
├── peqat_deploy/              OUR deployment package
│   ├── config/
│   │   ├── schema.py          ExportJob dataclass
│   │   └── loader.py          load_export_job() — reads training YAML + adapter_config.json
│   ├── io/
│   │   ├── discover.py        discover_source_dir() — scans training dir for all files
│   │   ├── hf_loader.py       load_base_model() — AutoModelForCausalLM in FP16
│   │   ├── peft_loader.py     load_adapter() — PEFT adapter with key remapping
│   │   └── save_artifacts.py  save_artifacts() — writes export tree
│   ├── quant/
│   │   ├── merge_lora.py      merge_lora_weights() — W_eff = W_base + (α/r)·B·A
│   │   ├── extract_quant_params.py  — parse quantization_params.pth, group by module
│   │   ├── fold_smoothing.py  fold_smoothing() — in-place W' = W_eff × smooth
│   │   ├── quantize_weights.py  — per-row INT4: scale = max|W'|/7, clamp round, save error stats
│   │   ├── pack_int4.py       — 2× INT4 per byte (low_high nibble), 16-byte aligned
│   │   ├── manifest.py        — build manifest.json with all metadata
│   │   └── reference.py       peqat_linear_reference() — Algorithm 1 in pure PyTorch
│   ├── runtime/
│   │   ├── cpu_model.py       PackedQuantizedLinear + build_packed_model() — CPU inference (sanity)
│   │   └── loader.py          CUDAQuantizedLinear + build_cuda_model() — GPU inference via kernels
│   └── eval/
│       └── harness.py         measure_prefill, measure_decode (KV-cached), measure_memory, run_suite
│
├── csrc/                      OUR CUDA kernels
│   ├── peqat_preprocess.cu    Activation path: x/smooth → clamp(-clip,clip) → rint(x/(clip/7)) → clamp(-8,7)
│   ├── peqat_gemm.cu          Phase B reference: unpack INT4 nibbles → dequantize FP32 → cuBLAS matmul
│   ├── peqat_fused_gemm.cu    Fused INT4 GEMM: tiled kernel with on-the-fly INT4→FP32 unpack (v2)
│   ├── bindings.cpp           pybind11: preprocess_forward(), linear_forward(), fused_linear_forward()
│   └── setup.py               Build with torch.utils.cpp_extension.CUDAExtension
│
├── scripts/
│   ├── export_peqat.py        Full export pipeline CLI
│   ├── inspect_peqat_dir.py   Inspect training checkpoint directory
│   ├── validate_export.py     Pack/unpack round-trip validation
│   ├── sanity_check.py        CPU model text generation test
│   ├── weight_spotcheck.py    Compare expected vs packed weights
│   ├── weight_compare.py      Alternative weight comparison (all layers)
│   ├── compare_peqat_vs_packed.py  Compare original PEQAT vs packed forward pass
│   ├── test_preprocess_kernel.py   CUDA preprocess test vs reference.py
│   ├── test_linear_kernel.py       CUDA linear_forward test (synthetic + real export)
│   ├── test_cuda_runtime.py        End-to-end CUDA forward + generation test
│   ├── test_fused_kernel.py        Fused kernel correctness test (synthetic vs reference + old path)
│   ├── bench_fused_kernel.py       Single-layer microbenchmark (old vs fused, per-shape)
│   └── run_benchmarks.py           Full model benchmark: prefill/decode/memory suite
│
├── plans/
│   └── fused_int4_gemm_design.md  Detailed design doc for the tiled fused kernel
│
├── STAGE1_COMPLETE.md         Stage 1 summary
├── peqat_kernel_project_plan.md  Full project plan
└── REFERENCE.md               This file
```

## PE-QAT Algorithm (what a trained checkpoint contains)

Each target linear layer has:

```
W_merged = W_base + (α/r) · B @ A              # LoRA merge (permanent)
x' = x / smooth_factor                          # SmoothQuant activation migration
W' = W_merged * smooth_factor                   # SmoothQuant weight migration (folded at export)
x_clipped = clamp(x', -activation_clip, activation_clip)  # PACT learned clipping
s_x = activation_clip / 7                       # Activation scale
x_q = clamp(round(x_clipped / s_x), -8, 7)     # INT4 activation quantization
W_q = clamp(round(W' / s_w), -8, 7)            # INT4 weight quantization (s_w = max|W'|/7 per row)
y = x_q * s_x @ (W_q * s_w)^T                  # Dequantize + matmul
```

**Trainable params per module:** LoRA A (r×K), LoRA B (D×r), smooth_factor (K), activation_clip (K).

## Export Pipeline (Stage 1 — complete, verified)

### discover_source_dir(source_dir, checkpoint)
- Scans for: training YAML, adapter_config.json, adapter_model.safetensors, quantization_params.pth, tokenizer, config
- Returns: dict of normalized absolute paths

### load_export_job(discovered)
- Reads YAML + adapter_config.json → ExportJob dataclass
- Default export_dir = `<parent>/<source_basename>_export`

### merge_lora_weights(peft_model, target_modules, lora_rank, lora_alpha)
- For each target module: `w_eff = base_weight + (lora_B @ lora_A) * (lora_alpha / lora_rank)`

### extract_quant_params(quant_params_path, merged_weights)
- Strips `.qat_core.activation_clip` / `.qat_core.smooth_factor` to get module_name
- Returns: `{module_name: {"smooth": tensor[K], "clip": tensor[K]}}`

### fold_smoothing(merged_weights, quant_params)
- In-place: `w.mul_(smooth.unsqueeze(0))` for each module (mutates, saves ~15GB)

### quantize_weights(smoothed_weights)
- Per-row: `scale = max|W'| / 7`, `W_q = clamp(round(W'/scale), -8, 7)`

### pack_int4(quantized)
- low_high nibble order: low nibble = first value, high = second value
- Pads to even K, 16-byte alignment
- Packed size per module: `D * ((K+1)/2)` bytes

### validate_export.py
- Pack/unpack round-trip lossless (verified on 224 Llama-3.1-8B modules)

## CUDA Kernels

### peqat_preprocess.cu — preprocess_forward_kernel (Phase A)

```
Input:  x[N,K] float32, smooth[K] float32, clip[K] float32
Output: x_q[N,K] int8, x_dq[N,K] float32, x_scale[K] float32
```

Per element: `x/smooth → clamp(-clip,clip) → rint(x/(clip/7)) → clamp(-8,7)`
- 1D grid, 256 threads, one element per thread
- Uses `__float2int_rn` (round-to-nearest, ties-to-even, matches PyTorch)

### peqat_gemm.cu — unpack_dequant_kernel (Phase B, reference path)

```
Input:  packed[P] uint8, w_scale[D] float32
Output: w_dq[D,K] float32
```
- Unpacks low_high nibbles → signed [-8,7] → multiplies by per-row scale
- Followed by `torch::matmul(x_dq, w_dq.T)` in bindings (cuBLAS)

### peqat_fused_gemm.cu — Fused INT4 GEMM (Phase F, v2)

Two kernels:

#### 1. fused_int4_gemm_kernel (prefill, M ≥ 8)

| Parameter | Value |
|-----------|-------|
| Tile (BM×BN×BK) | 128×128×32 |
| BK_STRIDE | 33 (BK+1, bank-conflict mitigation) |
| Threads | 256 |
| Per-thread fragment | 8×8 output elements (64 MACs/K-tile) |
| Groups (M×N) | 16×16 |
| Shared memory | s_A[4224], s_B[4224], s_scale[128] = 34KB |

Algorithm per K-tile:
1. **A-tile load**: cooperative, each thread loads 16 activation elements → s_A (scalar, stride 33)
2. **B-tile load (byte-pair)**: each thread loads 8 packed bytes, each byte → 2 INT4 values, dequantize with `s_scale[row]`, store as FP32 to s_B (stride 33)
3. **syncthreads**
4. **Compute**: 3-deep loop (kk → tm → tn), manual FP32 MACs, registers only
5. **syncthreads**

Key optimizations:
- **Byte-pair loading**: 8 packed bytes/thread instead of 16 values, halves global reads
- **s_scale[BN]**: w_scale cached in shared memory, loaded once per K-tile (512 bytes)
- **BK_STRIDE = 33**: breaks shared memory bank conflicts (stride 32 → all threads hit bank kk%32)
- **No FP32 weight materialization**: weights live in s_B tile-by-tile, never in DRAM

#### 2. fused_int4_gemm_smallm_kernel (decode, M ≤ 8)

| Parameter | Value |
|-----------|-------|
| Tile (BM×BN×BK) | 8×128×32 |
| BK_STRIDE | 33 |
| Threads | 256 |
| Per-thread output | 1×4 elements |
| Mapping | m_local = tid/32, n_base = (tid%32)×4 |

Dispatch logic: `fused_linear_forward` selects kernel based on `M ≤ 8`.
- M=1 during KV-cached decode: 32 threads compute valid output (1 row × 128 cols), 224 threads compute zeros
- s_A zero-initialized for invalid rows (fixes UB from uninitialized shared memory reads)

### bindings.cpp — PyTorch interface

Three exposed functions:

```python
peqat_cuda.preprocess_forward(x, smooth, clip) -> (x_q, x_dq, x_scale)
peqat_cuda.linear_forward(x_dq, packed_w, w_scale, rows, cols) -> y       # old reference path
peqat_cuda.fused_linear_forward(x_dq, packed_w, w_scale, rows, cols) -> y # new fused path
```

`fused_linear_forward` dispatches to prefill or decode kernel based on M ≤ 8.

## Runtime Models

### loader.py — CUDAQuantizedLinear + build_cuda_model

- `CUDAQuantizedLinear.forward(x)`:
  1. Flatten 3D→2D (handles batch×seq_len)
  2. `x_dq = peqat_cuda.preprocess_forward(x_f32, smooth, clip)[1]` (separate kernel launch)
  3. `y = peqat_cuda.fused_linear_forward(x_dq, ...)` (fused tile kernel)
  4. Restore 3D shape, cast to input dtype
- `build_cuda_model(export_dir, base_model_name)`: loads base model, replaces 224 target nn.Linear with CUDAQuantizedLinear, moves to GPU (~4GB for packed weights)

**Critical:** Preprocess and GEMM are still separate kernel launches. x_dq is written to and read from global memory between them. Future work: fuse preprocess into GEMM tile load.

### cpu_model.py — PackedQuantizedLinear (sanity check, not fast)

- Pure PyTorch reference: unpack all weights, do FP32 matmul
- `build_packed_model(export_dir, base_model_name, device)`: for CPU or GPU

## Benchmark Harness (eval/harness.py)

### measure_prefill(model, input_ids, warmup=3, repeats=10)
- Full forward pass on `input_ids` shape [1, prompt_len]
- Returns: ms, tokens, ms_per_token

### measure_decode(model, prompt_len=128, decode_tokens=50, warmup=3, repeats=5)
- **Uses KV cache** (fixed): prefill full prompt → then each step feeds `[1, 1]` (last token) + `past_key_values`
- Without KV cache, each step processes the full growing sequence (M=128+ → prefill kernel, small-M kernel never fires)
- Returns: ms_per_token, tokens_per_sec

### run_suite(model, tokenizer, suite_name, output_dir, seq_lens)
- Runs prefill + decode + memory for each seq_len
- Writes prefill.csv and decode.csv to output_dir

## Scripts

### run_benchmarks.py
- **Usage:** `python scripts/run_benchmarks.py [--device cpu|cuda] <export_dir>`
- Loads FP16 baseline, measures prefill/decode/memory → unloads
- Loads CUDA PEQAT model, measures same metrics → unloads
- **Output dir auto-increments:** `<base>_benchmarks`, `_benchmarks_1`, `_benchmarks_2`, etc.
- Fixed argument parsing bug: `--device` can appear before or after export_dir

## Export Format

### weight_int4.bin
- Flat uint8, row-major, low_high nibble: `[val0_low | val1_high]`
- Unsigned [0,15] → signed [-8,7]: `val = nibble > 7 ? nibble - 16 : nibble`
- K padded to even then packed, 16-byte aligned

### weight_scale.fp16
- Flat float16, one per row: D × 2 bytes
- Row i scale = `max|W'[i,:]| / 7`

### shape.json
```json
{"rows": D, "cols": K}
```

### smooth.safetensors / clip.safetensors
- Keyed by `base_model.model.model.layers.N.{attn|mlp}.{proj}`
- Float32 tensors of length K (input dimension)

## Benchmark Results (DGX Spark GB10)

### Per-Token Activation Quantization (best results) — Llama-2-7B
Checkpoint: `output_qat_lora_llama2_7b_per_tok_ac_05fa_001_export` (per-token activation quantization path).

| Metric | FP16 | CUDA PEQAT | Ratio |
|--------|------|------------|-------|
| Prefill (seq 128) | 90.5ms | 112.3ms | 1.2× |
| Prefill (seq 512) | — | 344.5ms | — |
| Decode (prompt 128) | 76.4ms/tok | **28.2ms/tok** | **0.37×** (2.7× faster) |
| Decode (prompt 512) | — | 32.4ms/tok | — |
| GPU memory (alloc) | 13.29 GB | **6.89 GB** | **1.9× less** |
| GPU memory (resv) | 13.32 GB | 7.63 GB | 1.7× less |

Decode is **2.7× faster than FP16** — the per-token activation quantization path makes the small-M kernel efficient enough to beat cuBLAS FP16 at M=1. Prefill is close to FP16 (1.2× slower vs 4.4× previously).

### Legacy: v1 Kernel (unpack+cuBLAS, no KV cache, seq_len 128) — Llama-3.1-8B
| Metric | FP16 | CUDA PEQAT | Ratio |
|--------|------|------------|-------|
| Prefill | 99.7ms | 455.3ms | 4.6× |
| Decode (no KV) | 106.2ms/tok | 467.7ms/tok | 4.4× |
| GPU memory | 17.24 GB | 6.69 GB | 2.6× less |

### Legacy: v2 Fused Kernel (no KV cache, seq_len 128) — Llama-3.1-8B
| Metric | FP16 | CUDA PEQAT | Ratio |
|--------|------|------------|-------|
| Prefill | 95.3ms | 417.4ms | 4.4× |
| Decode (no KV) | 99.3ms/tok | 623.8ms/tok | 6.3× |

### Legacy: v2 Fused Kernel (WITH KV cache, prompt_len 128) — Llama-3.1-8B
| Metric | FP16 | CUDA PEQAT | Ratio |
|--------|------|------------|-------|
| Prefill | 95ms | 420ms | 4.4× |
| Decode (KV cached) | 75.6ms/tok | 148ms/tok | 2.0× |
| Decode (KV cached, seq 512) | — | 155.2ms/tok | — |
| GPU memory | 15.8 GB | 5.4 GB | 2.9× less |

### Observations
- Per-token activation quantization decode (28.2ms/tok) is **2.7× faster than FP16 baseline** (76.4ms/tok) — the INT4 fused kernel with KV cache now outperforms cuBLAS FP16 at M=1
- Prefill narrowed to 1.2× slower (112.3ms vs 90.5ms), down from 4.4× in the legacy path
- Memory savings: 1.9× less GPU memory vs baseline (6.89 GB vs 13.29 GB allocated)
- Legacy Llama-3.1-8B numbers included for reference — larger model shifts absolute latencies but confirms the per-token quantization path is the key improvement

## Known Issues

1. **PyTorch 12.1 warning**: DGX Spark GB10 has compute cap 12.1, PyTorch supports up to 12.0. Non-fatal.
2. **Non-writable buffer warning**: `torch.frombuffer()` on file bytes. Harmless — `.clone()` makes it writable.
3. **No epsilon guard in preprocess**: If clip[k] == 0, s_x = 0 causes division by zero. Training clamps clip ≥ 1e-6.
4. **packed_idx formula**: `(r*cols + c) / 2` works for even cols only (all Llama projections are even).
5. **Fused kernel slower than cuBLAS**: For prefill (M=128+), the tiled kernel's overhead exceeds bandwidth savings. Needs activation fusion + INT4 tensor cores.
6. **x_dq global-memory round-trip**: Preprocess writes x_dq to DRAM, then GEMM reads it back. Eliminate by fusing activation quantization into GEMM tile load.
7. **Benchmark suite runs out of memory for seq_len 2048**: Default seq_lens=[128, 512]; 2048 requires ~70GB for prefill.
8. **Unused variable warning**: `int N` in `linear_forward` bindings — harmless.

## Remaining Work

### Phase E — Accuracy Evaluation
- Install lm-evaluation-harness (`pip install lm-eval`)
- `eval/accuracy.py`: wrap PIQA, HellaSwag, WinoGrande, ARC-E, ARC-C, OpenBookQA
- Compare FP16 baseline vs CUDA PEQAT

### Phase F.2 — Fuse Activation Preprocess into GEMM (HIGH for performance)
- Instead of `x_dq = preprocess(x) → fused_linear(x_dq, ...)`, accept `x`, `smooth`, `clip` directly
- Load `x` tile + apply `x/smooth → clamp → quantize → dequantize` during A-tile load
- Eliminates: preprocess kernel launch + x_dq global-memory write + x_dq global-memory read
- Saves 224 kernel launches and `2×M×K×4` bytes per layer

### Phase F.3 — True INT4 Tensor Core Path (HIGH for performance)
- WMMA with INT4 input types, INT32 accumulation, scale epilogue
- Eliminates all FP32 operations in the GEMM
- Target: match or beat cuBLAS throughput

### Fuse All 7 Linears Per Layer (MEDIUM for performance)
- Single kernel that reads layer input once, computes QKV + O + gate/up/down, writes output
- Reduces 14 kernel launches per layer → 1

### Paper Outputs (LOW priority)
- Generate CSVs for: main results table, accuracy table, runtime breakdown, speed-vs-accuracy scatter, sequence-length scaling
- Write COLM 2026 Efficient Reasoning Workshop paper

## Platform Details

**DGX Spark (GB10):**
- NVIDIA GB10 chip, compute capability 12.1 (Blackwell), aarch64
- 128 GB unified memory, ~120 GB GPU-accessible
- PyTorch 2.10.0.dev20251021+cu130 (supports max compute 12.0 — warning non-fatal)
- CUDA 13.0, driver 580.95.05
- Shared with vLLM Docker containers (Qwen 35B + Embedding 8B) consuming ~58 GB GPU memory when running
