# PE-QAT Deployment — Complete Reference

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
│       └── harness.py         measure_prefill, measure_decode, measure_memory, run_suite
│
├── csrc/                      OUR CUDA kernels
│   ├── peqat_preprocess.cu    Activation path: x/smooth → clamp(-clip,clip) → rint(x/(clip/7)) → clamp(-8,7)
│   ├── peqat_gemm.cu          Unpack INT4 nibbles → dequantize with per-row FP16 scale → matmul via cuBLAS
│   ├── bindings.cpp           pybind11: preprocess_forward(), linear_forward()
│   └── setup.py               Build with torch.utils.cpp_extension.CUDAExtension
│
├── scripts/                   OUR CLI scripts
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
│   └── run_benchmarks.py           Latency/memory benchmark (CUDA only or with FP16 baseline)
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
- Input: training directory path, optional checkpoint name
- Scans for: training YAML, adapter_config.json, adapter_model.safetensors, quantization_params.pth, tokenizer files, model config
- Returns: dict of normalized absolute paths

### load_export_job(discovered)
- Input: discovered dict
- Reads training YAML + adapter_config.json
- Populates ExportJob dataclass (base_model_name, lora_rank, lora_alpha, target_modules, bits, etc.)
- Default export_dir = `<parent>/<source_basename>_export`
- Returns: ExportJob

### merge_lora_weights(peft_model, target_modules, lora_rank, lora_alpha)
- For each module name containing any target_modules substring:
  - Loads base_weight from module.weight
  - Loads lora_A.default.weight, lora_B.default.weight
  - lora_delta = (lora_B @ lora_A) * (lora_alpha / lora_rank)
  - w_eff = base_weight + lora_delta (matching dtype/device)
  - Stores w_eff.cpu().half()
- Returns: dict[module_name] → tensor[D, K]
- Also saves debug_<name>_merged_fp16.pt if debug_layer matches

### extract_quant_params(quant_params_path, merged_weights)
- Loads quantization_params.pth (dict: `{name}.qat_core.{param}` → tensor)
- Strips `.qat_core.activation_clip` / `.qat_core.smooth_factor` to get module_name
- Validates smooth/clip size == input dim K of each module
- Returns: dict[module_name] → {"smooth": tensor[K], "clip": tensor[K]}

### fold_smoothing(merged_weights, quant_params)
- In-place: `w.mul_(smooth.unsqueeze(0))` for each module
- No clone (saves ~15GB for 8B model)
- Returns: same dict (mutated)

### quantize_weights(smoothed_weights)
- Per module [D, K]:
  - row_max = W.abs().amax(dim=1)
  - scale = (row_max / 7).clamp(min=1e-8)
  - W_q = clamp(round(W / scale.unsqueeze(1)), -8, 7).int8()
  - W_dq = W_q.float() * scale.unsqueeze(1)
- Computes error stats: max_abs_error, mean_abs_error, relative_frobenius
- Returns: quantized dict, error_stats dict

### pack_int4(quantized, smoothing_folded=True)
- Packing format (low_high): low nibble = first value, high nibble = second value
- Pads to even K if needed
- 16-byte alignment with zero padding
- pack_meta records: layout_version, rows, cols, pack_order, signed_range, smoothing_folded, etc.
- Returns: dict[module_name] → {weight_int4_bytes, weight_scale_bytes, shape, pack_meta}

### Manifest & Save
- manifest.json: source/export dirs, base_model_name, bit widths, target_modules list, num_exported_layers, weight_artifact_paths per module
- Export dir: `<basename>_export/`
  - manifest.json, export_config.json, model_config.json (optional)
  - metadata/: adapter_config, source_training_config, export_log, layer_map, selected_checkpoint
  - tokenizer/: model, config, special_tokens
  - activation/: smooth.safetensors (keyed by full module name), clip.safetensors, activation_meta.json
  - weights/<module_safe_name>/: weight_int4.bin, weight_scale.fp16, shape.json, pack_meta.json
  - validation/: layer_error_stats.json, sample_outputs.json (from validate), export_summary.json

### validate_export.py
- For each module: loads packed weight, unpacks, compares against export's error stats
- Pack/unpack round-trip is lossless (verified on 224 modules)
- Writes sample_outputs.json

## CUDA Kernels (Stage 2 Phases A-B — built, correctness-verified)

### peqat_preprocess.cu — preprocess_forward_kernel

```
Input:  x[N,K] float32, smooth[K] float32, clip[K] float32
Output: x_q[N,K] int8, x_dq[N,K] float32, x_scale[K] float32
```

Per element (n,k):
```
x_s = x[n,k] / smooth[k]
x_c = clamp(x_s, -clip[k], clip[k])
s_x = clip[k] / 7.0
q = rint(x_c / s_x)
q = clamp(q, -8, 7)
x_q[n,k] = (int8)q
x_dq[n,k] = q * s_x
if n == 0: x_scale[k] = s_x
```

- 1D grid: `total = N*K`, `threads=256`, one element per thread
- Uses `__float2int_rn` (round to nearest, ties to even — matches PyTorch)
- `__restrict__` pointers
- Uses current CUDA stream

### peqat_gemm.cu — unpack_dequant_kernel

```
Input:  packed[P] uint8, w_scale[D] float32
Output: w_dq[D,K] float32
```

Per element (d,k):
```
packed_idx = (d*K + k) / 2
is_low = (k % 2 == 0)
nibble = is_low ? (byte & 0xF) : ((byte >> 4) & 0xF)
q = nibble > 7 ? (nibble - 16) : nibble   # unsigned [0,15] → signed [-8,7]
w_dq[d,k] = q * w_scale[d]
```

### PyTorch Bindings (bindings.cpp)

Two functions exposed via pybind11:

```python
peqat_cuda.preprocess_forward(x, smooth, clip) -> (x_q, x_dq, x_scale)
peqat_cuda.linear_forward(x_dq, packed_w, w_scale, rows, cols) -> y
  # internally: launch_unpack_dequant → w_dq → torch::matmul(x_dq, w_dq.T)
```

- Shape/dtype/device validation on all inputs
- x_dq: float32, packed_w: uint8, w_scale: float16 (auto-cast to float32 internally)
- Returns float32

## Runtime Models

### cpu_model.py — CPU/GPU sanity check (PackedQuantizedLinear)

- `_load_packed_weight(layer_dir)`: reads weight_int4.bin, unpacks nibbles, converts signed
- `PackedQuantizedLinear.forward(x)`:
  - `w_dq = w_q.float() * scale.unsqueeze(1)`  # dequantize weights
  - `x_smooth = x / smooth`                     # activation smoothing
  - `x_clipped = clamp(x_smooth, -clip, clip)`  # PACT clip
  - `s_x = clip / 7.0`                          # activation scale
  - `x_q = clamp(round(x_clipped / s_x), -8, 7)`
  - `x_dq = x_q * s_x`
  - `return x_dq @ w_dq.T`
- `build_packed_model(export_dir, base_model_name, device)`:
  - Loads HF model
  - For each nn.Linear whose safe_name matches a smooth_dict key:
    - Creates PackedQuantizedLinear with that layer's data
    - Replaces the module via `setattr(parent, child_name, new_module)`
  - Returns: model, tokenizer

### loader.py — CUDA runtime (CUDAQuantizedLinear + build_cuda_model)

- `CUDAQuantizedLinear.forward(x)`:
  - Flattens 3D→2D if needed (handles batch×seq_len)
  - `x_dq = peqat_cuda.preprocess_forward(x_f32, smooth, clip)[1]`
  - `y = peqat_cuda.linear_forward(x_dq, packed_w, w_scale, out_features, in_features)`
  - Reshapes 2D→3D if needed
  - Casts output back to input dtype
- `build_cuda_model(export_dir, base_model_name)`:
  - Same replacement pattern as cpu_model.py
  - But creates CUDAQuantizedLinear with pre-loaded packed_w, w_scale, smooth, clip
  - Loads all 224 layers' artifacts to GPU at init (~4GB)
  - Returns: model, tokenizer

## Benchmark Harness (eval/harness.py)

### measure_prefill(model, input_ids, warmup=3, repeats=10)
- Runs full forward pass on input_ids
- Returns: ms, tokens, ms_per_token

### measure_decode(model, prompt_len=128, decode_tokens=50, warmup=3, repeats=5)
- Autoregressive generation: for each step, model(input_ids), argmax next token, append
- Returns: ms_per_token, tokens_per_sec

### measure_memory(model)
- torch.cuda.max_memory_allocated() + max_memory_reserved()
- Returns: gpu_allocated_gb, gpu_reserved_gb

### run_suite(model, tokenizer, suite_name, output_dir, seq_lens)
- Runs prefill + decode + memory for each seq_len
- Writes prefill.csv and decode.csv to output_dir

## Export Format (binary layout)

### weight_int4.bin
- Flat uint8 buffer, row-major
- For each row: pairs of consecutive columns packed into one byte
- Byte layout: `[val0_nibble_low | val1_nibble_high]` (low_high order)
- If K is odd, padded to K+1 before packing
- 16-byte zero-padded at end
- Values: unsigned nibble [0,15], mapped to signed [-8,7] by subtracting 16 if > 7

### weight_scale.fp16
- Flat float16 buffer, one per row: `D * 2 bytes`
- Row i scale = max|W'[i,:]| / 7

### shape.json
```json
{"rows": D, "cols": K}
```

### smooth.safetensors / clip.safetensors
- Keyed by full module name: `base_model.model.model.layers.N.{attn|mlp}.{proj}`
- Values: float32 tensors of length K (input dimension)

## Known Issues

1. **PyTorch 12.1 warning**: DGX Spark GB10 has compute cap 12.1, PyTorch supports up to 12.0. Non-fatal.
2. **Non-writable buffer warning**: `torch.frombuffer()` on file bytes. Harmless — `.clone()` makes it writable.
3. **Full dequant in GEMM**: Current linear_forward materializes FP32 weight matrix (~8GB per forward). Fused kernel needed for speed (Phase F).
4. **No epsilon guard in preprocess kernel**: If clip[k] == 0, s_x = 0 causes division by zero. Training clamps clip ≥ 1e-6, so not triggered in practice.
5. **packed_idx formula**: `(r*cols + c) / 2` works for even cols only (all Llama projections are even). Safer: `r * ((cols+1)/2) + c/2`.
6. **Benchmark only on one checkpoint**: Results may not generalize to other models/tasks.

## Remaining Work

### Phase E — Accuracy Evaluation (HIGH for paper, LOW for kernel correctness)
- Install lm-evaluation-harness (`pip install lm-eval`)
- `eval/accuracy.py`: wrap PIQA, HellaSwag, WinoGrande, ARC-E, ARC-C, OpenBookQA
- Run on FP16 baseline → compare with CUDA PEQAT → check accuracy preservation

### Phase F — Fused GEMM Kernel (HIGH for speedup)
- Replace current unpack+dequant+cuBLAS with a tiled kernel that reads INT4 directly
- Unpack nibbles in registers during matmul tile loading
- Avoid materializing FP32 weight matrix
- Targets: 2× faster than FP16 decode (from INT4 bandwidth advantage)
- Potential approaches: custom tiled GEMM, CUTLASS epilogue fusion, or cuBLASLt with custom type

### Paper Outputs (LOW priority)
- Generate CSVs for: main results table, accuracy table, runtime breakdown, speed-vs-accuracy scatter, sequence-length scaling
- Write COLM 2026 Efficient Reasoning Workshop paper
