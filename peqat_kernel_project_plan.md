# PE-QAT Kernel Project Plan for COLM 2026 Efficient Reasoning Workshop

This plan defines an exact two-stage implementation and evaluation roadmap for turning a PE-QAT training checkpoint into a deployable runtime with measurable inference speedups on reasoning-oriented benchmarks. The workshop explicitly welcomes quantized execution, inference acceleration, and empirical latency/throughput/memory studies for reasoning models under real-world constraints.[cite:1] The original PE-QAT work provides the algorithmic forward pass and states that current results are based on fake quantization, leaving real low-bit deployment as future work.[cite:2]

## Project objective

Implement a deployment path for PE-QAT that converts a trained PEFT+QAT checkpoint into packed inference artifacts and executes PE-QAT-style inference with custom CUDA kernels, then evaluate end-to-end speed, memory, and accuracy on reasoning-oriented benchmarks suitable for the workshop.[cite:1][cite:2]

Success criteria:
- Export a PE-QAT training folder into a deterministic inference artifact folder with the same basename plus `_export`.[cite:2]
- Support exact PE-QAT activation preprocessing semantics at runtime: smoothing, clipping, per-channel activation quantization.[cite:2]
- Support per-output-channel weight quantization and packed INT4 storage for every targeted linear layer.[cite:2]
- Run benchmarked inference and compare against FP16 and at least one deployable quantized baseline.[cite:1][cite:2]
- Produce paper-ready tables and figures for a 4–10 page non-archival workshop submission.[cite:1]

## Top-level deliverables

1. A host-side export pipeline that reads a PE-QAT training directory and writes an inference artifact directory named `<original_folder>_export`.[cite:2]
2. A CUDA runtime integrated with PyTorch that loads packed artifacts and runs PE-QAT inference-critical kernels.[cite:2]
3. A reproducible benchmark suite measuring latency, throughput, memory, and accuracy on reasoning-oriented tasks such as PIQA, HellaSwag, WinoGrande, ARC, and OpenBookQA.[cite:1]
4. A workshop paper centered on the deployment gap between fake quantization and real low-bit reasoning inference.[cite:1][cite:2]

---

## Stage 1 — Offline export pipeline

Stage 1 converts a PE-QAT training checkpoint folder into a clean deployment artifact tree. This stage is host-side and should be implemented in Python first, not CUDA, because it is not latency-critical and only needs to run once per exported checkpoint.[cite:2]

### Stage 1 goal

Given a source PE-QAT folder such as:

```text
output_qat_lora_llama2_7b_max_init_s_57e7_001/
```

produce:

```text
output_qat_lora_llama2_7b_max_init_s_57e7_001_export/
```

with all information required for runtime PE-QAT inference.[cite:2]

### Stage 1 scope

The exporter must:
- Discover model, adapter, tokenizer, training config, and quantization parameter files in the source folder.
- Optionally select a specific `checkpoint-*` subdirectory.
- Load the base model name from training config or adapter metadata.
- Load LoRA adapter weights from PEFT-format files (`adapter_config.json`, `adapter_model.safetensors`). Hugging Face PEFT expects adapter loading from a directory containing `adapter_config.json` plus adapter weights.[cite:3]
- Merge LoRA into each target linear layer.
- Extract learned PE-QAT smoothing and clipping parameters from `quantization_params.pth`.
- Apply weight-side preprocessing needed for deployment.
- Quantize each target linear layer to per-output-channel INT4.
- Pack quantized weights into the runtime layout expected by Stage 2 kernels.
- Save runtime metadata, activation metadata, tokenizer files, and validation outputs.

### Stage 1 required file structure

Implement the repository like this:

```text
peqat-deploy/
├── scripts/
│   ├── inspect_peqat_dir.py
│   ├── export_peqat.py
│   ├── validate_export.py
│   └── run_benchmarks.py
├── peqat_deploy/
│   ├── __init__.py
│   ├── config/
│   │   ├── loader.py
│   │   └── schema.py
│   ├── io/
│   │   ├── discover.py
│   │   ├── hf_loader.py
│   │   ├── peft_loader.py
│   │   └── save_artifacts.py
│   ├── quant/
│   │   ├── extract_quant_params.py
│   │   ├── merge_lora.py
│   │   ├── fold_smoothing.py
│   │   ├── quantize_weights.py
│   │   ├── pack_int4.py
│   │   └── manifest.py
│   ├── runtime/
│   │   ├── loader.py
│   │   ├── metadata.py
│   │   └── layout.py
│   └── eval/
│       ├── harness.py
│       ├── latency.py
│       └── accuracy.py
├── csrc/
│   ├── bindings.cpp
│   ├── peqat_preprocess.cu
│   ├── peqat_gemm.cu
│   └── CMakeLists.txt
└── exports/
```

### Stage 1 exported artifact layout

The exporter must produce this exact directory pattern:

```text
<source_basename>_export/
├── manifest.json
├── export_config.json
├── model_config.json
├── metadata/
│   ├── source_training_config.yaml
│   ├── adapter_config.json
│   ├── export_log.json
│   ├── layer_map.json
│   └── selected_checkpoint.json
├── tokenizer/
│   ├── tokenizer.model
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── activation/
│   ├── smooth.safetensors
│   ├── clip.safetensors
│   └── activation_meta.json
├── weights/
│   ├── model.layers.0.self_attn.q_proj/
│   │   ├── weight_int4.bin
│   │   ├── weight_scale.fp16
│   │   ├── pack_meta.json
│   │   └── shape.json
│   ├── model.layers.0.self_attn.k_proj/
│   ├── model.layers.0.self_attn.v_proj/
│   ├── model.layers.0.self_attn.o_proj/
│   ├── model.layers.0.mlp.gate_proj/
│   ├── model.layers.0.mlp.up_proj/
│   ├── model.layers.0.mlp.down_proj/
│   └── ... all remaining targeted layers ...
└── validation/
    ├── layer_error_stats.json
    ├── sample_outputs.json
    └── export_summary.json
```

### Stage 1 exact implementation requirements

#### 1. Source discovery

Implement `peqat_deploy/io/discover.py` with a function:

```python
def discover_source_dir(source_dir: str, checkpoint: str | None) -> dict:
    ...
```

It must:
- Resolve the source basename.
- Identify whether top-level adapter files exist.
- Identify available `checkpoint-*` directories.
- If `checkpoint` is provided, use that subdirectory; otherwise use top-level adapter files.
- Locate, in order of preference:
  - training YAML config,
  - PEFT adapter config,
  - adapter weights,
  - quantization params,
  - tokenizer files,
  - model config files.
- Return normalized absolute paths.

#### 2. Config normalization

Implement `peqat_deploy/config/loader.py` with a function:

```python
def load_export_job(discovered: dict) -> ExportJob:
    ...
```

The `ExportJob` object must include at minimum:
- `source_dir`
- `source_basename`
- `selected_checkpoint`
- `export_dir`
- `base_model_name`
- `weight_bits`
- `activation_bits`
- `lora_rank`
- `lora_alpha`
- `target_modules`
- `block_size`
- `calibration_percentile`
- `learn_smooth`
- `calibrate_smooth`

The exporter must default `export_dir = source_parent / (source_basename + "_export")` when no explicit output dir is given.[cite:2]

#### 3. Base model and adapter loading

Implement `peqat_deploy/io/hf_loader.py` and `peqat_deploy/io/peft_loader.py`.

Required behavior:
- Load the base Hugging Face model from `base_model_name`.
- Load the PEFT adapter from the selected directory containing `adapter_config.json` and `adapter_model.safetensors`, which is the standard PEFT layout for local adapters.[cite:3]
- Materialize the merged model in memory.

#### 4. LoRA merge

Implement `peqat_deploy/quant/merge_lora.py`.

For every targeted module, compute:

\[
W_{\text{eff}} = W_0 + \frac{\alpha_{\text{LoRA}}}{r} B A
\]

This follows Algorithm 1 exactly.[cite:2]

Implementation requirements:
- Resolve target modules from config names such as `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- Save a `layer_map.json` recording source module names and output artifact paths.
- Optional debug mode: save merged FP16 weights for one layer.

#### 5. Extract PE-QAT quantization parameters

Implement `peqat_deploy/quant/extract_quant_params.py`.

It must load `quantization_params.pth` and normalize it into:

```python
{
  "model.layers.0.self_attn.q_proj": {
    "smooth": tensor[K],
    "clip": tensor[K]
  },
  ...
}
```

Requirements:
- Handle missing keys with explicit errors.
- Validate that `smooth` and `clip` length match the input dimension `K` of each target linear layer.
- Save global tensors to `activation/smooth.safetensors` and `activation/clip.safetensors` keyed by exact module name.

#### 6. Fold smoothing onto weights

Implement `peqat_deploy/quant/fold_smoothing.py`.

According to Algorithm 1, per-channel smoothing scales weight columns and inversely scales input activations.[cite:2] The exporter must pre-apply the weight-side transformation:

\[
W'(:, c) = W_{\text{eff}}(:, c) \cdot s_{\text{smooth}}^{(c)}
\]

Requirements:
- Produce `W_prime` for each layer.
- Preserve original `smooth` tensor for runtime activation-side division.
- Record in `pack_meta.json` that weight-side smoothing has already been folded.

#### 7. Quantize weights per output channel

Implement `peqat_deploy/quant/quantize_weights.py`.

For each layer with shape `[d, k]`, compute per-row weight scale:

\[
s_w^{(j)} = \max_c |W'(j,c)| / 7
\]

then quantize:

\[
W_q(j,:) = \mathrm{clip}(\mathrm{round}(W'(j,:)/s_w^{(j)}), -8, 7)
\]

This matches the paper’s 4-bit signed range for `Qmin=-8`, `Qmax=7`.[cite:2]

Requirements:
- Store integer weights as signed values before packing.
- Store scales as FP16 or FP32; prefer FP16 unless accuracy debugging requires FP32.
- Save error metrics in `validation/layer_error_stats.json`:
  - max absolute error,
  - mean absolute error,
  - relative Frobenius norm error.

#### 8. Pack INT4 weights

Implement `peqat_deploy/quant/pack_int4.py`.

Requirements:
- Take signed INT4 values in `[-8, 7]` and pack two values per byte.
- Define one canonical nibble order and document it in `pack_meta.json`.
- Pack layout must be deterministic and kernel-specific.
- Save:
  - `weight_int4.bin`
  - `weight_scale.fp16`
  - `shape.json`
  - `pack_meta.json`

`pack_meta.json` must include:
- `layout_version`
- `rows`
- `cols`
- `pack_order`
- `signed_range`
- `weight_scale_dtype`
- `smoothing_folded`
- `transpose_required`
- `alignment_bytes`

#### 9. Manifest and metadata

Implement `peqat_deploy/quant/manifest.py` and `peqat_deploy/io/save_artifacts.py`.

`manifest.json` must include:
- source directory name,
- export directory name,
- selected checkpoint,
- base model name,
- bit widths,
- target module list,
- number of exported layers,
- activation file paths,
- per-layer weight artifact paths,
- runtime layout version,
- tokenizer path,
- timestamp,
- git commit if available.

#### 10. Export validation

Implement `scripts/validate_export.py`.

Validation pass requirements:
- For a small batch of random activations, compare:
  1. float merged layer output,
  2. fake-quant reference implementation of Algorithm 1,
  3. exported packed-weight CPU reconstruction.
- Save results to `validation/sample_outputs.json`.
- Fail if reconstructed packed-weight output deviates beyond configured tolerance.

### Stage 1 CLI contract

Implement:

```bash
python scripts/export_peqat.py \
  --source-dir ~/peqat-checkpoints/output_qat_lora_llama2_7b_max_init_s_57e7_001 \
  --checkpoint checkpoint-1320 \
  --output-dir ~/peqat-checkpoints/output_qat_lora_llama2_7b_max_init_s_57e7_001_export \
  --dtype float16 \
  --save-validation \
  --overwrite
```

If `--output-dir` is omitted, it must default to `<source_basename>_export` in the same parent directory.[cite:2]

### Stage 1 done definition

Stage 1 is complete when:
- The exporter runs end-to-end on the provided PE-QAT folder.
- The export folder name is exactly source basename plus `_export`.[cite:2]
- Every targeted linear layer has packed weight artifacts.
- `smooth` and `clip` tensors are exported and keyed by exact module name.
- Validation passes on at least one selected checkpoint.

---

## Stage 2 — CUDA runtime and benchmarking

Stage 2 implements runtime kernels and the benchmark/evaluation harness. This stage is CUDA-focused because it contains the latency-critical inference path: activation preprocessing and low-bit matrix multiplication.[cite:2]

### Stage 2 goal

Load the exported folder, run PE-QAT inference on GPU, and measure real runtime improvements while preserving benchmark accuracy well enough to support the workshop paper’s main claim.[cite:1][cite:2]

### Stage 2 scope

The runtime must:
- Load packed INT4 weight artifacts and metadata.
- Load per-layer `smooth` and `clip` tensors.
- Expose PyTorch-callable CUDA extension ops.
- Implement activation preprocessing exactly or near-exactly.
- Execute low-bit GEMM against packed weights.
- Produce benchmarkable inference results.

### Stage 2 kernel split

Implement two CUDA modules first:

1. `csrc/peqat_preprocess.cu`
   - Runtime activation smoothing, clipping, and activation quantization.
2. `csrc/peqat_gemm.cu`
   - Packed INT4 weight matmul and output reconstruction.

`bindings.cpp` must expose these as PyTorch extension functions.

### Stage 2 exact kernel requirements

#### 1. Runtime loader

Implement `peqat_deploy/runtime/loader.py`.

Required API:

```python
class ExportedPEQATModel:
    def __init__(self, export_dir: str, device: str = "cuda"):
        ...

    def load_layer(self, layer_name: str):
        ...

    def forward_linear(self, layer_name: str, x: torch.Tensor) -> torch.Tensor:
        ...
```

Responsibilities:
- Parse `manifest.json`.
- Load packed weights and scales for a layer.
- Load `smooth` and `clip` tensors for that layer.
- Materialize device tensors and cache them.

#### 2. Preprocess kernel

Implement `csrc/peqat_preprocess.cu`.

Required input:
- `x`: `[N, K]` float tensor
- `smooth`: `[K]`
- `clip`: `[K]`

Required outputs:
- `x_q`: `[N, K]` int8 tensor containing signed INT4 values in `[-8, 7]`
- `x_dequant`: `[N, K]` float tensor for debugging and reference mode
- `x_scale`: `[K]` float tensor where `x_scale[c] = clip[c] / 7`

Required runtime semantics from Algorithm 1:[cite:2]
- `x_s[:, c] = x[:, c] / smooth[c]`
- `x_clipped[:, c] = clip(x_s[:, c], -clip[c], clip[c])`
- `s_x[c] = clip[c] / 7`
- `x_q[:, c] = clip(round(x_clipped[:, c] / s_x[c]), -8, 7)`
- `x_tilde[:, c] = x_q[:, c] * s_x[c]`

Implementation notes:
- First implementation can return both `x_q` and `x_dequant`.
- Later optimization may fuse more steps into GEMM.
- Use current PyTorch CUDA stream.

#### 3. GEMM kernel

Implement `csrc/peqat_gemm.cu`.

Phase 1 implementation may be correctness-first, even if it is not fully tensor-core optimized.

Required input:
- `x_q` or `x_dequant`
- packed INT4 weights
- per-row weight scales
- layer metadata

Required output:
- `y`: `[N, D]`

Required behavior:
- Unpack signed INT4 weights correctly.
- Apply weight scales per output row.
- Multiply with activation-side data and reconstruct output.
- Match the exported layout exactly.

Preferred progression:
- Phase 1: unpack + dequantize + reference GEMM path for correctness.
- Phase 2: fused packed-weight kernel or CUTLASS-backed kernel for performance.

#### 4. PyTorch extension interface

Implement `csrc/bindings.cpp` with at least these bindings:

```cpp
preprocess_forward(x, smooth, clip) -> (x_q, x_dequant, x_scale)
linear_forward(x, packed_w, w_scale, smooth, clip, meta) -> y
```

Python wrapper in `peqat_deploy/runtime/loader.py` must compile or load the extension using PyTorch custom extension tooling.[cite:2]

#### 5. Reference path

Implement a pure PyTorch reference version of Algorithm 1 for one linear layer.

Required use cases:
- unit tests,
- numeric validation,
- debugging exact-vs-runtime discrepancies.

### Stage 2 benchmark harness

Implement `scripts/run_benchmarks.py` plus `peqat_deploy/eval/`.

Required benchmark modes:
- single-layer microbenchmarks,
- end-to-end model inference,
- task accuracy evaluation.

#### Microbenchmark metrics

For each representative layer and shape, measure:
- preprocess latency,
- GEMM latency,
- total linear layer latency,
- effective throughput,
- peak memory usage.

Run across:
- batch sizes: `1, 4, 8, 16`
- sequence lengths: `128, 512, 2048`
- modes: prefill and decode-like small-batch regime

#### End-to-end inference metrics

Measure:
- prefill latency,
- decode tokens/sec,
- end-to-end throughput,
- peak memory,
- model load time.

#### Accuracy benchmarks

Evaluate at minimum on:
- PIQA,
- HellaSwag,
- WinoGrande,
- ARC-E,
- ARC-C,
- OpenBookQA,
- optional BoolQ.

These are widely treated as commonsense or science reasoning-oriented benchmarks and are acceptable for the workshop when framed as reasoning evaluation under efficiency constraints.[cite:1]

#### Baselines

Implement comparisons against:
- FP16/BF16 base model,
- fake-quant PE-QAT reference,
- one practical quantized baseline such as W4A16 or W8A8 if available.

### Stage 2 required outputs

The benchmark harness must write:

```text
results/
├── latency/
│   ├── layer_microbench.csv
│   ├── prefill_decode.csv
│   └── runtime_breakdown.csv
├── accuracy/
│   ├── benchmark_scores.csv
│   └── benchmark_predictions.jsonl
├── memory/
│   └── memory_stats.csv
└── paper/
    ├── table_main_results.csv
    ├── table_accuracy.csv
    ├── fig_speed_vs_accuracy.csv
    ├── fig_runtime_breakdown.csv
    └── fig_seq_scaling.csv
```

### Stage 2 tests

Implement automated tests for:
- packed weight unpack correctness,
- preprocess kernel numeric correctness,
- one-layer forward match against the pure PyTorch Algorithm 1 reference,
- tolerance thresholds for each layer family,
- manifest compatibility and artifact loading.

### Stage 2 done definition

Stage 2 is complete when:
- Exported artifacts load successfully on GPU.
- At least one end-to-end PE-QAT inference path runs on the target hardware.
- Latency, throughput, memory, and accuracy results are generated reproducibly.
- Paper-ready CSV outputs exist for the planned tables and figures.

---

## Exact development sequence

Implement in this exact order:

1. `scripts/inspect_peqat_dir.py`
2. `peqat_deploy/io/discover.py`
3. `peqat_deploy/config/loader.py`
4. `peqat_deploy/io/hf_loader.py`
5. `peqat_deploy/io/peft_loader.py`
6. `peqat_deploy/quant/merge_lora.py`
7. `peqat_deploy/quant/extract_quant_params.py`
8. `peqat_deploy/quant/fold_smoothing.py`
9. `peqat_deploy/quant/quantize_weights.py`
10. `peqat_deploy/quant/pack_int4.py`
11. `peqat_deploy/io/save_artifacts.py`
12. `scripts/export_peqat.py`
13. Pure PyTorch reference implementation of one PE-QAT linear layer
14. `scripts/validate_export.py`
15. `csrc/peqat_preprocess.cu`
16. `csrc/bindings.cpp` for preprocess binding
17. `peqat_deploy/runtime/loader.py`
18. One-layer CUDA-vs-reference numeric test
19. `csrc/peqat_gemm.cu` reference-correctness version
20. one-layer end-to-end runtime path
21. model-level integration
22. `scripts/run_benchmarks.py`
23. accuracy evaluation on benchmark suite
24. result CSV generation for paper tables/figures

Do not start the optimized GEMM path before export validation and preprocess-kernel correctness are passing.

---

## Paper mapping

The workshop is interested in quantized execution, inference acceleration, benchmark methodologies, and empirical latency/throughput/memory analysis for reasoning models under real-world constraints.[cite:1] The project should therefore map implementation artifacts directly to paper sections:

- **Method section**: Stage 1 export semantics + Stage 2 runtime kernels.[cite:2]
- **Systems section**: packed layout, preprocessing kernel, GEMM path, exact vs approximate implementation choices.[cite:1][cite:2]
- **Experiments section**: latency, throughput, memory, and benchmark accuracy.[cite:1]
- **Discussion section**: whether exact PE-QAT semantics are practically worth their runtime overhead on Hopper-class hardware.[cite:1][cite:2]

Required paper tables and figures to generate from the benchmark outputs:
- Main results table: FP16 vs baseline quant vs PE-QAT runtime.
- Benchmark accuracy table by dataset.
- Runtime breakdown table for smoothing, clipping, activation quantization, GEMM, and output reconstruction.
- Speed-vs-accuracy scatter plot.
- Sequence-length scaling figure.

---

## Non-goals for the first version

Do not include these in the first working milestone:
- KV-cache quantization, even though the original paper lists it as future work.[cite:2]
- Triton-based exact INT4 runtime.
- Multi-GPU serving.
- Full engine integration with TensorRT-LLM or vLLM.
- Approximate grouped/per-tensor activation variants until the exact path is working.

---

## Final acceptance checklist

The project is ready for workshop paper writing when all items below are true:
- Exporter works on the provided PE-QAT checkpoint tree and writes `<source>_export`.[cite:2]
- Merged LoRA, smooth, and clip tensors are correctly reflected in exported artifacts.[cite:2]
- Runtime preprocess kernel matches the Algorithm 1 reference within agreed tolerance.[cite:2]
- Packed weight runtime path matches exported CPU reconstruction within agreed tolerance.
- End-to-end inference runs on at least one target model and hardware platform.
- Benchmark results include latency, throughput, memory, and reasoning benchmark accuracy.[cite:1]
- CSV outputs needed for the workshop paper are generated and reproducible.[cite:1]
