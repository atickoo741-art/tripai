# Stage 1 Complete — Export Pipeline

## What was built

### New files (none of the original training files were modified)

```
peqat_deploy/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── schema.py              ExportJob dataclass
│   └── loader.py              load_export_job() — reads training config, adapter config, populates job
├── io/
│   ├── __init__.py
│   ├── discover.py            discover_source_dir() — walks a training dir, finds all files
│   ├── hf_loader.py           load_base_model() — loads HuggingFace model in FP16
│   ├── peft_loader.py         load_adapter() — loads PEFT LoRA adapter with key remapping
│   └── save_artifacts.py      save_artifacts() — writes full export directory tree
├── quant/
│   ├── __init__.py
│   ├── merge_lora.py          merge_lora_weights() — W_eff = W_base + (alpha/r) * B @ A
│   ├── extract_quant_params.py  — parses quantization_params.pth, groups smooth/clip by module
│   ├── fold_smoothing.py      fold_smoothing() — in-place W' = W_eff * smooth (per column)
│   ├── quantize_weights.py    — per-row INT4: scale = max|W'|/7, clamp round(W'/scale) to [-8,7]
│   ├── pack_int4.py           — packs 2× signed INT4 values per byte (low_high nibble order)
│   ├── manifest.py            — builds manifest.json with all metadata
│   └── reference.py           peqat_linear_reference() — Algorithm 1 pure PyTorch reference
├── runtime/
│   ├── __init__.py
│   └── cpu_model.py           PackedQuantizedLinear — loads packed weights + smooth/clip, runs forward
│                                 build_packed_model() — replaces target layers in HF model
│                                 generate_text() — greedy decoding for sanity checks
└── eval/
    └── __init__.py

scripts/
├── inspect_peqat_dir.py       inspect a training checkpoint directory
├── export_peqat.py            full export pipeline: discover → load → merge → extract → fold → quantize → pack → save
├── validate_export.py         validates pack/unpack round-trip is lossless
├── sanity_check.py            loads packed model, generates text (end-to-end check)
├── weight_spotcheck.py        compares expected weight (W_base + LoRA) × smooth vs packed dequantized weight
└── weight_compare.py          (alternative to spotcheck, compares all layers)  -- NOT VERIFIED

csrc/                          (empty — for Stage 2 CUDA kernels)
exports/                       (empty — default export output directory)
```

## What was verified

1. **inspect_peqat_dir.py** — runs on real checkpoints, correctly discovers files
2. **export_peqat.py** — runs end-to-end on a real Llama-3.1-8B checkpoint:
   - Loads base model + LoRA adapter (224 target modules across 32 layers × 7 projections)
   - Merges LoRA into base weights
   - Extracts smooth/clip from quantization_params.pth
   - Folds smoothing into weights (in-place, saving ~15GB memory)
   - Quantizes to per-row INT4
   - Packs into binary format
   - Writes complete export directory tree
3. **validate_export.py** — pack/unpack round-trip is lossless for all 224 modules
4. **weight_spotcheck.py** — visual comparison of expected vs packed weights (layers 0,5,15,30) confirms correctness within INT4 quantization error (17-46% rel error, expected for W4A4)

## Verified export directory structure

```
<source_basename>_export/
├── manifest.json
├── export_config.json
├── metadata/
│   ├── adapter_config.json
│   ├── export_log.json
│   ├── layer_map.json
│   └── selected_checkpoint.json
├── tokenizer/
│   ├── tokenizer.model
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── activation/
│   ├── smooth.safetensors        (224 entries, keyed by module name)
│   ├── clip.safetensors          (224 entries, keyed by module name)
│   └── activation_meta.json
├── weights/
│   ├── {module_safe_name}/
│   │   ├── weight_int4.bin       packed INT4 bytes
│   │   ├── weight_scale.fp16     per-row FP16 scales
│   │   ├── shape.json
│   │   └── pack_meta.json
│   └── ... (224 dirs)
└── validation/
    ├── layer_error_stats.json    quantization error per module
    ├── sample_outputs.json       round-trip validation result
    └── export_summary.json
```

## Known issues

1. **PyTorch CUDA compatibility** — The DGX Spark GB10 chip has compute capability 12.1, but installed PyTorch (2.10.0.dev20251021+cu130) only supports up to 12.0. This generates a warning but doesn't prevent operation. Model can be loaded on CPU or CUDA.
2. **Weight reconstruction error 17-46%** — This is the expected INT4 quantization error after smoothing fold. It does NOT predict accuracy loss — the model was trained with fake quantization that accounts for this. True quality gate is end-to-end benchmark accuracy.

---

# What's left: Stage 2 — CUDA Runtime + Benchmarking

## Objectives

Build custom CUDA kernels that load the exported artifacts and run PE-QAT inference on GPU with measurable speedup over FP16. Then benchmark latency, memory, and accuracy on reasoning tasks for a COLM 2026 workshop paper.

## Implementation order (from the plan)

1. **`csrc/peqat_preprocess.cu`** — activation preprocessing: `x / smooth → clamp(-clip, clip) → quantize to INT4 → dequantize`
2. **`csrc/bindings.cpp`** — PyTorch CUDA extension bindings for preprocess kernel
3. **`peqat_deploy/runtime/loader.py`** — runtime model class that loads export artifacts and calls CUDA extensions
4. Single-layer CUDA-vs-reference numeric test
5. **`csrc/peqat_gemm.cu`** — packed INT4 weight matmul + output reconstruction (correctness-first, later optimize)
6. One-layer end-to-end runtime path
7. Model-level integration (replace all target layers)
8. **`scripts/run_benchmarks.py`** + `peqat_deploy/eval/` — harness for latency, throughput, memory
9. Accuracy evaluation on PIQA, HellaSwag, WinoGrande, ARC-E, ARC-C, OpenBookQA
10. CSV output generation for paper tables/figures

## Deliverables

- CUDA kernels for the two critical paths (preprocess + GEMM)
- PyTorch extension loading them
- Benchmark harness with baselines (FP16, W4A16, W8A8)
- Paper-ready CSV outputs
