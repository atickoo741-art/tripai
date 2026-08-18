# PE-QAT: Parameter-Efficient Quantization-Aware Training for W4A4 LLM Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A full training-to-inference stack for **4-bit weight / 4-bit activation (W4A4)**
quantization of large language models — from parameter-efficient
quantization-aware training (QAT) to native INT4 Tensor Core deployment on
consumer Blackwell (DGX Spark / GB10, SM 12.1).

The stack spans two papers, each covering one half of the pipeline:

| Paper | Venue | Focus |
|---|---|---|
| [PE-QAT: Parameter-Efficient Quantization-Aware Training for Large Language Models](https://aclanthology.org/2026.acl-srw.63/) | **ACL SRW 2026** | The training method: frozen backbone + LoRA, fake W4A4 quantization, learned SmoothQuant scales and activation clips |
| [Deployable W4A4 LLM Inference on Consumer Blackwell: Per-Token PE-QAT and Native INT4 `mma.sync` Kernels for SM121](https://openreview.net/forum?id=03keF5I9V5) | **COLM 2026 Workshop on Efficient Reasoning** | The deployment method: per-token activation quantization and native INT4 Tensor Core kernels on consumer Blackwell |

The ACL SRW paper proposes the training algorithm; the COLM workshop paper is
a deployment-oriented extension of it, reformulating the per-channel method
into a per-token format consumable by INT4 Tensor Cores. They are two
distinct papers with distinct contributions.

---

## 1. PE-QAT — training (ACL SRW 2026)

[Paper](https://aclanthology.org/2026.acl-srw.63/) · [PDF](https://aclanthology.org/2026.acl-srw.63.pdf)

> As large language models (LLMs) grow, their compute and memory demands become prohibitive for on-device deployment. Quantization is a crucial technique to shrink model footprint and accelerate inference, but aggressively low-bit weight-activation quantization schemes often sacrifice accuracy. Quantization Aware Training (QAT) is a commonly used paradigm to minimize quantization noise, but is extremely expensive to train and often unscalable to large models. We introduce PE-QAT, a parameter-efficient framework targeting per-channel 4-bit weight-activation quantization of LLMs, which aims to preserve model accuracy while significantly reducing resource requirements. The proposed method freezes the base model and trains lightweight LoRA adapters by fake quantizing the merged-weight model, enabling PE-QAT to scale efficiently unlike full QAT. We apply fake quantization with Straight-Through Estimators (STE) to the merged weights, allowing the adapters to explicitly compensate for quantization noise during training. One of the biggest challenges with quantizing activations alongside weights is addressing outliers that are orders of magnitude larger than other activations, which inflate quantization scales and suppress lower-magnitude values. To mitigate the impact of severe activation outliers, PE-QAT jointly learns per-channel smoothing factors and symmetric activation clipping thresholds. PE-QAT retains accuracy within 0.11 percentage points of the full-precision baseline on Llama-2-7B zero-shot tasks while training only 1.26% of total parameters.

### How PE-QAT works

**Forward pass** (per linear layer, base weights $W_0 \in \mathbb{R}^{d \times k}$ frozen, LoRA $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$):

1. **Merge LoRA:** $W_{\mathrm{eff}} = W_0 + \frac{\alpha_{\mathrm{LoRA}}}{r} BA$
2. **Smooth (per input channel $c$):** $W'^{(j,c)} = W_{\mathrm{eff}}^{(j,c)} \cdot s_{\mathrm{smooth}}^{(c)}$ and $X'^{(t,c)} = X^{(t,c)} / s_{\mathrm{smooth}}^{(c)}$. The two factors cancel exactly in $X' W'^{\top} = X W_{\mathrm{eff}}^{\top}$, so smoothing is lossless in full precision — it only migrates quantization difficulty from outlier activations to the (better-behaved) weight columns.
3. **Clip (PACT-style, per channel):** $X_{\mathrm{clipped}}^{(t,c)} = \mathrm{clip}\left(X'^{(t,c)}, -\alpha_{\mathrm{clip}}^{(c)}, +\alpha_{\mathrm{clip}}^{(c)}\right)$
4. **Fake-quantize to INT4 (symmetric):** weights per output row with $s_w^{(j)} = \max_j |W'^{(j,:)}| / 7$; activations per channel with $s_x^{(c)} = \alpha_{\mathrm{clip}}^{(c)} / 7$; round with straight-through gradients.
5. **Matmul:** $Y = \tilde{X} \tilde{W}^{\top}$ — the only trainable tensors are $A$, $B$, $s_{\mathrm{smooth}}$, and $\alpha_{\mathrm{clip}}$.

**Calibration-free initialization** (128 calibration batches):
- Smoothing factors: $s_{\mathrm{smooth}}^{(c)} = \frac{A_c^{0.5}}{(B_c + 10^{-6})^{0.5}}$, where $A_c$ is the 99.5th percentile of $|X_c|$ and $B_c = \max |W_0^{(:,c)}|$ — a percentile-robust variant of SmoothQuant (ablation: percentile beats max-based init by 0.62 PPL).
- Clips: $\alpha_{\mathrm{clip}}^{(c)} = \mathrm{quantile}_{0.995}\left(|X_c|\right)$, kept positive with a $10^{-6}$ floor.

**Clip gradients** decompose into a PACT-style boundary term (pushes walls outward where clipped activations carry loss gradients) and an LSQ-style scale term (shrinks the quantization bin $s_x$ to reduce reconstruction error).

**Optimization.** AdamW with three parameter groups — LoRA at base LR, clips and smoothing factors at **10× LR** (ablation: 1× costs +2.31 PPL, 5× +1.10, 25× destabilizes) — plus global grad-norm clipping at 5.0. LoRA rank 32 = 84.7M trainable parameters (**1.26%** of Llama-2-7B), trained for 1500 steps on Alpaca-52K.

**Scope.** W4A4KV16 on Llama-2 (7B, 13B), Llama-3-8B, and Mistral-7B-v0.3; evaluated on 7 zero-shot tasks + WikiText-2 against GPTQ, SpinQuant-H, FlatQuant, and L4Q. Learned clips matter: freezing them at their calibrated values costs +3.00 PPL.

## 2. Per-Token PE-QAT + INT4 Tensor Cores — deployment (COLM ER Workshop 2026)

[Paper](https://openreview.net/forum?id=03keF5I9V5)

> Weight-activation quantization reduces memory and inference cost for LLMs, but practical W4A4 deployment
> remains difficult: activation outliers degrade accuracy, and low-bit inference paths designed for datacenter
> Blackwell do not carry over directly to consumer GB10/SM121 devices such as DGX Spark. This hardware distinction is
> fundamental rather than incremental: public SM100 Blackwell kernels rely on tcgen05 instructions and tensor memory
> (TMEM), whereas SM121 lacks these execution paths and instead requires a deployment strategy built around
> warp-level `mma.sync` and shared-memory tiling. This paper presents a deployment-oriented extension
> of Parameter-Efficient Quantization-Aware Training (PE-QAT) that reformulates the original per-channel W4A4
> method into a per-token activation quantization pipeline compatible with native INT4 matrix multiply-accumulate
> on SM121. The method preserves the parameter-efficient training recipe of frozen base weights with LoRA adaptation
> while replacing non-deployable per-channel activation scaling with a single scale per token, enabling packed INT4
> activations and weights to be consumed directly by an inline PTX `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32`
> kernel. A Hadamard variant further replaces learned smoothing and clipping with a fixed orthogonal rotation that
> diffuses activation outliers before per-token quantization and is calibration-free at deployment. The resulting
> runtime uses a two-kernel preprocessing plus INT4 tensor-core GEMM pipeline for multi-token inference and a
> dedicated INT4 GEMV path for single-token decode, applying floating-point scaling only after INT32 accumulation.
> On Llama-2-7B, the per-token PE-QAT variant achieves 67.11 average zero-shot accuracy across seven commonsense
> reasoning tasks, compared with 66.80 for FP16 and 66.69 for the original per-channel PE-QAT, while the Hadamard
> variant reaches 66.53 without deployment-time calibration. On DGX Spark GB10, both deployable variants reduce GPU memory from 15.9 GB to 7.6 GB,
> achieve 2.7--2.8x total speedup at 128-token prompt length and 2.2x at 512 tokens.

### Why per-token?

Tensor Core `mma.sync` takes a single per-token activation scale shared across
all features. The original per-channel PE-QAT uses one scale per feature,
which cannot be represented in the INT4 MMA instruction. The per-token
variant computes `s_x[t] = max_k(|x[t,k]|) / 7`, making activations directly
consumable by `mma.sync.m16n8k64.s4` while keeping accuracy at or above FP16.

### Hadamard variant

Replaces all learned quantization parameters with a fixed Walsh-Hadamard
rotation `H` (orthogonal, calibration-free). Because `H(x) @ H(W)^T = x @ W^T`,
the rotation is transparent in full precision — it only diffuses activation
outliers so per-token quantization captures true dynamic range.

## Results

Accuracy (zero-shot, Llama-2-7B, 7 commonsense reasoning tasks):

| Variant | PIQA | WinoGrande | HellaSwag | BoolQ | ARC-E | ARC-C | OBQA | **Avg** |
|---|---|---|---|---|---|---|---|---|
| FP16 | 78.13 | 69.46 | 76.17 | 79.33 | 75.46 | 45.05 | 44.0 | **66.80** |
| Original PE-QAT (per-channel) | 78.78 | 70.24 | 74.75 | 77.28 | 74.46 | 46.93 | 44.4 | **66.69** |
| Per-token PE-QAT | 78.76 | 68.82 | 74.85 | 79.05 | 76.36 | 45.58 | 46.4 | **67.11** |
| Hadamard PE-QAT | 78.78 | 69.30 | 73.75 | 78.78 | 76.09 | 44.45 | 44.6 | **66.53** |

Inference on DGX Spark (GB10, SM 12.1):

| Metric | FP16 | Per-token PE-QAT | Hadamard PE-QAT |
|---|---|---|---|
| Prefill 128 (ms/tok) | 0.66 | 0.79 | 0.80 |
| Prefill 512 (ms/tok) | 0.29 | 0.66 | 0.66 |
| Decode 128 (ms/tok) | 79.4 | **27.4** (2.9x) | **27.3** (2.9x) |
| Decode 512 (ms/tok) | 82.5 | **32.0** (2.6x) | **32.2** (2.6x) |
| GPU memory | 15.9 GB | **7.6 GB** (2.1x) | **7.6 GB** (2.1x) |

## Repository layout

```
fake_quant.py                   # QAT core: W4A4 fake quant, LoRA wrappers, STE, load/save
peqat_training.py               # PE-QAT training (per-channel / per-token variants)
peqat_hadamard_training.py      # Hadamard PE-QAT training
calibrations.py                 # SmoothQuant + PACT clip calibration
config_loader.py                # YAML config loader
data_preparation.py             # Alpaca / WikiText data pipelines
config*.yaml                    # Training configs
csrc/                           # CUDA kernels + pybind bindings
  peqat_fused_int4_tc_sm121.cu  # Native INT4 mma.sync.m16n8k64.s4 GEMM (the core kernel)
  peqat_fused_int4_gemv.cu      # Single-token INT4 decode path
  peqat_preprocess_per_token.cu # Activation preprocess: smooth, clip, per-token quant, pack
  peqat_hadamard_preprocess.cu  # Hadamard rotation preprocess
  hadamard_butterfly.cu         # Fused butterfly Hadamard transform
  peqat_preprocess.cu           # bf16 per-channel preprocess (fallback path)
  peqat_gemm_bf16_wmma.cu       # bf16 WMMA GEMM/GEMV (fallback path)
scripts/export_peqat.py         # Merge LoRA, fold scales, pack INT4, emit inference artifacts
scripts/run_benchmarks.py       # Prefill/decode latency + memory benchmarks
scripts/validate_export.py      # Export validation
peqat_deploy/                   # Deployment package: export, manifest, CUDA/CPU runtimes
eval_cs.py                      # lm-eval harness (commonsense / MMLU)
ppl_eval.py                     # Perplexity evaluation
```

## Quickstart

### 1. Train (PE-QAT, per-token variant)

```bash
pip install -r requirements.txt          # see note below on the peqat_cuda egg
# set the config path in peqat_training.py (defaults to alpaca_config.yaml)
python peqat_training.py
```

Outputs: LoRA adapter (`adapter_model.safetensors`) + quantization parameters
(`quantization_params.pth`, per-module `activation_clip` / `smooth_factor`).

Hadamard variant:

```bash
python peqat_hadamard_training.py        # no learned clip/smooth, no calibration
```

### 2. Export to packed INT4 artifacts

```bash
python scripts/export_peqat.py --source-dir <training_output> \
    --checkpoint checkpoint-100 --fold-weights --save-validation
python scripts/export_peqat.py --source-dir <hadamard_output> --hadamard --fold-weights
```

This merges LoRA into the base weights, folds smooth scales (or H for
Hadamard) into the weights, quantizes per-row to INT4, and packs 8 nibbles
per int32 in the MMA-consumable layout.

### 3. Build the CUDA kernels

```bash
pip install -e csrc/                     # torch CUDAExtension (sm120/121)
```

Kernel entry points: `preprocess_per_token_forward`,
`fused_int4_tc_per_token_forward`, `fused_int4_gemv_per_token_forward`,
`preprocess_hadamard_forward`, `hadamard_butterfly_forward` (plus a bf16
fallback: `preprocess_forward_bf16`, `fused_gemv_bf16_forward`,
`fused_gemm_bf16_wmma_forward`).

### 4. Evaluate

```bash
python eval_cs.py --checkpoint <training_output> --tasks commonsense
python ppl_eval.py --checkpoint <training_output>
```

## Checkpoints

Trained LoRA adapters and quantization parameters for the checkpoints whose
results are reported in the papers are hosted on Hugging Face (links coming
soon).

## Hardware requirements

The INT4 Tensor Core kernels target **DGX Spark (NVIDIA GB10, SM 12.1)** and
use inline PTX `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32`. Training
runs on any CUDA GPU; the deployment kernels require sm120/sm121.

## Citation

```bibtex
@inproceedings{mishra-2026-pe,
    title = "{PE}-{QAT}: Parameter-Efficient Quantization-Aware Training for Large Language Models",
    author = "Mishra, Shresth",
    editor = "T.Y.S.S., Santosh and Rodriguez, Juan Diego and de Gibert, Ona",
    booktitle = "Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-srw.63/",
    doi = "10.18653/v1/2026.acl-srw.63",
    pages = "701--711",
}

@inproceedings{mishra-2026-deployable,
    title = {Deployable W4A4 LLM Inference on Consumer Blackwell: Per-Token PE-QAT and Native INT4 mma.sync Kernels for SM121},
    author = {Mishra, Shresth},
    booktitle = {COLM 2026 Workshop on Efficient Reasoning},
    year = {2026},
    url = {https://openreview.net/forum?id=03keF5I9V5},
}
```

## License

Code: [MIT](LICENSE). 
