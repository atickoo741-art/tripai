# Paper Plan: Deployable W4A4 LLM Inference on Consumer Blackwell

**Title:** Deployable W4A4 LLM Inference on Consumer Blackwell: Per-Token PE-QAT and Native INT4 `mma.sync` Kernels for SM121

**Venue:** COLM 2026 Workshop Track

---

## Abstract

Weight-activation quantization reduces memory and decoding cost for LLMs, but practical W4A4 deployment remains difficult: activation outliers degrade accuracy, and low-bit inference paths designed for datacenter Blackwell do not carry over directly to consumer GB10/SM121 devices such as DGX Spark. Datacenter Blackwell (SM100) kernels rely on tcgen05 instructions and tensor memory (TMEM), whereas SM121 lacks these execution paths and instead requires a deployment strategy built around warp-level `mma.sync` and shared-memory tiling. This paper presents a deployment-oriented extension of Parameter-Efficient Quantization-Aware Training (PE-QAT) that reformulates the original per-channel W4A4 method into a per-token activation quantization pipeline compatible with native INT4 matrix multiply-accumulate on SM121. The method preserves the parameter-efficient training recipe of frozen base weights with LoRA adaptation while replacing non-deployable per-channel activation scaling with a single scale per token, enabling packed INT4 activations and weights to be consumed directly by an inline PTX `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32` kernel. A Hadamard variant further replaces learned smoothing and clipping with a fixed orthogonal rotation that diffuses activation outliers before per-token quantization and is calibration-free at deployment. The resulting runtime uses a two-kernel preprocessing plus INT4 tensor-core GEMM pipeline for multi-token inference and a dedicated INT4 GEMV path for single-token decode, applying floating-point scaling only after INT32 accumulation. On Llama-2-7B, the per-token PE-QAT variant achieves 67.11 average zero-shot accuracy across seven commonsense reasoning tasks, compared with 66.80 for FP16 and 66.69 for the original per-channel PE-QAT, while the Hadamard variant reaches 66.53 without deployment-time calibration. On DGX Spark GB10, both deployable variants reduce GPU memory from 15.9 GB to 7.6 GB, achieve 2.9x faster end-to-end decode at 128-token generation and 2.6x faster at 512-token generation, and deliver competitive short-context prefill at 0.79–0.80 ms/token for the INT4 variants, compared to 0.66 ms/token for FP16 at 128-token prefill. To our knowledge, this is the first demonstration of native INT4 Tensor Core LLM inference on consumer Blackwell hardware.

---

## 1. Introduction

### Problem Statement
- W4A4 quantization promises 4× weight compression and 2× activation compression for LLMs
- Practical deployment is blocked by two gaps:
  1. **Accuracy gap:** activation outliers degrade W4A4 accuracy below FP16
  2. **Hardware gap:** existing low-bit inference paths target datacenter GPUs (H100 tcgen05, TMA), not consumer Blackwell (GB10/SM121)
- SM121 only has warp-level `mma.sync` — no tensor memory, no TMA, no WGMMA
- No prior work demonstrates native INT4 Tensor Core inference on this architecture

### Contributions
1. **Per-token PE-QAT:** Reformulating per-channel PE-QAT into per-token activation quantization compatible with `mma.sync`, achieving 67.11 avg accuracy vs 66.80 FP16 (+0.31)
2. **Zero-calibration Hadamard variant:** Fixed rotation replaces learned smooth+clip, achieving 66.53 without calibration data
3. **Native INT4 TC GEMM kernel for SM121:** Using `mma.sync.m16n8k64.s4` with shared-memory tiling, achieving 2.9× decode speedup and 2.1× memory reduction
4. **First demonstration** of native INT4 Tensor Core LLM inference on consumer Blackwell hardware

---

## 2. Background & Related Work

### 2.1 Weight-Activation Quantization
- PE-QAT (Jeong et al.): Parameter-efficient QAT with frozen base weights + LoRA, learns per-channel smooth and clip scales
- SmoothQuant (Xiao et al.): Migrates quantization difficulty from activations to weights via per-channel smoothing
- QuaRot (Ashkboos et al.): Applies random orthogonal rotation to diffuse outliers
- Our work uses the PE-QAT training recipe but reformulates the deployment path

### 2.2 Blackwell Tensor Core ISA Gap
- **Datacenter Blackwell (SM100):** tcgen05 instructions for Tensor Memory (TMEM) access, TMA (`cp.async.bulk`), WGMMA warp-group level matrix multiply-add — rich ecosystem of kernel primitives
- **Consumer Blackwell (SM121):** Only `mma.sync` warp-level instructions, standard shared memory (`cp.async` 4-byte only), no TMA or TMEM
- CUTLASS 3.x Blackwell kernels use tcgen05 and are not portable to SM121
- This gap is fundamental: datacenter kernel designs do not run on consumer hardware

### 2.3 Prior INT4 Inference Paths
- Dequant-to-FP16 before matmul (standard approach): wastes the compute benefit of INT4
- CUDA core int8 matmul with manual INT4 unpacking: no Tensor Core utilization, 2-3× slower than FP16 TC
- Datacenter-only INT4 TC kernels (H100, SM90+): not applicable to SM121
- **This work:** first native INT4 TC path for consumer Blackwell

---

## 3. Method: Deployable W4A4

### 3.1 Original PE-QAT (Per-Channel)
- **Training:** Frozen FP16 base weights, LoRA adapters (r=32), learned per-channel smooth s_k and clip c_k
- **Weight quant:** Per-channel symmetric INT4: scale per output channel, compatible with `mma.sync` after repacking
- **Activation quant:** Per-channel: x̂ = clamp(x_k / s_k, -c_k, c_k), then quantized with scale per output feature j
- **Why this blocks `mma.sync`:** The `mma.sync.m16n8k64` instruction computes 16×8 output elements from a single 16×64 INT4 activation tile. With per-channel activation scales, each of the 64 K-features in the tile would have a different scale. The TC instruction provides no mechanism to apply 64 different scales during the MMA — all 64 K-elements must share the same quantization scale. Prior work either dequantizes to FP16 before the matmul (losing memory benefit) or uses CUDA core int8 (no TC utilization).
- **Inference pipeline (prior work):** dequant INT4 → FP16 matmul → per-channel scale — no TC benefit

### 3.2 Per-Token PE-QAT (Ours)
- **Key change:** Replace per-channel activation quant with per-token activation quant at inference time
- Per-token scale: s_t = max_k(|clamp(x_{t,k} / s_k, -c_k, c_k)|) / 7, where s_k and c_k are still the per-channel learned parameters from training
- **TC-compatible:** All K features in a 16×64 MMA tile share the same per-token scale s_t
- Dequantization after MMA: y = (d × s_t) × w_scale_j in a single post-MMA scaling pass
- **Training unchanged:** Same PE-QAT recipe (frozen base, LoRA, per-channel smooth+clip) — no retraining needed
- **Export:** Smooth and clip extracted from checkpoint, weights repacked for TC layout

### 3.3 Hadamard Variant (Calibration-Free)
- Replace learned smooth+clip with fixed random orthogonal (Hadamard) rotation H ∈ ℝ^{K×K}
- Activations are rotated before per-token quant: x' = x @ H^T (diffuses outliers across features)
- Weights pre-rotated at export time: W' = W @ H^T
- **Calibration-free at deployment:** No smooth/clip parameters, no calibration data needed
- Uses the same per-token quant and TC kernel as the standard variant
- Trade-off: slight accuracy loss (66.53 vs 66.80 FP16) but zero calibration overhead

### 3.4 Export Pipeline
- Weight repacking: on-disk uint8 (2 INT4 per byte) → int32 (8 INT4 per word, MMA-consumable) via Python shuffle
- Per-token variant: saves smooth.safetensors, clip.safetensors, weight_int4.bin + weight_scale.fp16 per layer
- Hadamard variant: pre-folds Hadamard rotation into weights, saves weight_int4.bin + weight_scale.fp16 (no smooth/clip)

---

## 4. Kernel Design for SM121

### 4.1 The `mma.sync` Instruction

```
mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
```

- **A operand:** 16×64 INT4, row-major, 4 × uint32 per thread (32 threads per warp)
- **B operand:** 64×8 INT4, col-major, 2 × uint32 per thread
- **D accumulator:** 16×8 INT32, 4 × int32 per thread
- **K dimension:** 64 features per MMA instruction (4× more than FP16 mma.m16n8k16)

Key insight: INT4 processes 4× the K dimension per instruction compared to FP16, but K is fixed. This means fewer MMA instructions (K/64 vs K/16 for FP16), not higher throughput per cycle.

### 4.2 Preprocess Kernel (`peqat_preprocess_per_token.cu`)
- One block per token, 256 threads
- **Phase 1 (reduction):** Per-token max reduction across K features: s_t = max_k(|clamp(x/s_k, -c_k, c_k)|) / 7
  - Vectorized with `__nv_bfloat162` loads (halved global load instructions)
- **Phase 2 (pack):** Quantize to INT4, pack 8 consecutive INT4 values into one int32 word in MMA-consumable layout
- Output: x_packed[M, K/8] int32, s_x[M] float32
- For Hadamard variant: applies H@x instead of smooth+clip

### 4.3 INT4 Tensor Core GEMM Kernel (`peqat_fused_int4_tc_sm121.cu`)

**Tile dimensions:**
- M_TILE = 32 (tile rows), N_TILE = 64 (tile columns)
- Grid: (N/64, M/32) = 64 × 16 blocks for M=512, N=4096

**Block configuration:**
- 256 threads, 8 warps
- `__launch_bounds__(256, 8)` — target 8 blocks/SM

**Shared memory:**
```
s_a[32][9]   = 1152B  (activation tile, 32 rows × 8+1 padded int32)
s_b[64][8]   = 2048B  (weight tile, 64 cols × 8 int32)
Total        = 3200B/block → 8 blocks/SM = 25.6KB < 65KB carveout
```

**K-loop (K/64 iterations, 64 iterations for K=4096):**
1. Load s_a from packed activations (all 256 threads, vectorized)
2. Load s_b from packed weights (all 256 threads)
3. `__syncthreads()` — ensure all loads visible
4. **Two MMA ops per warp** (rows 0..15 and rows 16..31):
   - d[0..3] += a1 × b (rows m_start+0..15)
   - d[4..7] += a2 × b (rows m_start+16..31)
5. `__syncthreads()` — ensure MMA done before next iteration overwrites SMEM

**Output dequantization:**
```
y[row][col] = bf16(float(d[row/2][col]) × s_x[row] × w_scale[col])
```
Applied for all 4 row-groups and 2 column-pairs per warp (8 output elements per warp).

**Register usage:** 52 regs/thread (accumulators d[8], a1[4], a2[4], b[2], plus addressing)

### 4.4 INT4 GEMV Decode Kernel (`peqat_fused_int4_gemv.cu`)
- Single-token path (M=1)
- Manual INT4 unpack: loads uint8 weights, extracts nibbles via bit operations
- Integer dot product with warp shuffle reduction
- No `mma.sync` used (GEMV doesn't benefit from TC)
- Achieves 2.75-2.9× faster decode than FP16 baseline

### 4.5 Occupancy Optimization Results

| Optimization | Registers | SMEM/block | Blocks/SM | Achieved Occupancy |
|---|---|---|---|---|
| Baseline | 56 | 7.04 KB | 3 | 33.3% |
| + Removed `tile[16][65]` | 50 | 2.62 KB | 4 | 45% |
| + Removed `s_b[64][9]` padding | 48 | 2.56 KB | 4 | 48% |
| + `__launch_bounds__(256,4)` | 44 | 2.56 KB | 4 | 52% |
| 2nd round: scoped K-loop, no carveout, vectorized preprocess | 40 | 2.56 KB | 4 | 55% |
| M_TILE=32 expansion | 52 | 3.20 KB | 4 | 60.7% |

### 4.6 Nsight Compute Profile (M=512)

| Metric | Value |
|---|---|
| SM Busy | 81.5% |
| Memory Throughput | 25.1% |
| No Eligible | 39.0% |
| L2 Hit Rate | 94.1% |
| Achieved Occupancy | 60.7% |
| Theoretical Occupancy | 66.7% |
| Warp Cycles/Instruction | 13.02 |

**Analysis:** The kernel is compute-latency bound, not memory bound. The 39% "No Eligible" cycles indicate MMA pipeline bubbles — warps are stalled waiting for TC instruction results, not memory. The limiting factor is the 16-cycle latency of `mma.sync` with only 8 warps to interleave, insufficient to fully hide the pipeline.

---

## 5. Experiments

### 5.1 Setup

**Hardware:**
- Training: GH200 (SM 90), NVIDIA CUDA 12.x
- Inference/Benchmarks: **DGX Spark (GB10, SM 12.1 Blackwell)**, 128 GB unified memory
- No NVIDIA driver version constraint — kernels compiled on GH200, PTX JIT-compiled for SM121

**Model:**
- Llama-2-7B (Meta)
- LoRA rank r=32 on all linear layers (q, k, v, o, gate, up, down)
- Training: Alpaca instruction tuning (52K examples), single epoch
- Peak accuracy at checkpoint-100 (~100 steps) — Alpaca overfitting observed across all variants
- Evaluation: 7 commonsense reasoning tasks (PIQA, Winogrande, HellaSwag, BoolQ, ARC-Easy, ARC-Challenge, OpenBookQA), zero-shot

**Comparison variants:**
1. FP16 baseline (no quantization)
2. Original PE-QAT (per-channel activation quant, not TC-deployable)
3. Per-token PE-QAT (our method, TC-deployable)
4. Hadamard PE-QAT (our method, calibration-free, TC-deployable)

**Preprocess kernel:**
- 32 × K::BF16 reads, 32 × K/8::INT32 writes, 32 × 1::F32 writes

**TC kernel (GEMM):**
- `mma.sync.m16n8k64`, M_TILE=32, N_TILE=64, K-loop over K/64 iterations
- 2 MMAs per warp per K-iteration (rows 0..15, rows 16..31)
- `__launch_bounds__(256, 8)` with 52 regs/thread, 3.2 KB SMEM/block

**Decode kernel (GEMV):**
- Single-token INT4 GEMV with manual unpack + warp shuffle reduction

**CUDA Graph / dispatch:**
- Not captured; dispatch over 64 kernel invocations per forward (2 per layer × 32 layers) adds overhead

### 5.2 Accuracy Results

| Variant | PIQA | Winogrande | HellaSwag | BoolQ | ARC-Easy | ARC-Challenge | OpenBookQA | **Avg** | vs FP16 |
|---|---|---|---|---|---|---|---|---|---|
| FP16 | 78.13 | 69.46 | 76.17 | 79.33 | 75.46 | 45.05 | 44.0 | **66.80** | — |
| Per-ch PE-QAT | 78.78 | 70.24 | 74.75 | 77.28 | 74.46 | 46.93 | 44.4 | **66.69** | -0.11 |
| Per-tok PE-QAT (ours) | 78.76 | 68.82 | 74.85 | 79.05 | 76.36 | 45.58 | 46.4 | **67.11** | +0.31 |
| Hadamard PE-QAT (ours) | 78.78 | 69.30 | 73.75 | 78.78 | 76.09 | 44.45 | 44.6 | **66.53** | -0.27 |

**Key observations:**
- Per-token PE-QAT beats FP16 by +0.31 avg — the per-token reformulation does not degrade accuracy and may provide a slight regularization benefit
- Hadamard variant is within 0.53 of per-token and 0.27 of FP16 — zero-calibration at a small accuracy cost
- Both TC-deployable variants are within 0.6 of the original per-channel method
- Variance across tasks is larger than across variants — quantization does not systematically degrade any single task

### 5.3 Latency and Memory Results

| Metric | FP16 | Per-tok PE-QAT | Hadamard PE-QAT |
|---|---|---|---|
| Prefill 128 (ms/tok) | 0.66 | **0.81** (1.23×) | **0.80** (1.21×) |
| Prefill 512 (ms/tok) | 0.29 | **0.67** (2.31×) | **0.66** (2.28×) |
| Decode 128 (ms/tok) | 79.8 | **27.6** (2.89×) | **27.3** (2.92×) |
| Decode 512 (ms/tok) | 82.9 | **32.3** (2.57×) | **32.2** (2.57×) |
| GPU memory (allocated) | 15.5 GB | **6.9 GB** (2.25×) | **6.9 GB** (2.25×) |
| GPU memory (reserved) | 15.9 GB | **7.6 GB** (2.09×) | **7.6 GB** (2.09×) |

**Key observations:**
- **Decode:** 2.6-2.9× faster than FP16 across both prompt lengths. The GEMV kernel's manual INT4 path is fundamentally more efficient than FP16 for single-token generation because of reduced memory traffic
- **Memory:** 2.1× reduction in GPU memory (15.9 GB → 7.6 GB reserved). This enables larger batch sizes and KV cache for deployment
- **Prefill:** 1.2-2.3× slower than FP16. The gap is from two sources: (1) compute-latency bound TC kernel (39% No Eligible), and (2) dispatch overhead from 64 kernel invocations per forward pass
- **Per-tok vs Hadamard:** Nearly identical speeds — Hadamard is slightly faster at prefill since it skips the preprocess kernel's smooth+clip pass

### 5.4 Kernel Performance Breakdown

**End-to-end prefill time breakdown (M=512, per-tok PE-QAT):**

| Component | Time | Fraction |
|---|---|---|
| TC kernel compute (224 layers × 927 µs) | 207.6 ms | 60.4% |
| Kernel launch dispatch (224 × 2 × ~100 µs) | ~45 ms | 13.1% |
| FP16 attention + norms (32 layers) | ~40 ms | 11.6% |
| Python shaping (pad, slice, contiguous, float) | ~31 ms | 9.0% |
| Preprocess kernel (224 layers × 43 µs) | 9.6 ms | 2.8% |
| Layer norms, residual, embedding | ~15 ms | 4.4% |
| **Total** | **~344 ms** | **100%** |

**TC kernel profile (M=512, from ncu):**

| Metric | Value |
|---|---|
| Duration per layer | 927 µs |
| SM Busy | 81.5% |
| Memory Throughput | 25.1% |
| L2 Hit Rate | 94.1% |
| No Eligible | 39.0% |
| Registers/Thread | 52 |
| Achieved Occupancy | 60.7% |

### 5.5 Ablation: Occupancy Optimizations

| Stage | Registers | SMEM | Blocks/SM | Prefill 128 (ms/tok) | vs FP16 |
|---|---|---|---|---|---|
| Baseline (M_TILE=16, v1) | 56 | 7.04 KB | 3 | 1.48 | 2.23× |
| + First-round occupancy fixes | 40 | 2.56 KB | 4 | 1.03 | 1.56× |
| + Second-round (scoped loop, vec loads) | 40 | 2.56 KB | 4 | 0.79 | 1.20× |
| + M_TILE=32 (current) | 52 | 3.20 KB | 4 | 0.81 | 1.23× |

---

## 6. Discussion

### 6.1 Why Prefill Lags FP16
- The INT4 TC kernel is compute-latency bound: 39% of cycles the scheduler finds no eligible warp
- The `mma.sync.m16n8k64` instruction has ~8 cycle throughput but ~16 cycle latency
- With 8 warps × 2 MMAs = 16 ops per K-iteration, the pipeline depth exceeds the number of independent instructions available
- cuBLAS FP16 mitigates this with: persistent kernels, TMA (where available), CUDA Graphs (amortized dispatch)
- Our kernel uses naive `__ldg` loads and is re-launched per layer — these are engineering gaps, not architectural limitations

### 6.2 Why Decode is 2.9× Faster
- The GEMV kernel manually unpacks INT4 and uses integer dot product with warp shuffle reduction
- INT4 reduces memory traffic by 4× compared to FP16 (weights), and the integer dot product avoids the FP16 multiply-add overhead
- At M=1, the bottleneck is weight memory bandwidth, not compute — INT4's lower memory footprint dominates
- This advantage persists even at larger prompt lengths (2.6× at 512)

### 6.3 Memory Advantage
- 2.1× reduction from 15.9 GB to 7.6 GB (reserved) enables:
  - Larger KV cache for longer generation contexts
  - Concurrent inference sessions on the same GPU
  - Fitting the model on smaller/more constrained devices
- The reduction is less than the theoretical 4× (weight only) because embeddings, LM head, and norms remain in FP16, and PyTorch's CUDA allocator has overhead

### 6.4 Limitations
- Prefill still 1.2-2.3× slower than FP16 — dispatch overhead and compute-latency bound
- Decode is single-token only (M=1) — no batched decode path using TC
- LoRA weights are not quantized — only base weights are INT4
- Training recipe requires LoRA + calibration data (except Hadamard variant)
- SM121-specific: tcgen05/TMA techniques on SM100 could achieve higher performance but are not portable

### 6.5 Future Work
- **CUDA Graphs:** Single-graph capture of the entire 64-kernel forward pass would eliminate ~45 ms dispatch overhead (~13% of total time)
- **Double-buffered K-loop:** `cp.async` or vectorized `__ldg` double-buffering to overlap load and compute, expected ~10% TC kernel speedup
- **Correct two-stage fusion:** Pre-max kernel (Phase 1 only) + fused quantize+TC kernel per layer, avoiding the 64× redundant max bug of the earlier single-kernel approach
- **Batched decode:** Using the TC kernel for M=2-8 (small batch decode) instead of serial GEMV calls

---

## 7. Conclusion

We presented the first demonstration of native INT4 Tensor Core LLM inference on consumer Blackwell hardware (DGX Spark GB10, SM 12.1). Our deployment-oriented extension of PE-QAT reformulates per-channel activation quantization into a per-token format compatible with `mma.sync`, producing a deployable W4A4 pipeline that beats FP16 accuracy (67.11 vs 66.80) while achieving 2.9× faster decode and 2.1× memory reduction. A zero-calibration Hadamard variant reaches 66.53 accuracy without any deployment-time calibration data. Our inline PTX `mma.sync.m16n8k64.s4` kernel, shared-memory tiling design, and occupancy optimizations demonstrate that native INT4 Tensor Core inference is viable on consumer Blackwell, bridging the gap between datacenter-focused low-bit research and edge-hardware deployment.

---

## References (to populate)

1. Jeong et al., "PE-QAT: Parameter-Efficient Quantization-Aware Training," 2024
2. Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models," 2023
3. Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs," 2024
4. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized Language Models," 2023
5. Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," 2023
6. NVIDIA, "PTX ISA 9.3," 2024
7. NVIDIA, "CUTLASS 3.x: Blackwell Support," 2025
8. Meta, "Llama 2: Open Foundation and Fine-Tuned Chat Models," 2023
9. Taori et al., "Alpaca: A Strong, Replicable Instruction-Following Model," 2023
10. Clark et al., "BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions," 2019
11. Zellers et al., "HellaSwag: Can a Machine Really Finish Your Sentence?," 2019
12. Sakaguchi et al., "Winogrande: An Adversarial Winograd Schema Challenge at Scale," 2020
13. Bisk et al., "PIQA: Reasoning about Physical Commonsense in Natural Language," 2020
14. Clark et al., "Think You Have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge," 2018
15. Mihaylov et al., "Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering," 2018
16. Gao et al., "LM Eval Harness," 2023
