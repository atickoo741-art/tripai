"""
test_fusing.py - Benchmark unfused vs fused PEQAT inference speed

Loads the same QAT+LoRA model twice and compares inference throughput.
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os
import sys

# Import from existing codebase
from fake_quant import load_peqat_model, quantize_model, PEQATLoRAModule
from fused_ops import convert_to_fused, PEQATLoRAModuleFused
from data_preparation import get_concatenated_text


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "./qat_lora_model_24"  # Your trained model
BASE_MODEL = "meta-llama/Llama-2-7b-hf"  # Base model used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Benchmark settings
NUM_WARMUP = 5      # Warmup iterations (for JIT compilation)
NUM_SAMPLES = 20    # Actual benchmark iterations
SEQ_LEN = 2048      # Sequence length for inference
BATCH_SIZE = 2      # Batch size


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_model_unfused(model_path, base_model, device):
    """Load model with standard unfused PEQAT operations."""
    print(f"\n{'='*70}")
    print("LOADING UNFUSED MODEL")
    print('='*70)
    
    model, tokenizer = load_peqat_model(model_path, base_model, device)

    model = model.to(torch.bfloat16)
    
    # Verify all parameters are bfloat16
    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16:
            print(f"⚠️  Converting {name} from {param.dtype} to bfloat16")
            param.data = param.data.to(torch.bfloat16)
    
    # Verify it's unfused
    unfused_count = sum(1 for _, m in model.named_modules() 
                       if 'PEQATLoRAModule' in str(type(m).__name__) 
                       and 'Fused' not in str(type(m).__name__))
    
    print(f"✓ Model loaded with {unfused_count} unfused PEQAT modules")
    return model, tokenizer


def load_model_fused(model_path, base_model, device):
    """Load model and convert to fused PEQAT operations."""
    print(f"\n{'='*70}")
    print("LOADING FUSED MODEL")
    print('='*70)
    
    model, tokenizer = load_peqat_model(model_path, base_model, device)

    model = model.to(torch.bfloat16)
    
    # Verify all parameters are bfloat16
    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16:
            print(f"⚠️  Converting {name} from {param.dtype} to bfloat16")
            param.data = param.data.to(torch.bfloat16)
    
    # Convert to fused
    model = convert_to_fused(model)
    
    # Verify it's fused
    fused_count = sum(1 for _, m in model.named_modules() 
                     if isinstance(m, PEQATLoRAModuleFused))
    
    print(f"✓ Model loaded with {fused_count} fused PEQAT modules")
    return model, tokenizer


def prepare_test_data(tokenizer, seq_len, batch_size, num_samples):
    """Prepare test inputs from WikiText-2 validation set."""
    print(f"\n{'='*70}")
    print("PREPARING TEST DATA")
    print('='*70)
    
    # Load WikiText-2 validation
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    
    # Concatenate and tokenize
    input_ids_full = get_concatenated_text(dataset, tokenizer)
    
    # Create batches
    test_batches = []
    total_tokens = len(input_ids_full)
    
    for i in range(num_samples):
        # Sample random starting position
        start_idx = (i * seq_len) % (total_tokens - seq_len)
        end_idx = start_idx + seq_len
        
        # Extract sequence and repeat for batch
        input_ids = input_ids_full[start_idx:end_idx].unsqueeze(0).repeat(batch_size, 1)
        test_batches.append(input_ids.to(DEVICE))
    
    print(f"✓ Created {len(test_batches)} batches")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Total tokens per batch: {batch_size * seq_len}")
    
    return test_batches


def benchmark_model(model, test_batches, num_warmup, model_name="Model"):
    """Benchmark inference speed of a model."""
    print(f"\n{'='*70}")
    print(f"BENCHMARKING {model_name.upper()}")
    print('='*70)
    
    model.eval()
    
    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for i, batch in enumerate(test_batches[:num_warmup]):
            outputs = model(batch)
            if i == 0:
                print(f"  First warmup output shape: {outputs.logits.shape}")
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Benchmark
    print(f"\nRunning benchmark ({len(test_batches) - num_warmup} iterations)...")
    
    times = []
    with torch.no_grad():
        for batch in test_batches[num_warmup:]:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            outputs = model(batch)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append(end - start)
    
    # Statistics
    times = torch.tensor(times)
    mean_time = times.mean().item()
    std_time = times.std().item()
    min_time = times.min().item()
    max_time = times.max().item()
    
    tokens_per_batch = BATCH_SIZE * SEQ_LEN
    throughput = tokens_per_batch / mean_time
    
    print(f"\n{'─'*70}")
    print(f"Results for {model_name}:")
    print(f"{'─'*70}")
    print(f"  Mean time:     {mean_time*1000:.2f} ms/batch")
    print(f"  Std dev:       {std_time*1000:.2f} ms")
    print(f"  Min time:      {min_time*1000:.2f} ms")
    print(f"  Max time:      {max_time*1000:.2f} ms")
    print(f"  Throughput:    {throughput:.1f} tokens/sec")
    print(f"  Throughput:    {throughput*3600/1e6:.2f} M tokens/hour")
    
    return {
        'mean': mean_time,
        'std': std_time,
        'min': min_time,
        'max': max_time,
        'throughput': throughput,
        'times': times
    }


def compare_results(unfused_stats, fused_stats):
    """Print comparison between unfused and fused."""
    print(f"\n{'='*70}")
    print("COMPARISON: UNFUSED vs FUSED")
    print('='*70)
    
    speedup = unfused_stats['mean'] / fused_stats['mean']
    throughput_gain = (fused_stats['throughput'] - unfused_stats['throughput']) / unfused_stats['throughput'] * 100
    
    print(f"\nLatency (lower is better):")
    print(f"  Unfused:  {unfused_stats['mean']*1000:.2f} ms/batch")
    print(f"  Fused:    {fused_stats['mean']*1000:.2f} ms/batch")
    print(f"  Speedup:  {speedup:.3f}x")
    
    print(f"\nThroughput (higher is better):")
    print(f"  Unfused:  {unfused_stats['throughput']:.1f} tokens/sec")
    print(f"  Fused:    {fused_stats['throughput']:.1f} tokens/sec")
    print(f"  Gain:     {throughput_gain:+.1f}%")
    
    print(f"\nVariability:")
    print(f"  Unfused std:  {unfused_stats['std']*1000:.2f} ms")
    print(f"  Fused std:    {fused_stats['std']*1000:.2f} ms")
    
    # Verdict
    print(f"\n{'─'*70}")
    if speedup > 1.1:
        print(f"✅ FUSED IS FASTER: {speedup:.2f}x speedup")
        print(f"   Recommendation: Use fused operations for inference")
    elif speedup < 0.9:
        print(f"❌ FUSED IS SLOWER: {1/speedup:.2f}x slower")
        print(f"   Recommendation: Stick with unfused operations")
    else:
        print(f"⚠️  SIMILAR PERFORMANCE: {speedup:.2f}x")
        print(f"   Recommendation: Either version is fine")
    print(f"{'─'*70}")
    
    return speedup


def verify_output_equivalence(model_unfused, model_fused, test_batch):
    """Verify that unfused and fused produce same outputs."""
    print(f"\n{'='*70}")
    print("VERIFYING OUTPUT EQUIVALENCE")
    print('='*70)
    
    model_unfused.eval()
    model_fused.eval()
    
    with torch.no_grad():
        out_unfused = model_unfused(test_batch)
        out_fused = model_fused(test_batch)
    
    logits_unfused = out_unfused.logits
    logits_fused = out_fused.logits
    
    # Check shapes
    assert logits_unfused.shape == logits_fused.shape, \
        f"Shape mismatch: {logits_unfused.shape} vs {logits_fused.shape}"
    print(f"✓ Output shapes match: {logits_unfused.shape}")
    
    # Check numerical equivalence
    max_diff = (logits_unfused - logits_fused).abs().max().item()
    mean_diff = (logits_unfused - logits_fused).abs().mean().item()
    
    print(f"✓ Max absolute difference:  {max_diff:.6f}")
    print(f"✓ Mean absolute difference: {mean_diff:.6f}")
    
    # Check predictions
    pred_unfused = logits_unfused.argmax(dim=-1)
    pred_fused = logits_fused.argmax(dim=-1)
    
    agreement = (pred_unfused == pred_fused).float().mean().item() * 100
    print(f"✓ Token prediction agreement: {agreement:.2f}%")
    
    if max_diff < 1e-3 and agreement > 99:
        print(f"\n✅ OUTPUTS ARE EQUIVALENT")
    elif max_diff < 0.01 and agreement > 95:
        print(f"\n⚠️  OUTPUTS ARE SIMILAR (acceptable for quantized model)")
    else:
        print(f"\n❌ WARNING: Outputs differ significantly!")
    
    return max_diff, agreement


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():
    print(f"\n{'#'*70}")
    print("# PEQAT INFERENCE SPEED TEST: UNFUSED vs FUSED")
    print(f"{'#'*70}")
    print(f"\nConfiguration:")
    print(f"  Model path:    {MODEL_PATH}")
    print(f"  Base model:    {BASE_MODEL}")
    print(f"  Device:        {DEVICE}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Sequence len:  {SEQ_LEN}")
    print(f"  Warmup iters:  {NUM_WARMUP}")
    print(f"  Bench iters:   {NUM_SAMPLES - NUM_WARMUP}")
    
    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ ERROR: Model not found at {MODEL_PATH}")
        print(f"   Please train a model first or update MODEL_PATH")
        sys.exit(1)
    
    # Load models
    model_unfused, tokenizer = load_model_unfused(MODEL_PATH, BASE_MODEL, DEVICE)
    model_fused, _ = load_model_fused(MODEL_PATH, BASE_MODEL, DEVICE)
    
    # Prepare test data
    test_batches = prepare_test_data(tokenizer, SEQ_LEN, BATCH_SIZE, NUM_SAMPLES)
    
    # Verify outputs are equivalent
    verify_output_equivalence(model_unfused, model_fused, test_batches[0])
    
    # Benchmark unfused
    unfused_stats = benchmark_model(model_unfused, test_batches, NUM_WARMUP, "UNFUSED")
    
    # Clear memory
    del model_unfused
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Benchmark fused
    fused_stats = benchmark_model(model_fused, test_batches, NUM_WARMUP, "FUSED")
    
    # Compare
    speedup = compare_results(unfused_stats, fused_stats)
    
    # Save results
    results = {
        'unfused': {k: v.tolist() if isinstance(v, torch.Tensor) else v 
                   for k, v in unfused_stats.items()},
        'fused': {k: v.tolist() if isinstance(v, torch.Tensor) else v 
                 for k, v in fused_stats.items()},
        'speedup': speedup,
        'config': {
            'batch_size': BATCH_SIZE,
            'seq_len': SEQ_LEN,
            'num_samples': NUM_SAMPLES - NUM_WARMUP,
            'device': str(DEVICE)
        }
    }
    
    import json
    output_file = 'fusing_benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    print(f"\n{'#'*70}")
    print("# BENCHMARK COMPLETE")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
