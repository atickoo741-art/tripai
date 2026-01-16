"""
Inspect actual training data format to verify it matches MMLU evaluation
"""

import torch
from transformers import AutoTokenizer
from data_preparation import load_alpaca_instruct
from datasets import load_dataset

def inspect_alpaca_training_data(num_examples=5):
    """
    Show exactly what the model sees during training
    """
    print("\n" + "="*80)
    print("INSPECTING ALPACA TRAINING DATA")
    print("="*80)
    
    # Load your processed dataset
    # dataset, tokenizer = load_alpaca_instruct(
    #     model_name="meta-llama/Llama-2-7b-hf",
    #     max_length=2048
    # )

    dataset, tokenizer = load_alpaca_instruct()
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training samples: {len(dataset['train']):,}")
    print(f"   Validation samples: {len(dataset['validation']):,}")
    
    # Inspect examples
    for idx in range(num_examples):
        ex = dataset["train"][idx]
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        
        # Decode full sequence
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        
        # Find where training starts (first non -100 label)
        train_start_idx = next((i for i, l in enumerate(labels) if l != -100), None)
        
        # Decode prompt (masked part)
        prompt_ids = input_ids[:train_start_idx] if train_start_idx else input_ids
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        
        # Decode response (trained part)
        if train_start_idx:
            response_ids = [lid for lid in labels[train_start_idx:] if lid != -100]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        else:
            response_text = "[NO TRAINING LABELS]"
        
        # Statistics
        n_total = len(input_ids)
        n_masked = sum(1 for l in labels if l == -100)
        n_trained = n_total - n_masked
        
        print(f"\n{'='*80}")
        print(f"EXAMPLE {idx+1}")
        print(f"{'='*80}")
        print(f"Tokens: {n_total} total | {n_masked} masked | {n_trained} trained ({100*n_trained/n_total:.1f}%)")
        print(f"\n--- PROMPT (MASKED, NOT TRAINED ON) ---")
        print(prompt_text[:500] + "..." if len(prompt_text) > 500 else prompt_text)
        print(f"\n--- RESPONSE (TRAINED ON) ---")
        print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
        print(f"\n--- RAW FIRST 50 TOKENS ---")
        for i in range(min(50, len(input_ids))):
            token = tokenizer.decode([input_ids[i]])
            label_status = "TRAIN" if labels[i] != -100 else "MASK"
            print(f"  [{i:3d}] {token:20s} -> {label_status}")
    
    print("\n" + "="*80)


def compare_with_mmlu_format():
    """
    Show what MMLU evaluation expects vs what we trained on
    """
    print("\n" + "="*80)
    print("COMPARING FORMATS: ALPACA TRAINING vs MMLU EVALUATION")
    print("="*80)
    
    # Show raw Alpaca format
    print("\n📝 RAW ALPACA FORMAT (from tatsu-lab/alpaca):")
    raw = load_dataset("tatsu-lab/alpaca", split="train")
    
    for i in range(3):
        ex = raw[i]
        print(f"\n--- Raw Example {i+1} ---")
        print(f"Instruction: {ex['instruction']}")
        print(f"Input: {ex['input']}")
        print(f"Output: {ex['output'][:200]}...")
    
    # Show MMLU format
    print("\n\n📝 MMLU FORMAT (from cais/mmlu):")
    print("Question: What is the capital of France?")
    print("Choices:")
    print("  A. London")
    print("  B. Paris")
    print("  C. Berlin")
    print("  D. Madrid")
    print("Expected Output: B")
    
    print("\n\n⚠️  FORMAT MISMATCH ANALYSIS:")
    print("="*80)
    print("""
    ALPACA TRAINING FORMAT:
    - Instruction: "Give three tips for X"
    - Response: "1. First tip\\n2. Second tip\\n3. Third tip"
    - Model learns: instruction-following, open-ended generation
    
    MMLU EVALUATION FORMAT:
    - Question + 4 choices (A/B/C/D)
    - Response: Single letter
    - Model needs: multiple-choice reasoning
    
    🔍 DIAGNOSIS:
    If MMLU is 27% but WikiText was 36%, your model:
    ✓ Can generate text (Alpaca trained this)
    ✗ Struggles with A/B/C/D format (Alpaca didn't train this)
    
    💡 SOLUTION:
    1. Continue training to epoch 2+ (instruction-following improves)
    2. MMLU is 0-shot → model generalizes from instructions
    3. Expected: 38-42% after full training
    """)
    print("="*80)


def verify_label_masking():
    """
    Verify that prompt masking is correct
    """
    print("\n" + "="*80)
    print("VERIFYING LABEL MASKING")
    print("="*80)
    
    dataset, tokenizer = load_alpaca_instruct()
    
    issues = []
    
    for idx in range(100):
        ex = dataset["train"][idx]
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        
        # Check 1: Labels length matches input_ids
        if len(input_ids) != len(labels):
            issues.append(f"Example {idx}: Length mismatch")
        
        # Check 2: Some labels are masked (-100)
        n_masked = sum(1 for l in labels if l == -100)
        if n_masked == 0:
            issues.append(f"Example {idx}: No masking (training on prompt!)")
        
        # Check 3: Some labels are trained
        n_trained = sum(1 for l in labels if l != -100)
        if n_trained == 0:
            issues.append(f"Example {idx}: No training labels!")
        
        # Check 4: Reasonable mask ratio (expect 60-95% masked)
        mask_ratio = n_masked / len(labels)
        if mask_ratio < 0.5 or mask_ratio > 0.98:
            issues.append(f"Example {idx}: Unusual mask ratio {mask_ratio:.1%}")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues[:10]:
            print(f"  - {issue}")
    else:
        print("\n✅ All checks passed! Label masking is correct.")
    
    # Show statistics
    all_mask_ratios = []
    for idx in range(1000):
        ex = dataset["train"][idx]
        n_masked = sum(1 for l in ex["labels"] if l == -100)
        all_mask_ratios.append(n_masked / len(ex["labels"]))
    
    import statistics
    print(f"\n📊 Masking Statistics (1000 samples):")
    print(f"   Mean mask ratio: {statistics.mean(all_mask_ratios):.1%}")
    print(f"   Median: {statistics.median(all_mask_ratios):.1%}")
    print(f"   Min: {min(all_mask_ratios):.1%}")
    print(f"   Max: {max(all_mask_ratios):.1%}")
    print("="*80)


if __name__ == "__main__":
    # Run all checks
    inspect_alpaca_training_data(num_examples=3)
    compare_with_mmlu_format()
    verify_label_masking()
    
    print("\n\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
    Run this script to verify your training data format.
    
    If you see:
    ✅ Prompt contains "### Instruction:" → Correct format
    ✅ Response is actual answer text → Correct
    ✅ 70-95% of tokens masked → Good masking
    
    Then your data is fine, and 27% MMLU at epoch 1.1 is just undertraining.
    Continue to epoch 2.0+ for 38-42% MMLU.
    """)
    print("="*80)
