"""
MMLU & Commonsense Evaluation for PEQAT Models
Uses lm-evaluation-harness with your trained checkpoints
"""

import torch
import argparse
from transformers import AutoTokenizer
from fake_quant import PEQATLoRAModule, load_peqat_model
import subprocess
import json
import os
import shutil


def clean_adapter_config(checkpoint_path):
    """Temporarily clean adapter_config.json of unsupported keys"""
    config_path = os.path.join(checkpoint_path, "adapter_config.json")
    backup_path = os.path.join(checkpoint_path, "adapter_config.json.backup")
    
    # Read original config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Backup original
    with open(backup_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Keep only valid LoraConfig keys
    valid_keys = {
        'r', 'lora_alpha', 'target_modules', 'lora_dropout', 'bias', 
        'task_type', 'inference_mode', 'modules_to_save', 'init_lora_weights',
        'layers_to_transform', 'layers_pattern', 'rank_pattern', 'alpha_pattern',
        'fan_in_fan_out', 'peft_type', 'base_model_name_or_path'
    }
    
    cleaned_config = {k: v for k, v in config.items() if k in valid_keys}
    
    # Write cleaned config
    with open(config_path, 'w') as f:
        json.dump(cleaned_config, f, indent=2)
    
    return backup_path


def restore_adapter_config(checkpoint_path, backup_path):
    """Restore original adapter_config.json"""
    config_path = os.path.join(checkpoint_path, "adapter_config.json")
    
    if os.path.exists(backup_path):
        shutil.copy(backup_path, config_path)
        os.remove(backup_path)


def run_lm_eval(checkpoint_path: str, 
                base_model: str = "meta-llama/Llama-2-7b-hf",
                tasks: str = "mmlu",
                num_fewshot: int = 0,
                batch_size: int = 8,
                output_path: str = None):
    """
    Run lm-evaluation-harness on PEQAT checkpoint
    
    Args:
        checkpoint_path: Path to PEQAT checkpoint directory
        base_model: Base model name (e.g., meta-llama/Llama-2-13b-hf)
        tasks: Comma-separated task names
        num_fewshot: Number of few-shot examples
        batch_size: Batch size for evaluation
        output_path: Where to save results JSON
    
    Returns:
        True if successful, False otherwise
    """
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from fake_quant import PEQATLoRAModule
    
    # Determine output path
    if output_path is None:
        checkpoint_name = os.path.basename(checkpoint_path)
        parent_dir = os.path.dirname(checkpoint_path)
        output_path = os.path.join(parent_dir, f"eval_results_{checkpoint_name}.json")
    
    print(f"\n{'='*70}")
    print("🔍 PEQAT EVALUATION")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Base Model: {base_model}")
    print(f"Tasks:      {tasks}")
    print(f"Few-shot:   {num_fewshot}")
    print(f"Batch size: {batch_size}")
    print(f"Output:     {output_path}")
    print(f"{'='*70}\n")
    
    # Check if this is a PEFT checkpoint or base model
    is_peft = os.path.isdir(checkpoint_path) or "checkpoint" in checkpoint_path

    if is_peft:
        print("Type: PEFT Adapter (requires merging)")
        print(f"{'='*70}\n")
        
        # Step 0: Clean adapter config
        print("[1/5] Cleaning adapter_config.json...")
        backup_path = clean_adapter_config(checkpoint_path)
        print("      ✓ Config cleaned\n")
        
        try:
            # Step 1: Load PEQAT model
            print("[2/5] Loading PEQAT model with quantization...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, tokenizer = load_peqat_model(
                adapter_dir=checkpoint_path,
                base_model_name=base_model,
                device=device
            )
            print(f"      ✓ Model loaded on {device}\n")
            
            # Step 2: Set use_quant_grad flag (required for forward pass)
            print("[3/5] Setting quantization gradient flags...")
            for module in model.modules():
                if isinstance(module, PEQATLoRAModule):
                    module.use_quant_grad = True
            print("      ✓ Quantization flags set\n")
            
            # Step 3: Merge LoRA into base model
            print("[4/5] Merging LoRA adapters...")
            model = model.merge_and_unload()
            print("      ✓ LoRA merged\n")
            
            # Step 4: Ensure dtype consistency
            print("      Converting to BFloat16...")
            model = model.to(torch.bfloat16)
            # Double-check all parameters
            for param in model.parameters():
                if param.dtype != torch.bfloat16:
                    param.data = param.data.to(torch.bfloat16)
            for buffer in model.buffers():
                if buffer.is_floating_point() and buffer.dtype != torch.bfloat16:
                    buffer.data = buffer.data.to(torch.bfloat16)
            print("      ✓ All tensors converted to BFloat16\n")
            
        finally:
            # Always restore original config
            restore_adapter_config(checkpoint_path, backup_path)
            print("      ✓ Original config restored\n")
        
        # Step 5: Create evaluation wrapper
        print("[5/5] Creating evaluation wrapper...")
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        print("      ✓ Model wrapped for evaluation\n")
        
    else:
        # Base model evaluation
        print("Type: Base Model (HuggingFace Hub)\n")
        print("[1/1] Loading model...")
        lm = HFLM(pretrained=checkpoint_path, batch_size=batch_size)
        print("      ✓ Model loaded\n")
    
    # Run evaluation
    print(f"{'='*70}")
    print("RUNNING EVALUATION")
    print(f"{'='*70}\n")
    
    task_list = tasks.split(',')
    
    try:
        results = simple_evaluate(
            model=lm,
            tasks=task_list,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            log_samples=False,
        )
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}\n")
    
    # Prepare output structure
    output_dict = {
        "results": results.get("results", {}),
        "versions": results.get("versions", {}),
        "config": {
            "model": checkpoint_path,
            "base_model": base_model,
            "model_args": f"checkpoint={checkpoint_path}",
            "tasks": task_list,
            "num_fewshot": num_fewshot,
            "batch_size": batch_size,
        }
    }
    
    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save results (use default=str to handle any non-serializable types)
    try:
        with open(output_path, 'w') as f:
            json.dump(output_dict, f, indent=2, default=str)
        print(f"✅ Results saved to: {output_path}\n")
    except Exception as e:
        print(f"❌ Failed to save JSON: {e}")
        # Fallback: save as pickle
        import pickle
        pickle_path = output_path.replace('.json', '.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"⚠️  Saved as pickle instead: {pickle_path}\n")
    
    # Display results summary
    display_results(results, tasks)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Evaluate PEQAT model on MMLU and Commonsense tasks")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PEQAT checkpoint"
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Base model name"
    )
    
    parser.add_argument(
        "--tasks",
        type=str,
        default="mmlu",
        help="Tasks to evaluate (comma-separated)"
    )
    
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    run_lm_eval(
        checkpoint_path=args.checkpoint,
        base_model=args.base_model,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
