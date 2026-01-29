"""
MMLU & Commonsense Evaluation for PEQAT Models
Uses lm-evaluation-harness with your trained checkpoints
"""

import torch
import argparse
from transformers import AutoTokenizer
from fake_quant import load_peqat_model
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
    Merges LoRA adapters before evaluation to avoid lm_eval PEFT bugs
    """
    
    # Determine output path
    if output_path is None:
        checkpoint_name = os.path.basename(checkpoint_path)
        parent_dir = os.path.dirname(checkpoint_path)
        output_path = os.path.join(parent_dir, f"eval_results_{checkpoint_name}.json")
    
    print(f"\n🔍 Running LM Evaluation Harness")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Tasks: {tasks}")
    print(f"   Few-shot: {num_fewshot}")
    print(f"   Output: {output_path}\n")
    
    is_peft = os.path.isdir(checkpoint_path) or "checkpoint" in checkpoint_path

    if is_peft:
        print(f"Type: PEFT Adapter (will merge before evaluation)")
        print("=" * 70)
        
        # Step 0: Clean adapter config
        print("Step 0/3: Cleaning adapter_config.json...")
        backup_path = clean_adapter_config(checkpoint_path)
        print("✓ Config cleaned")
        
        try:
            # Step 1: Load PEQAT model
            print("Step 1/3: Loading PEQAT model with quantization...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, tokenizer = load_peqat_model(
                adapter_dir=checkpoint_path,
                base_model_name=base_model,
                device=device
            )
            print(f"✓ Model loaded on {device}")
            
            # Step 2: Merge LoRA into base model
            print("Step 2/3: Merging LoRA adapters into base weights...")
            model = model.merge_and_unload()
            print("✓ LoRA merged")
            
            # Step 3: Save merged model temporarily
            temp_merged_path = f"{checkpoint_path}_merged_temp"
            os.makedirs(temp_merged_path, exist_ok=True)
            
            print(f"Step 3/3: Saving merged model to {temp_merged_path}...")
            model.save_pretrained(temp_merged_path)
            tokenizer.save_pretrained(temp_merged_path)
            print("✓ Merged model saved")
            print("=" * 70 + "\n")
            
            # Evaluate the merged model (not PEFT)
            model_args = f"pretrained={temp_merged_path}"
            cleanup_after = True
            
            # Free memory
            del model
            torch.cuda.empty_cache()
            
        finally:
            # Always restore original config
            restore_adapter_config(checkpoint_path, backup_path)
            print("✓ Original config restored")
        
    else:
        # It's a baseline HF model
        print(f"Type: Base Model (HF Hub)")
        model_args = f"pretrained={checkpoint_path}"
        cleanup_after = False
        temp_merged_path = None
    
    print(f"Model Args: {model_args}")

    # Build lm_eval command
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", tasks,
        "--num_fewshot", str(num_fewshot),
        "--device", "cuda",
        "--batch_size", str(batch_size),
        "--output_path", output_path,
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    print("=" * 70)
    
    # Run evaluation
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    print("=" * 70)
    
    # Cleanup temporary merged model
    if cleanup_after and temp_merged_path and os.path.exists(temp_merged_path):
        print(f"\n🧹 Cleaning up temporary merged model at {temp_merged_path}...")
        shutil.rmtree(temp_merged_path)
        print("✓ Cleanup complete")
    
    if result.returncode == 0:
        print(f"\n✅ Evaluation complete! Results saved to {output_path}")
        
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
                display_results(results, tasks)
        except Exception as e:
            print(f"Note: Could not display results ({e})")
    else:
        print(f"\n❌ Evaluation failed with return code {result.returncode}")
        return False
    
    return True


def display_results(results: dict, tasks: str):
    """Pretty print evaluation results"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    if "results" in results:
        for task_name, metrics in results["results"].items():
            print(f"\n{task_name}:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")
    
    print("="*70 + "\n")


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
