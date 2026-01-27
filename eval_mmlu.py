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

def prepare_model_for_eval(checkpoint_path: str, base_model: str = "meta-llama/Llama-2-7b-hf"):
    """
    Load PEQAT checkpoint and prepare for evaluation
    
    Args:
        checkpoint_path: Path to PEQAT checkpoint (e.g., checkpoint-1544)
        base_model: Base model name
    
    Returns:
        model, tokenizer
    """
    print("\n" + "="*60)
    print(f"Loading PEQAT Model for MMLU Evaluation")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use existing load_peqat_model from fakequant.py
    model, tokenizer = load_peqat_model(
        adapter_dir=checkpoint_path,
        base_model_name=base_model,
        device=device
    )
    
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    print(f"✓ Model dtype: {next(model.parameters()).dtype}")
    print("="*60 + "\n")
    
    return model, tokenizer


def run_lm_eval(checkpoint_path: str, 
                base_model: str = "meta-llama/Llama-2-7b-hf",
                tasks: str = "mmlu",
                num_fewshot: int = 0,
                batch_size: int = 8,
                output_path: str = None):
    """
    Run lm-evaluation-harness on PEQAT checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint
        base_model: Base model name  
        tasks: Comma-separated task names
        num_fewshot: Number of few-shot examples
        batch_size: Batch size for evaluation
        output_path: Where to save results JSON
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
    is_peft = os.path.isdir(checkpoint_path) or "checkpoint" in checkpoint_path

    if is_peft:
        print(f"Type: PEFT Adapter (Local path)")
        model_args = f"pretrained={base_model},peft={checkpoint_path},torch_dtype=bfloat16"
    else:
        # It's a baseline HF model (e.g. meta-llama/Llama-2-7b-hf)
        print(f"Type: Base Model (HF Hub)")
        model_args = f"pretrained={checkpoint_path},torch_dtype=bfloat16"
    
    print(f"Model Args: {model_args}")  # Debug print


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
    
    # Run evaluation
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"\n✅ Evaluation complete! Results saved to {output_path}")
        
        # Try to load and display key results
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
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    if "results" in results:
        for task_name, metrics in results["results"].items():
            print(f"\n{task_name}:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PEQAT model on MMLU and Commonsense tasks")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PEQAT checkpoint (e.g., ../peqat-checkpoints/output_qat_lora_alpaca_first_try_92aa_001/checkpoint-1544)"
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
        help="Tasks to evaluate (comma-separated). Options: mmlu, boolq, piqa, hellaswag, winogrande, arc_easy, arc_challenge, openbookqa"
    )
    
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples (0 for zero-shot, 5 for 5-shot)"
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
