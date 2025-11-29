import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import argparse
from fake_quant import load_peqat_model, Evaluation, naive_quantize_model
from datasets import load_dataset
from fake_quant import QATW4A4LinearCore, _find_quant_params_path



parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=False, default=None)
parser.add_argument("--base_model_name", type=str, default="gpt2", help="HF base model to use when model_path is a PEFT adapter dir")
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--seq_len", type=int, default=1024)
parser.add_argument("--n_samples", type=int, default=None)

def main():
    args = parser.parse_args()
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

        
    if args.model_path is None:
        # Load base full precision model for benchmarking
        print(f"Loading base full precision model: {args.base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(args.base_model_name, torch_dtype=torch.float32, device_map=None).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
        
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Model loaded on {device}")
    else:
        # Detect if model_path is a PEFT adapters directory
        is_peft = os.path.isfile(os.path.join(args.model_path, "adapter_config.json"))

        model, tokenizer = load_peqat_model(args.model_path, args.base_model_name, device)

        # model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32, device_map=None).to(device)
        # tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        # naive_quantize_model(model)
        # diagnose_model_loading(model, args.model_path)


    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    eval_obj = Evaluation(tokenizer, dataset, device, num_samples=args.n_samples, block_size=args.seq_len)
    # eval_obj.evaluate(model, seq_len=args.seq_len, n_samples=args.n_samples)
    eval_obj.evaluate(model)


def diagnose_model_loading(model, adapter_dir):
    """Verify model loaded correctly"""
    print("\n" + "="*60)
    print("MODEL LOADING DIAGNOSTICS")
    print("="*60)
    
    # Check LoRA parameters
    lora_params_found = 0
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_params_found += 1
            print(f"LoRA param: {name}")
            print(f"  Shape: {param.shape}, Mean: {param.mean().item():.6f}, Std: {param.std().item():.6f}")
            # if lora_params_found >= 4:  # Just show first few
            #     break
    
    print(f"\nTotal LoRA parameters found: {lora_params_found}")
    
    # Check quantization parameters
    qat_modules_found = 0
    for name, module in model.named_modules():
        if isinstance(module, QATW4A4LinearCore):
            qat_modules_found += 1
            print(f"\nQAT module: {name}")
            print(f"  activation_clip: mean={module.activation_clip.mean().item():.4f}, "
                  f"min={module.activation_clip.min().item():.4f}, "
                  f"max={module.activation_clip.max().item():.4f}")
            print(f"  smooth_factor: mean={module.smooth_factor.mean().item():.4f}, "
                  f"min={module.smooth_factor.min().item():.4f}, "
                  f"max={module.smooth_factor.max().item():.4f}")
            # if qat_modules_found >= 3:
            #     break
    
    print(f"\nTotal QAT modules found: {qat_modules_found}")
    
    # Check if quantization params file exists
    qp_path = _find_quant_params_path(adapter_dir)
    if qp_path:
        print(f"\n✓ Quantization params found at: {qp_path}")
    else:
        print(f"\n✗ WARNING: No quantization_params.pth found in {adapter_dir}")
    
    # Check adapter files
    adapter_config = os.path.join(adapter_dir, "adapter_config.json")
    adapter_model = os.path.join(adapter_dir, "adapter_model.safetensors")
    print(f"\n✓ adapter_config.json: {os.path.exists(adapter_config)}")
    print(f"✓ adapter_model.safetensors: {os.path.exists(adapter_model)}")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

