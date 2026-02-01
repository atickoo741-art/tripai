# validate_outputs.py
"""
Generate reference outputs using trained PE-QAT model
Compare with CUDA kernel outputs for validation
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM
from peft import PeftModel
import argparse

def generate_reference_output(adapter_dir, base_model_name, input_path, output_path):
    """Generate reference output using trained PE-QAT model"""
    
    print("Loading trained PE-QAT model...")
    device = torch.device('cuda')
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16
    ).to(device)
    
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
    
    # Load input
    X = torch.from_numpy(np.load(input_path)).to(device).to(torch.bfloat16)
    
    print(f"Input shape: {X.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = peft_model(inputs_embeds=X)
        Y = outputs.logits
    
    # Save output
    np.save(output_path, Y.cpu().float().numpy())
    print(f"✓ Saved reference output to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--input", default="test_input.npy")
    parser.add_argument("--output", default="reference_output.npy")
    args = parser.parse_args()
    
    generate_reference_output(
        args.adapter_dir,
        args.base_model,
        args.input,
        args.output
    )
