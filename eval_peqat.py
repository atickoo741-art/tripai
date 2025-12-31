import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from fakequant import QATW4A4LinearCore
import sys

def load_peqat_model(checkpoint_path, base_model="meta-llama/Llama-2-7b-hf"):
    """Load PEQAT quantized model with LoRA adapters"""
    print(f"Loading base model: {base_model}")
    
    # Load base model in bf16
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Load LoRA adapters
    print(f"Loading PEQAT checkpoint: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    
    # Apply quantization (your custom layers should already be there)
    # Load quantization parameters if saved separately
    import os
    quant_params_path = os.path.join(checkpoint_path, "quantization_params.pth")
    if os.path.exists(quant_params_path):
        quant_params = torch.load(quant_params_path)
        for name, module in model.named_modules():
            if isinstance(module, QATW4A4LinearCore):
                prefix = f"{name}."
                if f"{prefix}activation_clip" in quant_params:
                    module.activation_clip.data = quant_params[f"{prefix}activation_clip"]
                if f"{prefix}smooth_factor" in quant_params:
                    module.smooth_factor.data = quant_params[f"{prefix}smooth_factor"]
        print(f"✓ Loaded quantization parameters from {quant_params_path}")
    
    model.eval()
    
    # Verify quantization is active
    print("\n--- Quantization Status ---")
    qat_modules = 0
    for name, module in model.named_modules():
        if isinstance(module, QATW4A4LinearCore):
            qat_modules += 1
            if qat_modules <= 3:  # Print first 3
                print(f"{name}: clip={module.activation_clip.item():.4f}")
    print(f"Total QAT modules: {qat_modules}\n")
    
    return model

if __name__ == "__main__":
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "./cloud_qat_lora/checkpoint-220"
    model = load_peqat_model(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # Test forward pass
    test_input = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**test_input)
    print("✓ Model loaded successfully")
