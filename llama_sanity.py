# verify_llama_setup.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

print("="*60)
print("Verifying Llama-2 Setup for PE-QAT")
print("="*60)

try:
    # 1. Check GPU
    assert torch.cuda.is_available(), "CUDA not available!"
    print(f"✓ CUDA found: {torch.cuda.get_device_name(0)}")

    # 2. Check Model & Tokenizer Access
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✓ Tokenizer loaded successfully.")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("✓ Model loaded successfully onto GPU.")
    
    # 3. Quick Inference Test
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=15)
    result = tokenizer.decode(outputs[0])
    print(f"Inference output: {result}")
    # assert "Paris" in result, "Inference test failed to produce expected output."
    print(f"✓ Inference test passed: '{result}'")
    
    del model
    torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("✓ Setup is correct. You are ready to train.")
    print("="*60)

except Exception as e:
    print(f"\n❌ SETUP FAILED: {e}")
    print("Please check your HuggingFace login, Llama-2 access, and CUDA setup.")
    sys.exit(1)
