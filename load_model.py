from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from fake_quant import load_peqat_model
# 1. Load the GPT-2 small model
# We'll prepare for 4-bit quantization later by defining the BitsAndBytesConfig now.
# This will ensure compatibility when we move to the W4A4 quantization step.

# Quantization configuration for 4-bit (W4A4)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config, # Uncomment this line when you are ready for 4-bit loading
    torch_dtype=torch.bfloat16, # Use bfloat16 for better precision if not using 4-bit for now
    device_map="auto"
)
model_path = "qat_lora_model_19"
model, tokenizer = load_peqat_model(model_path, "gpt2", "cuda")

# Add a pad token to the tokenizer if it doesn't have one, as it's common for causal LMs
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

print(f"Model '{model_name}' loaded successfully.")
print(f"Model device: {model.device}")
print(f"Number of parameters: {model.num_parameters()}")

# 2. Test text generation
print("\n--- Testing Text Generation ---")

# Set the model to evaluation mode
model.eval()

prompt = "The quick brown fox jumps over the"

# Encode the prompt and get attention mask
input_dict = tokenizer.encode_plus(
    prompt,
    return_tensors="pt",
    padding=True, # Ensure padding is applied
    truncation=True
).to(model.device)
input_ids = input_dict["input_ids"]
attention_mask = input_dict["attention_mask"]

# Generate text
# Using no_grad() for inference to save memory and speed up computation
with torch.no_grad():
    output = model.generate(
        input_ids,
        attention_mask=attention_mask, # Pass the attention mask here
        max_new_tokens=50, # Generate up to 50 new tokens
        num_return_sequences=1,
        no_repeat_ngram_size=2, # Avoid repeating n-grams
        do_sample=True, # Enable sampling
        top_k=50, # Sample from top 50 probable words
        top_p=0.95, # Sample from words that make up 95% of probability mass
        temperature=0.7, # Control randomness
        pad_token_id=tokenizer.eos_token_id # Use eos_token as pad_token_id
    )

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Generated text: {generated_text}") 