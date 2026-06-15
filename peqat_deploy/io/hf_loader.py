import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base_model(model_name: str, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    if device != "cpu":
        try:
            model = model.to(device)
        except torch.AcceleratorError:
            print(f"  Warning: could not move model to {device}, falling back to cpu")
            device = "cpu"
    model.eval()
    return model, tokenizer
