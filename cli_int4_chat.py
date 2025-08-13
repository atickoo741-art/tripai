import os
import argparse
import torch
import time
from transformers import AutoTokenizer
from torch.serialization import add_safe_globals
from peft.peft_model import PeftModelForCausalLM
from export_qat_int4 import Int4LinearRuntime
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load already exported INT4 model


def load_exported_int4(model_dir: str, device: torch.device):
    # Load full pickled model first if available, else state_dict
    model_path = os.path.join(model_dir, 'int4_model.pt')
    state_path = os.path.join(model_dir, 'int4_model_state.pt')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if os.path.isfile(model_path):
        # Allowlist custom classes for safe unpickling and disable weights_only
        add_safe_globals([Int4LinearRuntime, PeftModelForCausalLM])
        model = torch.load(model_path, map_location=device, weights_only=False)
        model = model.to(device).eval()
        return model, tokenizer
    elif os.path.isfile(state_path):
        # Only state_dict exists. Please export again to include int4_model.pt
        raise RuntimeError("Only int4_model_state.pt found. Please export again to include int4_model.pt for easy loading.")
    else:
        raise FileNotFoundError(f"No INT4 export artifacts found in {model_dir}")


def generate_reply(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int = 128):
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
    text = tokenizer.decode(output_ids[len(inputs['input_ids'][0]):], skip_special_tokens=True)
    return text


def load_full_precision_model(fp_model_name: str, fp_qat_dir: str | None, device: torch.device):
    # Load base FP model
    base = AutoModelForCausalLM.from_pretrained(
        fp_model_name,
        torch_dtype=torch.float32,
        device_map=None,
    ).to(device)
    if fp_qat_dir:
        # Load LoRA adapters and merge for full-precision inference parity
        peft_model = PeftModel.from_pretrained(base, fp_qat_dir, torch_dtype=torch.float32).to(device)
        try:
            merged = peft_model.merge_and_unload()
            return merged.eval()
        except Exception:
            # Fallback: use PEFT wrapper directly
            return peft_model.eval()
    return base.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--int4_export_dir', type=str, required=True, help='Directory with int4_model.pt from export script')
    parser.add_argument('--fp_model_name', type=str, default='gpt2', help='HF model name/path for full-precision reference')
    parser.add_argument('--fp_qat_dir', type=str, default=None, help='Optional LoRA/QAT adapters to load into FP model (merged)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    int4_model, tokenizer = load_exported_int4(args.int4_export_dir, device)
    fp_model = load_full_precision_model(args.fp_model_name, args.fp_qat_dir, device)

    print("Loaded INT4 runtime model. Type your prompt and press Enter. Type 'exit' to quit.")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()  # newline
            break
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue
        # INT4
        t0 = time.perf_counter()
        reply_int4 = generate_reply(int4_model, tokenizer, prompt, device)
        t1 = time.perf_counter()
        # FP
        t2 = time.perf_counter()
        reply_fp = generate_reply(fp_model, tokenizer, prompt, device)
        t3 = time.perf_counter()
        print(f"INT4 [{(t1 - t0)*1000:.0f} ms]: {reply_int4}")
        print(f"FP   [{(t3 - t2)*1000:.0f} ms]: {reply_fp}")


if __name__ == '__main__':
    main()


