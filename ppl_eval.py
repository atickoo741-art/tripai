import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import argparse
from fake_quant import load_peqat_model, Evaluation
from datasets import load_dataset




parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
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

    # Detect if model_path is a PEFT adapters directory
    is_peft = os.path.isfile(os.path.join(args.model_path, "adapter_config.json"))

    model, tokenizer = load_peqat_model(args.model_path, args.base_model_name, device)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_obj = Evaluation(tokenizer, dataset, device, num_samples=args.n_samples)
    eval_obj.evaluate(model, seq_len=args.seq_len, n_samples=args.n_samples)


if __name__ == "__main__":
    main()

