import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
from datasets import load_dataset
import argparse
from fake_quant import load_peqat_model

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--base_model_name", type=str, default="gpt2", help="HF base model to use when model_path is a PEFT adapter dir")
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--seq_len", type=int, default=1024)
parser.add_argument("--n_samples", type=int, default=None)

class Evaluation:
    def __init__(self, tokenizer, dataset, device, num_samples = 40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

def main():
    args = parser.parse_args()
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    # Detect if model_path is a PEFT adapters directory
    is_peft = os.path.isfile(os.path.join(args.model_path, "adapter_config.json"))

    # Load tokenizer
    # tok_source = args.base_model_name if is_peft else args.model_path
    # tokenizer = AutoTokenizer.from_pretrained(tok_source)
    
    model, tokenizer = load_peqat_model(args.model_path, args.base_model_name, device)


    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Tokenize entire test set into one long tensor, then stride windows of seq_len
    input_ids = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids.to(device)
    n_samples = args.n_samples if args.n_samples is not None else (input_ids.size(1) // args.seq_len)


    model.eval()

    nlls = []
    with torch.no_grad():
        for i in range(n_samples):
            start = i * args.seq_len
            end = (i + 1) * args.seq_len
            batch = input_ids[:, start:end]
            logits = model(batch).logits
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = batch[:, 1:]
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss.float() * (args.seq_len - 1))
    denom = n_samples * (args.seq_len - 1)
    ppl = torch.exp(torch.stack(nlls).sum() / denom)
    print(f"Perplexity: {ppl.item():.2f}")

if __name__ == "__main__":
    main()


    # Removed helper; using raw WikiText-2 evaluation method now

