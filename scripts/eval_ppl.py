"""WikiText-2 perplexity eval for deployed PE-QAT model (CUDA runtime)."""
import sys, os, json, torch, torch.nn as nn, tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from datasets import load_dataset
from peqat_deploy.runtime.loader import build_cuda_model


@torch.no_grad()
def evaluate(model, tokenizer, device="cuda", seq_len=2048, stride=2048, n_samples=None):
    model.eval()
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids.to(device)
    n = n_samples if n_samples else enc.size(1) // seq_len
    nlls = []
    for i in tqdm.trange(n, desc="PPL eval"):
        batch = enc[:, i * seq_len : (i + 1) * seq_len]
        logits = model(batch).logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:]
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss.float() * seq_len)
    ppl = torch.exp(torch.stack(nlls).sum() / (n * seq_len))
    return ppl.item()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("export_dir", type=str)
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    device = "cuda"
    model, tokenizer = build_cuda_model(args.export_dir, args.base_model, device)
    ppl = evaluate(model, tokenizer, device, args.seq_len, n_samples=args.n_samples)
    print(f"\nWikiText-2 perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()
