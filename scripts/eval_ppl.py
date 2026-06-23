"""WikiText-2 perplexity eval for PE-QAT models (deployed CUDA runtime or original fake-quant)."""
import sys, os, json, time, torch, torch.nn as nn, tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from datasets import load_dataset
from peqat_deploy.runtime.loader import build_cuda_model


@torch.no_grad()
def evaluate(model, tokenizer, device="cuda", seq_len=2048, stride=2048, n_samples=None):
    model.eval()
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
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
        torch.cuda.synchronize()
        time.sleep(1)
    ppl = torch.exp(torch.stack(nlls).sum() / (n * seq_len))
    return ppl.item()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate PE-QAT model perplexity on WikiText-2.")
    parser.add_argument("model_path", type=str, help="Path to export dir (CUDA runtime) or original PE-QAT checkpoint dir")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--hadamard", action="store_true", help="Use Hadamard PE-QAT export format")
    parser.add_argument("--mode", type=str, choices=["auto", "export", "export-folded", "original"], default="auto",
                        help="'export' for deployed CUDA runtime, 'export-folded' for INT4 TC folded-weights validation, 'original' for fake-quant PyTorch, 'auto' to detect")
    args = parser.parse_args()

    device = "cuda"

    if args.hadamard:
        from peqat_deploy.runtime.loader import build_hadamard_cuda_model

    if args.mode == "export":
        if args.hadamard:
            model, tokenizer = build_hadamard_cuda_model(args.model_path, args.base_model, device)
        else:
            model, tokenizer = build_cuda_model(args.model_path, args.base_model, device)
    elif args.mode == "export-folded":
        model, tokenizer = build_cuda_model(args.model_path, args.base_model, device, use_folded=True)
    elif args.mode == "original":
        from fake_quant import load_peqat_model
        import torch
        model, tokenizer = load_peqat_model(args.model_path, args.base_model, torch.device(device))
        model.to(torch.bfloat16)
    else:
        is_original = os.path.isfile(os.path.join(args.model_path, "quantization_params.pth"))
        is_export  = os.path.isfile(os.path.join(args.model_path, "manifest.json"))
        if is_export:
            if args.hadamard:
                model, tokenizer = build_hadamard_cuda_model(args.model_path, args.base_model, device)
            else:
                model, tokenizer = build_cuda_model(args.model_path, args.base_model, device)
        elif is_original:
            from fake_quant import load_peqat_model
            import torch
            model, tokenizer = load_peqat_model(args.model_path, args.base_model, torch.device(device))
            model.to(torch.bfloat16)
        else:
            print(f"Error: {args.model_path} is neither a valid export dir (manifest.json) "
                  f"nor an original checkpoint dir (quantization_params.pth)")
            sys.exit(1)

    ppl = evaluate(model, tokenizer, device, args.seq_len, n_samples=args.n_samples)
    print(f"\nWikiText-2 perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()
