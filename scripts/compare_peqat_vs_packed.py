import argparse
import json
import os
import sys
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from peqat_deploy.runtime.cpu_model import build_packed_model as build_packed


def _load_original(
    adapter_dir: str,
    base_model_name: str,
    device: str = "cpu",
):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fake_quant import load_peqat_model
    model, tokenizer = load_peqat_model(
        adapter_dir=adapter_dir,
        base_model_name=base_model_name,
        device=torch.device(device),
    )
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Compare original PE-QAT output vs packed export output"
    )
    parser.add_argument(
        "export_dir",
        type=str,
        help="Path to exported PE-QAT directory",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        required=True,
        help="Path to original PE-QAT checkpoint (e.g., checkpoint-114)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (default: read from manifest)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Prompt for comparison",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device",
    )
    args = parser.parse_args()

    with open(os.path.join(args.export_dir, "manifest.json")) as f:
        manifest = json.load(f)
    base_model = args.base_model or manifest["base_model_name"]

    tokenizer = None

    # --- Path 1: Packed export model ---
    print("=" * 60)
    print("Path 1: Packed export model")
    print("=" * 60)
    packed_model, tokenizer, _ = build_packed(
        args.export_dir, base_model, device=args.device
    )
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        packed_logits = packed_model(inputs["input_ids"]).logits
    packed_output = packed_logits[0, -1, :]  # last token logits
    packed_pred = packed_output.argmax().item()
    packed_text = tokenizer.decode(packed_pred)
    print(f"  Prompt: {args.prompt}")
    print(f"  Predicted next token: '{packed_text}' (id={packed_pred})")
    print(f"  Top-5 logits: {packed_output.topk(5).indices.tolist()}")
    print(f"  Top-5 tokens: {[tokenizer.decode(i) for i in packed_output.topk(5).indices.tolist()]}")

    del packed_model
    gc.collect()

    # --- Path 2: Original PE-QAT model ---
    print("\n" + "=" * 60)
    print("Path 2: Original PE-QAT (from checkpoint)")
    print("=" * 60)
    orig_model, _ = _load_original(args.adapter_dir, base_model, device=args.device)
    with torch.no_grad():
        orig_logits = orig_model(inputs["input_ids"]).logits
    orig_output = orig_logits[0, -1, :]
    orig_pred = orig_output.argmax().item()
    orig_text = tokenizer.decode(orig_pred)
    print(f"  Prompt: {args.prompt}")
    print(f"  Predicted next token: '{orig_text}' (id={orig_pred})")
    print(f"  Top-5 logits: {orig_output.topk(5).indices.tolist()}")
    print(f"  Top-5 tokens: {[tokenizer.decode(i) for i in orig_output.topk(5).indices.tolist()]}")

    del orig_model
    gc.collect()

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    cos_sim = torch.nn.functional.cosine_similarity(
        packed_output.unsqueeze(0), orig_output.unsqueeze(0)
    ).item()
    mse = (packed_output - orig_output).pow(2).mean().item()
    max_diff = (packed_output - orig_output).abs().max().item()
    same_pred = packed_pred == orig_pred
    print(f"  Same prediction: {same_pred}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  Max logit diff: {max_diff:.6f}")
    print(f"\n  {'MATCH' if same_pred and cos_sim > 0.99 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
