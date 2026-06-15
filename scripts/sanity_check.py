import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from peqat_deploy.runtime.cpu_model import build_packed_model, generate_text


def main():
    parser = argparse.ArgumentParser(
        description="Sanity-check exported packed weights via CPU inference"
    )
    parser.add_argument(
        "export_dir",
        type=str,
        help="Path to exported PE-QAT directory",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (reads from manifest if not provided)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Prompt for text generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=10,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu recommended for 8B model)",
    )
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.export_dir, "manifest.json")):
        print(f"Error: {args.export_dir} is not a valid export directory")
        sys.exit(1)

    with open(os.path.join(args.export_dir, "manifest.json")) as f:
        manifest = json.load(f)

    base_model = args.base_model or manifest["base_model_name"]
    device = args.device

    print(f"\nBuilding packed model from: {args.export_dir}")
    print(f"Base model: {base_model}")
    print(f"Device: {device}")
    model, tokenizer, replacements = build_packed_model(
        args.export_dir, base_model, device=device
    )

    print(f"\nGenerating with prompt: \"{args.prompt}\"")
    output = generate_text(model, tokenizer, args.prompt, args.max_new_tokens, device)
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
