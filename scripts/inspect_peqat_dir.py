import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from peqat_deploy.io.discover import discover_source_dir


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a PE-QAT training directory and print its contents"
    )
    parser.add_argument("source_dir", type=str, help="Path to PE-QAT output directory")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint subdirectory to inspect",
    )
    args = parser.parse_args()

    discovered = discover_source_dir(args.source_dir, args.checkpoint)

    print(f"Source directory: {discovered['source_dir']}")
    print(f"Source basename: {discovered['source_basename']}")
    print()

    checkpoints = discovered["checkpoints"]
    print(f"Checkpoints ({len(checkpoints)}):")
    for cp in checkpoints:
        marker = "  ← selected" if cp == discovered["selected_checkpoint"] else ""
        print(f"  {os.path.basename(cp)}{marker}")
    print()

    if discovered["selected_checkpoint"]:
        print(f"Selected checkpoint: {os.path.basename(discovered['selected_checkpoint'])}")
    else:
        print("Selected checkpoint: (none)")
    print()

    print("Files discovered:")
    for key in ("adapter_config", "adapter_weights", "quantization_params", "training_config", "model_config"):
        val = discovered.get(key)
        if val:
            rel = os.path.relpath(val, args.source_dir)
            print(f"  {key}: {rel}")
        else:
            print(f"  {key}: (not found)")

    print(f"  tokenizer_files ({len(discovered['tokenizer_files'])}):")
    for tf in discovered["tokenizer_files"]:
        rel = os.path.relpath(tf, args.source_dir)
        print(f"    {rel}")


if __name__ == "__main__":
    main()
