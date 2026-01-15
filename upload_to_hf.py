# upload_latest_checkpoint.py
from huggingface_hub import HfApi
import os
import sys
from pathlib import Path

def find_latest_checkpoint(base_dir="./"):
    """Find the most recently modified output_* directory"""
    checkpoints = [d for d in Path(base_dir).iterdir() 
                   if d.is_dir() and d.name.startswith("output_")]
    
    if not checkpoints:
        return None
    
    # Sort by modification time, most recent first
    latest = max(checkpoints, key=lambda d: d.stat().st_mtime)
    return latest

def upload_checkpoint(checkpoint_path, repo_id="shresthmishra/peqat-checkpoints-v2"):
    api = HfApi()
    checkpoint_name = os.path.basename(checkpoint_path)
    
    print(f"Uploading: {checkpoint_path}")
    print(f"To repo: {repo_id}")
    print(f"As: {checkpoint_name}\n")
    
    api.upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=checkpoint_name,
        commit_message=f"Upload {checkpoint_name}",
        ignore_patterns=["*.git*", "runs/*"]
    )
    
    print(f"\n✅ Upload complete!")
    print(f"View at: https://huggingface.co/{repo_id}/tree/main/{checkpoint_name}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path:
            print(f"Auto-detected latest checkpoint: {checkpoint_path}")
        else:
            print("No checkpoints found. Usage: python upload_latest_checkpoint.py <checkpoint_path>")
            sys.exit(1)
    
    upload_checkpoint(checkpoint_path)
