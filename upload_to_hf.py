from huggingface_hub import HfApi
import os

api = HfApi()
username = "shresthmishra1"  # Change to your username
repo_id = f"{username}/peqat-llama2-7b-checkpoints"

# Create PRIVATE repository
print(f"Creating private repository: {repo_id}")
api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

# Upload best checkpoint
checkpoint_dir = "./output_qat_lora_20/checkpoint-440"
if os.path.exists(checkpoint_dir):
    print(f"\nUploading {checkpoint_dir}...")
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        path_in_repo="checkpoint-440",
        repo_type="model",
    )
    print("✓ Checkpoint uploaded!")

# Upload final model
model_dir = "./qat_lora_model_20"
if os.path.exists(model_dir):
    print(f"\nUploading {model_dir}...")
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        path_in_repo="final_model",
        repo_type="model",
    )
    print("✓ Final model uploaded!")

print(f"\nSUCCESS! Weights at: https://huggingface.co/{repo_id}")
