import sys
import torch
sys.path.insert(0, ".")
from peqat_deploy.runtime.loader import build_cuda_model

export_dir = sys.argv[1]
device = "cuda"

model, tokenizer = build_cuda_model(export_dir, "meta-llama/Llama-3.1-8B", device=device)

input_ids = torch.randint(0, 32000, (1, 1), device=device)
with torch.no_grad():
    for _ in range(3):
        model(input_ids)  # warmup

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(5):
            model(input_ids)

print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=20
))
