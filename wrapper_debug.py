import torch
import torch.nn as nn
from fake_quant import load_peqat_model, Evaluation, QATW4A4Linear

model_path = "qat_lora_model_5"
(model, tokenizer) = load_peqat_model(model_path, "gpt2", torch.device("mps"))

for name, m in model.named_modules():
    if isinstance(m, QATW4A4Linear):
        print(name, "has QAT wrapper")

