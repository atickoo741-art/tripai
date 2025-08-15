# import os
# from transformers import TrainerCallback
# from fake_quant import QATW4A4Linear
# import torch

# def find_available_dir(base_dir):
#     """Find the next available directory with an index to prevent overwrites."""
#     index = 1
#     while True:
#         dir_name = f"{base_dir}_{index}" if index > 1 else base_dir
#         if not os.path.exists(dir_name):
#             return dir_name
#         index += 1


# class SaveQuantParamsCallback(TrainerCallback):
#     def on_save(self, args, state, control, **kwargs):
#         model = kwargs.get("model")
#         try:
#             ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
#             if os.path.isdir(ckpt_dir):
#                 quant_params = {}
#                 for name, module in model.named_modules():
#                     if isinstance(module, QATW4A4Linear):
#                         prefix = f"{name}."
#                         quant_params[f"{prefix}activation_clip"] = module.activation_clip.data.detach().cpu()
#                         if hasattr(module, 'smooth_factor'):
#                             quant_params[f"{prefix}smooth_factor"] = module.smooth_factor.data.detach().cpu()
#                         if hasattr(module, 'input_min') and hasattr(module, 'input_max'):
#                             quant_params[f"{prefix}input_min"] = module.input_min.data.detach().cpu()
#                             quant_params[f"{prefix}input_max"] = module.input_max.data.detach().cpu()
#                         quant_params[f"{prefix}weight_scale"] = module.weight_scale.data.detach().cpu()
#                         quant_params[f"{prefix}weight_zero_point"] = module.weight_zero_point.data.detach().cpu()
#                         quant_params[f"{prefix}act_scale"] = module.act_scale.data.detach().cpu()
#                         quant_params[f"{prefix}act_zero_point"] = module.act_zero_point.data.detach().cpu()
#                 torch.save(quant_params, os.path.join(ckpt_dir, "quantization_params.pth"))
#         except Exception as e:
#             print(f"Warning: failed to save quant params at checkpoint: {e}")

# def save_quantization_parameters(model, save_dir):
#         os.makedirs(save_dir, exist_ok=True)

#         quant_params = {}
#         i = 0
#         for name, module in model.named_modules():
#             i += 1
#             print(i)
#             if isinstance(module, QATW4A4Linear):
#                 # Save all quantization parameters
#                 param_prefix = f"{name}."
#                 quant_params[f"{param_prefix}activation_clip"] = module.activation_clip.data
#                 quant_params[f"{param_prefix}smooth_factor"] = module.smooth_factor.data
#                 quant_params[f"{param_prefix}input_min"] = module.input_min.data
#                 quant_params[f"{param_prefix}input_max"] = module.input_max.data
#                 quant_params[f"{param_prefix}weight_scale"] = module.weight_scale.data
#                 quant_params[f"{param_prefix}weight_zero_point"] = module.weight_zero_point.data
#                 quant_params[f"{param_prefix}act_scale"] = module.act_scale.data
#                 quant_params[f"{param_prefix}act_zero_point"] = module.act_zero_point.data
        
#         # Save quantization parameters
#         torch.save(quant_params, os.path.join(save_dir, "quantization_params.pth"))
#         print(f"Saved {len(quant_params)} quantization parameters")

# class Evaluation:
#     def __init__(self, tokenizer, dataset, device, num_samples = 40):
#         self.dataset = dataset
#         self.tokenizer = tokenizer
#         self.device = device
    
#     def evaluate(self, model, seq_len=1024, n_samples=None):
#         input_ids = self.tokenizer("\n\n".join(self.dataset["text"]), return_tensors="pt").input_ids.to(self.device)
#         n_samples = n_samples if n_samples is not None else (input_ids.size(1) // seq_len)

#         model.eval()

#         nlls = []
#         with torch.no_grad():
#             for i in range(n_samples):
#                 start = i * seq_len
#                 end = (i + 1) * seq_len
#                 batch = input_ids[:, start:end]
#                 logits = model(batch).logits
#                 shift_logits = logits[:, :-1, :].contiguous().float()
#                 shift_labels = batch[:, 1:]
#                 loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#                 nlls.append(loss.float() * (seq_len - 1))
#         denom = n_samples * (seq_len - 1)
#         ppl = torch.exp(torch.stack(nlls).sum() / denom)
#         print(f"Perplexity: {ppl.item():.2f}")

