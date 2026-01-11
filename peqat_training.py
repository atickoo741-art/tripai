import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from data_preparation import load_and_preprocess_wikitext2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
print(f"Vocab size: {len(tokenizer)}") 
print(f"PAD token: {tokenizer.pad_token}")  # Should be <|finetune_right_pad_id|> or similar
from calibrations import unified_calibration
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from typing import List
from fake_quant import quantize_model, PEQATLoRAModule, find_available_dir, SaveQuantParamsCallback, save_quantization_parameters, Evaluation, GradientMonitorCallback, QATW4A4LinearCore
import torch.nn as nn
from data_preparation import load_wikitext2_sliding_window

from config_loader import Config

import torch.cuda



# def save_model(peft_model, tokenizer, model_dir, model_name, device):
    # # Save the model, tokenizer, and quantization parameters
    
    # print(f"\n--- Saving W4A4 QAT LoRA model to {model_dir} ---")
    # os.makedirs(model_dir, exist_ok=True)
    
    # # Save quantization parameters after training
    # # save_quantization_parameters(peft_model, model_dir)
    
    # # Save the LoRA adapters
    # print(f"--- Saving W4A4 QAT LoRA adapters to {model_dir} ---")
    # peft_model.save_pretrained(model_dir)

    # # Save the tokenizer
    # tokenizer.save_pretrained(model_dir)
    
    # best_checkpoint = trainer.state.best_model_checkpoint
    # if best_checkpoint and os.path.exists(best_checkpoint):
    #     import shutil
    #     src = os.path.join(best_checkpoint, "quantization_params.pth")
    #     dst = os.path.join(model_dir, "quantization_params.pth")
        
    #     if os.path.exists(src):
    #         shutil.copy(src, dst)
    #         print(f"✓ Copied quantization params from: {best_checkpoint}")
            
    #         # Verify the copy
    #         quant_params = torch.load(dst)
    #         print(f"  {len(quant_params)} quantization parameters")
    #     else:
    #         print(f"⚠️  Warning: quantization_params.pth not found in best checkpoint")
    #         print(f"  Falling back to current model state...")
            
    #         # Fallback: save current state
    #         quant_params = {}
    #         for name, module in peft_model.named_modules():
    #             if isinstance(module, QATW4A4LinearCore):
    #                 prefix = f"{name}."
    #                 quant_params[f"{prefix}activation_clip"] = module.activation_clip.data.detach().cpu()
    #                 quant_params[f"{prefix}smooth_factor"] = module.smooth_factor.data.detach().cpu()
            
    #         torch.save(quant_params, dst)
    #         print(f"  ✓ Saved {len(quant_params)} parameters from current state")
    # else:
    #     print("⚠️  No best checkpoint found, saving current model state...")
        
    #     # Save current quantization state
    #     quant_params = {}
    #     for name, module in peft_model.named_modules():
    #         if isinstance(module, QATW4A4LinearCore):
    #             prefix = f"{name}."
    #             quant_params[f"{prefix}activation_clip"] = module.activation_clip.data.detach().cpu()
    #             quant_params[f"{prefix}smooth_factor"] = module.smooth_factor.data.detach().cpu()
        
    #     torch.save(quant_params, os.path.join(model_dir, "quantization_params.pth"))
    #     print(f"✓ Saved {len(quant_params)} quantization parameters")


    # # Save the model config
    # peft_model.config.to_json_file(os.path.join(model_dir, "config.json"))
    
    
    # print(f"\n--- Model, tokenizer, and quantization config saved to {model_dir} ---")
    # print("To load this model for inference, use:")
    # print(f"model = AutoModelForCausalLM.from_pretrained('{model_name}').to('{device}')")
    # print(f"model = PeftModel.from_pretrained(model, '{model_dir}')")
    # print("# Make sure to apply the same quantization setup when loading")

def ensure_dtype_consistency(model, target_dtype=torch.bfloat16):
    """Ensure all model parameters have the same dtype"""
    print(f"\nEnsuring all parameters are {target_dtype}...")
    
    mismatches = []
    for name, param in model.named_parameters():
        if param.dtype != target_dtype:
            mismatches.append((name, param.dtype))
            param.data = param.data.to(target_dtype)
    
    for name, buffer in model.named_buffers():
        if buffer.dtype != target_dtype and buffer.is_floating_point():
            buffer.data = buffer.data.to(target_dtype)
    
    if mismatches:
        print(f"Fixed {len(mismatches)} dtype mismatches:")
        for name, dtype in mismatches[:5]:
            print(f"  {name}: {dtype} → {target_dtype}")
    else:
        print("✓ All parameters already consistent")
    
    return model


def custom_eval(peft_model, tokenizer, device, trainer_ppl):
    print("\n--- Evaluating with Proper Perplexity Metric (Test Set) ---")
    from datasets import load_dataset
    
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # proper_eval = Evaluation(tokenizer, test_dataset, device, num_samples=None)
    proper_eval = Evaluation(test_dataset, tokenizer, device, num_samples=None)
    
    # Use stride=2048 (no overlap) to match paper baseline
    custom_ppl = proper_eval.evaluate(peft_model, seq_len=2048, stride=2048)
    print(f"\nTest Perplexity (Standard Protocol): {custom_ppl:.2f}")
    print(f"Target for FP16 Llama-2-7b: ~5.47")
    print(f"Target for W4A4 QAT: ~7-9 (realistic for aggressive quantization)")


    print(f"\nComparison:")
    print(f"  Trainer perplexity: {trainer_ppl:.2f}")
    print(f"  Custom eval perplexity: {custom_ppl:.2f}")
    print(f"  Difference: {abs(trainer_ppl - custom_ppl):.2f}")


if __name__ == "__main__":
    config = Config("config.yaml")
    # config = Config("alpaca_config.yaml")
    print("=" * 60)
    print(f"Experiment: {config.get('experiment.name')}")
    print(f"Run ID: {config.run_id}")
    print(f"Notes: {config.get('experiment.notes')}")
    print("=" * 60)

    torch.manual_seed(config.get('system.seed'))



    model_name = config.get('model.name')
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    # MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    BLOCK_SIZE = 2048
    NUM_EPOCHS = 10 
    BATCH_SIZE = 16 
    GRAD_ACCUM = 2 
    LEARNING_RATE = 5e-5
    LORA_RANK = 32     

    # use_smooth = True
    learn_smooth = config.get('calibration.learn_smooth', True)
    calibrate_smooth = config.get('calibration.calibrate_smooth', False)
    torch.backends.cuda.matmul.allow_tf32 = config.get('system.tf32', True)
    # ============================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Set pad_token to eos_token (ID: {tokenizer.pad_token_id})")

    device = torch.device("cuda")
    # torch.set_default_dtype(torch.bfloat16) # Use bfloat16 for Llama

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if bool(config.get('system.bf16')) else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    print(f"Model loaded on device: {model.device}")

    model = model.to(torch.bfloat16 if config.get('system.bf16') else torch.float32)
    print(f"Model dtype: {next(model.parameters()).dtype}")


    # Add a pad token to the tokenizer if it doesn't have one
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        # model.resize_token_embeddings(len(tokenizer))

    print(f"Model '{model_name}' loaded successfully for QAT.")
    print(f"Model device: {model.device}")
    print(f"Number of parameters: {model.num_parameters()}")

    # print("\n--- Applying quantization to model ---")

    # apply_quantization_to_model(model)
    # 3. Prepare the model for LoRA fine-tuning

    # LORA_RANK = 32
    print("\n--- Applying LoRA Adapters to quantized model ---")
    lora_config = LoraConfig(
    r=config.get('lora.rank'),
    lora_alpha=config.get('lora.alpha'),
    target_modules=config.get('lora.target_modules'),
    lora_dropout=config.get('lora.dropout'),
    bias="none",
    task_type="CAUSAL_LM"
)


    

    # 4. Apply LoRA adapters to the quantized model
    # After creating peft_model but BEFORE quantization
    peft_model = get_peft_model(model, lora_config)

    target_dtype = torch.bfloat16 if bool(config.get('system.bf16')) else torch.float32
    peft_model = peft_model.to(target_dtype)

    # for name, module in peft_model.named_modules():
    #     print(name)

    # Freeze base model (this is already done by PEFT)
    for param in peft_model.base_model.parameters():
        param.requires_grad = False

    # Enable LoRA parameters (PEFT does this by default, but let's be explicit)
    for name, param in peft_model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True


    quantize_model(peft_model)

    target_dtype = torch.bfloat16 if bool(config.get('system.bf16')) else torch.float32
    peft_model = peft_model.to(target_dtype)
    # print("\n--- Converting to Fused Operations ---")
    # from fused_ops import convert_to_fused
    # peft_model = convert_to_fused(peft_model)
    # print("✓ Model ready for accelerated training\n")


    # if torch.__version__ >= "2.0":
    #     print("Compiling model with torch.compile...")
    #     peft_model = torch.compile(peft_model, mode="reduce-overhead")
    # for name, module in peft_model.named_modules():
    #     if isinstance(module, PEQATLoRAModule):
    #         print(name)
    # FREEZE ALL SMOOTH FACTORS AT 1.0
    # use_smooth = False
    if not calibrate_smooth:
        # print("\n=== ABLATION: Freezing SmoothQuant factors at 1.0 ===")
        print("\n=== INITIALIZING: Smooth factors at 1.0 ===")
        with torch.no_grad():
            for name, module in peft_model.named_modules():
                if isinstance(module, QATW4A4LinearCore):
                    module.smooth_factor.data.fill_(1.0)
                    module.smooth_factor.requires_grad = True
        print("✓ All smooth factors initialized at 1.0\n")

    if not learn_smooth:
        print("\n=== FREEZING all Smooth Factors ===")
        with torch.no_grad():
            for name, module in peft_model.named_modules():
                if isinstance(module, QATW4A4LinearCore):
                    module.smooth_factor.requires_grad = False
        print("✓ All smooth factors frozen\n")



    # print("\n--- LoRA Initialization Check ---")
    # for name, module in peft_model.named_modules():
    #     if isinstance(module, PEQATLoRAModule):
    #         adapter_name = module.active_adapter[0]
    #         lora_A = module.lora_A[adapter_name].weight
    #         lora_B = module.lora_B[adapter_name].weight
            
    #         print(f"{name}:")
    #         print(f"  lora_A: norm={lora_A.norm().item():.4f}, "
    #             f"mean={lora_A.mean().item():.6f}, "
    #             f"std={lora_A.std().item():.6f}")
    #         print(f"  lora_B: norm={lora_B.norm().item():.4f}, "
    #             f"mean={lora_B.mean().item():.6f}, "
    #             f"std={lora_B.std().item():.6f}")
    #         print(f"  scaling: {module.scaling[adapter_name]}")
            
    #         # Check if B is actually zero
    #         # if lora_B.abs().max().item() < 1e-6:
            #     print(f"  ⚠️  lora_B is zero-initialized (PEFT default)")


    # # FIX: Reinitialize lora_B to break the zero-gradient deadlock
    # print("\n--- Fixing LoRA Gradient Flow ---")
    # for name, module in peft_model.named_modules():
    #     if isinstance(module, PEQATLoRAModule):
    #         adapter_name = module.active_adapter[0]
    #         lora_B_module = module.lora_B[adapter_name]
            
    #         # Reinitialize lora_B with small non-zero values
    #         with torch.no_grad():
    #             # Use small random initialization instead of zeros
    #             nn.init.normal_(lora_B_module.weight, mean=0.0, std=0.001)
            
    #         print(f"Reinitialized {name}.lora_B[{adapter_name}]: "
    #             f"mean={lora_B_module.weight.mean():.6f}, "
    #             f"std={lora_B_module.weight.std():.6f}")

    # print("LoRA gradient flow fix applied\n")


    # After quantization, configure QAT parameters to be trainable
    for name, param in peft_model.named_parameters():
        lname = name.lower()
        is_lora = ("lora" in lname)
        is_qat = ("activation_clip" in lname) or ("smooth_factor" in lname)
        param.requires_grad = bool(is_lora or is_qat)

    print("\n=== PARAMETER FILTERING DEBUG ===")
    lora_count = 0
    qat_count = 0
    other_count = 0

    for name, param in peft_model.named_parameters():
        lname = name.lower()
        is_lora = ("lora" in lname)
        is_qat = ("activation_clip" in lname) or ("smooth_factor" in lname)
        
        # DEBUG: Print what's happening
        if is_lora:
            # print(f"LoRA FOUND: {name} -> {param.numel()} params")
            lora_count += param.numel()
        elif is_qat:
            if param.requires_grad:
                # print(f"QAT FOUND: {name} -> {param.numel()} params")
                qat_count += param.numel()
        else:
            # print(f"OTHER: {name} -> {param.numel()} params (will be frozen)")
            other_count += param.numel()
        
        param.requires_grad = bool(is_lora or is_qat)

    print(f"LoRA params found: {lora_count}")
    print(f"QAT params found: {qat_count}")
    print(f"Other params: {other_count}")
    print(f"Total trainable should be: {lora_count + qat_count}")

    # Sanity checks: assert only intended params are trainable
    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    unexpected = [n for n in trainable if ("lora" not in n.lower() and "activation_clip" not in n and "smooth_factor" not in n)]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable params found (should be only LoRA/activation_clip/smooth_factor): {unexpected[:8]}")
    print(f"Trainable parameter count: {len(trainable)}")

    # The final step will be setting up the training loop.

    # 6. Setup and Run Training Loop
    print("\n--- Setting up Training Loop for W4A4 QAT ---")
    
    # Create unique output directories
    # base_output_dir = "./cloud_qat_lora"
    # base_model_dir = "./qat_lora_model"
    # base_output_dir = "./cloud_output_qat_lora"
    # base_model_dir = "./cloud_qat_lora_model"
    
    # output_dir = find_available_dir(base_output_dir)
    # model_dir = find_available_dir(base_model_dir)

    output_dir = f"{config.get('training.output_dir')}_{config.run_id}"
    # output_dir = f"{config.get('training.output_dir')}/{config.run_id}"

    
    
    print(f"Using output directory: {output_dir}")
    # print(f"Model will be saved to: {model_dir}")


    # Load and preprocess the WikiText-2 dataset
    # datasets, _ = load_and_preprocess_wikitext2(model_name=model_name, block_size=2048) # Use _ for tokenizer as it\'s already loaded
    # Load with sliding window (matches eval protocol)
    # datasets, _ = load_wikitext2_sliding_window(
    #     model_name=model_name,
    #     seq_len=config.get('model.block_size'),     
    #     stride=config.get('model.block_size')
    # )

    # train_dataset = datasets["train"]
    # eval_dataset = datasets["validation"]

    dataset_name = config.get('dataset.name', 'wikitext')
    print(f"\nDataset selected: {dataset_name}")

    if dataset_name == "alpaca":
        # Load Alpaca for instruction tuning (MMLU preparation)
        from data_preparation import load_alpaca_instruct
        datasets, _ = load_alpaca_instruct(
            model_name=model_name,
            max_length=config.get('model.block_size', 2048)
        )
        
    elif dataset_name == "wikitext":
        # Load WikiText-2 for perplexity benchmarking (original setup)
        from data_preparation import load_wikitext2_sliding_window
        datasets, _ = load_wikitext2_sliding_window(
            model_name=model_name,
            seq_len=config.get('model.block_size'),     
            stride=config.get('model.block_size')
        )
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'wikitext' or 'alpaca'")

    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]

    print(f"✓ Train samples: {len(train_dataset):,}")
    print(f"✓ Eval samples:  {len(eval_dataset):,}")

    
    # Choose memory-friendly settings based on device
    is_cuda = torch.cuda.is_available() and not torch.backends.mps.is_available()
    # per_device_bs = 16 if is_cuda else 8
    # grad_accum = 2 if is_cuda else 8

    # training_args = TrainingArguments(
    #     output_dir= output_dir,
    #     overwrite_output_dir=False,
    #     num_train_epochs=NUM_EPOCHS,
    #     per_device_train_batch_size=BATCH_SIZE,
    #     per_device_eval_batch_size=BATCH_SIZE,
    #     gradient_accumulation_steps=GRAD_ACCUM,
    #     max_grad_norm=5.0,
    #     gradient_checkpointing=True,
    #     # gradient_checkpointing=False,
    #     gradient_checkpointing_kwargs={'use_reentrant': True},
    #     save_strategy="epoch",
    #     eval_strategy="epoch",
    #     logging_steps=10,
    #     learning_rate=LEARNING_RATE,
    #     weight_decay=0.01,
    #     warmup_ratio=0.3,
    #     lr_scheduler_type="cosine_with_min_lr",
    #     lr_scheduler_kwargs={"min_lr_rate": 0.1},
    #     # lr_scheduler_kwargs={"num_cycles": 2},
    #     # lr_scheduler_type="constant_with_warmup",
    #     bf16=True, # Enable bfloat16
    #     fp16=False,
    #     tf32=True, # Enable for Blackwell/Ampere GPUs 
    #     report_to="tensorboard",
    #     load_best_model_at_end=True,
    #     metric_for_best_model="eval_loss",
    #     dataloader_num_workers=16,
    #     dataloader_prefetch_factor=4,
    #     dataloader_pin_memory=True
    # )

    config.save_to_output(output_dir)
    torch.cuda.is_bf16_supported = lambda: True

    training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=bool(config.get('training.overwrite_output_dir', False)),
    num_train_epochs=config.get('training.num_epochs'),
    per_device_train_batch_size=config.get('training.batch_size'),
    per_device_eval_batch_size=config.get('training.batch_size'),
    gradient_accumulation_steps=config.get('training.gradient_accumulation_steps'),
    max_grad_norm=float(config.get('training.max_grad_norm')),
    gradient_checkpointing=bool(config.get('training.gradient_checkpointing', True)),
    gradient_checkpointing_kwargs={'use_reentrant': config.get('training.use_reentrant', False)},
    save_strategy=config.get('training.save_strategy'),
    eval_strategy=config.get('training.eval_strategy'),
    logging_steps=config.get('training.logging_steps'),
    learning_rate=float(config.get('training.learning_rate')),
    warmup_ratio=float(config.get('training.warmup_ratio')),
    lr_scheduler_type=config.get('training.lr_scheduler_type'),
    # lr_scheduler_kwargs={'min_lr_rate': float(config.get('training.min_lr_rate')), "num_cycles": int(config.get('training.num_cycles'))},
    # lr_scheduler_kwargs={'min_lr_rate': float(config.get('training.min_lr_rate'))},
    lr_scheduler_kwargs={"num_cycles": int(config.get('training.num_cycles'))},
    bf16=bool(config.get('system.bf16', True)),
    fp16=bool(config.get('system.fp16', False)),
    tf32=bool(config.get('system.tf32', True)),
    report_to="tensorboard",
    load_best_model_at_end=bool(config.get('training.load_best_model_at_end', True)),
    metric_for_best_model=config.get('training.metric_for_best_model', 'eval_loss'),
    dataloader_num_workers=config.get('system.dataloader_num_workers'),
    dataloader_prefetch_factor=config.get('system.dataloader_prefetch_factor', 4),
    dataloader_pin_memory=bool(config.get('system.dataloader_pin_memory', True)),
)



    

    # DEBUG: Print what TrainingArguments actually got
    print("\n" + "="*60)
    print("TRAINING ARGS DIAGNOSTIC")
    print("="*60)
    print(f"training_args.bf16 = {training_args.bf16}")
    print(f"training_args.fp16 = {training_args.fp16}")
    print(f"training_args.fp16_full_eval = {training_args.fp16_full_eval}")
    print(f"config.get('system.bf16') = {config.get('system.bf16')}")
    print(f"type(config.get('system.bf16')) = {type(config.get('system.bf16'))}")
    print("="*60 + "\n")

    # If bf16 is False but should be True, something is wrong
    if not training_args.bf16 and config.get('system.bf16'):
        raise RuntimeError("❌ bf16 not enabled in TrainingArguments despite config!")

    # Define Data Collator
    data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False, 
    pad_to_multiple_of=8 
)

    # Percentile-based calibration of activation clips (short pass)
    print("\n--- Calibrating activation clips (percentile init) ---")

    # for name, module in peft_model.named_modules():
    #     if isinstance(module, PEQATLoRAModule):
            # print(name)
        # print(name)

    peft_model = peft_model.to(device)
    peft_model = ensure_dtype_consistency(peft_model, torch.bfloat16)

    calib_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=data_collator)
    # per_module_abs = calibrate_activation_clips(peft_model, calib_loader, steps=200, percentile=99.5, device=device)
    # # Initialize SmoothQuant scaling factors using activation percentiles and weight stats
    # calibrate_smooth_factors(peft_model, per_module_abs, alpha=0.5)
    import gc
    gc.collect()
    unified_calibration(peft_model, calib_loader, steps=config.get('calibration.steps'), percentile=config.get('calibration.percentile'), alpha=config.get('calibration.alpha'), device=device, use_smooth=config.get('calibration.calibrate_smooth'))  # TO INIT SMOOTH FACTORS AT 1.0
    torch.cuda.empty_cache()



    # Build optimizer with parameter groups: LoRA vs PACT clip vs SmoothQuant factors
    lora_params: List[torch.nn.Parameter] = []
    clip_params: List[torch.nn.Parameter] = []
    smooth_params: List[torch.nn.Parameter] = []

    for name, param in peft_model.named_parameters():
        name_lower = name.lower()
        
        # Explicitly check for LoRA patterns and force enable
        is_lora = any(pattern in name for pattern in [
            'lora_A.default.weight', 
            'lora_B.default.weight'
        ])
        
        # Check for QAT patterns
        is_qat = any(pattern in name for pattern in [
            'activation_clip', 
            'smooth_factor'
        ])
        
        if is_lora:
            param.requires_grad_(True)  # Force enable
            lora_params.append(param)
            # print(f"Added LoRA param: {name} ({param.numel()} params)")
        elif is_qat:
            param.requires_grad_(True)  # Force enable  
            if "activation_clip" in name:
                clip_params.append(param)
                # print(f"Added clip param: {name} ({param.numel()} params)")
            elif "smooth_factor" in name:
                smooth_params.append(param)
                # print(f"Added smooth param: {name} ({param.numel()} params)")

    print(f"Optimizer setup: {len(lora_params)} LoRA, {len(clip_params)} clip, {len(smooth_params)} smooth params")


    # base_lr = config.get('training.learning_rate')
    # param_groups = []
    # # Set learning rates: LoRA uses the TrainingArguments LR; quant params use a higher LR
    # lora_lr = training_args.learning_rate 
    # clip_lr = training_args.learning_rate * 5
    # smooth_lr = training_args.learning_rate * 3 


    # param_groups = []
    # if lora_params:
    #     param_groups.append({
    #         "params": lora_params, 
    #         "lr": lora_lr, 
    #         "weight_decay": 0.01  # Small L2 regularization for LoRA
    #     })
    # if clip_params:
    #     param_groups.append({
    #         "params": clip_params, 
    #         "lr": clip_lr, 
    #         "weight_decay": 0.0  # No weight decay on clip values
    #     })
    # if smooth_params and learn_smooth:
    #     param_groups.append({
    #         "params": smooth_params, 
    #         "lr": smooth_lr, 
    #         "weight_decay": 0.0  # No weight decay on smooth factors
    #     })

    # Optimizer setup
    base_lr = float(config.get('training.learning_rate')) 
    optimizer = torch.optim.AdamW([
        {
            'params': lora_params, 
            'lr': base_lr * config.get('training.optimizer.lora_lr_multiplier'),
            'weight_decay': config.get('training.optimizer.weight_decay')
        },
        {
            'params': clip_params, 
            'lr': base_lr * config.get('training.optimizer.clip_lr_multiplier'),
            'weight_decay': 0.0
        },
        {
            'params': smooth_params, 
            'lr': base_lr * config.get('training.optimizer.smooth_lr_multiplier'),
            'weight_decay': 0.0
        }
    ], 
    betas=(config.get('training.optimizer.adam_beta1'), 
        config.get('training.optimizer.adam_beta2')),
    eps=float(config.get('training.optimizer.adam_eps')))


    # # Fallback: if for some reason groups are empty, include all trainable params
    # if not param_groups:
    #     all_trainable = [p for p in peft_model.parameters() if p.requires_grad]
    #     param_groups = [{"params": all_trainable, "lr": lora_lr, "weight_decay": 0.0}]

    # optimizer = torch.optim.AdamW(
    # param_groups, 
    # betas=(0.9, 0.95),  
    # eps=1e-8,
    # fused=True)

    
    
    config.save_to_output(output_dir)

    print(f"\n--- Starting W4A4 QAT Training (Run ID: {config.run_id}) ---\n")

    # Initialize Trainer with custom optimizer (Trainer will create scheduler)
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )
    

    trainer.add_callback(SaveQuantParamsCallback())
    trainer.add_callback(GradientMonitorCallback(
    log_every_n_steps=10,
    # clip_lora=5.0,   # LoRA can have larger gradients
    # clip_qat=10.0     # QAT should have smaller gradients
    ))

    
    # Ensure model on device and enable input grads for all devices to allow LoRA/SQ to learn
    # peft_model = peft_model.to(device)
    peft_model.enable_input_require_grads()
    # Disable KV cache to avoid conflicts with potential checkpointing and ensure full backprop
    if hasattr(peft_model, 'config'):
        peft_model.config.use_cache = False

    # print("\n=== Pre-Training Gradient Check (Detailed) ===")
    # peft_model.train()
    # dummy_batch = next(iter(trainer.get_train_dataloader()))
    # dummy_batch = {k: v.to(device) for k, v in dummy_batch.items()}

    # # First forward-backward pass
    # outputs = peft_model(**dummy_batch)
    # loss = outputs.loss
    # loss.backward()

    # print("\nLoRA A matrices:")
    # for name, param in peft_model.named_parameters():
    #     if 'lora_A' in name and param.grad is not None:
    #         print(f"  {name}: norm={param.grad.norm().item():.6f}, "
    #             f"max={param.grad.abs().max().item():.6f}")

    # print("\nLoRA B matrices:")
    # for name, param in peft_model.named_parameters():
    #     if 'lora_B' in name and param.grad is not None:
    #         print(f"  {name}: norm={param.grad.norm().item():.6f}, "
    #             f"max={param.grad.abs().max().item():.6f}")

    # peft_model.zero_grad()
    # print("=" * 50)


    # 7. Train the model (fresh run). Checkpoints will save quant params via callback.
    print("\n--- Starting W4A4 QAT with LoRA Fine-tuning ---")
    trainer.train()
    import json
    with open('training_log.json', 'w') as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print("Loss curve data saved to training_log.json")

    # 8. Evaluate and Save
    print("\n--- Evaluating W4A4 QAT LoRA Model ---")
    eval_results = trainer.evaluate()
    trainer_loss = eval_results['eval_loss']
    trainer_ppl = torch.exp(torch.tensor(trainer_loss))

    print(f"Trainer Loss: {trainer_loss:.4f}")
    print(f"Trainer Perplexity: {trainer_ppl:.2f}")

    # Now evaluate with YOUR evaluation class on the SAME split
    # print("\n--- Cross-checking with custom Evaluation class ---")
    # from datasets import load_dataset
    # val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    # custom_eval = Evaluation(tokenizer, val_dataset, device, num_samples=None, block_size=2048)
    # custom_ppl = custom_eval.evaluate(peft_model)
        # Now evaluate with proper sliding window on TEST set

    # Save the model, tokenizer, and quantization parameters
    
    print(f"\n--- Saving W4A4 QAT LoRA model to {output_dir} ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save quantization parameters after training
    # save_quantization_parameters(peft_model, model_dir)
    
    # Save the LoRA adapters
    print(f"--- Saving W4A4 QAT LoRA adapters to {output_dir} ---")
    peft_model.save_pretrained(output_dir)

    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)
    
    best_checkpoint = trainer.state.best_model_checkpoint
    if best_checkpoint and os.path.exists(best_checkpoint):
        import shutil
        src = os.path.join(best_checkpoint, "quantization_params.pth")
        dst = os.path.join(output_dir, "quantization_params.pth")
        
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"✓ Copied quantization params from: {best_checkpoint}")
            
            # Verify the copy
            quant_params = torch.load(dst)
            print(f"  {len(quant_params)} quantization parameters")
        else:
            print(f"⚠️  Warning: quantization_params.pth not found in best checkpoint")
            print(f"  Falling back to current model state...")
            
            # Fallback: save current state
            quant_params = {}
            for name, module in peft_model.named_modules():
                if isinstance(module, QATW4A4LinearCore):
                    prefix = f"{name}."
                    quant_params[f"{prefix}activation_clip"] = module.activation_clip.data.detach().cpu()
                    quant_params[f"{prefix}smooth_factor"] = module.smooth_factor.data.detach().cpu()
            
            torch.save(quant_params, dst)
            print(f"  ✓ Saved {len(quant_params)} parameters from current state")
    else:
        print("⚠️  No best checkpoint found, saving current model state...")
        
        # Save current quantization state
        quant_params = {}
        for name, module in peft_model.named_modules():
            if isinstance(module, QATW4A4LinearCore):
                prefix = f"{name}."
                quant_params[f"{prefix}activation_clip"] = module.activation_clip.data.detach().cpu()
                quant_params[f"{prefix}smooth_factor"] = module.smooth_factor.data.detach().cpu()
        
        torch.save(quant_params, os.path.join(output_dir, "quantization_params.pth"))
        print(f"✓ Saved {len(quant_params)} quantization parameters")


    # Save the model config
    peft_model.config.to_json_file(os.path.join(output_dir, "config.json"))
    
    
    print(f"\n--- Model, tokenizer, and quantization config saved to {output_dir} ---")
    print("To load this model for inference, use:")
    print(f"model = AutoModelForCausalLM.from_pretrained('{model_name}').to('{device}')")
    print(f"model = PeftModel.from_pretrained(model, '{output_dir}')")
    print("# Make sure to apply the same quantization setup when loading")

    custom_eval(peft_model, tokenizer, device, trainer_ppl)
    
    # save_model()