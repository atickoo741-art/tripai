import os
import json
from data_preparation import load_and_preprocess_wikitext2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from calibrations import unified_calibration
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from typing import List
from fake_quant import quantize_model, PEQATLoRAModule, find_available_dir, SaveQuantParamsCallback, save_quantization_parameters, Evaluation, GradientMonitorCallback, QATW4A4LinearCore
import torch.nn as nn
import multiprocessing
import triton
from fake_quant import get_dtype




def get_data_loader():
    if torch.cuda.is_available():
        optimal_workers = min(multiprocessing.cpu_count() // 2, 8)
        pin_memory = True
    else:
        optimal_workers = min(multiprocessing.cpu_count() // 4, 4)
        pin_memory = False
    return optimal_workers, pin_memory


if __name__ == "__main__":

    # 1. Load the GPT-2 small model and tokenizer
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set up device
    print(torch.cuda.is_available())
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.set_default_dtype(torch.float32)  # MPS works best with float32
        print("Using MPS device (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_dtype(torch.float32)  # Change from bfloat16 to float32
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        torch.set_default_dtype(torch.float32)
        print("Using CPU")

    use_bf16,use_fp16,use_tf32 = get_dtype()
    
    if use_bf16:
        print("✓ Using BFloat16 mixed precision training")
    elif use_fp16:
        print("✓ Using Float16 mixed precision training")
    else:
        print("✓ Using Float32 training")

    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype= torch.float32, 
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":  # For MPS and CPU, explicitly move to device
        model = model.to(device)

    # Add a pad token to the tokenizer if it doesn't have one
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    print(f"Model '{model_name}' loaded successfully for QAT.")
    print(f"Model device: {model.device}")
    print(f"Number of parameters: {model.num_parameters()}")

    # print("\n--- Applying quantization to model ---")

    # apply_quantization_to_model(model)
    # 3. Prepare the model for LoRA fine-tuning
    print("\n--- Applying LoRA Adapters to quantized model ---")
    lora_config = LoraConfig(
        r=32,  # Rank of the update matrices (lower = less parameters)
        lora_alpha=64,  # Alpha parameter for LoRA scaling
        target_modules=["c_attn", "c_proj", "c_fc"],  # Target attention and feed-forward layers
        lora_dropout=0.05,
        bias="none",  # Don't train bias parameters
        task_type=TaskType.CAUSAL_LM,
    )


    

    # 4. Apply LoRA adapters to the quantized model
    # After creating peft_model but BEFORE quantization
    peft_model = get_peft_model(model, lora_config)

    for name, module in peft_model.named_modules():
        print(name)

    # Freeze base model (this is already done by PEFT)
    for param in peft_model.base_model.parameters():
        param.requires_grad = False

    # Enable LoRA parameters (PEFT does this by default, but let's be explicit)
    for name, param in peft_model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True


    quantize_model(peft_model)
    # for name, module in peft_model.named_modules():
    #     if isinstance(module, PEQATLoRAModule):
    #         print(name)
    # FREEZE ALL SMOOTH FACTORS AT 1.0
    use_smooth = False
    if not use_smooth:
        print("\n=== ABLATION: Freezing SmoothQuant factors at 1.0 ===")
        with torch.no_grad():
            for name, module in peft_model.named_modules():
                if isinstance(module, QATW4A4LinearCore):
                    module.smooth_factor.data.fill_(1.0)
                    module.smooth_factor.requires_grad = False
        print("✓ All smooth factors frozen\n")


    print("\n--- LoRA Initialization Check ---")
    for name, module in peft_model.named_modules():
        if isinstance(module, PEQATLoRAModule):
            adapter_name = module.active_adapter[0]
            lora_A = module.lora_A[adapter_name].weight
            lora_B = module.lora_B[adapter_name].weight
            
            print(f"{name}:")
            print(f"  lora_A: norm={lora_A.norm().item():.4f}, "
                f"mean={lora_A.mean().item():.6f}, "
                f"std={lora_A.std().item():.6f}")
            print(f"  lora_B: norm={lora_B.norm().item():.4f}, "
                f"mean={lora_B.mean().item():.6f}, "
                f"std={lora_B.std().item():.6f}")
            print(f"  scaling: {module.scaling[adapter_name]}")
            
            # Check if B is actually zero
            if lora_B.abs().max().item() < 1e-6:
                print(f"  ⚠️  lora_B is zero-initialized (PEFT default)")


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
            print(f"LoRA FOUND: {name} -> {param.numel()} params")
            lora_count += param.numel()
        elif is_qat:
            if param.requires_grad:
                print(f"QAT FOUND: {name} -> {param.numel()} params")
                qat_count += param.numel()
        else:
            print(f"OTHER: {name} -> {param.numel()} params (will be frozen)")
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
    base_output_dir = "./output_qat_lora"
    base_model_dir = "./qat_lora_model"
    
    output_dir = find_available_dir(base_output_dir)
    model_dir = find_available_dir(base_model_dir)
    
    print(f"Using output directory: {output_dir}")
    print(f"Model will be saved to: {model_dir}")

    # Load and preprocess the WikiText-2 dataset
    datasets, _ = load_and_preprocess_wikitext2(model_name="gpt2", block_size=1024) # Use _ for tokenizer as it\'s already loaded
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]

    # Set up Training Arguments
    
    # Choose memory-friendly settings based on device
    is_cuda = torch.cuda.is_available() and not torch.backends.mps.is_available()
    per_device_bs = 32 if is_cuda else 8
    grad_accum = 1 if is_cuda else 8

    
    optimal_workers, pin_memory = get_data_loader()

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        num_train_epochs=30,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs * 2,
        gradient_accumulation_steps=grad_accum,
        max_grad_norm=1.0,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={'use_reentrant': False},  # Required for PEFT,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_dir=os.path.join(output_dir, "tensorboard_logs"),
        logging_steps=10,
        learning_rate=2.5e-4,
        weight_decay=0.01,
        warmup_ratio=0.15,
        lr_scheduler_type="cosine",
        lr_scheduler_kwargs={"num_cycles": 3},
        fp16=use_fp16,
        bf16=use_bf16,
        tf32=use_tf32,
        report_to="tensorboard",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=optimal_workers,
        dataloader_pin_memory=pin_memory,
        dataloader_prefetch_factor=2,  # ✓ Prefetch batches
    )

    # Define Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Percentile-based calibration of activation clips (short pass)
    print("\n--- Calibrating activation clips (percentile init) ---")

    for name, module in peft_model.named_modules():
        if isinstance(module, PEQATLoRAModule):
            print(name)
        # print(name)

    peft_model = peft_model.to(device)
    calib_loader = DataLoader(train_dataset, batch_size=per_device_bs, shuffle=False, collate_fn=data_collator)
    # per_module_abs = calibrate_activation_clips(peft_model, calib_loader, steps=200, percentile=99.5, device=device)
    # # Initialize SmoothQuant scaling factors using activation percentiles and weight stats
    # calibrate_smooth_factors(peft_model, per_module_abs, alpha=0.5)
    unified_calibration(peft_model, calib_loader, steps=64, percentile=99.5, alpha=0.6, device=device, use_smooth = use_smooth)
    del calib_loader
    torch.cuda.empty_cache()
    print(f"✓ Memory after cleanup: {torch.cuda.memory_allocated()/1e9:.1f} GB")


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
            param.requires_grad_(True)
            lora_params.append(param)
            print(f"Added LoRA param: {name} ({param.numel()} params)")
    
        elif "activation_clip" in name:
            param.requires_grad_(True)
            clip_params.append(param)
            print(f"Added clip param: {name} ({param.numel()} params)")
        
        elif "smooth_factor" in name:
            # CHANGED: Only add if use_smooth is True
            if use_smooth:
                param.requires_grad_(True)
                smooth_params.append(param)
                print(f"Added smooth param: {name} ({param.numel()} params)")
            else:
                # Keep frozen
                param.requires_grad_(False)
                print(f"Skipped smooth param (frozen): {name}")

                
    print(f"Optimizer setup: {len(lora_params)} LoRA, {len(clip_params)} clip, {len(smooth_params)} smooth params")

    param_groups = []
    # Set learning rates: LoRA uses the TrainingArguments LR; quant params use a higher LR
    lora_lr = training_args.learning_rate 
    qat_lr = training_args.learning_rate * 10 


    param_groups = []
    if lora_params:
        param_groups.append({
            "params": lora_params, 
            "lr": lora_lr, 
            "weight_decay": 0.00  # Small L2 regularization for LoRA
        })
    if clip_params:
        param_groups.append({
            "params": clip_params, 
            "lr": qat_lr, 
            "weight_decay": 0.0  # No weight decay on clip values
        })
    if smooth_params:
        param_groups.append({
            "params": smooth_params, 
            "lr": qat_lr, 
            "weight_decay": 0.0  # No weight decay on smooth factors
        })


    # Fallback: if for some reason groups are empty, include all trainable params
    if not param_groups:
        all_trainable = [p for p in peft_model.parameters() if p.requires_grad]
        param_groups = [{"params": all_trainable, "lr": lora_lr, "weight_decay": 0.0}]

    optimizer = torch.optim.AdamW(
    param_groups, 
    betas=(0.9, 0.999),  
    eps=1e-8,
    fused=True)

    # Verification: Check smooth_factor is actually frozen
    print("\n=== Smooth Factor Status ===")
    for name, param in peft_model.named_parameters():
        if "smooth_factor" in name:
            print(f"{name}:")
            print(f"  requires_grad: {param.requires_grad}")
            print(f"  value: {param.data[0].item():.4f}")
            print(f"  in optimizer: {any(param is p for group in optimizer.param_groups for p in group['params'])}")

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
    peft_model = peft_model.to(device)
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
    
    # peft_model = torch.compile(peft_model, mode='reduce-overhead', fullgraph=False)
    if torch.cuda.is_available() and torch.__version__ >= "2.0":
        print("\n🚀 Compiling model...")
        try:
            peft_model = torch.compile(
                peft_model,
                mode='reduce-overhead',
                fullgraph=False,  # CRITICAL for PEFT!
                dynamic=True      # Handles variable seq length
            )
            print("✓ Compilation successful")
        except Exception as e:
            print(f"⚠️  Failed: {e}")
            # Training continues without it (slower but works)

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
    print("\n--- Cross-checking with custom Evaluation class ---")
    from datasets import load_dataset
    val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    # # After training completes
    # print("--- Evaluating W4A4 QAT + LoRA Model ---")

    # # ✓ Disable torch.compile for evaluation
    # if hasattr(peft_model, '_orig_mod'):
    #     peft_model = peft_model._orig_mod
    # print("✓ Unwrapped compiled model for evaluation")
    # custom_eval = Evaluation(tokenizer, val_dataset, device, num_samples=None, block_size=1024)
    # custom_ppl = custom_eval.evaluate(peft_model)

    # print(f"\nComparison:")
    # print(f"  Trainer perplexity: {trainer_ppl:.2f}")
    # print(f"  Custom eval perplexity: {custom_ppl:.2f}")
    # print(f"  Difference: {abs(trainer_ppl - custom_ppl):.2f}")


    
    
    # Save the model, tokenizer, and quantization parameters
    
    print(f"\n--- Saving W4A4 QAT LoRA model to {model_dir} ---")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save quantization parameters after training
    # save_quantization_parameters(peft_model, model_dir)
    
    # Save the LoRA adapters
    print(f"--- Saving W4A4 QAT LoRA adapters to {model_dir} ---")
    peft_model.save_pretrained(model_dir)

    # Save the tokenizer
    tokenizer.save_pretrained(model_dir)
    
    best_checkpoint = trainer.state.best_model_checkpoint
    if best_checkpoint and os.path.exists(best_checkpoint):
        import shutil
        src = os.path.join(best_checkpoint, "quantization_params.pth")
        dst = os.path.join(model_dir, "quantization_params.pth")
        
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
        
        torch.save(quant_params, os.path.join(model_dir, "quantization_params.pth"))
        print(f"✓ Saved {len(quant_params)} quantization parameters")


    # Save the model config
    peft_model.config.to_json_file(os.path.join(model_dir, "config.json"))
    
    
    print(f"\n--- Model, tokenizer, and quantization config saved to {model_dir} ---")
    print("To load this model for inference, use:")
    print(f"model = AutoModelForCausalLM.from_pretrained('{model_name}').to('{device}')")
    print(f"model = PeftModel.from_pretrained(model, '{model_dir}')")
    print("# Make sure to apply the same quantization setup when loading")