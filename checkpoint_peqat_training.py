import os
import glob
import json
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from typing import List

# ============================================================================
# MULTI-CORE OPTIMIZATION - ADD THIS AT THE TOP
# ============================================================================
import torch
torch.set_num_threads(16)  # Use all 16 cores for PyTorch ops
torch.set_num_interop_threads(16)  # Parallelism for operations

# Also set environment variables before any imports
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
# ============================================================================

from fake_quant import (
    PEQATLoRAModule, 
    find_available_dir, 
    SaveQuantParamsCallback, 
    GradientMonitorCallback, 
    QATW4A4LinearCore, 
    load_peqat_model
)
from data_preparation import load_and_preprocess_wikitext2


if __name__ == "__main__":
    resume_from_dir = "./output_qat_lora_12"
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    
    print("\n" + "="*60)
    print("CHECKPOINT RESUME SCRIPT (MEMORY + CPU OPTIMIZED)")
    print("="*60)

    # === FIND CHECKPOINT ===
    print("\n[1/14] Finding checkpoint...")
    
    checkpoints = glob.glob(os.path.join(resume_from_dir, "checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {resume_from_dir}")
    
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"✓ Checkpoint: {latest_checkpoint}")

    # === READ CHECKPOINT STATE ===
    print("\n[2/14] Reading checkpoint state...")
    
    trainer_state_path = os.path.join(latest_checkpoint, "trainer_state.json")
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)
    
    checkpoint_epoch = trainer_state.get('epoch', 0)
    checkpoint_step = trainer_state.get('global_step', 0)
    total_epochs = trainer_state.get('num_train_epochs', 10)
    original_batch_size = trainer_state.get('train_batch_size', 32)
    original_lr = trainer_state.get('learning_rate', 1e-4)
    
    remaining_epochs = total_epochs - checkpoint_epoch
    
    print(f"  Checkpoint: Epoch {checkpoint_epoch:.2f}, Step {checkpoint_step}")
    print(f"  Total epochs: {total_epochs}")
    print(f"  Remaining: {remaining_epochs:.2f} epochs")
    
    # === MEMORY OPTIMIZATION ===
    EFFECTIVE_BATCH = original_batch_size  # 32
    PER_DEVICE_BATCH = 32
    GRAD_ACCUM = EFFECTIVE_BATCH // PER_DEVICE_BATCH  # 2
    
    print(f"\n  OPTIMIZATION:")
    print(f"    Batch: {original_batch_size} → {PER_DEVICE_BATCH} × {GRAD_ACCUM} = {EFFECTIVE_BATCH}")
    print(f"    Memory: ~{85 * PER_DEVICE_BATCH / original_batch_size:.0f}GB (was 85GB)")
    print(f"    CPU threads: {torch.get_num_threads()}")
    
    LEARNING_RATE = original_lr
    BLOCK_SIZE = 2048
    use_smooth = True

    # === DEVICE ===
    print("\n[3/14] Setting up device...")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        print("✓ Using CUDA")
    else:
        device = torch.device("cpu")
        torch.set_default_dtype(torch.float32)
        print("✓ Using CPU")

    # === LOAD MODEL ===
    print("\n[4/14] Loading model from checkpoint...")
    
    peft_model, tokenizer = load_peqat_model(latest_checkpoint, MODEL_NAME, device)
    peft_model.train()
    
    for module in peft_model.modules():
        if hasattr(module, 'training'):
            module.training = True
    
    print("✓ Model loaded")

    # === CONFIGURE TRAINABLE PARAMS ===
    print("\n[5/14] Configuring trainable parameters...")
    
    for param in peft_model.parameters():
        param.requires_grad = False
    
    lora_count = 0
    qat_count = 0
    
    for name, param in peft_model.named_parameters():
        is_lora = ('lora_A' in name or 'lora_B' in name)
        is_qat = ('activation_clip' in name or 'smooth_factor' in name)
        
        if is_lora or is_qat:
            param.requires_grad = True
            if is_lora:
                lora_count += 1
            else:
                qat_count += 1
    
    print(f"  LoRA: {lora_count}, QAT: {qat_count}")

    # === BUILD OPTIMIZER ===
    print("\n[6/14] Building optimizer...")
    
    lora_params = []
    clip_params = []
    smooth_params = []
    
    for name, param in peft_model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
        elif 'activation_clip' in name:
            clip_params.append(param)
        elif 'smooth_factor' in name:
            smooth_params.append(param)
    
    lora_lr = LEARNING_RATE
    qat_lr = LEARNING_RATE * 10
    smooth_lr = LEARNING_RATE * 5
    
    param_groups = [
        {"params": lora_params, "lr": lora_lr, "weight_decay": 0.00},
        {"params": clip_params, "lr": qat_lr, "weight_decay": 0.0},
    ]
    
    if smooth_params and use_smooth:
        param_groups.append({"params": smooth_params, "lr": smooth_lr, "weight_decay": 0.0})
    
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8, fused=True)
    print("✓ Optimizer built")

    # === LOAD DATASET ===
    print("\n[7/14] Loading dataset...")
    
    datasets, _ = load_and_preprocess_wikitext2(model_name=MODEL_NAME, block_size=BLOCK_SIZE)
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
    
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # === TRAINING ARGS WITH MULTI-CORE DATA LOADING ===
    print("\n[8/14] Configuring training...")
    
    training_args = TrainingArguments(
        output_dir=resume_from_dir,
        overwrite_output_dir=False,
        num_train_epochs=total_epochs,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        tf32=True,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_pin_memory=True,  # Enable pinned memory for faster GPU transfer
        dataloader_num_workers=16,   # Use 16 workers (was 8)
        dataloader_prefetch_factor=4,  # Prefetch 4 batches per worker
        logging_dir=os.path.join(resume_from_dir, "logs"),
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    print(f"  Dataloader workers: {training_args.dataloader_num_workers}")
    print(f"  Prefetch factor: {training_args.dataloader_prefetch_factor}")
    print("✓ Config done")

    # === CREATE TRAINER ===
    print("\n[9/14] Creating trainer...")
    
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
    trainer.add_callback(GradientMonitorCallback(log_every_n_steps=10))
    print("✓ Trainer created")

    # === PREPARE ===
    print("\n[10/14] Preparing model...")
    
    peft_model.enable_input_require_grads()
    if hasattr(peft_model, 'config'):
        peft_model.config.use_cache = False
    
    print("✓ Ready")

    # === VERIFY ===
    print("\n[11/14] Pre-training check...")
    print(f"  Resuming from: Epoch {checkpoint_epoch:.2f}")
    print(f"  Will train to: Epoch {total_epochs}")
    print(f"  Batch: {PER_DEVICE_BATCH} × {GRAD_ACCUM} = {EFFECTIVE_BATCH}")
    print(f"  CPU utilization: 16 data workers + 20 PyTorch threads")

    # === TRAIN ===
    print("\n[12/14] RESUMING TRAINING...")
    print("="*60 + "\n")
    
    try:
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n✓ Training completed")

    # === EVALUATE ===
    print("\n[13/14] Final evaluation...")
    
    eval_results = trainer.evaluate()
    eval_loss = eval_results['eval_loss']
    ppl = torch.exp(torch.tensor(eval_loss))
    
    print(f"  Loss: {eval_loss:.4f}, PPL: {ppl:.2f}")

    # === SAVE ===
    print("\n[14/14] Saving final model...")
    
    model_dir = find_available_dir("./qat_lora_final")
    os.makedirs(model_dir, exist_ok=True)
    
    peft_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt:
        src = os.path.join(best_ckpt, "quantization_params.pth")
        if os.path.exists(src):
            import shutil
            dst = os.path.join(model_dir, "quantization_params.pth")
            shutil.copy(src, dst)
            print("  ✓ Copied quant params")
    else:
        quant_params = {}
        for name, module in peft_model.named_modules():
            if isinstance(module, QATW4A4LinearCore):
                quant_params[f"{name}.activation_clip"] = module.activation_clip.data.cpu()
                quant_params[f"{name}.smooth_factor"] = module.smooth_factor.data.cpu()
        torch.save(quant_params, os.path.join(model_dir, "quantization_params.pth"))
    
    print(f"\n✓ Model: {model_dir}")
    print("\n" + "="*60)
    print("✓ COMPLETE")
    print("="*60 + "\n")
