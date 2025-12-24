import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import DataLoader
from typing import List
from fake_quant import (
    PEQATLoRAModule,
    quantize_model,
    find_available_dir,
    SaveQuantParamsCallback,
    GradientMonitorCallback,
    load_peqat_model,
    QATW4A4LinearCore,
)
from data_preparation import load_and_preprocess_wikitext2
from calibrations import unified_calibration


if __name__ == "__main__":
    # === 1. Setup ===
    base_model_name = "meta-llama/Llama-2-7b-hf"  # ← FIXED: Was "gpt2"
    resume_from_dir = "./output_qat_lora_12"  # ← CHANGE THIS to your crashed run's output dir
    
    # Find the latest checkpoint in that directory
    checkpoints = glob.glob(os.path.join(resume_from_dir, "checkpoint-*"))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {resume_from_dir}!")
    
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"\n{'='*60}")
    print(f"Found latest checkpoint: {latest_checkpoint}")
    print(f"Resuming training from this checkpoint...")
    print(f"{'='*60}\n")
    
    # Continue saving to the SAME output directory (not a new one)
    new_output_dir = resume_from_dir  # ← FIXED: Keep same dir so epochs continue correctly
    

    # === 2. Device setup ===
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")


    # === 3. Load tokenizer and model from checkpoint ===
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading PEFT+QAT model from checkpoint...")
    model, _ = load_peqat_model(latest_checkpoint, base_model_name, device)
    model.resize_token_embeddings(len(tokenizer))
    
    # Ensure model is in correct dtype
    model = model.to(torch.bfloat16)
    print(f"Model dtype: {next(model.parameters()).dtype}")


    # === 4. Sanity check ===
    print(f"\nModel parameters (trainable):")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,}")
    print(f"  Total: {n_total:,}")


    # === 5. Reload dataset ===
    datasets, _ = load_and_preprocess_wikitext2(model_name=base_model_name, block_size=2048)
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    # === 6. Training arguments (MATCH YOUR ORIGINAL RUN) ===
    BATCH_SIZE = 32  # ← MATCH what you used originally
    GRAD_ACCUM = 1
    LEARNING_RATE = 1e-4  # ← MATCH what you used originally
    
    training_args = TrainingArguments(
        output_dir=new_output_dir,
        overwrite_output_dir=False,
        num_train_epochs=10,  # Total epochs (Trainer will continue from where it left off)
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_grad_norm=1.0,
        gradient_checkpointing=True,  # ← MATCH original
        gradient_checkpointing_kwargs={'use_reentrant': True},
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,  # ← MATCH original
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        tf32=True,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=16,  # ← MATCH original
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
        logging_dir=os.path.join(new_output_dir, "logs"),
    )


    # === 7. Optimizer setup (MATCH YOUR ORIGINAL MULTIPLIERS) ===
    lora_params, clip_params, smooth_params = [], [], []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        lname = name.lower()
        if "lora" in lname:
            lora_params.append(param)
        elif "activation_clip" in lname:
            clip_params.append(param)
        elif "smooth_factor" in lname:
            smooth_params.append(param)

    print(f"\nOptimizer groups:")
    print(f"  LoRA params: {len(lora_params)}")
    print(f"  Clip params: {len(clip_params)}")
    print(f"  Smooth params: {len(smooth_params)}")

    lora_lr = training_args.learning_rate       # 1e-4
    qat_lr = training_args.learning_rate * 10   # 1e-3
    smooth_lr = training_args.learning_rate * 5 # 5e-4  ← YOUR WORKING CONFIG

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": lora_lr, "weight_decay": 0.01},
            {"params": clip_params, "lr": qat_lr, "weight_decay": 0.0},
            {"params": smooth_params, "lr": smooth_lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )


    # === 8. Trainer setup ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )

    trainer.add_callback(SaveQuantParamsCallback())
    
    # Add the fixed gradient monitor
    class DetailedGradientMonitorCallback(TrainerCallback):
        def __init__(self, log_every_n_steps=10):
            self.log_every_n_steps = log_every_n_steps
        
        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step % self.log_every_n_steps != 0:
                return
            
            # Unwrap model
            unwrapped_model = model
            while hasattr(unwrapped_model, 'module'):
                unwrapped_model = unwrapped_model.module
            
            lora_grad = 0.0
            clip_grad = 0.0
            smooth_grad = 0.0
            
            for name, param in unwrapped_model.named_parameters():
                if param.grad is None or not param.requires_grad:
                    continue
                
                grad_sq = (param.grad.norm() ** 2).item()
                
                if 'lora' in name.lower():
                    lora_grad += grad_sq
                elif 'activation_clip' in name:
                    clip_grad += grad_sq
                elif 'smooth_factor' in name:
                    smooth_grad += grad_sq
            
            print(f"\n{'='*60}")
            print(f"Step {state.global_step} Grad Norms:")
            print(f"  LoRA: {lora_grad**0.5:8.4f}")
            print(f"  Clips: {clip_grad**0.5:8.4f}")
            print(f"  Smooth: {smooth_grad**0.5:8.4f}")
            print(f"{'='*60}\n")
    
    trainer.add_callback(DetailedGradientMonitorCallback(log_every_n_steps=10))


    # === 9. Resume training ===
    model.enable_input_require_grads()
    if hasattr(model, 'config'):
        model.config.use_cache = False
    
    print("\n--- Resuming W4A4 QAT with LoRA Fine-tuning ---")
    trainer.train(resume_from_checkpoint=latest_checkpoint)  # ← FIXED: Pass actual checkpoint


    # === 10. Final evaluation ===
    print("\n--- Evaluating resumed model ---")
    eval_results = trainer.evaluate()
    loss = eval_results["eval_loss"]
    ppl = torch.exp(torch.tensor(loss))
    print(f"Final Eval Loss: {loss:.4f}, Perplexity: {ppl:.2f}")


    # === 11. Save final model ===
    final_model_dir = find_available_dir("./qat_lora_model")
    print(f"\nSaving final model to {final_model_dir}")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # Copy quantization params from best checkpoint
    best_checkpoint = trainer.state.best_model_checkpoint
    if best_checkpoint and os.path.exists(best_checkpoint):
        import shutil
        src = os.path.join(best_checkpoint, "quantization_params.pth")
        dst = os.path.join(final_model_dir, "quantization_params.pth")
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"✓ Copied quantization params from best checkpoint")
    
    print("\n✓ Training resumed and completed successfully!")
