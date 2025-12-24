import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback
from peft import PeftModel
from torch.utils.data import DataLoader
from typing import List
from fake_quant import (
    SaveQuantParamsCallback,
    find_available_dir,
)
from data_preparation import load_and_preprocess_wikitext2


if __name__ == "__main__":
    # === 1. Setup ===
    base_model_name = "meta-llama/Llama-2-7b-hf"
    resume_from_dir = "./output_qat_lora_12"  # ← YOUR crashed run's folder
    
    # Find the latest checkpoint
    checkpoints = glob.glob(os.path.join(resume_from_dir, "checkpoint-*"))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {resume_from_dir}!")
    
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"\n{'='*60}")
    print(f"Found checkpoint: {latest_checkpoint}")
    print(f"{'='*60}\n")
    
    # Verify checkpoint integrity
    required_files = ["adapter_config.json", "optimizer.pt", "trainer_state.json", "adapter_model.safetensors"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(latest_checkpoint, f))]
    if missing:
        raise ValueError(f"Checkpoint missing files: {missing}")
    print("✓ Checkpoint integrity verified\n")
    

    # === 2. Device setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # === 3. Load BASE model first (with quantization) ===
    print("Loading base model...")
    from fake_quant import apply_qat_to_model
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Apply quantization wrappers
    print("Applying quantization wrappers...")
    base_model = apply_qat_to_model(
        base_model,
        w_bits=4,
        a_bits=4,
        calibrate_smooth=True,
        lora_alpha=16,
        lora_rank=32,
    )
    
    # NOW load PEFT adapters from checkpoint
    print(f"Loading LoRA adapters from {latest_checkpoint}...")
    model = PeftModel.from_pretrained(
        base_model,
        latest_checkpoint,
        is_trainable=True,
    )
    
    # Load quantization params
    quant_params_path = os.path.join(latest_checkpoint, "quantization_params.pth")
    if os.path.exists(quant_params_path):
        print("Loading quantization parameters...")
        quant_state = torch.load(quant_params_path, map_location=device)
        model.load_state_dict(quant_state, strict=False)
        print("✓ Quantization params loaded")
    else:
        print("⚠ Warning: No quantization_params.pth found")
    
    model = model.to(device)
    

    # === 4. Tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.resize_token_embeddings(len(tokenizer))


    # === 5. Sanity check ===
    print(f"\n{'='*60}")
    lora_params, clip_params, smooth_params = 0, 0, 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora' in name.lower():
            lora_params += 1
        elif 'activation_clip' in name:
            clip_params += 1
        elif 'smooth_factor' in name:
            smooth_params += 1
    
    print(f"Trainable parameter counts:")
    print(f"  LoRA params: {lora_params}")
    print(f"  Clip params: {clip_params}")
    print(f"  Smooth params: {smooth_params}")
    print(f"{'='*60}\n")
    
    if lora_params == 0:
        raise ValueError("ERROR: No LoRA parameters found! Check model loading.")


    # === 6. Dataset ===
    datasets, _ = load_and_preprocess_wikitext2(model_name=base_model_name, block_size=2048)
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    # === 7. Training arguments (MUST MATCH ORIGINAL) ===
    training_args = TrainingArguments(
        output_dir=resume_from_dir,  # Same dir
        overwrite_output_dir=False,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': True},
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=10,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        tf32=True,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=16,
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
        logging_dir=os.path.join(resume_from_dir, "logs"),
    )


    # === 8. DON'T create custom optimizer - let Trainer load it ===
    # When resume_from_checkpoint is set, Trainer will:
    # 1. Create optimizer based on model params
    # 2. Load optimizer.pt from checkpoint
    # 3. Restore exact optimizer state
    
    print("Note: Optimizer will be loaded from checkpoint by Trainer\n")


    # === 9. Trainer (NO custom optimizer) ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # NO optimizers=(optimizer, None) here!
    )

    trainer.add_callback(SaveQuantParamsCallback())
    
    # Gradient monitor
    class DetailedGradientMonitorCallback(TrainerCallback):
        def __init__(self, log_every_n_steps=10):
            self.log_every_n_steps = log_every_n_steps
        
        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step % self.log_every_n_steps != 0:
                return
            
            unwrapped_model = model
            while hasattr(unwrapped_model, 'module'):
                unwrapped_model = unwrapped_model.module
            
            lora_grad = clip_grad = smooth_grad = 0.0
            
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
            
            print(f"Step {state.global_step:4d} | LoRA: {lora_grad**0.5:8.4f} | Clips: {clip_grad**0.5:8.4f} | Smooth: {smooth_grad**0.5:8.4f}")
    
    trainer.add_callback(DetailedGradientMonitorCallback(log_every_n_steps=10))


    # === 10. Resume ===
    model.enable_input_require_grads()
    if hasattr(model, 'config'):
        model.config.use_cache = False
    
    print("\n--- Resuming training ---")
    trainer.train(resume_from_checkpoint=latest_checkpoint)


    # === 11. Final eval ===
    print("\n--- Final evaluation ---")
    eval_results = trainer.evaluate()
    loss = eval_results["eval_loss"]
    ppl = torch.exp(torch.tensor(loss))
    print(f"Eval Loss: {loss:.4f}, Perplexity: {ppl:.2f}")


    # === 12. Save final ===
    final_dir = find_available_dir("./qat_lora_final")
    print(f"\nSaving to {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # Copy quant params from best checkpoint
    if trainer.state.best_model_checkpoint:
        import shutil
        src = os.path.join(trainer.state.best_model_checkpoint, "quantization_params.pth")
        dst = os.path.join(final_dir, "quantization_params.pth")
        if os.path.exists(src):
            shutil.copy(src, dst)
    
    print("\n✓ Done!")
