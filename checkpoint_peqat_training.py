import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
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
    base_model_name = "gpt2"
    resume_checkpoint = "./output_qat_lora_8"  # folder containing prior checkpoint
    new_output_dir = find_available_dir(resume_checkpoint)
    print(f"Resuming training from: {resume_checkpoint}")
    print(f"New checkpoints will be saved to: {new_output_dir}")

    # === 2. Device setup ===
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # === 3. Load tokenizer and model ===
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    print("Loading PEFT+QAT model from previous checkpoint...")
    model, tokenizer = load_peqat_model(resume_checkpoint, base_model_name, device)
    model.resize_token_embeddings(len(tokenizer))

    # === 4. Sanity print ===
    print(f"Model parameters (trainable):")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable:,}")

    # === 5. Reload dataset ===
    datasets, _ = load_and_preprocess_wikitext2(model_name=base_model_name)
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # === 6. Training arguments ===
    use_cuda = torch.cuda.is_available()
    per_device_bs = 128 if use_cuda else 8
    grad_accum = 1 if use_cuda else 8

    training_args = TrainingArguments(
        output_dir=new_output_dir,
        overwrite_output_dir=False,
        num_train_epochs=10,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=5e-5,      # Resume with lower LR for stability
        weight_decay=0.01,
        warmup_steps = 200,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=use_cuda,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_dir=os.path.join(new_output_dir, "tensorboard_logs"),
        logging_steps=20,
        report_to="tensorboard",
        remove_unused_columns=False,
        load_best_model_at_end=True,
    )

    # === 7. Optimizer setup (reuse LoRA + QAT logic) ===
    lora_params, clip_params, smooth_params = [], [], []
    for name, param in model.named_parameters():
        lname = name.lower()
        if "lora_" in lname:
            lora_params.append(param)
        elif "activation_clip" in lname:
            clip_params.append(param)
        elif "smooth_factor" in lname:
            smooth_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": training_args.learning_rate, "weight_decay": 0.01},
            {"params": clip_params, "lr": training_args.learning_rate * 2, "weight_decay": 0.0},
            {"params": smooth_params, "lr": training_args.learning_rate * 2, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=False,
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
    trainer.add_callback(GradientMonitorCallback(
        log_every_n_steps=10,
        clip_lora=1.0,
        clip_qat=0.5
    ))

    # === 9. Resume training ===
    print("\n--- Resuming W4A4 QAT LoRA fine-tuning ---")
    trainer.train(resume_from_checkpoint=None)

    # === 10. Final evaluation ===
    print("\n--- Evaluating resumed model ---")
    eval_results = trainer.evaluate()
    loss = eval_results["eval_loss"]
    ppl = torch.exp(torch.tensor(loss))
    print(f"Final Eval Loss: {loss:.4f}, Perplexity: {ppl:.2f}")

    # === 11. Save ===
    print(f"Saving continued checkpoint to {new_output_dir}")
    model.save_pretrained(new_output_dir)
    tokenizer.save_pretrained(new_output_dir)
