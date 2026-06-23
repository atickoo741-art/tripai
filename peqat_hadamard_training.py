"""
Hadamard PE-QAT Training Script.

Trains a causal LM with:
  - LoRA adapters on all target projections
  - Hadamard rotation of inputs and weights (diffuses outliers)
  - Per-token dynamic activation quantization (TC-compatible)
  - Per-row symmetric weight quantization

No learned clip, smooth, or per-channel calibration params.
The rotation H is fixed (Walsh-Hadamard), no learned rotation correction.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import json
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from typing import List

from fake_quant import (
    hadamard_quantize_model,
    HadamardPEQATLoRAModule,
    Evaluation,
    find_available_dir,
)
from data_preparation import load_wikitext2_sliding_window
from config_loader import Config


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
    test_dataset = load_dataset("wikitext/wikitext-2-raw-v1", split="test")
    proper_eval = Evaluation(test_dataset, tokenizer, device, num_samples=None)
    custom_ppl = proper_eval.evaluate(peft_model, seq_len=2048, stride=2048)
    print(f"\nTest Perplexity (Standard Protocol): {custom_ppl:.2f}")
    print(f"Target for FP16 Llama-2-7b: ~5.47")
    print(f"Target for Hadamard PE-QAT: ~6-7 (competitive with SpinQuant QAT)")
    print(f"\nComparison:")
    print(f"  Trainer perplexity: {trainer_ppl:.2f}")
    print(f"  Custom eval perplexity: {custom_ppl:.2f}")
    print(f"  Difference: {abs(trainer_ppl - custom_ppl):.2f}")


if __name__ == "__main__":
    config = Config("config_hadamard.yaml")
    print("=" * 60)
    print(f"Experiment: {config.get('experiment.name')}")
    print(f"Run ID: {config.run_id}")
    print(f"Notes: {config.get('experiment.notes')}")
    print("=" * 60)

    torch.manual_seed(config.get('system.seed'))
    torch.backends.cuda.matmul.allow_tf32 = config.get('system.tf32', True)

    model_name = config.get('model.name')
    target_dtype = torch.bfloat16 if bool(config.get('system.bf16')) else torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load tokenizer ---
    print(f"Loading {model_name} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Set pad_token to eos_token (ID: {tokenizer.pad_token_id})")

    # --- Load model ---
    print(f"Loading {model_name} model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=target_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = model.to(target_dtype)
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Number of parameters: {model.num_parameters():,}")

    # --- LoRA config ---
    print("\n--- Applying LoRA Adapters ---")
    lora_config = LoraConfig(
        r=config.get('lora.rank'),
        lora_alpha=config.get('lora.alpha'),
        target_modules=config.get('lora.target_modules'),
        lora_dropout=config.get('lora.dropout'),
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model = peft_model.to(target_dtype)

    # Freeze base model
    for param in peft_model.base_model.parameters():
        param.requires_grad = False

    # --- Apply Hadamard quantization ---
    print("\n--- Applying Hadamard PE-QAT Quantization ---")
    hadamard_quantize_model(peft_model)

    # Confirm no clip/smooth params exist
    for name, param in peft_model.named_parameters():
        if "activation_clip" in name or "smooth_factor" in name:
            raise RuntimeError(f"Unexpected clip/smooth param found: {name}")
    print("✓ No clip/smooth parameters — Hadamard rotation handles outlier diffusion")

    # Enable only LoRA params
    print("\n=== Parameter Setup ===")
    lora_count = 0
    for name, param in peft_model.named_parameters():
        lname = name.lower()
        is_lora = "lora" in lname
        param.requires_grad = bool(is_lora)
        if is_lora:
            lora_count += param.numel()
    print(f"LoRA trainable params: {lora_count:,}")

    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    unexpected = [n for n in trainable if "lora" not in n.lower()]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable params: {unexpected[:8]}")
    print(f"✓ Only LoRA params are trainable ({len(trainable)} tensors)")

    # --- Output dir ---
    output_dir = f"{config.get('training.output_dir')}_{config.run_id}"
    print(f"Using output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Dataset ---
    dataset_name = config.get('dataset.name', 'wikitext')
    print(f"\nDataset selected: {dataset_name}")

    if dataset_name == "wikitext":
        from data_preparation import load_wikitext2_sliding_window
        datasets, _ = load_wikitext2_sliding_window(
            model_name=model_name,
            seq_len=config.get('model.block_size'),
            stride=config.get('model.block_size'),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
    print(f"✓ Train samples: {len(train_dataset):,}")
    print(f"✓ Eval samples:  {len(eval_dataset):,}")

    config.save_to_output(output_dir)

    # --- Optimizer ---
    lora_params = []
    for name, param in peft_model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad_(True)
            lora_params.append(param)
    print(f"Optimizer: {len(lora_params)} LoRA param groups")

    base_lr = float(config.get('training.learning_rate'))
    optimizer = torch.optim.AdamW(
        [{'params': lora_params, 'lr': base_lr, 'weight_decay': config.get('training.optimizer.weight_decay')}],
        betas=(config.get('training.optimizer.adam_beta1'),
               config.get('training.optimizer.adam_beta2')),
        eps=float(config.get('training.optimizer.adam_eps')),
    )

    # --- Training arguments ---
    torch.cuda.is_bf16_supported = lambda: True

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get('training.num_epochs'),
        per_device_train_batch_size=config.get('training.batch_size'),
        per_device_eval_batch_size=config.get('training.batch_size'),
        gradient_accumulation_steps=config.get('training.gradient_accumulation_steps'),
        max_grad_norm=float(config.get('training.max_grad_norm')),
        gradient_checkpointing=bool(config.get('training.gradient_checkpointing', True)),
        gradient_checkpointing_kwargs={'use_reentrant': config.get('training.use_reentrant', False)},
        save_strategy=config.get('training.save_strategy'),
        eval_strategy=config.get('training.eval_strategy'),
        logging_steps=config.get('training.logging_steps', None),
        learning_rate=float(config.get('training.learning_rate')),
        warmup_ratio=float(config.get('training.warmup_ratio')),
        lr_scheduler_type=config.get('training.lr_scheduler_type'),
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

    # --- Data collator ---
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    # --- No calibration needed ---
    print("\n--- No calibration needed (H is fixed, no clip/smooth to calibrate) ---")

    # --- Move to device ---
    peft_model = peft_model.to(device)
    peft_model = ensure_dtype_consistency(peft_model, target_dtype)
    gc.collect()
    torch.cuda.empty_cache()

    # --- Trainer ---
    print(f"\n--- Starting Hadamard PE-QAT Training (Run ID: {config.run_id}) ---\n")

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )

    peft_model.enable_input_require_grads()
    if hasattr(peft_model, 'config'):
        peft_model.config.use_cache = False

    # Train
    print("\n--- Starting W4A4 Hadamard QAT with LoRA Fine-tuning ---")
    trainer.train()
    with open('training_log.json', 'w') as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print("Loss curve data saved to training_log.json")

    # Evaluate
    print("\n--- Evaluating Hadamard PE-QAT Model ---")
    eval_results = trainer.evaluate()
    trainer_loss = eval_results['eval_loss']
    trainer_ppl = torch.exp(torch.tensor(trainer_loss))

    print(f"Trainer Loss: {trainer_loss:.4f}")
    print(f"Trainer Perplexity: {trainer_ppl:.2f}")

    # Save
    print(f"\n--- Saving Hadamard PE-QAT model to {output_dir} ---")
    os.makedirs(output_dir, exist_ok=True)
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    peft_model.config.to_json_file(os.path.join(output_dir, "config.json"))

    print(f"\n--- Model and tokenizer saved to {output_dir} ---")
    print("No quantization_params.pth needed — no clip/smooth to save.")

    # Custom eval on test set
    custom_eval(peft_model, tokenizer, device, trainer_ppl)
