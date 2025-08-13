import os
import json
from data_preparation import load_and_preprocess_wikitext2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from calibrations import calibrate_activation_clips, calibrate_smooth_factors
from peft import LoraConfig, get_peft_model, TaskType
from fake_quant import apply_quantization_to_peft_model, QATW4A4Linear
from torch.utils.data import DataLoader
from typing import List
from utils import find_available_dir, SaveQuantParamsCallback, save_quantization_parameters



if __name__ == "__main__":

    # 1. Load the GPT-2 small model and tokenizer
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set up device
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

    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Change from bfloat16 to float32
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
        r=8,  # Rank of the update matrices (lower = less parameters)
        lora_alpha=32,  # Alpha parameter for LoRA scaling
        target_modules=["c_attn", "c_proj", "c_fc"],  # Target attention and feed-forward layers
        lora_dropout=0.05,
        bias="none",  # Don't train bias parameters
        task_type=TaskType.CAUSAL_LM,
    )

    # 4. Apply LoRA adapters to the quantized model
    peft_model = get_peft_model(model, lora_config)
    """
    Ensure base model remains frozen by default. We will explicitly enable
    LoRA and QAT per-channel parameters later.
    """
    for param in peft_model.base_model.parameters():
        param.requires_grad = False
    # print("\n--- LoRA Adapters Applied ---") 
    # peft_model.print_trainable_parameters()
    apply_quantization_to_peft_model(peft_model)

    # First apply quantization to the base model
    # print("\n--- Applying W4A4 quantization to base model ---")

    # apply_quantization_to_peft_model(peft_model)
    # print("\n--- W4A4 Quantization Applied to LoRA-enabled model ---")



    # 5. Configure Trainable Parameters
    # Ensure only LoRA adapters and PACT thresholds are trainable.
    print("\n--- Configuring Trainable Parameters (LoRA, PACT clip, SmoothQuant) ---")
    for name, param in peft_model.named_parameters():
        lname = name.lower()
        is_lora = ("lora" in lname)
        is_qat = ("activation_clip" in lname) or ("smooth_factor" in lname)
        param.requires_grad = bool(is_lora or is_qat)

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
    datasets, _ = load_and_preprocess_wikitext2(model_name="gpt2") # Use _ for tokenizer as it\'s already loaded
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]

    # Set up Training Arguments
    use_fp16 = torch.cuda.is_available() and not torch.backends.mps.is_available()
    
    # Choose memory-friendly settings based on device
    is_cuda = torch.cuda.is_available() and not torch.backends.mps.is_available()
    per_device_bs = 4 if is_cuda else 8
    grad_accum = 16 if is_cuda else 8

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        num_train_epochs=2,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs * 2,
        gradient_accumulation_steps=grad_accum,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_dir=os.path.join(output_dir, "tensorboard_logs"),
        logging_steps=20,
        learning_rate=5e-5,  # Reduced learning rate for stability
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=is_cuda,
        bf16=False,
        tf32=False,
        report_to="tensorboard",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )

    # Define Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Percentile-based calibration of activation clips (short pass)
    print("\n--- Calibrating activation clips (percentile init) ---")
    peft_model = peft_model.to(device)
    calib_loader = DataLoader(train_dataset, batch_size=per_device_bs, shuffle=False, collate_fn=data_collator)
    per_module_abs = calibrate_activation_clips(peft_model, calib_loader, steps=200, percentile=99.5, device=device)
    # Initialize SmoothQuant scaling factors using activation percentiles and weight stats
    calibrate_smooth_factors(peft_model, per_module_abs, alpha=0.5)

    # Build optimizer with parameter groups: LoRA vs PACT clip vs SmoothQuant factors
    lora_params: List[torch.nn.Parameter] = []
    clip_params: List[torch.nn.Parameter] = []
    smooth_params: List[torch.nn.Parameter] = []

    for name, param in peft_model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if "lora" in lname:
            lora_params.append(param)
        elif "activation_clip" in lname:
            clip_params.append(param)
        elif "smooth_factor" in lname:
            smooth_params.append(param)

    param_groups = []
    # Set learning rates: LoRA uses the TrainingArguments LR; quant params use a higher LR
    lora_lr = training_args.learning_rate
    quant_lr = 1e-3
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lora_lr, "weight_decay": 0.0})
    if clip_params:
        param_groups.append({"params": clip_params, "lr": quant_lr, "weight_decay": 0.0})
    if smooth_params:
        param_groups.append({"params": smooth_params, "lr": quant_lr, "weight_decay": 0.0})

    # Fallback: if for some reason groups are empty, include all trainable params
    if not param_groups:
        all_trainable = [p for p in peft_model.parameters() if p.requires_grad]
        param_groups = [{"params": all_trainable, "lr": lora_lr, "weight_decay": 0.0}]

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

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
    
    # Ensure model on device and enable input grads for all devices to allow LoRA/SQ to learn
    peft_model = peft_model.to(device)
    peft_model.enable_input_require_grads()
    # Disable KV cache to avoid conflicts with potential checkpointing and ensure full backprop
    if hasattr(peft_model, 'config'):
        peft_model.config.use_cache = False

    # 7. Train the model (fresh run). Checkpoints will save quant params via callback.
    print("\n--- Starting W4A4 QAT with LoRA Fine-tuning ---")
    trainer.train()

    # 8. Evaluate and Save
    print("\n--- Evaluating W4A4 QAT LoRA Model ---")
    eval_results = trainer.evaluate()
    print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss']))}")

    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model, tokenizer, and quantization parameters
    print(f"\n--- Saving W4A4 QAT LoRA model to {model_dir} ---")
    
    # Save quantization parameters
    
    # Save quantization parameters after training
    save_quantization_parameters(peft_model, model_dir)
    
    # Save the LoRA adapters
    peft_model.save_pretrained(model_dir)
    print(f"--- Saving W4A4 QAT LoRA adapters to {model_dir} ---")
    
    # Save the tokenizer
    tokenizer.save_pretrained(model_dir)
    
    # Save the model config
    peft_model.config.to_json_file(os.path.join(model_dir, "config.json"))
    
    
    print(f"\n--- Model, tokenizer, and quantization config saved to {model_dir} ---")
    print("To load this model for inference, use:")
    print(f"model = AutoModelForCausalLM.from_pretrained('{model_name}').to('{device}')")
    print(f"model = PeftModel.from_pretrained(model, '{model_dir}')")
    print("# Make sure to apply the same quantization setup when loading")