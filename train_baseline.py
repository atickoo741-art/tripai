from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import torch

from load_model import model, tokenizer # Import the pre-loaded model and tokenizer
from data_preparation import load_and_preprocess_wikitext2



# 1. Load and preprocess the WikiText-2 dataset
datasets, tokenizer = load_and_preprocess_wikitext2(model_name="gpt2")

train_dataset = datasets["train"]
eval_dataset = datasets["validation"]

# 2. Define LoRA configuration
# For the baseline, we're applying LoRA to the unquantized model.
lora_config = LoraConfig(
    r=8, # LoRA attention dimension
    lora_alpha=16, # Alpha parameter for LoRA scaling
    target_modules=["c_attn", "c_proj", "c_fc"], # Common target modules for GPT-2 LoRA
    lora_dropout=0.1, # Dropout probability for LoRA layers
    bias="none", # No bias in LoRA layers
    task_type=TaskType.CAUSAL_LM, # Causal Language Modeling
)

# 3. Get the PEFT model
# This will add the LoRA adapters to the base GPT-2 model.
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# 4. Set up Training Arguments
training_args = TrainingArguments(
    output_dir="./output_baseline_lora", # Output directory for checkpoints and logs
    overwrite_output_dir=True,
    num_train_epochs=1, # Train for one epoch as per plan
    per_device_train_batch_size=8, # Adjust batch size as per your GPU memory
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8, # Simulate larger batch size
    save_strategy="epoch",
    logging_dir="./logs_baseline_lora",
    logging_steps=50, # Log every 50 steps
    learning_rate=2e-4, # Standard learning rate for LoRA fine-tuning
    weight_decay=0.01,
    fp16=torch.cuda.is_available(), # Use FP16 if CUDA is available, for faster training
    report_to="none", # Disable reporting to external services like W&B for this fast-fail setup
    remove_unused_columns=False, # Important for DataCollatorForLanguageModeling
)

# 5. Define Data Collator
# This will dynamically pad sequences to the longest length in a batch.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6. Initialize Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 7. Train the model
print("\n--- Starting Baseline LoRA Fine-tuning ---")
trainer.train()

# 8. Evaluate and Save
print("\n--- Evaluating Baseline LoRA Model ---")
eval_results = trainer.evaluate()
print(f"Perplexity: {torch.exp(torch.tensor(eval_results["eval_loss"]))}")

# Save the LoRA adapters (not the full model)
peft_model.save_pretrained("./baseline_lora_adapters")
print("Baseline LoRA adapters saved to ./baseline_lora_adapters") 