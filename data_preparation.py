from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import torch
from transformers import DataCollatorForLanguageModeling


def load_and_preprocess_wikitext2(model_name="gpt2", block_size=128):
    """DEPRECATED: Old chunking method - do not use for training."""
    raise NotImplementedError(
        "Use load_wikitext2_sliding_window() instead to match eval protocol"
    )


def load_wikitext2_sliding_window(model_name="gpt2", seq_len=2048, stride=512):
    """
    Load WikiText-2 with sliding window preprocessing that matches evaluation protocol.
    
    This creates training samples where:
    - Each sample is a window of seq_len tokens
    - Windows overlap by (seq_len - stride) tokens
    - Labels are masked so only new tokens beyond the stride are predicted
    - Matches the exact evaluation methodology for fair comparison
    
    Args:
        model_name: HuggingFace model name for tokenizer
        seq_len: Length of each window (default 2048)
        stride: Stride between windows (default 512, meaning 1536 tokens of context)
    
    Returns:
        lm_datasets: DatasetDict with train/validation/test splits
        tokenizer: The tokenizer used
    """
    print(f"Loading WikiText-2 with sliding window (seq_len={seq_len}, stride={stride})...")
    raw_datasets = load_dataset("wikitext/wikitext-2-raw-v1")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    def create_sliding_windows(split_data):
        """
        Create sliding window samples matching EXACT evaluation protocol.
        
        With stride=seq_len (e.g., 2048):
        - No overlap between windows
        - Each window predicts all tokens
        - Matches standard perplexity evaluation
        
        With stride<seq_len (e.g., stride=512, seq_len=2048):
        - Windows overlap
        - Only new tokens beyond stride are predicted
        - First window predicts all tokens
        """
        print("Creating sliding window samples...")
        
        full_text = "\n\n".join(split_data["text"])
        encodings = tokenizer(full_text, return_tensors="pt")
        input_ids = encodings["input_ids"][0]
        total_length = len(input_ids)
        
        samples = []
        prev_end_loc = 0
        
        for begin_loc in range(0, total_length, stride):
            end_loc = min(begin_loc + seq_len, total_length)
            trg_len = end_loc - prev_end_loc  # New tokens in this window
            
            if end_loc - begin_loc < stride:
                break
            
            window_input_ids = input_ids[begin_loc:end_loc]
            labels = window_input_ids.clone()
            
            # Masking logic: EXACTLY matches Evaluation.evaluate()
            if begin_loc > 0:
                # Mask context: only predict last trg_len tokens
                context_len = len(labels) - trg_len
                labels[:context_len] = -100
            # else: first window predicts all tokens
            
            samples.append({
                "input_ids": window_input_ids.tolist(),
                "labels": labels.tolist(),
                "attention_mask": [1] * len(window_input_ids)
            })
            
            prev_end_loc = end_loc
        
        total_tokens = sum(len(s["input_ids"]) for s in samples)
        labeled_tokens = sum(sum(1 for l in s["labels"] if l != -100) for s in samples)
        
        print(f"  Samples: {len(samples)}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Labeled tokens: {labeled_tokens} ({100*labeled_tokens/total_tokens:.1f}%)")
        
        return samples



    # Process each split
    print("Creating sliding window samples...")
    processed_datasets = {}
    
    for split_name in ["train", "validation", "test"]:
        samples = create_sliding_windows(raw_datasets[split_name])
        processed_datasets[split_name] = Dataset.from_list(samples)
        
        # Calculate statistics
        total_tokens = sum(len(s["input_ids"]) for s in samples)
        labeled_tokens = sum(
            sum(1 for label in s["labels"] if label != -100) 
            for s in samples
        )
        
        print(f"{split_name}:")
        print(f"  Samples: {len(samples)}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Labeled tokens: {labeled_tokens:,} ({100*labeled_tokens/total_tokens:.1f}%)")
    
    lm_datasets = DatasetDict(processed_datasets)
    
    print(f"\n✓ WikiText-2 loaded with sliding window preprocessing")
    print(f"  seq_len={seq_len}, stride={stride}")
    print(f"  Context size: {seq_len - stride} tokens")
    return lm_datasets, tokenizer


def get_concatenated_text(dataset_split, tokenizer):
    """
    Helper function: Concatenate and tokenize dataset split.
    Used by both training data prep and evaluation.
    
    Args:
        dataset_split: A HuggingFace dataset split
        tokenizer: Tokenizer to use
    
    Returns:
        input_ids: Tensor of shape [total_length] with all tokens
    """
    # Concatenate with \n\n separators (preserves article boundaries)
    full_text = "\n\n".join(dataset_split["text"])
    
    # Tokenize
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings["input_ids"][0]  # Remove batch dimension
    
    return input_ids


from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer

ALPACA_PROMPT_WITH_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

ALPACA_PROMPT_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def load_alpaca_instruct(model_name="meta-llama/Llama-2-7b-hf", max_length=2048):
    """
    DYNAMIC PADDING version: Let DataCollator handle padding/labels.
    ~4x faster training.
    """
    print("\n" + "=" * 60)
    print("Loading Alpaca (DYNAMIC padding - FAST)")
    print("=" * 60)

    raw = load_dataset("tatsu-lab/alpaca", split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_prompts(examples):
        """Batch format Alpaca prompts"""
        texts = []
        for instruction, inp, output in zip(
            examples["instruction"], 
            examples["input"], 
            examples["output"]
        ):
            instruction = instruction.strip()
            inp = inp.strip() if inp else ""
            output = output.strip()
            
            if inp:
                prompt = ALPACA_PROMPT_WITH_INPUT.format(
                    instruction=instruction,
                    input=inp
                )
            else:
                prompt = ALPACA_PROMPT_NO_INPUT.format(
                    instruction=instruction
                )
            
            texts.append(prompt + output)
        return {"text": texts}

    # 1. Format prompts (batched)
    print("Formatting Alpaca prompts...")
    formatted = raw.map(
        format_prompts,
        batched=True,
        remove_columns=raw.column_names,
        desc="Formatting prompts"
    )

    # 2. Tokenize ONLY input_ids (no labels, no padding)
    def tokenize_prompts(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,  # CRITICAL: No padding here
        )

    print("Tokenizing (dynamic padding)...")
    tokenized = formatted.map(
        tokenize_prompts,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )

    # 3. Split
    split = tokenized.train_test_split(test_size=0.05, seed=42)
    dataset = DatasetDict({
        "train": split["train"],
        "validation": split["test"]
    })

    # 4. Verify structure (input_ids only, variable length)
    ex_lengths = []
    for i in range(min(10, len(dataset["train"]))):
        ex = dataset["train"][i]
        ex_lengths.append(len(ex["input_ids"]))
    print(f"\n✓ Dynamic padding dataset ready:")
    print(f"  Train examples:      {len(dataset['train']):,}")
    print(f"  Validation examples: {len(dataset['validation']):,}")
    print(f"  Example lengths:     {ex_lengths[:5]}... (variable ✓)")
    print(f"  DataCollator will pad within batches")
    print("=" * 60 + "\n")

    return dataset, tokenizer












if __name__ == "__main__":
    # # Test the new function
    # datasets, tokenizer = load_wikitext2_sliding_window(
    #     model_name="meta-llama/Llama-2-7b-hf",
    #     seq_len=2048,
    #     stride=2048
    # )

    # # Show example
    # train_dataset = datasets["train"]
    # print("\n" + "="*60)
    # print("Example training sample:")
    # print("="*60)
    # sample = train_dataset[0]
    # print(f"Input length: {len(sample['input_ids'])}")
    # print(f"Labels length: {len(sample['labels'])}")
    # print(f"Masked tokens: {sum(1 for l in sample['labels'] if l == -100)}")
    # print(f"Labeled tokens: {sum(1 for l in sample['labels'] if l != -100)}")
    
    

    dataset, tokenizer = load_alpaca_instruct()  # Your dynamic function

    print("\nTesting DataCollator...")
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    batch = collator([dataset["train"][i] for i in range(8)])

    print(f"✅ SUCCESS - Dynamic padding works:")
    print(f"  input_ids shape:     {batch['input_ids'].shape}")
    print(f"  labels shape:        {batch['labels'].shape}")
    print(f"  attention_mask:      {batch.get('attention_mask', 'N/A').shape}")
    print(f"  Max length in batch: {batch['input_ids'].shape[1]}")
    print("\n🚀 Ready for calibration/training!")



