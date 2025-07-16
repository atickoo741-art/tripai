from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_preprocess_wikitext2(model_name="gpt2", block_size=128):
    """Loads and preprocesses the WikiText-2 dataset for language modeling."""
    print("Loading WikiText-2 dataset...")
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    print("Tokenizing dataset...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4, # Use multiple processes for faster tokenization
        remove_columns=["text"],
    )

    print(f"Original dataset sizes: {raw_datasets}")
    print(f"Tokenized dataset sizes: {tokenized_datasets}")

    # Group texts into sequences of block_size
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this line.
        # For now, we just truncate data to a multiple of block_size
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    print(f"Grouping texts into blocks of size {block_size}...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4,
    )

    print(f"Final preprocessed dataset sizes: {lm_datasets}")
    return lm_datasets, tokenizer

if __name__ == "__main__":
    # Example usage:
    datasets, tokenizer = load_and_preprocess_wikitext2()

    # You can now access the training, validation, and test splits:
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
    test_dataset = datasets["test"]

    print("\nFirst example of processed training data:")
    print(tokenizer.decode(train_dataset[0]["input_ids"]))
    print(f"Length of first example: {len(train_dataset[0]["input_ids"])}")
    print(f"Number of training examples: {len(train_dataset)}") 