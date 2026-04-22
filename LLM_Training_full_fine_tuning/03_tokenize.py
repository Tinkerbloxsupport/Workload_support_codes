from datasets import load_from_disk
from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MAX_LENGTH = 512

print(f"Loading tokenizer from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading formatted dataset...")
dataset = load_from_disk("./alpaca_formatted")
print(f"Samples: {len(dataset)}")

def tokenize(sample):
    result = tokenizer(
        sample["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        add_special_tokens=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result

print("Tokenizing...")
tokenized = dataset.map(
    tokenize,
    remove_columns=["text"],
    batched=True,
    batch_size=1000,
)

print("\n=== Verification ===")
print(f"Features: {tokenized.features}")
print(f"Sample 0 input_ids shape: {len(tokenized[0]['input_ids'])}")
print(f"Sample 0 first 10 tokens: {tokenized[0]['input_ids'][:10]}")
print(f"Decoded sample 0 start: {tokenizer.decode(tokenized[0]['input_ids'][:30])}")

tokenized.save_to_disk("./alpaca_tokenized")
print(f"\nSaved to ./alpaca_tokenized")
