from datasets import load_dataset

print("Downloading yahma/alpaca-cleaned...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
print(f"Total samples: {len(dataset)}")

def format_sample(sample):
    if sample["input"].strip():
        user_msg = f"{sample['instruction']}\n\n{sample['input']}"
    else:
        user_msg = sample["instruction"]

    return {
        "text": (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            "You are a helpful assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{user_msg}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
            f"{sample['output']}<|eot_id|>"
        )
    }

print("Formatting into LLaMA 3 chat template...")
dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

print("\n=== Sample 0 ===")
print(dataset[0]["text"][:300])

print("\n=== Sample 1 ===")
print(dataset[1]["text"][:300])

dataset.save_to_disk("./alpaca_formatted")
print(f"\nSaved to ./alpaca_formatted")
print(f"Total formatted samples: {len(dataset)}")
