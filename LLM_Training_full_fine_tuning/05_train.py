import os
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer

MODEL_ID   = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "./llama3_finetuned"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="cuda",
)

print("Loading tokenized dataset...")
dataset = load_from_disk("./alpaca_tokenized")
train_dataset = dataset.select(range(2000))
print(f"Training on {len(train_dataset)} samples")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    bf16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    report_to="none",
    dataloader_pin_memory=False,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

print("\n=== Starting Training ===")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Steps per epoch: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")

trainer.train()

print("\nTraining complete. Saving...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}")
