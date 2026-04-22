import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "./llama3_finetuned"
FINAL_DIR = "./outputs/llama3_final"

print("Loading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype=torch.bfloat16,
    device_map="cpu",
)

total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total/1e9:.3f}B")

os.makedirs(FINAL_DIR, exist_ok=True)
print(f"\nSaving final model to {FINAL_DIR}...")
model.save_pretrained(FINAL_DIR, safe_serialization=True)
tokenizer.save_pretrained(FINAL_DIR)

print("\n=== Saved files ===")
for f in sorted(os.listdir(FINAL_DIR)):
    size = os.path.getsize(f"{FINAL_DIR}/{f}") / 1e6
    print(f"  {f:40s} {size:.1f} MB")

print("\nDone. Model ready for deployment or upload to HuggingFace Hub.")
