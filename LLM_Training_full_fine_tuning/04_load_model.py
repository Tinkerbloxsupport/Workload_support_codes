import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

print(f"Loading model: {MODEL_ID}")
print(f"VRAM before load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="cuda",
)

print(f"\nVRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters:     {total/1e9:.3f}B")
print(f"Trainable parameters: {trainable/1e9:.3f}B")
print(f"Trainable %:          {100 * trainable/total:.1f}%")

print("\n=== Model Architecture (top level) ===")
for name, module in list(model.named_children()):
    print(f"  {name}: {module.__class__.__name__}")

print("\nModel loaded successfully.")
