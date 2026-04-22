import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "./llama3_finetuned"

print(f"Loading fine-tuned model from {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype=torch.bfloat16,
    device_map="cuda",
)
model.eval()

def ask(instruction, input_text=""):
    if input_text.strip():
        user_msg = f"{instruction}\n\n{input_text}"
    else:
        user_msg = instruction

    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful assistant.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_msg}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response

print("\n=== Test 1 ===")
print(ask("Explain what photosynthesis is in simple terms."))

print("\n=== Test 2 ===")
print(ask("Write a Python function to reverse a string."))

print("\n=== Test 3 ===")
print(ask("What are three tips for better sleep?"))
