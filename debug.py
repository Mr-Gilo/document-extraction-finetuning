"""Quick debug: show raw model outputs to diagnose evaluation failure."""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import MODEL_NAME, TRAINING_CONFIG, DATA_CONFIG
from data_generator import format_prompt

ADAPTER_PATH = f"{TRAINING_CONFIG['output_dir']}/lora_adapter"

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

test_examples = load_jsonl(f"{DATA_CONFIG['data_dir']}/test.jsonl")[:3]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()

for i, ex in enumerate(test_examples):
    prompt = format_prompt(ex["text"])
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=200)

    input_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only new tokens
    new_tokens = outputs[0][input_length:]
    after_json = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"\n--- Example {i+1} (domain: {ex['domain']}) ---")
    print(f"EXPECTED: {json.dumps(ex['extraction'])[:150]}")
    print(f"GOT:      {after_json[:300]}")
    print()