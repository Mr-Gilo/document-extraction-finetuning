"""
LoRA Fine-tuning for Document Information Extraction

Uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters,
training only ~0.5% of model parameters while achieving strong
performance improvements on the extraction task.
"""

import json
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from trl import SFTTrainer
from config import MODEL_NAME, LORA_CONFIG, TRAINING_CONFIG, DATA_CONFIG
from data_generator import save_dataset


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def setup_model_and_tokenizer():
    """Load base model and apply LoRA adapters."""
    print(f"Loading base model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # float32 for CPU stability
        device_map="auto"
    )

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_CONFIG["target_modules"]
    )

    model = get_peft_model(model, lora_config)

    # Report trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    return model, tokenizer


def prepare_dataset(examples, tokenizer):
    """Convert examples to HuggingFace Dataset."""
    return Dataset.from_dict({"text": [ex["prompt"] for ex in examples]})


def train():
    # Generate data if not already done
    train_path = os.path.join(DATA_CONFIG["data_dir"], "train.jsonl")
    if not os.path.exists(train_path):
        print("Generating dataset...")
        save_dataset()

    train_examples = load_jsonl(train_path)
    print(f"Loaded {len(train_examples)} training examples")

    model, tokenizer = setup_model_and_tokenizer()
    train_dataset = prepare_dataset(train_examples, tokenizer)

    training_args = TrainingArguments(
        output_dir=TRAINING_CONFIG["output_dir"],
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        fp16=False,          # CPU training requires fp16=False
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_length=TRAINING_CONFIG["max_seq_length"],
        dataset_text_field="text",
        packing=False,
    )

    print("\nStarting LoRA fine-tuning...")
    print(f"Model: {MODEL_NAME}")
    print(f"LoRA rank: {LORA_CONFIG['r']}, alpha: {LORA_CONFIG['lora_alpha']}")
    print(f"Epochs: {TRAINING_CONFIG['num_train_epochs']}, "
          f"Batch size: {TRAINING_CONFIG['per_device_train_batch_size']}\n")

    trainer.train()

    # Save LoRA adapter weights only (not full model)
    adapter_path = os.path.join(TRAINING_CONFIG["output_dir"], "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nLoRA adapter saved to: {adapter_path}")


if __name__ == "__main__":
    train()