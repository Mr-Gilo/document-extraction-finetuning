"""
LoRA Fine-tuning for Document Information Extraction

Uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters,
training only ~0.18% of model parameters while achieving strong
performance improvements on the extraction task.

Uses standard HuggingFace Trainer for maximum compatibility.
"""

import json
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
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
        dtype=torch.float32,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_CONFIG["target_modules"]
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    return model, tokenizer


def tokenize_dataset(examples, tokenizer, max_length):
    """Tokenize prompts for causal language model training."""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def train():
    train_path = os.path.join(DATA_CONFIG["data_dir"], "train.jsonl")
    if not os.path.exists(train_path):
        print("Generating dataset...")
        save_dataset()

    train_examples = load_jsonl(train_path)
    print(f"Loaded {len(train_examples)} training examples")

    model, tokenizer = setup_model_and_tokenizer()

    # Build dataset
    raw_dataset = Dataset.from_dict({
        "text": [ex["prompt"] for ex in train_examples]
    })

    # Tokenize
    tokenized_dataset = raw_dataset.map(
        lambda x: tokenize_dataset(x, tokenizer,
                                   TRAINING_CONFIG["max_seq_length"]),
        batched=True,
        remove_columns=["text"]
    )
    tokenized_dataset.set_format("torch")

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=TRAINING_CONFIG["output_dir"],
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        fp16=False,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("\nStarting LoRA fine-tuning...")
    print(f"Model: {MODEL_NAME}")
    print(f"LoRA rank: {LORA_CONFIG['r']}, alpha: {LORA_CONFIG['lora_alpha']}")
    print(f"Epochs: {TRAINING_CONFIG['num_train_epochs']}, "
          f"Batch size: {TRAINING_CONFIG['per_device_train_batch_size']}\n")

    trainer.train()

    # Save LoRA adapter weights only
    adapter_path = os.path.join(TRAINING_CONFIG["output_dir"], "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nLoRA adapter saved to: {adapter_path}")


if __name__ == "__main__":
    train()