"""
LoRA Fine-tuning for Document Information Extraction

Uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters,
training only ~0.18% of model parameters while achieving strong
performance improvements on the extraction task.

Uses standard HuggingFace Trainer for maximum compatibility.

Uses masked label training: loss is computed ONLY on the JSON output tokens,
not the input prompt. This focuses learning on generating correct extractions.
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


def tokenize_with_masked_labels(examples, tokenizer, max_length):
    """
    Tokenize with masked labels for instruction fine-tuning.
    Labels are set to -100 for prompt tokens (ignored in loss).
    Only the JSON output tokens contribute to the loss.
    This focuses the model on learning to generate correct extractions.
    """
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for text in examples["text"]:
        # Split prompt from JSON response
        if "JSON:" in text:
            prompt_part = text.split("JSON:")[0] + "JSON:"
            json_part = text.split("JSON:", 1)[1].strip()
        else:
            prompt_part = text
            json_part = ""

        # Tokenize prompt to find its length
        prompt_tokens = tokenizer(
            prompt_part,
            add_special_tokens=True,
            return_tensors=None
        )
        prompt_len = len(prompt_tokens["input_ids"])

        # Tokenize full sequence (prompt + JSON response)
        full_tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]

        # Mask prompt tokens in labels (-100 = ignored by loss function)
        labels = input_ids.copy()
        mask_len = min(prompt_len, len(labels))
        for i in range(mask_len):
            labels[i] = -100

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels
    }


def train():
    train_path = os.path.join(DATA_CONFIG["data_dir"], "train.jsonl")
    if not os.path.exists(train_path):
        print("Generating dataset...")
        save_dataset()

    train_examples = load_jsonl(train_path)
    print(f"Loaded {len(train_examples)} training examples")

    model, tokenizer = setup_model_and_tokenizer()

    raw_dataset = Dataset.from_dict({
        "text": [ex["prompt"] for ex in train_examples]
    })

    print("Tokenising with masked labels (prompt tokens excluded from loss)...")
    tokenized_dataset = raw_dataset.map(
        lambda x: tokenize_with_masked_labels(
            x, tokenizer, TRAINING_CONFIG["max_seq_length"]
        ),
        batched=True,
        remove_columns=["text"]
    )
    tokenized_dataset.set_format("torch")

    # Verify masking worked
    sample_labels = tokenized_dataset[0]["labels"].tolist()
    n_masked = sum(1 for l in sample_labels if l == -100)
    n_active = sum(1 for l in sample_labels if l != -100)
    print(f"Label check — masked (prompt): {n_masked}, "
          f"active (JSON output): {n_active}")

    training_args = TrainingArguments(
        output_dir=TRAINING_CONFIG["output_dir"],
        num_train_epochs=10,          # More epochs with focused loss
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=500,
        fp16=False,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=lambda data: {
            "input_ids": torch.stack([d["input_ids"] for d in data]),
            "attention_mask": torch.stack([d["attention_mask"] for d in data]),
            "labels": torch.stack([d["labels"] for d in data]),
        },
    )

    print("\nStarting LoRA fine-tuning with masked labels...")
    print(f"Model: {MODEL_NAME}")
    print(f"LoRA rank: {LORA_CONFIG['r']}, alpha: {LORA_CONFIG['lora_alpha']}")
    print(f"Epochs: 10 (focused on JSON output tokens only)\n")

    trainer.train()

    adapter_path = os.path.join(TRAINING_CONFIG["output_dir"], "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nLoRA adapter saved to: {adapter_path}")


if __name__ == "__main__":
    train()