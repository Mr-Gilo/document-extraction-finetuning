"""
Evaluation: Before vs After Fine-tuning Comparison

Measures field-level extraction accuracy on the held-out test set,
demonstrating the tangible improvement from LoRA fine-tuning.
"""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import MODEL_NAME, TRAINING_CONFIG, DATA_CONFIG
from data_generator import format_prompt


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def extract_json_from_output(text):
    """Extract JSON object from model output."""
    # Find the JSON marker
    if "JSON:" in text:
        json_part = text.split("JSON:")[-1].strip()
    else:
        json_part = text.strip()

    # Find first { and last }
    start = json_part.find("{")
    end = json_part.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(json_part[start:end + 1])
    except json.JSONDecodeError:
        return None


def generate_extraction(model, tokenizer, report_text, max_new_tokens=150):
    """Generate extraction — decode only newly generated tokens."""
    prompt = format_prompt(report_text)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=TRAINING_CONFIG["max_seq_length"]
    )

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode ONLY the new tokens, not the prompt
    new_tokens = outputs[0][input_length:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return extract_json_from_output(generated)

def compute_field_accuracy(predicted, ground_truth):
    """Compute per-field extraction accuracy."""
    fields = ["incident_type", "date_reported", "location",
              "severity", "action_required"]
    scores = {}

    if predicted is None:
        return {f: 0.0 for f in fields}, False

    for field in fields:
        pred_val = str(predicted.get(field, "")).lower().strip()
        true_val = str(ground_truth.get(field, "")).lower().strip()

        # Exact match
        exact = 1.0 if pred_val == true_val else 0.0

        # Partial match: key words overlap
        if exact == 0 and pred_val and true_val:
            pred_words = set(pred_val.split())
            true_words = set(true_val.split())
            overlap = len(pred_words & true_words)
            if overlap > 0:
                exact = round(overlap / max(len(pred_words), len(true_words)), 2)

        scores[field] = exact

    valid_json = True
    return scores, valid_json


def evaluate_model(model, tokenizer, test_examples, label="Model"):
    """Evaluate model on test set, return per-field accuracy."""
    print(f"\nEvaluating: {label}")
    print(f"Test examples: {len(test_examples)}")

    field_scores = {
        "incident_type": [], "date_reported": [], "location": [],
        "severity": [], "action_required": []
    }
    valid_json_count = 0

    for i, example in enumerate(test_examples):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(test_examples)}")

        predicted = generate_extraction(model, tokenizer, example["text"])
        scores, valid = compute_field_accuracy(predicted, example["extraction"])

        if valid and predicted is not None:
            valid_json_count += 1

        for field, score in scores.items():
            field_scores[field].append(score)

    # Compute averages
    results = {
        "label": label,
        "valid_json_rate": round(100 * valid_json_count / len(test_examples), 1),
        "field_accuracy": {
            field: round(100 * sum(scores) / len(scores), 1)
            for field, scores in field_scores.items()
        }
    }
    results["avg_accuracy"] = round(
        sum(results["field_accuracy"].values()) / len(results["field_accuracy"]), 1
    )

    return results


def print_results(before, after):
    """Print comparison table."""
    fields = list(before["field_accuracy"].keys())

    print("\n" + "=" * 65)
    print("  EXTRACTION ACCURACY: BEFORE vs AFTER LoRA FINE-TUNING")
    print("=" * 65)
    print(f"{'Field':<22} {'Before':>10} {'After':>10} {'Improvement':>12}")
    print("-" * 65)

    for field in fields:
        b = before["field_accuracy"][field]
        a = after["field_accuracy"][field]
        improvement = a - b
        sign = "+" if improvement >= 0 else ""
        print(f"{field:<22} {b:>9.1f}% {a:>9.1f}% {sign}{improvement:>10.1f}%")

    print("-" * 65)
    print(f"{'Average accuracy':<22} {before['avg_accuracy']:>9.1f}% "
          f"{after['avg_accuracy']:>9.1f}% "
          f"+{after['avg_accuracy'] - before['avg_accuracy']:>9.1f}%")
    print(f"{'Valid JSON rate':<22} {before['valid_json_rate']:>9.1f}% "
          f"{after['valid_json_rate']:>9.1f}% "
          f"+{after['valid_json_rate'] - before['valid_json_rate']:>9.1f}%")
    print("=" * 65)


def run_evaluation():
    test_path = os.path.join(DATA_CONFIG["data_dir"], "test.jsonl")
    if not os.path.exists(test_path):
        from data_generator import save_dataset
        save_dataset()

    test_examples = load_jsonl(test_path)
    # Use subset for faster evaluation
    test_subset = test_examples[:30]

    print("Loading base tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Evaluate base model (before fine-tuning) ──
    print(f"\nLoading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    )
    base_results = evaluate_model(
        base_model, tokenizer, test_subset, label="Base model (no fine-tuning)"
    )
    del base_model

    # ── Evaluate fine-tuned model (after fine-tuning) ──
    adapter_path = os.path.join(TRAINING_CONFIG["output_dir"], "lora_adapter")
    if not os.path.exists(adapter_path):
        print(f"\nFine-tuned model not found at {adapter_path}.")
        print("Run train.py first.")
        return

    print(f"\nLoading fine-tuned model from: {adapter_path}")
    base_for_lora = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    )
    finetuned_model = PeftModel.from_pretrained(base_for_lora, adapter_path)
    finetuned_results = evaluate_model(
        finetuned_model, tokenizer, test_subset,
        label="Fine-tuned model (LoRA)"
    )

    print_results(base_results, finetuned_results)

    # Save results
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation_results.json", "w") as f:
        json.dump({"before": base_results, "after": finetuned_results}, f, indent=2)
    print("\nResults saved to outputs/evaluation_results.json")


if __name__ == "__main__":
    run_evaluation()