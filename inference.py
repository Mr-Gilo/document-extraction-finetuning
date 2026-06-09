"""
Demo inference: run the fine-tuned model on new reports.
Shows how the adapter can be loaded and deployed.
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import MODEL_NAME, TRAINING_CONFIG
from data_generator import format_prompt
from evaluate import extract_json_from_output

ADAPTER_PATH = f"{TRAINING_CONFIG['output_dir']}/lora_adapter"

DEMO_REPORTS = [
    """
    INCIDENT REPORT — Health and Safety
    Date of Report: 14/03/2024
    Location: Packing Hall, North Wing
    A moderate incident occurred when a forklift operator, T. Williams,
    reported near-miss contact with a pedestrian in a shared access zone.
    The site supervisor K. Osei was immediately notified.
    Action taken: Pedestrian barriers installed and forklift traffic
    rerouted pending permanent signage installation.
    """,
    """
    IT SECURITY INCIDENT REPORT
    Reported: 22/05/2024
    Affected System: Customer relationship management database
    Severity: High
    Incident type: Suspicious bulk data export detected.
    The Data Loss Prevention team identified and blocked an unusual
    export of 50,000 customer records outside business hours.
    Immediate response: User account suspended and forensic
    imaging of workstation initiated.
    """,
    """
    FIELD MAINTENANCE REPORT
    Date: 09/01/2024
    Site: Site Foxtrot — Bristol
    Equipment: Water treatment pump array
    Fault reported: Pressure drop across primary pump below operational
    threshold. Senior Engineer M. Abubakar identified worn impeller
    as likely cause. Severity assessed as High.
    Recommended action: Emergency impeller replacement authorised,
    backup pump brought online to maintain service continuity.
    """
]


def run_demo():
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("\n" + "=" * 60)
    print("  DOCUMENT EXTRACTION DEMO — Fine-tuned Model")
    print("=" * 60)

    for i, report in enumerate(DEMO_REPORTS, 1):
        print(f"\n--- Report {i} ---")
        print(report.strip())

        prompt = format_prompt(report.strip())
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=256
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        extraction = extract_json_from_output(generated)

        print("\nExtracted JSON:")
        if extraction:
            print(json.dumps(extraction, indent=2))
        else:
            print("(Could not parse JSON from output)")
            raw = generated.split("JSON:")[-1].strip()[:300]
            print(f"Raw output: {raw}")
        print()


if __name__ == "__main__":
    run_demo()