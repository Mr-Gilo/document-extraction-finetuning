# Document Extraction Fine-tuning with LoRA

Fine-tuning a causal language model to extract structured information
from incident and operational reports using Low-Rank Adaptation (LoRA).

Demonstrates parameter-efficient fine-tuning across three document domains:
Health and Safety incidents, IT Security incidents, and Field Maintenance reports.
The same extraction framework adapts to any document domain by replacing
the training data.

## What is LoRA?

Low-Rank Adaptation (LoRA) freezes the pre-trained model weights and injects
trainable low-rank matrices into the attention layers. This trains only ~0.5%
of model parameters while achieving strong task-specific performance,
making fine-tuning feasible on consumer hardware.

Base Model Weights (frozen)
+
LoRA Adapter Weights (trainable, ~0.5% of params)
=
Fine-tuned behaviour on target task

## Extraction Schema

All domains share a unified extraction schema:

| Field | Description |
|-------|-------------|
| incident_type | Classification of the reported event |
| date_reported | Date of the incident report (DD/MM/YYYY) |
| location | Physical or system location of the incident |
| parties_involved | People or teams named in the report |
| severity | Low / Medium / High / Critical |
| summary | One-sentence description |
| action_required | Recommended or taken action |

## Project Structure

document-extraction-finetuning/
├── config.py          Model, LoRA, and training configuration
├── data_generator.py  Synthetic multi-domain report generator
├── train.py           LoRA fine-tuning with PEFT + SFTTrainer
├── evaluate.py        Before vs after accuracy comparison
├── inference.py       Demo inference on new reports
├── requirements.txt
└── README.md

## Setup

```bash
git clone https://github.com/Mr-Gilo/document-extraction-finetuning.git
cd document-extraction-finetuning

conda create -n doc-extraction python=3.11 -y
conda activate doc-extraction
pip install -r requirements.txt
```

## Running

```bash
# Step 1: Generate synthetic training data
python data_generator.py

# Step 2: Fine-tune with LoRA (approx 10-15 mins on CPU)
python train.py

# Step 3: Evaluate before vs after accuracy
python evaluate.py

# Step 4: Demo inference on new reports
python inference.py
```

## Results

After 3 epochs of LoRA fine-tuning on 300 examples:

| Field | Before | After | Improvement |
|-------|--------|-------|-------------|
| incident_type | ~15% | ~75% | +60% |
| date_reported | ~20% | ~85% | +65% |
| severity | ~25% | ~80% | +55% |
| action_required | ~10% | ~65% | +55% |
| Valid JSON rate | ~10% | ~80% | +70% |

*Results on distilgpt2. Larger models (TinyLlama, Phi-3) achieve higher accuracy.*

## Scaling to Larger Models

Change `MODEL_NAME` in `config.py` and update `target_modules`:

```python
# For TinyLlama (GPU recommended)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_CONFIG["target_modules"] = ["q_proj", "v_proj", "k_proj", "o_proj"]

# For Phi-3-mini (GPU recommended)
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
LORA_CONFIG["target_modules"] = ["q_proj", "v_proj"]
```

## Applying to New Domains

Replace or extend `data_generator.py` with domain-specific examples
following the same prompt format. The training pipeline requires no
other changes. Examples of applicable domains:

- Medical incident reports
- Legal contract clause extraction
- Financial audit findings
- Customer complaint classification
- Insurance claim analysis and so on

## Related Projects

- [pdf-extractor](https://github.com/Mr-Gilo/pdf-extractor)
- [rag-document-assistant](https://github.com/Mr-Gilo/rag-document-assistant)
- [multimodal-risk-pipeline](https://github.com/Mr-Gilo/multimodal-risk-pipeline)

## Roadmap

- [x] Multi-domain synthetic data generator
- [x] LoRA fine-tuning with PEFT
- [x] Before/after evaluation comparison
- [x] Demo inference pipeline
- [ ] Weights and Biases training dashboard
- [ ] ONNX export for edge deployment
- [ ] Domain-specific fine-tuning examples