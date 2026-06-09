"""
Configuration for document extraction fine-tuning.
Change MODEL_NAME to scale to larger models on GPU.
"""

# Model configuration
MODEL_NAME = "distilgpt2"          # Fast CPU training, ~10 mins
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Better quality, needs GPU
# MODEL_NAME = "microsoft/phi-2"   # Strong small model, needs GPU

# LoRA configuration
LORA_CONFIG = {
    "r": 8,                        # Rank: higher = more capacity, more params
    "lora_alpha": 16,              # Scaling factor (typically 2x rank)
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    # Target modules for distilgpt2
    "target_modules": ["c_attn"],
    # For TinyLlama use: ["q_proj", "v_proj", "k_proj", "o_proj"]
    # For Phi-2 use: ["q_proj", "v_proj"]
}

# Training configuration
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 20,
    "logging_steps": 10,
    "save_steps": 100,
    "output_dir": "./outputs/model",
    "max_seq_length": 256,
}

# Data configuration
DATA_CONFIG = {
    "n_train": 300,
    "n_test": 60,
    "seed": 42,
    "data_dir": "./data",
    "domains": ["health_safety", "it_security", "field_maintenance"]
}