# config.py
# Author: Your Name
# Created: 2025-01-19
# Description: Add a description of what this script does.

"""
config.py
Central configuration for the Symptom Classifier Project.
"""

import os

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "clinicalbert_finetuned")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# === Model & Tokenizer ===
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"  # ClinicalBERT
TOKENIZER_NAME = MODEL_NAME  # Usually same as model name unless you have a custom tokenizer

# === Training Hyperparameters ===
NUM_EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 128
SAVE_TOTAL_LIMIT = 1  # Limit saved checkpoints to prevent clutter

# === Checkpoint Configs ===
SAVE_STRATEGY = "epoch"           # Save model each epoch or after certain steps
EVALUATION_STRATEGY = "epoch"     # Evaluate each epoch
LOAD_BEST_MODEL_AT_END = True     # Automatically load best model after training

# === Department Labels ===
# Option 1: Hardcode if you want direct referencing in your code.
# Option 2: Dynamically handle after training from label_mapping.json.
DEPARTMENT_LABELS = [
    "Cardiology", "Dermatology", "ENT", "Endocrinology",
    "Gastroenterology", "General Medicine", "Neurology",
    "Orthopedics", "Pediatrics", "Pulmonology", "Radiology","Hematology"
]

# Example usage:
# - Your training script can import these constants
# - Minimizes “magic numbers” or repeated strings in your code
