"""
evaluate.py
Evaluates the fine-tuned ClinicalBERT model on the test dataset.
"""

import os
import sys
import logging
import json
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ensure project root is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import PROCESSED_DATA_DIR, MODEL_DIR, TOKENIZER_NAME

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_label_mapping():
    """Load label mapping from JSON file and ensure keys are integers."""
    label_mapping_path = os.path.join(PROCESSED_DATA_DIR, "label_mapping.json")
    if os.path.exists(label_mapping_path):
        with open(label_mapping_path, "r") as f:
            mapping = json.load(f)
        return {int(k): v for k, v in mapping.items()}
    else:
        logger.warning("Label mapping file not found. Proceeding without label mapping.")
        return None

def prepare_data_for_evaluation(texts, tokenizer, batch_size=32):
    """Prepare and batch the test data for evaluation."""
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"])
    return DataLoader(dataset, batch_size=batch_size)

def main():
    logger.info("Starting evaluation...")

    # 1. Load test data
    test_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")
    if not os.path.exists(test_path):
        logger.error(f"Test file not found at: {test_path}")
        sys.exit(1)

    test_df = pd.read_csv(test_path)
    logger.info(f"Loaded {len(test_df)} test samples.")

    if test_df.empty or "symptoms" not in test_df.columns or "label" not in test_df.columns:
        logger.error("Test data is either empty or missing required columns.")
        sys.exit(1)

    # 2. Load model & tokenizer
    logger.info(f"Loading model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Running evaluation on device: {device}")

    # 3. Prepare test data
    texts = test_df["symptoms"].tolist()
    labels = test_df["label"].tolist()
    dataloader = prepare_data_for_evaluation(texts, tokenizer)

    # 4. Inference
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = [t.to(device) for t in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds_batch = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds_batch)

    logger.info(f"Labels in test set: {labels}")
    logger.info(f"Predictions (raw): {all_preds}")

    # 5. Metrics
    accuracy = accuracy_score(labels, all_preds)
    logger.info(f"Accuracy on test set: {accuracy:.4f}")

    # 6. Apply label mapping
    label_mapping = load_label_mapping()
    if label_mapping:
        preds_named = [label_mapping.get(pred, "Unknown") for pred in all_preds]
        labels_named = [label_mapping.get(label, "Unknown") for label in labels]
        logger.info(f"Mapped Predictions: {preds_named}")
        logger.info(f"Mapped Labels: {labels_named}")
    else:
        preds_named = all_preds
        labels_named = labels

    # 7. Generate classification report and confusion matrix
    if label_mapping:
        all_labels = list(label_mapping.values())
        report = classification_report(labels_named, preds_named, labels=all_labels, zero_division=0)
        cm = confusion_matrix(labels_named, preds_named, labels=all_labels)
    else:
        report = classification_report(labels, all_preds, zero_division=0)
        cm = confusion_matrix(labels, all_preds)

    logger.info("Classification Report:\n" + report)
    logger.info(f"Confusion Matrix:\n{cm}")

    # 8. Save metrics to a file
    metrics_output_path = os.path.join(MODEL_DIR, "evaluation_metrics.txt")
    with open(metrics_output_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("Classification Report:\n" + report + "\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n")
    logger.info(f"Saved evaluation results to: {metrics_output_path}")

    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
