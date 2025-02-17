"""
train.py
Trains a ClinicalBERT model to classify symptoms into hospital departments.
"""

import os
import sys
import warnings
import logging
from datetime import datetime
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Suppress warnings related to deprecated features
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Ensure the project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    MODEL_NAME,
    TOKENIZER_NAME,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_SEQ_LENGTH,
    SAVE_TOTAL_LIMIT,
    LOAD_BEST_MODEL_AT_END
)

# === Logging Configuration ===
log_file = os.path.join(MODEL_DIR, "logs", "training.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)

# === Dataset Definition ===
class SymptomDataset(Dataset):
    """ Custom PyTorch Dataset for symptom classification. """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Remove the batch dimension ([1, seq_len] -> [seq_len])
        for key, value in encoding.items():
            encoding[key] = value.squeeze(0)
        encoding["labels"] = torch.tensor(label, dtype=torch.long)
        return encoding

# === Metric Computation ===
def compute_metrics(eval_pred):
    """ Computes accuracy, precision, recall, and F1-score. """
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def log_metrics(metrics, epoch):
    """ Logs evaluation metrics for each epoch. """
    logger.info(f"Epoch {epoch} Evaluation Metrics:")
    logger.info(f"  Loss:       {metrics['eval_loss']:.4f}")
    logger.info(f"  Accuracy:   {metrics['eval_accuracy']:.2%}")
    logger.info(f"  Precision:  {metrics['eval_precision']:.4f}")
    logger.info(f"  Recall:     {metrics['eval_recall']:.4f}")
    logger.info(f"  F1 Score:   {metrics['eval_f1']:.4f}")

# === Main Training Process ===
def main():
    logger.info("Starting training process...")

    # === Load Train and Validation Data ===
    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DATA_DIR, "val.csv")
    
    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
    except Exception as e:
        logger.error(f"Error loading CSV files: {e}")
        sys.exit(1)

    logger.info(f"Loaded {len(train_df)} training samples and {len(val_df)} validation samples.")

    # === Initialize Tokenizer ===
    logger.info(f"Initializing tokenizer from {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # === Create Dataset Objects ===
    train_dataset = SymptomDataset(
        texts=train_df["symptoms"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LENGTH
    )
    val_dataset = SymptomDataset(
        texts=val_df["symptoms"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LENGTH
    )

    # === Data Collator ===
    data_collator = DataCollatorWithPadding(tokenizer)

    # === Load ClinicalBERT Model ===
    num_unique_labels = train_df["label"].nunique()
    logger.info(f"Loading model {MODEL_NAME} with {num_unique_labels} labels...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_unique_labels
    )

    # === Set Up Training Arguments ===
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        save_total_limit=SAVE_TOTAL_LIMIT,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=os.path.join(MODEL_DIR, "logs"),
        fp16=torch.cuda.is_available(),
        logging_steps=50
    )

    logger.info("Training hyperparameters:")
    logger.info(f"  Epochs: {NUM_EPOCHS}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Learning rate: {LEARNING_RATE}")
    logger.info(f"  Save total limit: {SAVE_TOTAL_LIMIT}")
    logger.info(f"  Using GPU: {torch.cuda.is_available()}")

    # === Create the Trainer ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # === Start Training ===
    start_time = datetime.now()
    logger.info(f"Training started at: {start_time}")
    trainer.train()
    end_time = datetime.now()

    # === Final Evaluation ===
    final_metrics = trainer.evaluate()
    log_metrics(final_metrics, NUM_EPOCHS)

    # === Save the Best Model ===
    logger.info(f"Saving best model to: {MODEL_DIR}")
    trainer.save_model(MODEL_DIR)

    duration = end_time - start_time
    logger.info(f"Training completed successfully at {end_time}")
    logger.info(f"Total training duration: {duration}")

if __name__ == "__main__":
    main()
