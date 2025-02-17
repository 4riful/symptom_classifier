"""
inference.py
------------
Defines 'classify_symptom' for real-time inference using a fine-tuned ClinicalBERT model,
and 'classify_with_fallback' that integrates threshold logic and a naive fallback mechanism.

Usage Example (in a FastAPI route):
-----------------------------------
from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import classify_with_fallback

app = FastAPI()

class SymptomRequest(BaseModel):
    symptoms: str

@app.post("/classify")
def classify_endpoint(request: SymptomRequest):
    result = classify_with_fallback(request.symptoms)
    return result
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ensure we can import from one directory above 'src/'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import MODEL_DIR  # Or wherever you keep your model directory path
from src.fallback import fallback_classification

# Update this path to wherever your label_mapping.json is stored
LABEL_MAP_PATH = os.path.join("data", "processed", "label_mapping.json")

# Load label mapping
# Example label_mapping.json might look like: {"0": "Cardiology", "1": "Dermatology", ...}
with open(LABEL_MAP_PATH, "r") as f:
    label_mapping = json.load(f)

# Convert string keys to int if necessary
label_mapping_int = {int(k): v for k, v in label_mapping.items()}

# Load the model and tokenizer (trained/fine-tuned)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model.eval()

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def classify_symptom(
    text: str,
    high_conf_threshold: float = 0.7,
    low_conf_threshold: float = 0.3
) -> dict:
    """
    Performs single-label classification on the input text using a softmax distribution
    over known department labels. Returns a dictionary that may contain:

    1) Single high-confidence department if 'max_conf' >= high_conf_threshold.
    2) A list of candidate departments if no single label meets the high threshold,
       but one or more labels exceed low_conf_threshold.
    3) A fallback signal if no label exceeds low_conf_threshold.

    :param text: Symptom description input.
    :param high_conf_threshold: Confidence above which we trust a single department.
    :param low_conf_threshold: Confidence above which we consider a label as a candidate.
    :return: dict with the following possible keys:
        - "department": str or None
        - "confidence": float
        - "candidates": list of { "department": str, "confidence": float } or None
        - "fallback": bool (indicates if the model is uncertain)
        - "message": str (explanation of the fallback)
    """
    # Tokenize the text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # Move inputs to the appropriate device (CPU or GPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze()  # shape: (num_labels,)

    # Convert to Python float
    probs_list = probs.tolist()

    max_conf = max(probs_list)
    max_idx = probs_list.index(max_conf)

    # CASE 1: Single high-confidence department
    if max_conf >= high_conf_threshold:
        return {
            "department": label_mapping_int[max_idx],
            "confidence": float(max_conf),
            "candidates": None,
            "fallback": False
        }

    # CASE 2: Possibly multiple candidates above low_conf_threshold
    candidates = [
        {
            "department": label_mapping_int[i],
            "confidence": float(p)
        }
        for i, p in enumerate(probs_list) if p >= low_conf_threshold
    ]

    if len(candidates) == 0:
        # CASE 3: No candidate above the low_conf_threshold
        return {
            "department": None,
            "confidence": 0.0,
            "candidates": None,
            "fallback": True,
            "message": "Model confidence too low for any department."
        }
    elif len(candidates) == 1:
        # We found exactly one label above the low threshold, but below the high threshold
        c = candidates[0]
        return {
            "department": c["department"],
            "confidence": c["confidence"],
            "candidates": None,
            "fallback": True,
            "message": f"Department is below high confidence threshold ({high_conf_threshold})."
        }
    else:
        # We have multiple possible departments
        return {
            "department": None,
            "confidence": 0.0,
            "candidates": candidates,
            "fallback": True,
            "message": "Multiple potential departments found. Further evaluation suggested."
        }


def classify_with_fallback(text: str) -> dict:
    """
    Wraps 'classify_symptom' with an additional step of naive keyword-based fallback
    if the model's result indicates uncertainty ('fallback': True and no single candidate).

    :param text: Symptom description from the user.
    :return: A dictionary containing at least:
        - 'department': str (final department guess or fallback)
        - 'confidence': float
        - 'candidates': list or None
        - 'message': str
    """
    result = classify_symptom(text)

    # If the model is uncertain and has NO single candidate (department is None), then fallback
    if result.get("fallback", False):
        # "department" might still be None or might be set with a single low-confidence guess.
        # We'll only call the naive fallback if "department" is None or confidence is 0.0
        if result["department"] is None:
            naive_dep = fallback_classification(text)
            result["department"] = naive_dep
            result["confidence"] = 0.0
            result["candidates"] = None
            # Modify message to indicate fallback usage
            result["message"] = (
                "Falling back to naive keyword-based inference. "
                "Please note this is not a definitive medical diagnosis."
            )

    return result
