"""
app.py
------
Basic FastAPI application for symptom classification with robust threshold handling
and integrated fallback logic.

This API endpoint calls the 'classify_with_fallback' function which:
  1. Uses a high-confidence threshold to return a single department.
  2. If the confidence is low, it may either return multiple candidate departments 
     or trigger a naive keyword-based fallback.
"""

import sys, os
# Ensure that the project root is in the Python path so that 'src' is found.
sys.path.insert(0, os.path.abspath("."))

from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import classify_with_fallback

app = FastAPI()

class SymptomInput(BaseModel):
    symptoms: str

@app.post("/classify")
def classify(input_data: SymptomInput):
    """
    Accepts a JSON payload with a 'symptoms' key.
    
    Example Request:
    {
        "symptoms": "I have fever, chills, and body aches."
    }

    Returns a JSON response with the predicted department, confidence score,
    any candidate suggestions, and a fallback message if applicable.
    """
    result = classify_with_fallback(input_data.symptoms)
    return result

@app.get("/")
def root():
    return {"message": "Welcome to the Symptom Classifier API!"}
