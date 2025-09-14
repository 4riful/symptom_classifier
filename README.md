# 🩺 Symptom Classifier

A machine learning system for classifying patient symptoms into appropriate hospital departments using ClinicalBERT and a hybrid fallback mechanism. Built with FastAPI for real-time inference, this project streamlines triage and healthcare workflow automation.

---

## 🌟 Features

- **ClinicalBERT-based classification**: Fine-tunes the Bio_ClinicalBERT model for robust symptom to department mapping.
- **Hybrid fallback logic**: If model confidence is low, a rule-based system ensures a department is suggested.
- **Production-ready API**: FastAPI-powered HTTP endpoints for real-time inference.
- **Extensive data pipeline**: From raw text to cleaned, encoded datasets and label mapping.
- **Complete training, evaluation, and inference scripts**.
- **Configurable and extensible** architecture.

---

## 📂 Directory Structure

```
.
├── data/             # Raw and processed datasets
│   ├── raw/
│   └── processed/
├── models/           # Fine-tuned models and checkpoints
├── notebooks/        # Jupyter notebooks for experiments/prototyping
├── src/              # Main source code
│   ├── config.py           # Centralized configs
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── fallback.py
│   └── utils.py
├── logs/             # Training/eval logs
├── venv_setup/       # Environment setup scripts/notes
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Create and activate a virtual environment (Python 3.9+ recommended)
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

- Place your raw symptom CSV as `data/raw/symptoms.csv` in the format:
  ```
  symptoms,department
  "itching, rash, ...",Dermatology
  "stomach pain, ...",Gastroenterology
  ...
  ```

- Run the data preprocessing:
  ```bash
  python src/data_preprocessing.py
  ```
  This will clean data, encode labels, and split into train/val/test sets in `data/processed/`.

### 3. Train the Model

```bash
python src/train.py
```
- The model will be fine-tuned and checkpoints/logs will be saved in `models/`.

### 4. Evaluate the Model

```bash
python src/evaluate.py
```
- Outputs accuracy, precision, recall, and F1 on your test set.

### 5. Run Inference API

```bash
uvicorn app:app --reload
```
- (Update `app.py` or use FastAPI example below.)

---

## 🧠 How it Works

- **Data Preprocessing**: Cleans, encodes, and splits symptom/department pairs.
- **Model Training**: Fine-tunes ClinicalBERT for multi-class text classification.
- **Evaluation**: Computes key metrics on held-out data.
- **Inference Logic**:
  - If ClinicalBERT predicts with high confidence, returns a single department.
  - For ambiguous cases, returns candidate departments.
  - If all confidences are low, falls back to a rule-based keyword system (see `src/fallback.py`).

---

## 🛠️ API Example (with FastAPI)

```python
from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import classify_with_fallback

app = FastAPI()

class SymptomRequest(BaseModel):
    symptoms: str

@app.post("/classify")
def classify_endpoint(request: SymptomRequest):
    """
    Input: {"symptoms": "itching, skin rash"}
    Output: {"department": "Dermatology", "confidence": 0.92, ...}
    """
    result = classify_with_fallback(request.symptoms)
    return result
```

---

## 📝 Configuration

Centralized in `src/config.py`:
- Paths for data/models/logs
- Model name (default: `emilyalsentzer/Bio_ClinicalBERT`)
- Hyperparameters (epochs, batch size, learning rate, max sequence length, etc.)
- Department label list

---

## 📊 Dataset Example

See `dataset.csv` for a sample:

```csv
symptoms,department
"itching, skin rash, nodal skin eruptions, dischromic  patches",Dermatology
"stomach pain, acidity, ulcers on tongue, vomiting, cough, chest pain",Gastroenterology
...
```

---

## ✅ Tips & Best Practices

- For best results, use high-quality, well-labeled symptom data.
- Fine-tune thresholds in `src/inference.py` for your use case.
- Extend the fallback logic for better real-world coverage.
- Use GPU for training and inference if available.

---

## 🤝 Contributing

Contributions are welcome! Open issues or pull requests for enhancements or bugfixes.

---

## 📄 License

[MIT License](LICENSE)

---

## 🔗 Further Reading

- [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Transformers (Hugging Face)](https://huggingface.co/docs/transformers/index)
