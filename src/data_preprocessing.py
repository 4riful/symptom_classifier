# Author: Your Name
# Created: 2025-01-19
# Description: Data preprocessing for symptom classification model.

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import logging
import re

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define file paths
RAW_DATA_PATH = "data/raw/symptoms.csv"
PROCESSED_DIR = "data/processed/"
TRAIN_FILE = os.path.join(PROCESSED_DIR, "train.csv")
VAL_FILE = os.path.join(PROCESSED_DIR, "val.csv")
TEST_FILE = os.path.join(PROCESSED_DIR, "test.csv")

# Function to validate the input CSV
def validate_input_data(df):
    """Check if the required columns are present and handle missing data."""
    required_columns = ['symptoms', 'department']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input CSV is missing required columns: {required_columns}")
    
    if df.isnull().any().any():
        logger.warning("Missing values found in data. Dropping rows with missing values.")
        df = df.dropna()

    return df

# Function to clean and preprocess the data
def clean_data(df):
    """Apply cleaning operations to the data."""
    logger.info("Cleaning data...")
    
    # Clean symptoms: Convert to lowercase, strip spaces, collapse multiple spaces
    df['symptoms'] = df['symptoms'].str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)
    
    # Clean department labels: Remove leading/trailing spaces and unify case if needed
    df['department'] = df['department'].str.strip()
    
    # Remove duplicates after cleaning
    df = df.drop_duplicates()

    return df

# Function to encode department labels
def encode_labels(df):
    """Encode the department labels."""
    logger.info("Encoding department labels...")
    
    # Detect and resolve duplicate or inconsistent labels before encoding
    department_counts = df['department'].value_counts()
    logger.info(f"Department label distribution after cleaning: \n{department_counts}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['department'])

    # Log and save the updated label mapping
    unique_labels = len(label_encoder.classes_)
    logger.info(f"Found {unique_labels} unique department labels after normalization.")
    
    label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    save_label_mapping(label_mapping)
    
    return df, label_encoder


# Function to save the label mapping
def save_label_mapping(label_mapping):
    """Save label mapping as a JSON file."""
    logger.info("Saving label mapping...")
    # Ensure the keys in the mapping are integers
    int_key_mapping = {int(k): v for k, v in label_mapping.items()}
    with open(os.path.join(PROCESSED_DIR, "label_mapping.json"), "w") as f:
        json.dump(int_key_mapping, f, indent=4)

# Function to split data into train, validation, and test sets
def split_data(df):
    """Split the data into train, validation, and test sets."""
    logger.info("Splitting data into train, validation, and test sets...")
    train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['label'])
    return train_data, val_data, test_data

# Function to save the processed data
def save_data(train_data, val_data, test_data):
    """Save train, validation, and test sets to CSV files."""
    logger.info("Saving processed data...")
    train_data.to_csv(TRAIN_FILE, index=False)
    val_data.to_csv(VAL_FILE, index=False)
    test_data.to_csv(TEST_FILE, index=False)

# Main function
def main():
    """Main data preprocessing function."""
    # Create processed directory if it doesn't exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    try:
        # Step 1: Load CSV data
        logger.info("Loading raw data...")
        if not os.path.exists(RAW_DATA_PATH):
            raise FileNotFoundError(f"Raw data file not found at: {RAW_DATA_PATH}")

        df = pd.read_csv(RAW_DATA_PATH)

        # Step 2: Validate and clean data
        df = validate_input_data(df)
        df = clean_data(df)

        # Step 3: Encode labels
        df, label_encoder = encode_labels(df)

        # Step 4: Split the data
        train_data, val_data, test_data = split_data(df)

        # Step 5: Save the processed files
        save_data(train_data, val_data, test_data)

        logger.info("Data preprocessing completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred during data preprocessing: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
