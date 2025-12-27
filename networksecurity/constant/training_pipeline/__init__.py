"""
SQL	       MongoDB
Database	Database (same)
Table	    Collection
Row	        Document
Column	    Field/Key


Training Pipeline Constants Module.

WHY constants in one place?
- Consistency (same values used everywhere)
- Easy updates (change value once, everywhere updates)
- Avoids hardcoding magic numbers in code
- Makes code more readable ("TARGET_COLUMN" vs "Result")
"""

import os  # For file path operations
import numpy as np  # NumPy for numerical arrays (used in DATA_TRANSFORMATION_IMPUTER_PARAMS)
# sys and pandas imports removed - not used in this constants file

# ============================================================
# GENERAL PIPELINE CONSTANTS
# ============================================================

TARGET_COLUMN = "Result"  # Column name being predicted (phishing=1, legitimate=0)

PIPELINE_NAME: str = "NetworkSecurity"  # Pipeline identifier for logging and tracking

ARTIFACT_DIR: str = "Artifacts"  # Folder where all outputs are saved (timestamped subfolders)

FILE_NAME: str = "phisingData.csv"  # Raw data file name

TRAIN_FILE_NAME: str = "train.csv"  # Training data file (after data ingestion)
TEST_FILE_NAME: str = "test.csv"  # Testing data file (after data ingestion)

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")  # Path to schema validation file

# Directory for production-ready models
SAVED_MODEL_DIR = os.path.join("saved_models")

# Trained model filename (pickle format)
MODEL_FILE_NAME = "model.pkl"


# ============================================================
# DATA INGESTION CONSTANTS
# ============================================================
"""
Data Ingestion stage fetches data from MongoDB and splits it
into training and testing datasets.
"""

# ============================================================
# DATA INGESTION CONSTANTS
# ============================================================
"""
Data Ingestion stage fetches data from MongoDB and splits it
into training and testing datasets.
"""

DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"  # MongoDB collection name (table with 9000 phishing records)

DATA_INGESTION_DATABASE_NAME: str = "KRISHAI"  # MongoDB database name (folder containing collections (tables))

DATA_INGESTION_DIR_NAME: str = "data_ingestion"  # Folder name inside Artifacts to store ingestion outputs

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"  # Subfolder storing raw data backup (for traceability)

DATA_INGESTION_INGESTED_DIR: str = "ingested"  # Subfolder storing split train/test data

DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2  # 0.2 = 20% test, 80% train (80/20 split)


# ============================================================
# DATA VALIDATION CONSTANTS
# ============================================================
"""
Data Validation stage checks data quality, schema compliance,
and detects data drift between train and test sets.
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"  # Folder name inside Artifacts for validation artifacts

DATA_VALIDATION_VALID_DIR: str = "validated"  # Subfolder for data that passed all validation checks

DATA_VALIDATION_INVALID_DIR: str = "invalid"  # Subfolder for data that failed validation checks

DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"  # Subfolder for drift analysis reports

DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"  # Drift report filename (YAML format)

PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"  # Preprocessing pipeline filename (pickle format)


# ============================================================
# DATA TRANSFORMATION CONSTANTS
# ============================================================
"""
Data Transformation stage handles feature engineering,
imputation (filling missing values), and conversion to numpy arrays.
"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"  # Folder name inside Artifacts for transformation outputs

DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"  # Subfolder for transformed numpy arrays (train.npy, test.npy)

DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"  # Subfolder for transformation objects (KNN imputer, etc.)

# KNN Imputer settings (fills missing values using K-nearest neighbors)
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,  # np.nan = Missing value indicator (NaN = Not a Number)
    "n_neighbors": 3,          # K=3 = Use 3 nearest neighbors to estimate missing values
    "weights": "uniform",      # All 3 neighbors vote equally (no weighted voting)
}

DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"  # Transformed training data filename (.npy = numpy format)
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"  # Transformed testing data filename (.npy = numpy format)


# ============================================================
# MODEL TRAINER CONSTANTS
# ============================================================
"""
Model Trainer stage handles model training, evaluation,
and performance validation.
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"  # Folder name inside Artifacts for trained model

MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"  # Subfolder for the final trained model file

MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"  # Trained model filename (pickle format)

MODEL_TRAINER_EXPECTED_SCORE: float = 0.6  # Minimum acceptable accuracy (60% = 0.60)

MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05  # Max difference between train/test accuracy (5%)
# If train accuracy >> test accuracy (by >5%), model is overfitting (memorized training data)

TRAINING_BUCKET_NAME = "netwworksecurity"  # Cloud storage bucket for model deployment