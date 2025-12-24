"""

1) sys means It provides access to system-specific parameters and functions.

Commonly used for interacting with the Python interpreter, command-line arguments, and system-level operations.



Training Pipeline Constants Module.

This module centralizes all constants, file names, directory names, and
configuration parameters used throughout the ML pipeline. Having constants
in one place ensures consistency and makes updates easier.

The constants are organized by pipeline stage:
- General pipeline settings
- Data Ingestion
- Data Validation
- Data Transformation
- Model Training
"""

import os
import sys
import numpy as np
import pandas as pd

# ============================================================
# GENERAL PIPELINE CONSTANTS
# ============================================================

# Target column name in the dataset
TARGET_COLUMN = "Result"

# Pipeline name for identification and logging
PIPELINE_NAME: str = "NetworkSecurity"

# Base directory where all artifacts are stored
ARTIFACT_DIR: str = "Artifacts"

# Raw data CSV filename
FILE_NAME: str = "phisingData.csv"

# Train and test split filenames
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

# Path to schema configuration file
SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

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

# MongoDB collection name containing the phishing data
DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"

# MongoDB database name
DATA_INGESTION_DATABASE_NAME: str = "KRISHAI"

# Directory name for data ingestion artifacts
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

# Subdirectory for raw data (feature store)
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

# Subdirectory for split train/test data
DATA_INGESTION_INGESTED_DIR: str = "ingested"

# Train-test split ratio (0.2 = 20% test, 80% train)
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2


# ============================================================
# DATA VALIDATION CONSTANTS
# ============================================================
"""
These are the inputs we are giving for data validation stage
Data Validation stage checks data quality, schema compliance,
and detects data drift between train and test sets.
"""

# Directory name for validation artifacts (Data Validation stage Dir)
DATA_VALIDATION_DIR_NAME: str = "data_validation"

# Subdirectory for validated (clean) data 
DATA_VALIDATION_VALID_DIR: str = "validated"

# Subdirectory for invalid (rejected) data
DATA_VALIDATION_INVALID_DIR: str = "invalid"

# Subdirectory for drift analysis reports 
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"

# Drift report filename (YAML format)  Drift Report File Path
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

# Preprocessing pipeline object filename
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"


# ============================================================
# DATA TRANSFORMATION CONSTANTS 
# ============================================================
"""
Data Transformation stage handles feature engineering,
scaling, encoding, and imputation.
"""

# Directory name for transformation artifacts
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"

# Subdirectory for transformed datasets (NumPy arrays)
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"

# Subdirectory for transformation objects (scalers, encoders)
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# KNN Imputer parameters for handling missing values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,  # Identify NaN as missing
    "n_neighbors": 3,          # Use 3 nearest neighbors for imputation
    "weights": "uniform",      # Equal weight to all neighbors
}

# Transformed data filenames (NumPy format)
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"


# ============================================================
# MODEL TRAINER CONSTANTS
# ============================================================
"""
Model Trainer stage handles model training, evaluation,
and performance validation.
"""

# Directory name for model training artifacts
MODEL_TRAINER_DIR_NAME: str = "model_trainer"

# Subdirectory for trained model files
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"

# Trained model filename
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"

# Minimum acceptable model accuracy (60%)
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6

# Maximum acceptable difference between train and test accuracy
# Used to detect overfitting/underfitting (5% threshold)
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05

# Cloud storage bucket name (for model deployment)
TRAINING_BUCKET_NAME = "netwworksecurity"