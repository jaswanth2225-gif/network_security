# === IMPORTS SECTION ===
import sys
import os
import pandas as pd  # For loading CSV files
import numpy as np  # For array operations (combining features with target)
from sklearn.impute import KNNImputer  # K-Nearest Neighbors: fills missing values using nearby data points
from sklearn.pipeline import Pipeline  # Container that chains preprocessing steps together

# Import constants (predefined values used across project)
from networksecurity.constant.training_pipeline import TARGET_COLUMN  # Name of target column (e.g., "Result")
from networksecurity.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS  # Settings for KNN (e.g., K=3)

from networksecurity.entity.artifact_entity import(
    DataTransformationArtifact,
    DataValidationArtifacts
)

from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    """
    WHAT THIS CLASS DOES: Prepares data for machine learning models
    
    TRANSFORMATIONS PERFORMED:
    1. Separates features (X) from target (y)
    2. Fills missing values using KNN imputation
    3. Converts DataFrames to numpy arrays (faster for model training)
    4. Saves preprocessing object for future predictions
    
    ANALOGY: Like a chef preparing ingredients:
    - Raw vegetables (raw data) → Cleaned, chopped, measured (transformed data)
    - Recipe card saved (preprocessing object) so you can prepare same way next time
    
    WHY THIS MATTERS:
    - ML models need clean, numerical data (no missing values)
    - Models train faster on numpy arrays than pandas DataFrames
    - Must use same preprocessing on future data (saved as pickle file)
    """
    
    def __init__(self, data_validation_artifact: DataValidationArtifacts,
                 data_transformation_config: DataTransformationConfig):
        """
        Initialize the transformation component
        
        PARAMETERS:
            data_validation_artifact = object with paths to validated train.csv and test.csv
            data_transformation_config = object with paths for saving transformed data
        """
        try:
            # Save artifacts so other methods can access them
            self.data_validation_artifact: DataValidationArtifacts = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Utility method to read a CSV file
        
        @staticmethod = Doesn't need access to self (works independently)
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def get_data_transformation_object(self) -> Pipeline:
        """
        WHAT THIS METHOD DOES: Creates a preprocessing pipeline with KNN Imputer
        
        KNN IMPUTER EXPLAINED:
        - KNN = K-Nearest Neighbors
        - K=3 means "look at 3 closest data points"
        - If a value is missing, average the values from 3 nearest neighbors
        
        EXAMPLE:
        - Row 100 has missing "byte_count"
        - Find 3 most similar rows (based on other features)
        - If neighbors have byte_count = [500, 600, 550]
        - Fill missing value with average: (500+600+550)/3 = 550
        
        WHY K=3:
        - Too small (K=1): Can be influenced by outliers (extreme values)
        - Too large (K=10): Over-smooths, loses patterns
        - K=3: Good balance (industry standard for this type of data)
        
        PIPELINE EXPLAINED:
        - Pipeline = Container that chains steps together
        - WHY: Can save entire pipeline as one object and reuse later
        """
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)  # ** unpacks dict into keyword arguments
            pipeline = Pipeline([("imputer", imputer)])  # Create pipeline with one step
            return pipeline
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        WHAT THIS METHOD DOES: The main method that transforms raw data into ML-ready format
        
        WORKFLOW (step by step):
        1. Load validated train.csv and test.csv
        2. Separate features (X) from target (y) for both datasets
        3. Create KNN Imputer and FIT it on training features only
        4. TRANSFORM both train and test using the fitted imputer
        5. Combine transformed features with target (stack horizontally)
        6. Save as numpy arrays (.npy files) for fast loading
        7. Save the fitted imputer for future predictions
        
        KEY CONCEPT - Data Leakage Prevention:
        - Imputer is FIT on training data ONLY (learns neighbors from train)
        - Then TRANSFORM is applied to both train and test
        - WHY: Test data can't influence the imputation (prevents cheating)
        
        ANALOGY:
        - Training = Learning phase (imputer learns patterns)
        - Testing = Exam phase (use learned patterns, don't learn new ones)
        
        NUMPY ARRAY FORMAT:
        - Train array shape: (7200 rows, 31 columns)
        - Columns: [feature1, feature2, ..., feature30, target]
        - Target in last column for easy splitting later
        
        RETURNS: DataTransformationArtifact with paths to:
            - Transformed train.npy
            - Transformed test.npy  
            - Preprocessing object (pickle file)
        """
        logging.info("Entered the initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            
            # STEP 1: Load validated CSV files from previous stage
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # STEP 2A: Separate features and target for TRAINING data
            # drop() removes the target column, leaving only features
            # EXAMPLE: If TARGET_COLUMN = "Result", this removes "Result" column
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)  # Features (30 columns)
            target_feature_train_df = train_df[TARGET_COLUMN]  # Target (1 column)
            
            # Normalize target values: replace 1 with 0
            # WHY: This project uses labels [-1, 0] instead of [0, 1]
            # BEFORE: [1, -1, 1, -1] → AFTER: [0, -1, 0, -1]
            target_feature_train_df = target_feature_train_df.replace(1, 0)

            # STEP 2B: Separate features and target for TESTING data (same process)
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(1, 0)

            # STEP 3: Get preprocessing pipeline (KNN Imputer with K=3)
            preprocessor_object = self.get_data_transformation_object()
            
            # STEP 4: FIT the imputer on training features ONLY
            # fit() = Learn patterns from training data (find neighbors, calculate means, etc.)
            # CRITICAL: We ONLY fit on training data to prevent data leakage
            preprocessor_object = preprocessor_object.fit(input_feature_train_df)
            
            # STEP 5: TRANSFORM both datasets using the FITTED imputer
            # transform() = Apply learned patterns (fill missing values using neighbors found during fit)
            # Training data: Fill missing values using neighbors from training data
            # Testing data: Fill missing values using neighbors from training data (NOT test data!)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            # STEP 6: Combine features with target using np.c_[]
            # np.c_[] = Horizontal stack (column-wise concatenation)
            # BEFORE: features=[30 cols], target=[1 col] → AFTER: combined=[31 cols]
            # Target goes in LAST column for easy separation later: data[:, :-1]=features, data[:, -1]=target
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # STEP 7: Save arrays as .npy files (NumPy's binary format - faster than CSV)
            # WHY .npy: Loads 10-100x faster than CSV for large datasets
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            
            # STEP 8: Save the preprocessing object as pickle file
            # WHY: Future predictions need SAME imputer (same neighbors, same K=3)
            # PICKLE = Python's way to save objects to disk (serialization)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)
            
            # Also save to final_model directory for deployment
            save_object("final_model/preprocessor.pkl", preprocessor_object)

            # WHY return artifact: Model trainer needs paths to arrays and preprocessor
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file=self.data_transformation_config.transformed_test_file_path
            )

            logging.info("Exited the initiate_data_transformation method of DataTransformation class")
            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
