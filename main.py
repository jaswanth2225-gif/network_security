"""
==============================================
NETWORK SECURITY ML PIPELINE - MAIN SCRIPT
==============================================

WHAT THIS SCRIPT DOES:
This is the "master controller" that runs the entire machine learning pipeline from start to finish.
Think of it like a factory assembly line where each station does one job.

THE PIPELINE (5 STAGES):
1. Data Ingestion    → Fetch data from MongoDB, split into train/test (80/20)
2. Data Validation   → Check data quality (31 columns?, all numbers?, train vs test similar?)
3. Data Transformation → Fill missing values with KNN, convert to numpy arrays
4. Model Training    → Try 5 algorithms, pick the best one, save it
5. DONE! Now we have a trained model ready to detect network attacks

ANALOGY - Building a Car:
1. Get raw materials from warehouse (Data Ingestion)
2. Quality check materials (Data Validation)
3. Process materials into car parts (Data Transformation)
4. Assemble car and test drive (Model Training)

ARCHITECTURE BENEFITS:
- Modular = Each stage is independent (easy to modify one without breaking others)
- Versioned = Each run gets timestamped folder (Artifacts/12_25_2025_10_30_45/)
- Traceable = Log files show exactly what happened when
"""

# === IMPORTS: The Components (Workers) ===
# Each import is a specialist who handles one stage of the pipeline
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

# === IMPORTS: The Configurations (Instructions) ===
# Each config tells a component WHERE to save files and WHAT settings to use
from networksecurity.entity.config_entity import (
    DataIngestionConfig,        # Instructions for stage 1: database name, file paths, split ratio
    TrainingPipelineConfig,     # Master config: creates timestamped folder for this run
    DataValidationConfig,       # Instructions for stage 2: what makes data "valid"
    DataTransformationConfig,   # Instructions for stage 3: how to fill missing values
    ModelTrainerConfig          # Instructions for stage 4: which models to try, quality thresholds
)

# === IMPORTS: Utilities ===
from networksecurity.exception.exception import NetworkSecurityException  # Custom error handler with detailed messages
from networksecurity.logging.logger import logging  # Writes execution history to logs/pipeline.log
import sys  # Python system module (needed for error details)


# ============================================================
# HELPER FUNCTIONS (Not used much, mostly legacy)
# ============================================================

def new_func(data_ingestion_config):
    """
    Legacy function - not actively used
    Could be used for config validation in future
    """
    return data_ingestion_config


# ============================================================
# MAIN EXECUTION - This is where the pipeline actually runs!
# ============================================================

if __name__ == "__main__":
    """
    WHAT THIS BLOCK DOES: Runs when you execute `python main.py`
    
    __name__ == "__main__" means:
    - This code runs when you execute THIS file directly
    - It WON'T run if someone imports this file as a module
    
    THE PIPELINE FLOW:
    Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → DONE!
    Each step creates "artifacts" (outputs) that the next step needs as inputs
    
    ERROR HANDLING:
    - Everything is wrapped in try-except
    - If ANY step fails, we catch the error, log it, and stop gracefully
    - No silent failures - all errors are logged and displayed
    """
    try:
        # ==============================================================
        # STEP 1: Initialize Master Configuration
        # ==============================================================
        # TrainingPipelineConfig creates the timestamped folder structure
        # EXAMPLE: Creates Artifacts/12_25_2025_10_30_45/
        # WHY: Every run gets its own folder so results don't overwrite each other
        training_pipeline_config = TrainingPipelineConfig()
        logging.info("Pipeline configuration initialized with timestamp: " 
                     f"{training_pipeline_config.timestamp}")
        
        # ==============================================================
        # STEP 2: Data Ingestion - Fetch data from MongoDB
        # ==============================================================
        print("\n" + "="*70)
        print(" "*20 + "STEP 2: DATA INGESTION")
        print("="*70)
        print("PURPOSE: Fetch data from MongoDB and split into train/test")
        print("INPUT: MongoDB (KRISHAI.NetworkData)")
        print("OUTPUT: train.csv (80%) and test.csv (20%)")
        print("="*70 + "\n")
        
        # Create config that tells ingestion WHERE to save files
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        logging.info(f"Data Ingestion Config: {data_ingestion_config}")
        
        # Run data ingestion (fetch 9000 records, split 7200/1800)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiating data ingestion process")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully")
        
        # Show what was created
        print("\n[OK] DATA INGESTION COMPLETE!")
        print(f"  Training file: {data_ingestion_artifact.trained_file_path}")
        print(f"  Testing file: {data_ingestion_artifact.tested_file_path}\n")

        # ==============================================================
        # STEP 3: Data Validation - Quality check the data
        # ==============================================================
        print("\n" + "="*70)
        print(" "*20 + "STEP 3: DATA VALIDATION")
        print("="*70)
        print("PURPOSE: Check data quality before training models")
        print("CHECKS: Column count (31?), Data types (numerical?), Drift (train vs test)")
        print("OUTPUT: Validated CSVs + Drift Report (YAML)")
        print("="*70 + "\n")
        
        """
        Output (DataValidationArtifacts):
        - validation_status: Pass/Fail status
        - valid_train_file_path: Validated training data
        - valid_test_file_path: Validated testing data
        - invalid_train_file_path: Records that failed validation
        - invalid_test_file_path: Records that failed validation
        - drift_report_file_path: YAML file with drift statistics
        
        These paths are passed to the next stage.
        """
        
        # WHY validate: Catch bad data (wrong columns, wrong types, corrupted values) before training
        # WHY check drift: Alert if new data looks very different from training data
        data_validation_config = DataValidationConfig(training_pipeline_config)
        logging.info(f"Data Validation Config: {data_validation_config}")
        
        # WHY pass artifacts: Validation needs the CSV paths from ingestion
        data_validation = DataValidation(data_validation_config, data_ingestion_artifact)
        logging.info('Initiating data validation')
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed successfully")
        
        # Display results
        print("\n" + "="*60)
        print("DATA VALIDATION ARTIFACTS")
        print("="*60)
        print(data_validation_artifact)
        print("="*60 + "\n")

        # ================================================================
        # STEP 4: Data Transformation
        # ================================================================
        # Data Transformation Stage:
        # 
        # Input:
        # - DataValidationArtifacts (validated train and test CSV paths)
        # 
        # Process:
        # - Load validated training and testing CSV files
        # - Separate features (X) from target (y) for both sets
        # - Create preprocessing pipeline with KNN Imputer
        # - Fit imputer on training features (IMPORTANT: fit only on train)
        # - Transform training features using fitted imputer
        # - Transform testing features using same fitted imputer (prevents data leakage)
        # - Combine transformed features with target variable
        # - Save as NumPy arrays (.npy format for efficient loading)
        # - Save preprocessing object (pickle) for model deployment
        # 
        # Output (DataTransformationArtifact):
        # - transformed_object_file: Pickled preprocessing pipeline
        # - transformed_train_file: NumPy array with transformed train data
        # - transformed_test_file: NumPy array with transformed test data
        # 
        # Data Format:
        # - Each row: [feature_1, feature_2, ..., feature_31, target]
        # - Ready for direct input to ML models
        
        # WHY transform: Fill missing values so models don't crash, convert to numbers
        # WHY KNN imputer: Uses nearby values to guess missing data (better than zeros)
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        logging.info(f"Data Transformation Config: {data_transformation_config}")
        
        # WHY fit only on train: Prevents model from "peeking" at test data
        # WHY save preprocessor: Need same transformations for new data in production
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        logging.info("Initiating data transformation")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed successfully")
        
        # Display results
        print("\n" + "="*60)
        print("DATA TRANSFORMATION ARTIFACTS")
        print("="*60)
        print(data_transformation_artifact)
        print("="*60 + "\n")

        # ================================================================
        # STEP 5: Model Training
        # ================================================================
        # Model Training Stage:
        # 
        # Input:
        # - DataTransformationArtifact (transformed train/test .npy + preprocessor)
        # 
        # Process:
        # - Evaluate multiple models with GridSearchCV-like tuning
        # - Select best model by validation score
        # - Wrap with preprocessing object and persist to disk
        # 
        # Output (ModelTrainerArtifact):
        # - trained_model_file_path
        # - train_metric_artifact
        # - test_metric_artifact

        # WHY train multiple models: Don't know which works best until we try them
        # WHY GridSearch: Automatically finds best settings (n_estimators, learning_rate, etc.)
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        logging.info(f"Model Trainer Config: {model_trainer_config}")

        # WHY try 5 models: Random Forest, Gradient Boosting, AdaBoost, Logistic, Decision Tree
        # WHY pick best: Use highest test accuracy to avoid overfitting
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        logging.info("Initiating model training")
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model training completed successfully")

        # WHY print summary: Show which model won and how good it is
        # WHY show train AND test: Spot overfitting (train >> test means memorization)
        print("\n" + "="*60)
        print("MODEL TRAINER ARTIFACTS")
        print("="*60)
        print(model_trainer_artifact)
        print(f"Best Model: {model_trainer_artifact.best_model_name}")  # WHY: Need to know which algorithm worked
        print(f"Train Accuracy: {model_trainer_artifact.train_metric_artifact.accuracy:.4f}")  # WHY: Check if model learned anything
        print(f"Test Accuracy: {model_trainer_artifact.test_metric_artifact.accuracy:.4f}")  # WHY: Real performance on unseen data
        print(f"Train F1: {model_trainer_artifact.train_metric_artifact.f1_score:.4f}")  # WHY: Balance of precision/recall
        print(f"Train Precision: {model_trainer_artifact.train_metric_artifact.precision:.4f}")  # WHY: How many predictions were correct
        print(f"Train Recall: {model_trainer_artifact.train_metric_artifact.recall:.4f}")  # WHY: How many actual threats we caught
        print(f"Test F1: {model_trainer_artifact.test_metric_artifact.f1_score:.4f}")
        print(f"Test Precision: {model_trainer_artifact.test_metric_artifact.precision:.4f}")
        print(f"Test Recall: {model_trainer_artifact.test_metric_artifact.recall:.4f}")
        # Log the same summary to file
        logging.info(f"Best Model: {model_trainer_artifact.best_model_name}")
        logging.info(f"Train Accuracy: {model_trainer_artifact.train_metric_artifact.accuracy:.4f}")
        logging.info(f"Test Accuracy: {model_trainer_artifact.test_metric_artifact.accuracy:.4f}")
        logging.info(f"Train F1: {model_trainer_artifact.train_metric_artifact.f1_score:.4f}")
        logging.info(f"Train Precision: {model_trainer_artifact.train_metric_artifact.precision:.4f}")
        logging.info(f"Train Recall: {model_trainer_artifact.train_metric_artifact.recall:.4f}")
        logging.info(f"Test F1: {model_trainer_artifact.test_metric_artifact.f1_score:.4f}")
        logging.info(f"Test Precision: {model_trainer_artifact.test_metric_artifact.precision:.4f}")
        logging.info(f"Test Recall: {model_trainer_artifact.test_metric_artifact.recall:.4f}")
        print("="*60 + "\n")

        # ================================================================
        # Future Pipeline Stages (Not Yet Implemented)
        # ================================================================
        # Next stages will be added as follows:
        # 
        # STEP 5: Model Training
        # - Load transformed train data from artifacts
        # - Load preprocessing object from artifacts
        # - Train multiple ML models (Random Forest, XGBoost, etc.)
        # - Save trained models
        # 
        # STEP 6: Model Evaluation
        # - Load trained models
        # - Evaluate on transformed test data
        # - Generate performance metrics and reports
        # - Compare model performance
        # 
        # STEP 7: Model Deployment
        # - Select best performing model
        # - Package with preprocessing object
        # - Deploy to production environment
        # - Create inference API
        logging.info("Pipeline execution completed successfully!")
        logging.info(f"All artifacts saved under: {training_pipeline_config.artifact_dir}")

    except Exception as e:
        # WHY catch all errors: Want clear message showing WHICH file and WHICH line failed
        # WHY NetworkSecurityException: Standard Python errors don't show enough detail
        # WHY sys parameter: Gives us traceback info (file path, line number)
        raise NetworkSecurityException(e, sys)
