"""
Main entry point for the Network Security ML Pipeline.

This script orchestrates the end-to-end machine learning pipeline including:
1. Data Ingestion - Fetching data from MongoDB and splitting into train/test
2. Data Validation - Schema checks and drift detection
3. [Future stages: Data Transformation, Model Training, Evaluation]

The pipeline uses a modular architecture where each component:
- Takes a configuration object
- Performs its designated task
- Returns artifacts (paths/metadata) for the next stage
"""

# Project component imports
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
# Configuration and artifact entity imports
from networksecurity.entity.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataValidationConfig,
    DataTransformationConfig
)

# Utility imports
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import sys


def new_func(data_ingestion_config):
    """
    Helper function to pass config (currently just a passthrough).
    
    Args:
        data_ingestion_config: DataIngestionConfig object
        
    Returns:
        DataIngestionConfig: Same config object
    """
    return data_ingestion_config


if __name__ == "__main__":
    """
    Main execution block for the ML pipeline.
    
    Pipeline Flow:
        1. Initialize master training pipeline configuration
        2. Run data ingestion (fetch and split data)
        3. Run data validation (schema and drift checks)
        4. [Future: transformation, training, evaluation]
    """
    try:
        # ============================================================
        # STEP 1: Initialize Pipeline Configuration
        # ============================================================
        # Create master configuration with timestamp for artifact versioning
        training_pipeline_config = TrainingPipelineConfig()
        
        # ============================================================
        # STEP 2: Data Ingestion
        # ============================================================
        # Configure data ingestion with database and file paths
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        
        # Initialize and run data ingestion component
        data_ingestion = DataIngestion(new_func(data_ingestion_config))
        logging.info("Initiating data ingestion process")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed")
        print(data_ingestion_artifact)  # Display train/test file paths

        # ============================================================
        # STEP 3: Data Validation
        # ============================================================
        # Configure validation with paths for validated data and reports
        data_validation_config = DataValidationConfig(training_pipeline_config)
        
        # Initialize validation component with ingestion artifacts
        data_validation = DataValidation(data_validation_config, data_ingestion_artifact)
        logging.info('Initiating data validation')
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        print(data_validation_artifact)  # Display validation results

        # ============================================================
        # STEP 4: Data Transformation
        # ============================================================
        # Configure data transformation with paths for transformed data
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        
        # Initialize and run data transformation component
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        logging.info("Data Transformation started")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data Transformation completed")
        print(data_transformation_artifact)  # Display transformation results

        # ============================================================
        # Future Pipeline Stages:
        # - Model Training (ML model fitting)
        # - Model Evaluation (performance metrics)
        # - Model Deployment (production serving)
        # ============================================================

    except Exception as e:
        # Catch any exception and wrap in custom NetworkSecurityException
        raise NetworkSecurityException(e, sys)

