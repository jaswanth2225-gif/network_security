"""
Artifact Entity Module

This module defines data classes (using @dataclass) that represent the output
artifacts of each pipeline stage. Artifacts are metadata/paths that describe
where processed data is stored and what the results of each stage are.

Artifacts serve as:
1. Output of one pipeline stage
2. Input to the next pipeline stage
3. Traceable records of pipeline execution
"""

from dataclasses import dataclass


@dataclass
class DataIngestionArtifacts:
    """
    Artifacts produced by the Data Ingestion stage.
    
    This class holds paths to the training and testing datasets
    created during data ingestion. These paths are used by subsequent
    validation and transformation stages.
    
    Attributes:
        trained_file_path (str): Absolute path to the training CSV file
        tested_file_path (str): Absolute path to the testing CSV file
        
    Example:
        >>> artifacts = DataIngestionArtifacts(
        ...     trained_file_path='Artifacts/12_23_2025/data_ingestion/train.csv',
        ...     tested_file_path='Artifacts/12_23_2025/data_ingestion/test.csv'
        ... )
    """
    trained_file_path: str
    tested_file_path: str


@dataclass
class DataValidationArtifacts:
    """
    Artifacts produced by the Data Validation stage.
    
    This class captures the results of data validation including:
    - Overall validation status (whether data passed validation)
    - Paths to validated (clean) data files
    - Paths to invalid (rejected) data files
    - Path to drift detection report
    
    Attributes:
        validation_status (bool): True if no drift detected, False if drift found
        valid_train_file_path (str): Path to validated training data CSV
        valid_test_file_path (str): Path to validated testing data CSV
        invalid_train_file_path (str): Path to rejected training data (if any)
        invalid_test_file_path (str): Path to rejected testing data (if any)
        drift_report_file_path (str): Path to YAML file with drift analysis
        
    Example:
        >>> artifacts = DataValidationArtifacts(
        ...     validation_status=True,
        ...     valid_train_file_path='Artifacts/data_validation/validated/train.csv',
        ...     valid_test_file_path='Artifacts/data_validation/validated/test.csv',
        ...     invalid_train_file_path=None,
        ...     invalid_test_file_path=None,
        ...     drift_report_file_path='Artifacts/data_validation/drift_report/report.yaml'
        ... )
    """
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    """
    Artifacts produced by the Data Transformation stage.
    
    This class holds paths to the preprocessed/transformed data
    and the preprocessing object (scaler, imputer, etc.) needed
    for model training and inference.
    
    Attributes:
        transformed_object_file (str): Path to the serialized preprocessing object
        transformed_train_file (str): Path to transformed training data
        transformed_test_file (str): Path to transformed testing data
    """
    transformed_object_file: str
    transformed_train_file: str
    transformed_test_file: str
