"""

1)training_pipeline is __init__.py(constant)
2) import os means joins the build file and folder paths
3) datetime is used to get current date and time

Configuration Entity Module.

This module defines configuration classes for each stage of the ML pipeline.
Each config class encapsulates paths, parameters, and settings needed by
its corresponding pipeline component.

Configuration classes serve to:
1. Centralize all file paths and parameters
2. Ensure consistent directory structure across pipeline runs
3. Enable versioned artifacts through timestamps
4. Decouple configuration from business logic
"""

from datetime import datetime
import os
from networksecurity.constant import training_pipeline

# Display pipeline configuration on module load (for debugging)
print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACT_DIR)


class TrainingPipelineConfig:
    """
    Master configuration for the entire training pipeline.
    
    This is the root configuration that creates a timestamped artifact
    directory for the current pipeline run. All subsequent stage configs
    derive their paths from this master config.
    
    Attributes:
        pipeline_name (str): Name of the ML pipeline
        artifact_name (str): Base directory name for artifacts
        artifact_dir (str): Timestamped directory for this pipeline run
        model_dir (str): Directory for final trained models
        timestamp (str): Timestamp string for versioning
        
    Example:
        >>> config = TrainingPipelineConfig()
        >>> print(config.artifact_dir)
        'Artifacts/12_23_2025_14_30_45'
    """
    
    def __init__(self, timestamp=datetime.now()):
        """
        Initialize master pipeline configuration.
        
        Args:
            timestamp (datetime, optional): Timestamp for versioning.
                                           Defaults to current time.
        """
        # Format timestamp as: MM_DD_YYYY_HH_MM_SS
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        
        # Load pipeline constants
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        
        # Create versioned artifact directory: Artifacts/<timestamp>
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        
        # Directory for final production models
        self.model_dir = os.path.join("final_model")
        
        # Store timestamp for reference
        self.timestamp: str = timestamp


class DataIngestionConfig:
    """
    Configuration for the Data Ingestion stage.
    
    This config defines all paths and parameters needed for:
    - Fetching data from MongoDB
    - Storing raw data in feature store
    - Splitting data into train/test sets
    
    Attributes:
        data_ingestion_dir (str): Base directory for ingestion artifacts
        feature_store_file_path (str): Path to raw data CSV (feature store)
        training_file_path (str): Path to training dataset CSV
        testing_file_path (str): Path to testing dataset CSV
        train_test_split_ratio (float): Proportion of data for testing (e.g., 0.2)
        collection_name (str): MongoDB collection name to fetch from
        database_name (str): MongoDB database name
        
    Example:
        >>> master_config = TrainingPipelineConfig()
        >>> config = DataIngestionConfig(master_config)
        >>> print(config.training_file_path)
        'Artifacts/12_23_2025_14_30_45/data_ingestion/ingested/train.csv'
    """
    
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize data ingestion configuration.
        
        Args:
            training_pipeline_config: Master pipeline config with artifact directory
        """
        # Base directory for all ingestion outputs
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME
        )
        
        # Path to raw data CSV in feature store
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FILE_NAME
        )
        
        # Path to training dataset
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME
        )
        
        # Path to testing dataset
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME
        )
        
        # Train-test split ratio (e.g., 0.2 means 20% test, 80% train)
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        
        # MongoDB collection and database names
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME

class DataValidationConfig:
    """
    Configuration for the Data Validation stage.
    
    This config defines paths for:
    - Validated (clean) data storage
    - Invalid (rejected) data storage
    - Drift detection reports
    
    Attributes:
        data_validation_dir (str): Base directory for validation artifacts
        valid_data_dir (str): Directory for validated datasets
        invalid_data_dir (str): Directory for invalid datasets
        valid_train_file_path (str): Path to validated training CSV
        valid_test_file_path (str): Path to validated testing CSV
        invalid_train_file_path (str): Path to rejected training CSV
        invalid_test_file_path (str): Path to rejected testing CSV
        drift_report_file_path (str): Path to drift analysis YAML report
    """
    
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize data validation configuration.
        
        Args:
            training_pipeline_config: Master pipeline config with artifact directory
        """
        # Base directory for validation outputs
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME
        )
        
        # Directory for validated (passed) datasets
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_VALID_DIR
        )
        
        # Directory for invalid (failed) datasets
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_INVALID_DIR
        )
        
        # Paths for validated datasets
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.TRAIN_FILE_NAME
        )
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.TEST_FILE_NAME
        )
        
        # Paths for invalid datasets (currently unused but available)
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TRAIN_FILE_NAME
        )
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TEST_FILE_NAME
        )
        
        # Path to drift detection report (YAML file with p-values)
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )

class DataTransformationConfig:
    """
    Configuration for the Data Transformation stage.
    
    This config defines paths for:
    - Transformed (preprocessed) datasets
    - Preprocessing objects (scalers, encoders)
    
    Attributes:
        data_transformation_dir (str): Base directory for transformation artifacts
        transformed_train_file_path (str): Path to transformed training array (.npy)
        transformed_test_file_path (str): Path to transformed testing array (.npy)
        transformed_object_file_path (str): Path to preprocessing pipeline object
    """
    
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize data transformation configuration.
        
        Args:
            training_pipeline_config: Master pipeline config with artifact directory
        """
        # Base directory for transformation outputs
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )
        
        # Path to transformed training data (NumPy array format)
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy"),  # Convert .csv to .npy
        )
        
        # Path to transformed testing data (NumPy array format)
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy"),  # Convert .csv to .npy
        )
        
        # Path to preprocessing pipeline object (pickled scikit-learn pipeline)
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME,
        )


class ModelTrainerConfig:
    """
    Configuration for the Model Training stage.
    
    This config defines:
    - Where to save trained models
    - Model performance thresholds
    - Overfitting/underfitting detection parameters
    
    Attributes:
        model_trainer_dir (str): Base directory for model training artifacts
        trained_model_file_path (str): Path to save trained model (.pkl)
        expected_accuracy (float): Minimum acceptable model accuracy
        overfitting_underfitting_threshold (float): Max acceptable difference
                                                    between train and test scores
    """
    
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize model trainer configuration.
        
        Args:
            training_pipeline_config: Master pipeline config with artifact directory
        """
        # Base directory for model training outputs
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        
        # Path to save the trained model file
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
            training_pipeline.MODEL_FILE_NAME
        )
        
        # Minimum acceptable accuracy for the model (e.g., 0.6 = 60%)
        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        
        # Maximum acceptable difference between train and test accuracy
        # Used to detect overfitting (e.g., 0.05 = 5% max difference)
        self.overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD