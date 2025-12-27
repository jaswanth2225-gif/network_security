# WHY config classes: Centralize all file paths and settings so components don't hardcode paths
# WHY timestamps: Each pipeline run gets its own folder (Artifacts/12_23_2025_14_30_45/)
# WHY versioning: Can compare different runs, keep history, reproduce results

from __future__ import annotations  # WHY: Enable forward references in type hints for reordered classes

from datetime import datetime  # WHY: Generate unique folder names for each run
import os  # WHY: Build cross-platform file paths (Windows vs Linux)
from networksecurity.constant import training_pipeline  # WHY: Load constants (database names, folder names)

# WHY print: Quick debug to see where artifacts will be saved
print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACT_DIR)


class DataIngestionConfig:
    """Paths and settings for fetching data from MongoDB and splitting train/test"""
    
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # WHY join with artifact_dir: Creates Artifacts/12_23_2025_14_30_45/data_ingestion/
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME
        )
        
        # WHY feature_store: Keep snapshot of raw data from MongoDB for reproducibility
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FILE_NAME
        )
        
        # WHY training_file_path: Separate 80% of data for model training
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME
        )
        
        # WHY testing_file_path: Separate 20% of data for model evaluation
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME
        )
        
        # WHY 0.2 ratio: Industry standard 80/20 split
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        
        # WHY save names: Tell data ingestion which MongoDB collection to read
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME


class DataValidationConfig:
    """Paths for validated data and drift reports"""
    
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # WHY join: Creates Artifacts/12_23_2025_14_30_45/data_validation/
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME
        )
        
        # WHY valid_data_dir: Store data that passed schema and drift checks
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_VALID_DIR
        )
        
        # WHY invalid_data_dir: Separate folder for rejected data (not used yet)
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_INVALID_DIR
        )
        
        # WHY valid paths: Transformation stage loads these validated CSVs
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.TRAIN_FILE_NAME
        )
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.TEST_FILE_NAME
        )
        
        # WHY invalid paths: Future enhancement to segregate bad data
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TRAIN_FILE_NAME
        )
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TEST_FILE_NAME
        )
        
        # WHY drift report: YAML file with KS test p-values for each column
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )


class DataTransformationConfig:
    """Paths for transformed numpy arrays and preprocessing objects"""
    
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # WHY join: Creates Artifacts/12_23_2025_14_30_45/data_transformation/
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )
        
        # WHY .npy format: NumPy arrays load faster than CSV for model training
        # WHY replace: Convert train.csv → train.npy
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy"),
        )
        
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy"),
        )
        
        # WHY save preprocessor: Predictions need same transformations (KNN imputer with K=3)
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME,
        )


class ModelTrainerConfig:
    """Paths and thresholds for model training"""
    
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # WHY join: Creates Artifacts/12_23_2025_14_30_45/model_trainer/
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        
        # WHY .pkl format: Pickled model can be loaded for predictions
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
            training_pipeline.MODEL_FILE_NAME
        )
        
        # WHY expected_accuracy: Reject models below 60% accuracy (too weak for production)
        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        
        # WHY overfitting threshold: If train accuracy >> test accuracy, model memorized training data
        # WHY 0.05: Max 5% difference allowed between train and test scores
        self.overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD


class TrainingPipelineConfig:
    """Root config that creates timestamped folder for this pipeline run"""
    
    def __init__(self, timestamp=datetime.now()):
        # WHY timestamp: Each run gets unique folder (Artifacts/12_23_2025_14_30_45/)
        # WHY unique folders: Can compare runs, won't overwrite previous results
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        
        # WHY join: Creates path like Artifacts/12_23_2025_14_30_45/
        # WHY timestamped: Every run is versioned and traceable
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        
        # WHY final_model: Separate folder for production-ready models
        self.model_dir = os.path.join("final_model")
        
        self.timestamp: str = timestamp