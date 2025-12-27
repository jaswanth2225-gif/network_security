# WHY artifact classes: Store output paths from each pipeline stage
# WHY dataclasses: Python's built-in way to create simple data containers
# WHY pass data between stages: Ingestion → Validation → Transformation → Training

from dataclasses import dataclass  # WHY: Auto-generate __init__, __repr__, etc.


@dataclass
class DataIngestionArtifacts:
    """
    Output from Data Ingestion: paths to train and test CSVs.
    
    WHY this class: Validation stage needs to know where ingestion saved the files
    """
    trained_file_path: str  # WHY: Path to train.csv (80% of data)
    tested_file_path: str   # WHY: Path to test.csv (20% of data)


@dataclass
class DataValidationArtifacts:
    """
    Output from Data Validation: paths to validated CSVs and drift report.
    
    WHY this class: Transformation stage needs to know where validated files are
    WHY validation_status: Tells pipeline if data has critical drift issues
    """
    validation_status: bool  # WHY: True if no drift, False if drift detected
    valid_train_file_path: str  # WHY: Path to validated train.csv
    valid_test_file_path: str   # WHY: Path to validated test.csv
    invalid_train_file_path: str  # WHY: Path for rejected data (not used yet)
    invalid_test_file_path: str   # WHY: Path for rejected data (not used yet)
    drift_report_file_path: str   # WHY: Path to YAML with KS test results


@dataclass
class DataTransformationArtifact:
    """
    Output from Data Transformation: paths to numpy arrays and preprocessing object.
    
    WHY this class: Model trainer needs transformed data and the fitted preprocessor
    WHY numpy format: Models train faster on arrays than CSVs
    WHY save preprocessor: Predictions need same transformations (KNN imputer with K=3)
    """
    transformed_object_file: str  # WHY: Path to pickled KNN imputer (fitted on train data)
    transformed_train_file: str   # WHY: Path to train.npy (features + target in last column)
    transformed_test_file: str    # WHY: Path to test.npy (features + target in last column)


@dataclass
class ClassificationMetricArtifact:
    """
    Classification metrics for evaluating model performance.
    
    WHY this class: Store all metrics together for train and test sets
    WHY accuracy added: Overall correctness (most intuitive metric)
    """
    f1_score: float    # WHY: Balance between precision and recall
    accuracy: float    # WHY: Percentage of correct predictions
    precision: float   # WHY: Of predicted positives, how many were correct
    recall: float      # WHY: Of actual positives, how many were caught


@dataclass
class ModelTrainerArtifact:
    """
    Output from Model Training: path to best model and performance metrics.
    
    WHY this class: Next stage needs model path and metrics to decide if model is good enough
    WHY best_model_name: Log which algorithm won (Random Forest, Gradient Boosting, etc.)
    """
    trained_model_file_path: str  # WHY: Path to pickled model wrapper (includes preprocessor)
    train_metric_artifact: ClassificationMetricArtifact  # WHY: Performance on training data
    test_metric_artifact: ClassificationMetricArtifact   # WHY: Performance on unseen test data
    best_model_name: str  # WHY: Which algorithm performed best (e.g., "Random Forest")







