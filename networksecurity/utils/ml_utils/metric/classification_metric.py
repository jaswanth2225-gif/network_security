from networksecurity.entity.artifact_entity import ClassificationMetricArtifact  # Dataclass to store metric results
from networksecurity.exception.exception import NetworkSecurityException  # Custom exception handler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score  # Classification performance metrics
import numpy as np  # NumPy for array operations
import sys  # System operations for error handling


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Calculate classification metrics (accuracy, precision, recall, F1-score).
    
    Metrics:
    - accuracy = % of correct predictions (but misleading if classes imbalanced)
    - precision = Of positive predictions, how many were correct (TP / (TP+FP))
    - recall = Of actual positives, how many did we catch (TP / (TP+FN))
    - F1-score = Harmonic mean of precision and recall (balanced metric)
    - pos_label = Which label is "positive" class (for binary classification)
    """
    try:
        labels = np.unique(y_true)  # Find unique class labels (e.g., [-1, 0] or [0, 1])
        if len(labels) == 2:  # Binary classification (two classes)
            pos_label = labels.max()  # pos_label = larger value is positive (e.g., if [-1,0] then 0 is positive)
            # average='binary' = Standard calculation for binary problems
            model_f1_score = f1_score(y_true, y_pred, average='binary', pos_label=pos_label)
            model_precision = precision_score(y_true, y_pred, average='binary', pos_label=pos_label)
            model_recall = recall_score(y_true, y_pred, average='binary', pos_label=pos_label)
        else:
            # Multi-class classification (more than 2 classes)
            # average='weighted' = Weight each class by its support (number of samples)
            model_f1_score = f1_score(y_true, y_pred, average='weighted')
            model_precision = precision_score(y_true, y_pred, average='weighted')
            model_recall = recall_score(y_true, y_pred, average='weighted')

        model_accuracy = accuracy_score(y_true, y_pred)  # accuracy = (TP + TN) / (TP+TN+FP+FN)

        # Create artifact object to store all metrics together
        classification_metric_artifact = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            accuracy=model_accuracy,
            precision=model_precision,
            recall=model_recall
        )

        return classification_metric_artifact

    except Exception as e:
        raise NetworkSecurityException(e, sys)