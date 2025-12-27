from networksecurity.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME  # Constants for model paths
import os  # File/folder operations
from networksecurity.exception.exception import NetworkSecurityException  # Custom exception handler
import sys  # System operations for error handling

from networksecurity.logging.logger import logging  # Logger for recording operations

class NetworkModel:
    """
    Wrapper class that combines preprocessor + model.
    
    Preprocessor = Transforms raw data (scales, imputes, encodes) before prediction
    Model = ML classifier (Random Forest, Logistic Regression, etc.)
    
    Why combine? So predictions work directly on raw data without manual preprocessing.
    Example: model.predict(raw_data) → automatically preprocesses then classifies
    """
    
    def __init__(self, preprocessor, model):
        """
        Initialize NetworkModel with preprocessor and trained model.
        
        preprocessor = Object with .transform() method (KNN imputer, scaler, encoder, etc.)
        model = Trained classifier with .predict() method
        """
        try:
            self.preprocessor = preprocessor  # Stores the preprocessor object (transforms data)
            self.model = model  # Stores the trained model object (makes predictions)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict(self, X):
        """
        Make predictions on raw data by preprocessing first, then classifying.
        
        Steps:
        1. X_transformed = preprocessor.transform(X)  - Transform raw data (fill NaNs, scale, etc.)
        2. y_hat = model.predict(X_transformed)        - Make predictions on transformed data
        """
        try:
            # Step 1: Apply preprocessor (imputation, scaling, encoding, etc.)
            X_transformed = self.preprocessor.transform(X)  # Transform raw input data
            
            # Step 2: Use trained model to make predictions on transformed data
            y_hat = self.model.predict(X_transformed)  # y_hat = predicted labels/classes
            
            return y_hat  # Return predictions (e.g., [0, 1, 0, 1, ...])
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
        
