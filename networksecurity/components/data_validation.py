# Project-specific imports
from networksecurity.entity.artifact_entity import DataIngestionArtifacts, DataValidationArtifacts
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils import read_yaml_file, write_yaml_file

# Third-party imports
from scipy.stats import ks_2samp  # Kolmogorov-Smirnov test for distribution comparison
import pandas as pd  # For DataFrame operations
import os, sys  # For file operations and system functions


class DataValidation:
    """
    DataValidation class handles the data quality and integrity checks.
    
    This class performs several validation tasks:
    1. Schema validation - checking if data has expected number of columns
    2. Data drift detection - identifying distribution shifts between train/test
    3. Saving validated data to designated directories
    4. Generating drift reports for monitoring
    
    Attributes:
        data_validation_config: Configuration with validation paths and thresholds
        data_ingestion_artifact: Artifact from previous ingestion stage with data paths
        schema_config: YAML schema defining expected data structure
    """
    
    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifacts):
        """
        Initialize the DataValidation component.
        
        Args:
            data_validation_config: Configuration object with validation settings
            data_ingestion_artifact: Artifact containing paths to train/test data
            
        Raises:
            NetworkSecurityException: If initialization or schema loading fails
        """
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            # Load expected schema from YAML configuration file
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e,sys)


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Static method to read CSV data into a DataFrame.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            NetworkSecurityException: If file reading fails
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has the expected number of columns.
        
        This method compares the actual column count in the DataFrame
        against the expected count defined in the schema YAML file.
        
        Args:
            dataframe (pd.DataFrame): DataFrame to validate
            
        Returns:
            bool: True if column count matches schema, False otherwise
            
        Raises:
            NetworkSecurityException: If validation check fails
        """
        try:
            # Get expected column count from schema configuration
            number_of_columns = len(self.schema_config.get("columns", []))
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")
            
            # Return True if counts match, False otherwise
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def validate_numerical_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate that all expected numerical columns exist in the DataFrame.
        
        This method checks if all numerical columns defined in the schema
        are present in the DataFrame. This ensures data integrity before
        processing numerical operations.
        
        Args:
            dataframe (pd.DataFrame): DataFrame to validate
            
        Returns:
            bool: True if all numerical columns exist, False otherwise
            
        Raises:
            NetworkSecurityException: If validation check fails
        """
        try:
            # Get list of expected numerical columns from schema
            numerical_columns = self.schema_config.get("numerical_columns", [])
            
            # Get actual columns in the dataframe
            dataframe_columns = dataframe.columns
            
            # Check if all expected numerical columns are present
            numerical_column_present = True
            missing_numerical_columns = []
            
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_column_present = False
                    missing_numerical_columns.append(num_column)
            
            logging.info(f"Required numerical columns: {len(numerical_columns)}")
            logging.info(f"Numerical columns present in dataframe: {numerical_column_present}")
            
            if not numerical_column_present:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")
            
            return numerical_column_present
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        """
        Detect data drift between training and testing datasets.
        
        This method uses the Kolmogorov-Smirnov (KS) two-sample test to compare
        distributions of each column between base (train) and current (test) datasets.
        Data drift indicates that the test data distribution significantly differs
        from training data, which may impact model performance.
        
        Statistical Test:
            - H0 (null hypothesis): Both samples come from the same distribution
            - If p-value < threshold: Reject H0, drift detected
            - If p-value >= threshold: Accept H0, no drift
        
        Args:
            base_df (pd.DataFrame): Reference dataset (typically training data)
            current_df (pd.DataFrame): Comparison dataset (typically test data)
            threshold (float, optional): P-value threshold for drift detection.
                                        Defaults to 0.05 (95% confidence)
                                        
        Returns:
            bool: True if no drift detected, False if drift found in any column
            
        Side Effects:
            Writes drift report YAML file with p-values and drift status per column
            
        Raises:
            NetworkSecurityException: If drift detection or report writing fails
        """
        try:
            status = True  # Assume no drift initially
            report = {}  # Dictionary to store drift report
            
            # Iterate through each column to perform KS test
            for column in base_df.columns:
                d1 = base_df[column]  # Base dataset column
                d2 = current_df[column]  # Current dataset column
                
                # Perform Kolmogorov-Smirnov two-sample test
                is_same_dist = ks_2samp(d1, d2)
                
                # Compare p-value with threshold
                if threshold <= is_same_dist.pvalue:
                    # P-value >= threshold: Distributions are similar (no drift)
                    is_found = False
                else:
                    # P-value < threshold: Distributions differ significantly (drift detected)
                    is_found = True
                    status = False  # Mark overall status as drift detected
                
                # Record results for this column in the report
                report.update({
                    column: {
                        "p_value": float(is_same_dist.pvalue),
                        "drift_status": is_found
                    }
                })
        
            # Get drift report file path from configuration
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            
            # Create directory for drift report if it doesn't exist
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Write drift report to YAML file for future reference
            write_yaml_file(file_path=drift_report_file_path, content=report)

        except Exception as e:
            raise NetworkSecurityException(e,sys)    
        

    def initiate_data_validation(self) -> DataValidationArtifacts:
        """
        Execute the complete data validation pipeline.
        
        This orchestrator method performs:
        1. Loading train and test datasets
        2. Schema validation (column count check)
        3. Data drift detection between train and test
        4. Saving validated datasets to designated paths
        5. Returning validation artifacts for next pipeline stage
        
        Workflow:
            - Read train/test data from ingestion artifacts
            - Validate schema for both datasets
            - Detect statistical drift between datasets
            - Save validated data to validation directory
            - Create and return validation artifact object
        
        Returns:
            DataValidationArtifacts: Object containing:
                - validation_status: Overall validation result (drift status)
                - valid_train_file_path: Path to validated training data
                - valid_test_file_path: Path to validated testing data
                - invalid_train_file_path: Path for invalid train data (None if valid)
                - invalid_test_file_path: Path for invalid test data (None if valid)
                - drift_report_file_path: Path to drift analysis report
                
        Raises:
            NetworkSecurityException: If any validation step fails
        """
        try:
            # Step 1: Get file paths from ingestion artifacts
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.tested_file_path
            
            # Step 2: Read train and test datasets
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # Initialize error message accumulator
            error_message = ""

            # Step 3: Validate schema for training data
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message += "Train data does not contain all required columns.\n"

            # Step 4: Validate schema for testing data
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message += "Test data does not contain all required columns.\n"

            # Step 5: Validate numerical columns in training data
            status = self.validate_numerical_columns(dataframe=train_dataframe)
            if not status:
                error_message += "Train data does not contain all required numerical columns.\n"
            
            # Step 6: Validate numerical columns in testing data
            status = self.validate_numerical_columns(dataframe=test_dataframe)
            if not status:
                error_message += "Test data does not contain all required numerical columns.\n"

            # Check if data exists for validation
            if isinstance(train_dataframe, pd.DataFrame):
                print("There is data to validate")
            
            # Log validation result
            if not error_message:
                print("Data validation completed successfully.")
         
            # Step 5: Detect data drift between train and test sets
            status = self.detect_dataset_drift(base_df=train_dataframe,
                                               current_df=test_dataframe)
            
            # Step 6: Create validation output directory
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Step 7: Save validated datasets to validation directory
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            # Step 8: Create and return validation artifacts
            data_validation_artifact = DataValidationArtifacts(
                validation_status=status,  # Overall drift status
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,  # Currently not segregating invalid data
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys)