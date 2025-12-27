# === IMPORTS SECTION ===
# Import our custom modules (code we wrote in other files)
from networksecurity.entity.artifact_entity import DataIngestionArtifacts, DataValidationArtifacts
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils import read_yaml_file, write_yaml_file

# Statistical library for distribution comparison
from scipy.stats import ks_2samp  # Kolmogorov-Smirnov test: checks if two datasets have same distribution (e.g., train vs test)
import pandas as pd  # For working with CSV data in table format
import os, sys  # For file operations and error handling


class DataValidation:
    """
    WHAT THIS CLASS DOES: Quality control inspector for our data
    
    CHECKS PERFORMED:
    1. Schema validation - Does data have correct number of columns? (Should be 31 for network data)
    2. Type validation - Are all columns numerical values?
    3. Drift detection - Are train and test data too different from each other?
    
    ANALOGY: Like a factory quality inspector who:
    - Checks if products have all parts (column count)
    - Verifies parts are correct type (numerical validation)
    - Compares batches to ensure consistency (drift detection)
    
    IF VALIDATION FAILS: We get a warning (drift detected) but pipeline continues
    """
    
    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifacts):
        """
        Initialize = set up the validation component
        
        PARAMETERS:
            data_validation_config = object with file paths for saving validation results
            data_ingestion_artifact = object containing paths to train.csv and test.csv (from previous stage)
        """
        try:
            # Save the configs so other methods can access them
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            
            # Load the schema file (YAML file that defines expected structure)
            # SCHEMA = Blueprint/rulebook that says:
            #   - How many columns should exist (31 for this project)
            #   - Which columns should be numerical
            # YAML = Human-readable format like JSON (key: value pairs)
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e,sys)


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Utility method to read a CSV file
        
        @staticmethod = This method doesn't need access to self (works independently)
        USAGE: DataValidation.read_data("path/to/file.csv")
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        WHAT THIS METHOD DOES: Counts columns in data and checks if it matches expected count
        
        WHY THIS MATTERS: 
        - If data has wrong number of columns, something went wrong in data collection
        - Missing columns = model can't work (needs all features)
        - Extra columns = data might be corrupted or from wrong source
        
        EXAMPLE:
        - Expected: 31 columns (30 features + 1 target)
        - If actual has 25 columns → returns False (validation failed)
        - If actual has 31 columns → returns True (validation passed)
        
        RETURNS: True if count matches, False otherwise
        """
        try:
            # Get expected column count from schema YAML file
            # schema_config.get("columns", []) means:
            #   - Get the "columns" key from schema dictionary
            #   - If key doesn't exist, return empty list [] as default
            # len() counts how many columns are defined in schema (should be 31)
            number_of_columns = len(self.schema_config.get("columns", []))
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")
            
            # Compare counts and return True/False
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def validate_numerical_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        WHAT THIS METHOD DOES: Checks if all expected numerical columns exist in the data
        
        WHY THIS MATTERS:
        - ML models for this project only work with numbers
        - If numerical columns are missing, model training will fail
        - Helps catch data corruption early
        
        EXAMPLE:
        - Schema says we need: ["byte_count", "packet_count", "duration", ...]
        - If "byte_count" is missing → returns False
        - If all numerical columns present → returns True
        
        RETURNS: True if all numerical columns exist, False otherwise
        """
        try:
            # Get list of column names that should be numerical
            # EXAMPLE: ["src_bytes", "dst_bytes", "src_pkts", "dst_pkts", ...]
            numerical_columns = self.schema_config.get("numerical_columns", [])
            
            # Get actual column names from the DataFrame
            # dataframe.columns returns something like: Index(['col1', 'col2', 'col3'])
            dataframe_columns = dataframe.columns
            
            # Start assuming all columns are present
            numerical_column_present = True
            missing_numerical_columns = []
            
            # Loop through each expected numerical column and check if it exists
            # LIKE: Going through a checklist and marking what's missing
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_column_present = False  # Found a missing column!
                    missing_numerical_columns.append(num_column)  # Add to missing list
            
            logging.info(f"Required numerical columns: {len(numerical_columns)}")
            logging.info(f"Numerical columns present in dataframe: {numerical_column_present}")
            
            # If any columns are missing, log which ones
            if not numerical_column_present:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")
            
            return numerical_column_present
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        """
        WHAT THIS METHOD DOES: Checks if training and testing data are statistically different
        
        DATA DRIFT = When test data distribution is significantly different from training data
        
        WHY THIS MATTERS:
        - Model learns patterns from training data
        - If test data is very different, model won't work well
        - Example: Train on summer weather, test on winter weather → model fails
        
        REAL-WORLD ANALOGY:
        - You study math problems with small numbers (1-10)
        - Exam has large numbers (1000-10000)
        - You'll struggle because the patterns are different!
        
        HOW IT WORKS (Kolmogorov-Smirnov Test):
        1. For each column (feature), compare train vs test distributions
        2. Get a p-value (probability they're from same distribution)
        3. If p-value < 0.05 → Distributions are different (DRIFT DETECTED!)
        4. If p-value >= 0.05 → Distributions are similar (NO DRIFT)
        
        PARAMETERS:
            base_df = Training data (reference/baseline)
            current_df = Testing data (what we're comparing)
            threshold = 0.05 means 95% confidence level (standard in statistics)
        
        RETURNS: True if NO drift (data is good), False if drift detected (warning!)
        """
        try:
            status = True  # Start optimistic: assume no drift
            report = {}  # Dictionary to store results for each column
            
            # Loop through each column (e.g., "byte_count", "packet_count", etc.)
            for column in base_df.columns:
                d1 = base_df[column]  # Training data for this column (e.g., [100, 200, 150, ...])
                d2 = current_df[column]  # Testing data for this column (e.g., [110, 190, 160, ...])
                
                # ks_2samp = Kolmogorov-Smirnov 2-sample test
                # RETURNS: An object with .pvalue attribute
                # P-VALUE INTERPRETATION:
                #   - p-value = 0.80 → 80% chance they're from same distribution (GOOD!)
                #   - p-value = 0.02 → 2% chance they're from same distribution (DIFFERENT!)
                is_same_dist = ks_2samp(d1, d2)
                
                # Check the p-value against our threshold (0.05)
                if threshold <= is_same_dist.pvalue:
                    # P-value >= 0.05 means distributions look similar
                    # EXAMPLE: p-value = 0.80 → No drift, data is consistent
                    is_found = False  # No drift found for this column
                else:
                    # P-value < 0.05 means distributions are significantly different
                    # EXAMPLE: p-value = 0.01 → Drift detected, distributions differ!
                    is_found = True  # Drift found for this column!
                    status = False  # Mark overall validation as failed
                
                # Save results for this column to the report dictionary
                # EXAMPLE: {"byte_count": {"p_value": 0.02, "drift_status": True}}
                report.update({
                    column: {
                        "p_value": float(is_same_dist.pvalue),  # Store the p-value
                        "drift_status": is_found  # Store if drift was detected
                    }
                })
        
            # Save the report to a YAML file so data scientists can review it
            # YAML = Easy to read file format (like JSON but more human-friendly)
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)  # Create folders if needed
            write_yaml_file(file_path=drift_report_file_path, content=report)
            
            # Return True if NO drift, False if drift detected
            # TRUE = Safe to proceed with model training
            # FALSE = Warning! Test data looks different from training data
            return status

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    def initiate_data_validation(self) -> DataValidationArtifacts:
        """
        WHAT THIS METHOD DOES: The main/master method that runs all validation checks
        
        THIS IS THE METHOD YOU CALL to run data validation. It calls other methods internally.
        
        WORKFLOW (step by step):
        1. Load train.csv and test.csv from previous stage
        2. Check if both have correct number of columns (31)
        3. Check if both have all required numerical columns
        4. Check if train and test distributions are similar (drift detection)
        5. Save validated files to new location
        6. Return file paths so next stage knows where to find them
        
        RETURNS: DataValidationArtifacts object containing:
            - validation_status: True if NO drift, False if drift detected
            - valid_train_file_path: Path to validated train.csv
            - valid_test_file_path: Path to validated test.csv
            - drift_report_file_path: Path to YAML report with p-values
        """
        try:
            # STEP 1: Get file paths from previous stage (Data Ingestion)
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.tested_file_path
            
            # STEP 2: Load the CSV files into pandas DataFrames
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            error_message = ""  # String to accumulate any error messages

            # STEP 3: Validate training data has correct number of columns
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message += "Train data does not contain all required columns.\n"

            # STEP 4: Validate testing data has correct number of columns
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message += "Test data does not contain all required columns.\n"

            # STEP 5: Validate training data has all numerical columns
            status = self.validate_numerical_columns(dataframe=train_dataframe)
            if not status:
                error_message += "Train data does not contain all required numerical columns.\n"
            
            # STEP 6: Validate testing data has all numerical columns
            status = self.validate_numerical_columns(dataframe=test_dataframe)
            if not status:
                error_message += "Test data does not contain all required numerical columns.\n"

            # Quick sanity check that data loaded successfully
            if isinstance(train_dataframe, pd.DataFrame):
                print("There is data to validate")
            
            # If no errors so far, log success
            if not error_message:
                print("Data validation completed successfully.")
         
            # STEP 7: Drift detection (most important check!)
            # Compares train vs test distributions using KS test
            # Returns True if NO drift, False if drift detected
            status = self.detect_dataset_drift(base_df=train_dataframe,
                                               current_df=test_dataframe)
            
            # STEP 8: Create output directory for validated files
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # STEP 9: Save validated data to new location
            # Even if drift detected, we still save the files (just with a warning)
            # Next stage (transformation) will use these files
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            # STEP 10: Create artifact object to pass to next stage
            # This is like a delivery receipt: "Here are your validated files and the drift status"
            data_validation_artifact = DataValidationArtifacts(
                validation_status=status,  # True = no drift, False = drift detected
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,  # Not used yet (future: separate bad data)
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys)
