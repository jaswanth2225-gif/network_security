# === IMPORTS SECTION ===
# These are like tools in a toolbox - we import libraries/modules we'll use below

import os  # Built-in Python library for file/folder operations (creating directories, checking if files exist)
import sys  # Built-in Python library to get system info (helps with error messages showing which line failed)

# Data manipulation libraries (most common in data science)
import pandas as pd  # Pandas = Excel on steroids. Works with tables/spreadsheets (called DataFrames)
import numpy as np  # NumPy = Fast math operations on arrays/matrices of numbers
from sklearn.model_selection import train_test_split  # Scikit-learn function: randomly splits data into training and testing sets

# Our custom modules (code we wrote in other files)
from networksecurity.exception.exception import NetworkSecurityException  # Our custom error class: shows helpful error messages
from networksecurity.logging.logger import logging  # Our custom logger: writes messages to log files so we can track what happened
from networksecurity.utils.main_utils import export_collection_as_dataframe  # Our function: connects to MongoDB and fetches data
from networksecurity.entity.artifact_entity import DataIngestionArtifacts  # Our dataclass: holds file paths to pass to next stage


class DataIngestion:
    """
    WHAT THIS CLASS DOES: Gets data from MongoDB database and splits it into separate train/test CSV files
    
    ANALOGY: Like a librarian who:
    1. Fetches books (data) from the library (MongoDB)
    2. Makes a photocopy (saves to CSV) 
    3. Splits the copy into two piles: 80% for studying (training), 20% for exam (testing)
    """
    
    def __init__(self, data_ingestion_config):
        """Initialize with data ingestion configuration."""
        self.data_ingestion_config = data_ingestion_config

    def export_data_into_feature_store(self):
        """
        WHAT THIS METHOD DOES: Downloads data from MongoDB and saves it as a CSV file
        
        FEATURE STORE = A folder where we keep a snapshot/backup of the raw data
        WHY: If MongoDB data changes later, we still have the original version we used
        
        RETURNS: A pandas DataFrame (think: Excel spreadsheet in Python memory)
        
        STEP-BY-STEP:
        1. Connect to MongoDB database
        2. Fetch all records from the collection (like a table in SQL)
        3. Convert MongoDB documents to pandas DataFrame (rows and columns)
        4. Save DataFrame as CSV file for backup
        5. Return the DataFrame so we can use it
        """
        try:
            logging.info("Exporting data from MongoDB to feature store")
            
            dataframe = export_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name
            )

            os.makedirs(os.path.dirname(self.data_ingestion_config.feature_store_file_path), exist_ok=True)

            dataframe.to_csv(self.data_ingestion_config.feature_store_file_path, index=False)
            logging.info("Saved feature store file at: %s", self.data_ingestion_config.feature_store_file_path)
            
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe):
        """
        WHAT THIS METHOD DOES: Divides the data into two separate sets - training and testing
        
        WHY WE NEED TWO SETS:
        - Training set (80%) = Data the model learns from (like study material)
        - Testing set (20%) = Data we use to test if model learned correctly (like an exam)
        - We NEVER let the model see test data during training (prevents cheating/memorization)
        
        PARAMETERS:
            dataframe = the complete dataset we fetched from MongoDB
        """
        try:
            logging.info("Starting train-test split")
            
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )
            
            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False)

            logging.info("Saved train and test files")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        """
        Run the complete data ingestion: fetch from MongoDB, save to feature store, split into train/test.
        
        WORKFLOW (step by step):
        1. Fetch data from MongoDB → save to feature_store/phisingData.csv
        2. Split that data → save to ingested/train.csv and ingested/test.csv
        3. Return the file paths so the next pipeline stage knows where to find the files
        
        RETURNS: DataIngestionArtifacts object containing:
            - trained_file_path: Path to train.csv
            - tested_file_path: Path to test.csv
        """
        try:
            dataframe = self.export_data_into_feature_store()
            
            self.split_data_as_train_test(dataframe)
            
            return DataIngestionArtifacts(
                trained_file_path=self.data_ingestion_config.training_file_path,
                tested_file_path=self.data_ingestion_config.testing_file_path,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)



  

