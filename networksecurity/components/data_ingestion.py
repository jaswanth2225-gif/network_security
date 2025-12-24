# Standard library imports
import os  # For file and directory operations
import sys  # For system-specific parameters and functions

# Third-party imports
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computing (currently unused)
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets

# Project-specific imports
from networksecurity.exception.exception import NetworkSecurityException  # Custom exception handling
from networksecurity.logging.logger import logging  # Custom logging configuration
from networksecurity.utils.main_utils import export_collection_as_dataframe  # MongoDB data export utility
from networksecurity.entity.artifact_entity import DataIngestionArtifacts  # Data class for ingestion outputs


class DataIngestion:
    """
    DataIngestion class handles the initial stage of the ML pipeline.
    
    This class is responsible for:
    1. Fetching data from MongoDB or CSV fallback
    2. Storing data in a feature store
    3. Splitting data into training and testing sets
    4. Returning paths to generated datasets as artifacts
    
    Attributes:
        data_ingestion_config: Configuration object containing paths and database details
    """
    
    def __init__(self, data_ingestion_config):
        """
        Initialize the DataIngestion component with configuration.
        
        Args:
            data_ingestion_config: DataIngestionConfig object with all necessary
                                  paths, database names, and split ratios
                                  
        Raises:
            NetworkSecurityException: If initialization fails
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self) -> pd.DataFrame:
        """
        Export data from MongoDB/CSV and save it to the feature store.
        
        This method fetches raw data from the configured MongoDB collection
        (or CSV fallback) and stores it in a centralized feature store location.
        The feature store acts as a versioned snapshot of raw data for the pipeline.
        
        Returns:
            pd.DataFrame: DataFrame containing the exported data
            
        Raises:
            NetworkSecurityException: If data export or file saving fails
        """
        try:
            logging.info("Exporting data from MongoDB to feature store")
            
            # Fetch data from MongoDB collection (with CSV fallback)
            dataframe = export_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name
            )

            # Create feature store directory structure if it doesn't exist
            os.makedirs(os.path.dirname(self.data_ingestion_config.feature_store_file_path), exist_ok=True)

            # Save the DataFrame to CSV in the feature store location
            # index=False prevents saving DataFrame index as a column
            dataframe.to_csv(self.data_ingestion_config.feature_store_file_path, index=False)
            logging.info("Saved feature store file at: %s", self.data_ingestion_config.feature_store_file_path)
            
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """
        Split the feature store data into training and testing datasets.
        
        This method performs stratified train-test split using scikit-learn's
        train_test_split function. The split ratio is configured in the
        data_ingestion_config. A fixed random_state ensures reproducibility.
        
        Args:
            dataframe (pd.DataFrame): Complete dataset to be split
            
        Raises:
            NetworkSecurityException: If splitting or file saving fails
        """
        try:
            logging.info("Starting train-test split")
            
            # Split data into train and test sets
            # test_size: Proportion of data for testing (e.g., 0.2 = 20%)
            # random_state=42: Fixed seed for reproducible splits
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            # Create directory for ingested data if it doesn't exist
            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)

            # Save training and testing datasets as CSV files
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False)

            logging.info("Saved train and test files")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        """
        Execute the complete data ingestion pipeline.
        
        This is the main orchestrator method that:
        1. Exports data from MongoDB to feature store
        2. Splits the data into train and test sets
        3. Returns artifact paths for downstream pipeline stages
        
        Returns:
            DataIngestionArtifacts: Object containing paths to:
                - trained_file_path: Path to training CSV
                - tested_file_path: Path to testing CSV
                
        Raises:
            NetworkSecurityException: If any step in the ingestion process fails
            
        Example:
            >>> ingestion = DataIngestion(config)
            >>> artifacts = ingestion.initiate_data_ingestion()
            >>> print(artifacts.trained_file_path)
        """
        try:
            # Step 1: Export data from MongoDB/CSV to feature store
            dataframe = self.export_data_into_feature_store()
            
            # Step 2: Split data into training and testing sets
            self.split_data_as_train_test(dataframe)
            
            # Step 3: Return artifact paths for next pipeline stage
            return DataIngestionArtifacts(
                trained_file_path=self.data_ingestion_config.training_file_path,
                tested_file_path=self.data_ingestion_config.testing_file_path,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)



  

