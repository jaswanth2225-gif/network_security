import os
import sys
import pandas as pd
import numpy as np
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils import export_collection_as_dataframe
from sklearn.model_selection import train_test_split
from networksecurity.entity.artifact_entity import DataIngestionArtifacts

class DataIngestion:
    def __init__(self, data_ingestion_config):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self) -> pd.DataFrame:
        try:
            logging.info("Exporting data from MongoDB to feature store")
            dataframe = export_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name
            )

            # Create feature store directory if not exists
            os.makedirs(os.path.dirname(self.data_ingestion_config.feature_store_file_path), exist_ok=True)

            # Save to CSV
            dataframe.to_csv(self.data_ingestion_config.feature_store_file_path, index=False)
            logging.info("Saved feature store file at: %s", self.data_ingestion_config.feature_store_file_path)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
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
        try:
            dataframe = self.export_data_into_feature_store()
            self.split_data_as_train_test(dataframe)
            return DataIngestionArtifacts(
                trained_file_path=self.data_ingestion_config.training_file_path,
                tested_file_path=self.data_ingestion_config.testing_file_path,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)



  

