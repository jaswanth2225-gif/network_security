from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation   
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig
import sys
from networksecurity.logging.logger import logging

def new_func(data_ingestion_config):
    return data_ingestion_config

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)

        data_ingestion = DataIngestion(new_func(data_ingestion_config))
        logging.info("Initiating data ingestion process")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed")
        print(data_ingestion_artifact)

        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_validation_config, data_ingestion_artifact)
        logging.info('Initiating data validation')
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        print(data_validation_artifact)

    except Exception as e:
        raise NetworkSecurityException(e, sys)

