from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
import sys
from networksecurity.logging.logger import logging

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()  # Create pipeline config
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)  # Create data ingestion config
        
        data_ingestion = DataIngestion(data_ingestion_config)  # Create DataIngestion object
        
        logging.info("Initiating data ingestion process")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()  # Call method
        print(data_ingestion_artifact)  # Output result

    except Exception as e:
        raise NetworkSecurityException(e, sys)

