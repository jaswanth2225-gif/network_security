import os
import sys
import yaml
import pandas as pd
import dill
import pickle
from dotenv import load_dotenv
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
        
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    



    
    """
    Reads a YAML file and returns its contents as a dictionary.
       Args:
        file_path: Path to the YAML file."""

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(content, yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def export_collection_as_dataframe(database_name: str, collection_name: str) -> pd.DataFrame:
    """
    Export data from a MongoDB collection into a pandas DataFrame.
    Falls back to reading CSV from Network_Data/phisingData.csv if MongoDB is unavailable.

    Args:
        database_name: Name of the MongoDB database
        collection_name: Name of the MongoDB collection

    Returns:
        pd.DataFrame: DataFrame with collection data or CSV data
    """
    try:
        load_dotenv()
        mongo_uri = (
            os.getenv("MONGO_DB_URL")
            or os.getenv("MONGODB_URL")
            or "mongodb://localhost:27017/"
        )

        try:
            from pymongo import MongoClient
        except Exception as e:
            logging.warning("pymongo not available: %s", e)
            raise e

        logging.info("Connecting to MongoDB")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        # Trigger server selection to validate connection early
        client.admin.command('ping')

        db = client[database_name]
        col = db[collection_name]

        logging.info("Fetching documents from %s.%s", database_name, collection_name)
        docs = list(col.find({}, {"_id": 0}))

        df = pd.DataFrame(docs)
        logging.info("Fetched %d records", len(df))
        return df

    except Exception as mongo_error:
        logging.warning("MongoDB fetch failed: %s", mongo_error)
        # Fallback to CSV
        try:
            csv_path = os.path.join(os.getcwd(), "Network_Data", "phisingData.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV fallback file not found: {csv_path}")
            logging.info("Reading fallback CSV: %s", csv_path)
            df = pd.read_csv(csv_path)
            return df
        except Exception as csv_error:
            raise NetworkSecurityException(
                Exception(
                    f"Failed to export data. MongoDB error: {mongo_error}. CSV fallback error: {csv_error}"
                ),
                sys,
            ) from csv_error