# Standard library imports for file operations and system utilities
import os
import sys
import numpy as np

# Third-party imports
import yaml  # For parsing and writing YAML configuration files
import pandas as pd  # For data manipulation and DataFrame operations
import dill  # For advanced object serialization (not currently used)
import pickle  # For Python object serialization (not currently used)
from dotenv import load_dotenv  # For loading environment variables from .env file

# Project-specific imports
from networksecurity.exception.exception import NetworkSecurityException  # Custom exception handler
from networksecurity.logging.logger import logging  # Custom logging configuration


def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML configuration file and returns its contents as a dictionary.
    
    This function safely loads YAML files using yaml.safe_load to prevent
    arbitrary code execution vulnerabilities.
    
    Args:
        file_path (str): Absolute or relative path to the YAML file to be read
        
    Returns:
        dict: Dictionary containing the parsed YAML content
        
    Raises:
        NetworkSecurityException: If the file cannot be read or parsed,
                                  wrapping the original exception
    
    Example:
        >>> schema = read_yaml_file('data_schema/schema.yaml')
        >>> print(schema['columns'])
    """
    try:
        # Open the YAML file in binary read mode
        with open(file_path, 'rb') as yaml_file:
            # Parse YAML content safely and return as dictionary
            return yaml.safe_load(yaml_file)
        
    except Exception as e:
        # Wrap any exception in custom NetworkSecurityException for consistent error handling
        raise NetworkSecurityException(e, sys) from e
    





def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes Python objects to a YAML file with optional file replacement.
    
    This function serializes Python dictionaries or other objects to YAML format
    and saves them to a file. It automatically creates parent directories if needed.
    
    Args:
        file_path (str): Destination path where the YAML file will be written
        content (object): Python object (typically dict) to serialize to YAML
        replace (bool, optional): If True, deletes existing file before writing.
                                  Defaults to False.
                                  
    Returns:
        None
        
    Raises:
        NetworkSecurityException: If file cannot be written or directory cannot be created,
                                  wrapping the original exception
    
    Example:
        >>> report = {'accuracy': 0.95, 'loss': 0.05}
        >>> write_yaml_file('reports/model_metrics.yaml', report, replace=True)
    """
    try:
        # If replace flag is set, remove existing file
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write content to YAML file
        with open(file_path, 'w') as yaml_file:
            yaml.dump(content, yaml_file)
            
    except Exception as e:
        # Wrap any exception in custom NetworkSecurityException
        raise NetworkSecurityException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e



    











def export_collection_as_dataframe(database_name: str, collection_name: str) -> pd.DataFrame:

    """
    Exports data from a MongoDB collection into a pandas DataFrame.
    
    This function attempts to connect to MongoDB using credentials from environment
    variables. If MongoDB is unavailable or connection fails, it falls back to
    reading data from a local CSV file (Network_Data/phisingData.csv).
    
    Connection Priority:
        1. MONGO_DB_URL environment variable
        2. MONGODB_URL environment variable
        3. Default localhost connection (mongodb://localhost:27017/)
    
    Args:
        database_name (str): Name of the MongoDB database to connect to
        collection_name (str): Name of the collection within the database to query
        
    Returns:
        pd.DataFrame: DataFrame containing all documents from the collection,
                     with MongoDB's _id field excluded
                     
    Raises:
        NetworkSecurityException: If both MongoDB connection and CSV fallback fail,
                                  containing details of both errors
    
    Example:
        >>> df = export_collection_as_dataframe('KRISHAI', 'NetworkData')
        >>> print(f"Loaded {len(df)} records")
    
    Notes:
        - Connection timeout is set to 5 seconds
        - The _id field is automatically excluded from the result
        - CSV fallback expects file at: <current_directory>/Network_Data/phisingData.csv
    """
    try:
        # Load environment variables from .env file
        load_dotenv()
        
        # Try to get MongoDB URI from environment variables with fallback to localhost
        mongo_uri = (
            os.getenv("MONGO_DB_URL")  # Primary environment variable
            or os.getenv("MONGODB_URL")  # Alternative environment variable
            or "mongodb://localhost:27017/"  # Default local connection
        )

        try:
            # Import pymongo (lazy import to avoid import errors if not installed)
            from pymongo import MongoClient
        except Exception as e:
            logging.warning("pymongo not available: %s", e)
            raise e

        logging.info("Connecting to MongoDB")
        # Create MongoDB client with 5-second timeout for connection attempts
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        
        # Trigger server selection to validate connection early (ping test)
        client.admin.command('ping')

        # Access the specified database and collection
        db = client[database_name]
        col = db[collection_name]

        logging.info("Fetching documents from %s.%s", database_name, collection_name)
        # Query all documents, excluding MongoDB's internal _id field
        docs = list(col.find({}, {"_id": 0}))

        # Convert list of documents to pandas DataFrame
        df = pd.DataFrame(docs)
        logging.info("Fetched %d records", len(df))
        return df

    except Exception as mongo_error:
        # If MongoDB connection or query fails, try CSV fallback
        logging.warning("MongoDB fetch failed: %s", mongo_error)
        
        try:
            # Construct path to fallback CSV file
            csv_path = os.path.join(os.getcwd(), "Network_Data", "phisingData.csv")
            
            # Check if CSV file exists
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV fallback file not found: {csv_path}")
            
            logging.info("Reading fallback CSV: %s", csv_path)
            # Read CSV file into DataFrame
            df = pd.read_csv(csv_path)
            return df
            
        except Exception as csv_error:
            # Both MongoDB and CSV failed - raise comprehensive error
            raise NetworkSecurityException(
                Exception(
                    f"Failed to export data. MongoDB error: {mongo_error}. CSV fallback error: {csv_error}"
                ),
                sys,
            ) from csv_error