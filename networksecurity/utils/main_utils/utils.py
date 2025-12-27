import os  # For file/folder operations (creating directories, checking if file exists)
import sys  # For system-level operations (error handling, traceback details)
import numpy as np  # NumPy = Fast numerical computing library (arrays, math operations)

import yaml  # YAML = Configuration file format (human-readable, easier than JSON)
import pandas as pd  # Pandas = DataFrame library (Excel-like tables in Python)
import pickle  # pickle = Python serialization (converts objects to binary format)
from dotenv import load_dotenv  # dotenv = Loads environment variables from .env file (keeps secrets safe)
from sklearn.model_selection import GridSearchCV  # GridSearchCV = Exhaustive search over all hyperparameter combinations
from sklearn.metrics import accuracy_score  # accuracy_score = % correct predictions

from networksecurity.exception.exception import NetworkSecurityException  # Shows clear error messages
from networksecurity.logging.logger import logging  # Records pipeline progress to console and file


def read_yaml_file(file_path: str) -> dict:
    """Read YAML config file and return as Python dictionary (like reading Excel spreadsheet)."""
    try:
        with open(file_path, 'rb') as yaml_file:  # Open file in binary read mode
            return yaml.safe_load(yaml_file)  # safe_load = Parse YAML safely, return as dict
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """Write Python dict to YAML file. replace=True = overwrite, False = keep existing."""
    try:   #if replace is true then only we will delete the existing file
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)  # Delete old file if replace=True
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create parent folders if missing
        
        with open(file_path, 'w') as yaml_file:  # Open in write mode
            yaml.dump(content, yaml_file)  # yaml.dump = Convert Python dict to YAML text and write
            
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save NumPy array to .npy file (binary format, super fast).
    .npy = NumPy native format (preserves data type and shape, loads quickly)
    Used for: saving transformed training/testing data
    """
    try:
        dir_path = os.path.dirname(file_path)  # Extract folder path from full file path
        os.makedirs(dir_path, exist_ok=True)  # Create folders if missing
        
        with open(file_path, "wb") as file_obj:  # Open in binary write mode
            np.save(file_obj, array)  # np.save = Serialize numpy array to .npy format
            
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    """
    Save Python object to pickle file (binary format).
    pickle = Python serialization (converts any Python object to binary)
    Used for: trained models, preprocessors, transformers
    Example: save_object('model.pkl', trained_model) → saves trained ML model to disk
    """
    try:
        logging.info("Saving object to pickle file")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create parent folders if missing
        
        with open(file_path, "wb") as file_obj:  # Open in binary write mode
            pickle.dump(obj, file_obj)  # pickle.dump = Serialize object to binary and write to file
            
        logging.info("Object saved successfully")
        
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def export_collection_as_dataframe(database_name: str, collection_name: str) -> pd.DataFrame:
    """
    Fetch data from MongoDB collection and return as DataFrame (Excel-like table).
    
    MongoDB = Document database (like digital filing cabinet)
    Collection = Table in MongoDB
    Falls back to CSV if MongoDB is unavailable
    """
    try:
        load_dotenv()  # Load environment variables from .env file (safe way to store secrets)
        
        mongo_uri = (
            os.getenv("MONGO_DB_URL")  # First try this environment variable
            or os.getenv("MONGODB_URL")  # Then try this one
            or "mongodb://localhost:27017/"  # Default local connection
        )

        try:
            from pymongo import MongoClient  # MongoClient = Tool to talk to MongoDB server
        except Exception as e:
            logging.warning("pymongo not available: %s", e)
            raise e

        logging.info("Connecting to MongoDB")
        # Create MongoDB client with 5-second timeout (don't wait forever for connection)
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        
        # Trigger server selection to validate connection early (ping test)
        client.admin.command('ping')

        # Access the specified database and collection
        db = client[database_name]  # Access folder (database)
        col = db[collection_name]  # Access table (collection) inside folder

        logging.info("Fetching documents from %s.%s", database_name, collection_name)
        # Query all documents, excluding MongoDB's internal _id field (we don't need it)
        docs = list(col.find({}, {"_id": 0}))  # find({}, {"_id": 0}) = Get all, skip _id field

        # Convert list of documents to pandas DataFrame (like Excel table)
        df = pd.DataFrame(docs)
        logging.info("Fetched %d records", len(df))
        return df

    except Exception as mongo_error:
        # If MongoDB fails, try CSV fallback
        logging.warning("MongoDB fetch failed: %s", mongo_error)
        
        try:
            # Construct path to fallback CSV file
            csv_path = os.path.join(os.getcwd(), "Network_Data", "phisingData.csv")
            
            # Check if CSV file exists
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV fallback file not found: {csv_path}")
            
            logging.info("Reading fallback CSV: %s", csv_path)
            # Read CSV file into DataFrame (like opening Excel file)
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


def load_object(file_path: str) -> object:
    """
    Load Python object from pickle file.
    pickle.load = Deserialize binary file back to Python object
    Used for: loading trained models, preprocessors, transformers
    Example: model = load_object('model.pkl') → loads trained ML model from disk
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
         
        with open(file_path, "rb") as file_obj:  # Open in binary read mode
            return pickle.load(file_obj)  # pickle.load = Deserialize binary to Python object
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    Load NumPy array from .npy file.
    np.load = Read .npy file and convert back to numpy array (fast, efficient)
    Used for: loading transformed training/testing data
    """
    try:
        with open(file_path, "rb") as file_obj:  # Open in binary read mode
            return np.load(file_obj)  # np.load = Deserialize .npy file to numpy array
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Train multiple ML models using GridSearchCV (exhaustive hyperparameter search).
    
    GridSearchCV = Tries all combinations of hyperparameters (like testing all recipes)
    Returns dictionary of {model_name: best_test_accuracy}
    """
    try:
        report = {}  # Dictionary to store results: {"Random Forest": 0.95, "Decision Tree": 0.92, ...}

        for i in range(len(list(models))):  # Loop through each model
            model = list(models.values())[i]  # Get the model object
            para = param[list(models.keys())[i]]  # Get hyperparameters to try for this model

            # GridSearchCV = Tests all parameter combinations with cross-validation
            # cv=3 = 3-fold cross validation (split data into 3 parts, test 3 times)
            # scoring='accuracy' = Use accuracy as the metric to optimize
            # n_jobs=-1 = Use all CPU cores (faster)
            gs = GridSearchCV(model, para, cv=3, scoring='accuracy', n_jobs=-1)
            gs.fit(X_train, y_train)  # Fit model with all parameter combinations

            # Set model to best parameters found by GridSearch
            model.set_params(**gs.best_params_)
            # Retrain with best parameters on full training data
            model.fit(X_train, y_train)

            # Make predictions on training data
            y_train_pred = model.predict(X_train)

            # Make predictions on test data
            y_test_pred = model.predict(X_test)

            # Calculate accuracy scores
            train_model_score = accuracy_score(y_train, y_train_pred)  # accuracy = % of correct predictions
            test_model_score = accuracy_score(y_test, y_test_pred)  # test accuracy = real performance on unseen data

            # Log both scores to detect overfitting (if train >> test, model memorized data)
            model_name = list(models.keys())[i]
            logging.info(f"{model_name}: Train accuracy={train_model_score:.4f}, Test accuracy={test_model_score:.4f}")

            # Store test score in report (test accuracy is what matters - unseen data performance)
            report[model_name] = test_model_score

        return report  # Return {"Random Forest": 0.95, "Decision Tree": 0.92, ...}

    except Exception as e:
        raise NetworkSecurityException(e, sys)
