import pandas as pd  # DataFrame library for data manipulation (like Excel in Python)
import os  # File/folder operations
from typing import List  # Type hints to document what data types functions expect


def export_collection_as_dataframe(database_name: str, collection_name: str) -> pd.DataFrame:
    """
    Export data from MongoDB collection as DataFrame.
    Falls back to CSV file if MongoDB is not available.
    
    MongoDB = A document database (like digital filing cabinet with folders inside)
    CSV = Comma-separated values file (like Excel spreadsheet saved as text)
    
    Args:
        database_name: Name of the MongoDB database (folder name in MongoDB)
        collection_name: Name of the MongoDB collection (table name inside the folder)
    
    Returns:
        pandas DataFrame containing the collection data (looks like Excel spreadsheet)
    """
    try:
        # Try to connect to MongoDB first
        from pymongo import MongoClient  # MongoClient = tool to talk to MongoDB server
        
        client = MongoClient('mongodb://localhost:27017/')  # Connect to MongoDB on this computer, port 27017
        database = client[database_name]  # Access specific database (folder)
        collection = database[collection_name]  # Access specific collection (table inside folder)
        
        # Fetch data from collection
        data = list(collection.find())  # find() = "get all documents" (like SELECT * in SQL)
        
        # Convert to DataFrame
        if data:
            # Remove MongoDB's internal _id field if present (MongoDB auto-creates this field)
            for record in data:
                if '_id' in record:
                    del record['_id']  # Delete the _id field (we don't need it)
            
            df = pd.DataFrame(data)  # Convert list of documents to table format
            return df
        else:
            return pd.DataFrame()  # Return empty DataFrame if no data
    
    except Exception as e:
        # Fallback: Try to read from CSV file
        try:
            csv_path = os.path.join(os.getcwd(), "Network_Data", "phisingData.csv")  # Path to backup CSV file
            
            if os.path.exists(csv_path):
                print(f"MongoDB connection failed. Reading from CSV file: {csv_path}")
                df = pd.read_csv(csv_path)  # Read CSV file into DataFrame (like opening Excel file)
                return df
            else:
                raise Exception(f"Error exporting collection {collection_name} from database {database_name}: {str(e)}")
        except Exception as csv_error:
            raise Exception(f"Error exporting collection {collection_name} from database {database_name}: {str(e)} | CSV fallback also failed: {str(csv_error)}")
