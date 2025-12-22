import pandas as pd
import os
from typing import List


def export_collection_as_dataframe(database_name: str, collection_name: str) -> pd.DataFrame:
    """
    Export data from MongoDB collection as DataFrame.
    Falls back to CSV file if MongoDB is not available.
    
    Args:
        database_name: Name of the MongoDB database
        collection_name: Name of the MongoDB collection
    
    Returns:
        pandas DataFrame containing the collection data
    """
    try:
        # Try to connect to MongoDB first
        from pymongo import MongoClient
        
        client = MongoClient('mongodb://localhost:27017/')
        database = client[database_name]
        collection = database[collection_name]
        
        # Fetch data from collection
        data = list(collection.find())
        
        # Convert to DataFrame
        if data:
            # Remove MongoDB's internal _id field if present
            for record in data:
                if '_id' in record:
                    del record['_id']
            
            df = pd.DataFrame(data)
            return df
        else:
            return pd.DataFrame()
    
    except Exception as e:
        # Fallback: Try to read from CSV file
        try:
            csv_path = os.path.join(os.getcwd(), "Network_Data", "phisingData.csv")
            
            if os.path.exists(csv_path):
                print(f"MongoDB connection failed. Reading from CSV file: {csv_path}")
                df = pd.read_csv(csv_path)
                return df
            else:
                raise Exception(f"Error exporting collection {collection_name} from database {database_name}: {str(e)}")
        except Exception as csv_error:
            raise Exception(f"Error exporting collection {collection_name} from database {database_name}: {str(e)} | CSV fallback also failed: {str(csv_error)}")
