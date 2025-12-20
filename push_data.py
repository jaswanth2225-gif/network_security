import os
import sys
import json
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

Mongo_DB_URI = os.getenv("MONGODB_URL")

if not Mongo_DB_URI:
    print("Error: MONGODB_URL not found in .env file")
    sys.exit(1)
    
print(f"MongoDB URI loaded successfully")

import certifi
ca = certifi.where()

import pandas as pd
import numpy as np
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkDataExtract():
    def __init__(self):
        try:
            self.mongo_client = None
            self.database = None
            self.collection = None
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def cv_to_json(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def insert_data_mongodb(self, records, database, collection):
        try:
            self.database = database
            self.collection = collection
            self.records = records

            self.mongo_client = pymongo.MongoClient(Mongo_DB_URI, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
            self.database = self.mongo_client[self.database]

            self.collection = self.database[self.collection]
            
            # Clear existing data to avoid duplicates
            print("Clearing existing data...")
            self.collection.delete_many({})
            
            # Insert in batches for better performance
            batch_size = 1000
            total_inserted = 0
            print(f"Inserting {len(self.records)} records in batches of {batch_size}...")
            
            for i in range(0, len(self.records), batch_size):
                batch = self.records[i:i+batch_size]
                self.collection.insert_many(batch)
                total_inserted += len(batch)
                print(f"Inserted {total_inserted}/{len(self.records)} records...")
            
            return total_inserted
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def export_data_to_excel(self, database, collection, output_file):
        try:
            self.mongo_client = pymongo.MongoClient(Mongo_DB_URI, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
            self.database = self.mongo_client[database]
            self.collection = self.database[collection]
            
            data = list(self.collection.find({}, {'_id': 0}))
            df = pd.DataFrame(data)
            df.to_excel(output_file, index=False)
            return f"Data exported to {output_file}"
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
              
if __name__ == "__main__":
    FILE_PATH = os.path.join("Network_Data", "phisingData.csv")
    
    # Validate file exists
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found: {FILE_PATH}")
        sys.exit(1)
    
    print(f"Reading CSV file: {FILE_PATH}")
    df = pd.read_csv(FILE_PATH)
    print(f"Records loaded: {len(df)}")
    
    # Export to Excel
    output_file = "Network_Data.xlsx"
    print(f"\nExporting to Excel: {output_file}")
    df.to_excel(output_file, index=False)
    print(f"SUCCESS! Data exported to {output_file}")
