# app/ingestion.py
import boto3
import pandas as pd
from elasticsearch import Elasticsearch
import os
from datetime import datetime, timedelta
import logging
from typing import Optional
from .utils import setup_logging

setup_logging()

class DataIngestion:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        self.es = Elasticsearch([{
            'host': os.getenv('ELASTICSEARCH_HOST'),
            'port': int(os.getenv('ELASTICSEARCH_PORT', 9200)),
            'scheme': os.getenv('ELASTICSEARCH_SCHEME', 'http')
        }])

    def get_sku_from_filename(self, filename: str) -> str:
        """Extract SKU from filename like '6SQ-BLUBRY_data_69_entries.csv'"""
        return filename.split('_')[0]

    def upload_csv_to_s3(self, file_path: str, sku: str) -> bool:
        """Upload CSV file to S3 bucket with SKU-based path"""
        try:
            # Read and standardize the CSV format
            df = pd.read_csv(file_path)
            
            # Ensure columns are correctly named
            df = df.rename(columns={
                'Date': 'Date',
                'Product Name': 'product_name',
                'Quantity': 'Quantity'
            })
            
            # Convert date to standard format
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            
            # Sort by date
            df = df.sort_values('Date')
            
            # Save to temporary file
            temp_file = f'/tmp/{sku}_processed.csv'
            df.to_csv(temp_file, index=False)
            
            # Upload to S3
            s3_key = f'historical_data/{sku}/{os.path.basename(file_path)}'
            self.s3.upload_file(temp_file, self.bucket_name, s3_key)
            
            # Cleanup
            os.remove(temp_file)
            
            logging.info(f"Successfully uploaded {file_path} to S3 for SKU {sku}")
            return True
        except Exception as e:
            logging.error(f"Error uploading CSV to S3: {e}")
            return False

    def get_latest_data(self, sku: str) -> Optional[pd.DataFrame]:
        """Get latest data from S3 for a specific SKU"""
        try:
            # List all files for this SKU
            historical_prefix = f'historical_data/{sku}/'
            latest_prefix = f'latest_data/{sku}/'
            
            all_data = []
            
            # Get historical data
            response = self.s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=historical_prefix
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    s3_object = self.s3.get_object(
                        Bucket=self.bucket_name,
                        Key=obj['Key']
                    )
                    df = pd.read_csv(s3_object['Body'])
                    all_data.append(df)
            
            if not all_data:
                return None
                
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            # Remove duplicates based on Date
            combined_df = combined_df.drop_duplicates(subset=['Date'])
            # Sort by Date
            combined_df = combined_df.sort_values('Date')
            
            return combined_df
            
        except Exception as e:
            logging.error(f"Error getting latest data from S3: {e}")
            return None

    def fetch_es_data_and_upload(self, sku: str) -> bool:
        """Fetch latest data from Elasticsearch and upload to S3"""
        try:
            # Query Elasticsearch for the past week's data
            now = datetime.now()
            week_ago = now - timedelta(days=7)
            
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"sku": sku}},
                            {"range": {
                                "Date": {
                                    "gte": week_ago.strftime("%Y-%m-%d"),
                                    "lte": now.strftime("%Y-%m-%d")
                                }
                            }}
                        ]
                    }
                }
            }
            
            res = self.es.search(
                index=os.getenv('ELASTICSEARCH_INDEX'),
                body=query,
                size=10000
            )
            
            # Convert to DataFrame
            records = [hit['_source'] for hit in res['hits']['hits']]
            df = pd.DataFrame(records)
            
            if df.empty:
                logging.warning(f"No new data found in Elasticsearch for SKU {sku}")
                return False
            
            # Save to temporary CSV
            temp_file = f'/tmp/{sku}_latest_data.csv'
            df.to_csv(temp_file, index=False)
            
            # Upload to S3
            s3_key = f'latest_data/{sku}/{now.strftime("%Y%m%d")}_data.csv'
            self.s3.upload_file(temp_file, self.bucket_name, s3_key)
            
            # Clean up
            os.remove(temp_file)
            
            logging.info(f"Successfully uploaded latest data to S3 for SKU {sku}")
            return True
            
        except Exception as e:
            logging.error(f"Error processing Elasticsearch data: {e}")
            return False

    # def get_latest_data(self, sku: str) -> Optional[pd.DataFrame]:
    #     """Combine historical and latest data from S3"""
    #     try:
    #         # List all files for this SKU
    #         historical_prefix = f'historical_data/{sku}/'
    #         latest_prefix = f'latest_data/{sku}/'
            
    #         all_data = []
            
    #         # Get historical data
    #         response = self.s3.list_objects_v2(
    #             Bucket=self.bucket_name,
    #             Prefix=historical_prefix
    #         )
            
    #         if 'Contents' in response:
    #             for obj in response['Contents']:
    #                 s3_object = self.s3.get_object(
    #                     Bucket=self.bucket_name,
    #                     Key=obj['Key']
    #                 )
    #                 df = pd.read_csv(s3_object['Body'])
    #                 all_data.append(df)
            
    #         # Get latest data
    #         response = self.s3.list_objects_v2(
    #             Bucket=self.bucket_name,
    #             Prefix=latest_prefix
    #         )
            
    #         if 'Contents' in response:
    #             # Get the most recent file
    #             latest_file = max(
    #                 response['Contents'],
    #                 key=lambda x: x['LastModified']
    #             )
    #             s3_object = self.s3.get_object(
    #                 Bucket=self.bucket_name,
    #                 Key=latest_file['Key']
    #             )
    #             df = pd.read_csv(s3_object['Body'])
    #             all_data.append(df)
            
    #         if not all_data:
    #             return None
                
    #         # Combine all data
    #         combined_df = pd.concat(all_data, ignore_index=True)
    #         # Remove duplicates based on Date
    #         combined_df = combined_df.drop_duplicates(subset=['Date'])
    #         # Sort by Date
    #         combined_df = combined_df.sort_values('Date')
            
    #         return combined_df
            
    #     except Exception as e:
    #         logging.error(f"Error getting latest data from S3: {e}")
    #         return None
