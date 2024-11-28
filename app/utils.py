# app/utils.py

import boto3
import os
import pickle
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from prophet import Prophet

def setup_logging():
    """Configure logging for the application"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO if os.getenv('LOG_LEVEL') != 'DEBUG' else logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler(
                os.path.join(log_dir, 'app.log')
            )  # Log to file
        ]
    )
def get_skus_from_s3(s3_client, bucket_name: str) -> List[str]:
        """Get list of SKUs from S3 historical data directory"""
        try:
            # List all 'directories' under historical_data/
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix='historical_data/',
                Delimiter='/'
            )
            
            skus = []
            if 'CommonPrefixes' in response:
                for prefix in response['CommonPrefixes']:
                    # Extract SKU from path like 'historical_data/SKU123/'
                    sku = prefix['Prefix'].split('/')[1]
                    if sku:  # Ensure not empty
                        skus.append(sku)
            
            logging.info(f"Found SKUs in S3: {skus}")
            return skus
        except Exception as e:
            logging.error(f"Error getting SKUs from S3: {e}")
            return []

class ModelRegistry:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        self.model_cache = {}  # In-memory cache for frequently used models
        self.cache_expiry = {}  # Track when cached models should be refreshed
        
    def get_available_skus(self) -> List[str]:
        """Get list of available SKUs"""
        return get_skus_from_s3(self.s3, self.bucket_name)

    def save_model(self, model: Prophet, sku: str, metrics: Dict[str, float], parameters: Dict[str, Any]) -> bool:
        """
        Save trained model and its metadata to S3
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version = f"v_{timestamp}"
            
            # Create model metadata
            metadata = {
                'version': version,
                'training_date': datetime.now().isoformat(),
                'metrics': metrics,
                'parameters': parameters,
                'last_used': datetime.now().isoformat()
            }
            
            # Save model to S3
            model_key = f'models/{sku}/{version}/model.pkl'
            metadata_key = f'models/{sku}/{version}/metadata.json'
            
            # Serialize model
            with open(f'/tmp/{sku}_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            
            # Upload model file
            self.s3.upload_file(
                f'/tmp/{sku}_model.pkl',
                self.bucket_name,
                model_key
            )
            
            # Upload metadata
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=metadata_key,
                Body=json.dumps(metadata)
            )
            
            # Save latest version reference
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=f'models/{sku}/latest_version.txt',
                Body=version
            )
            
            # Update cache
            self.model_cache[sku] = {
                'model': model,
                'metadata': metadata,
                'loaded_at': datetime.now()
            }
            
            # Cleanup
            os.remove(f'/tmp/{sku}_model.pkl')
            
            logging.info(f"Successfully saved model version {version} for SKU {sku}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving model for SKU {sku}: {e}")
            return False

    def get_latest_model(self, sku: str, force_reload: bool = False) -> Tuple[Optional[Prophet], Optional[Dict]]:
        """
        Get the latest model from cache or S3
        """
        try:
            # Check cache first if not forced to reload
            if not force_reload and sku in self.model_cache:
                cache_entry = self.model_cache[sku]
                cache_age = (datetime.now() - cache_entry['loaded_at']).total_seconds()
                
                # Return cached model if it's less than 1 hour old
                if cache_age < 3600:  # 1 hour in seconds
                    logging.info(f"Using cached model for SKU {sku}")
                    return cache_entry['model'], cache_entry['metadata']
            
            # Get latest version
            try:
                response = self.s3.get_object(
                    Bucket=self.bucket_name,
                    Key=f'models/{sku}/latest_version.txt'
                )
                latest_version = response['Body'].read().decode('utf-8')
            except:
                logging.error(f"No model version found for SKU {sku}")
                return None, None
            
            # Load model and metadata
            model_key = f'models/{sku}/{latest_version}/model.pkl'
            metadata_key = f'models/{sku}/{latest_version}/metadata.json'
            
            # Download model
            self.s3.download_file(
                self.bucket_name,
                model_key,
                f'/tmp/{sku}_model.pkl'
            )
            
            # Load model
            with open(f'/tmp/{sku}_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            # Get metadata
            response = self.s3.get_object(
                Bucket=self.bucket_name,
                Key=metadata_key
            )
            metadata = json.loads(response['Body'].read().decode('utf-8'))
            
            # Update cache
            self.model_cache[sku] = {
                'model': model,
                'metadata': metadata,
                'loaded_at': datetime.now()
            }
            
            # Cleanup
            os.remove(f'/tmp/{sku}_model.pkl')
            
            logging.info(f"Successfully loaded model version {latest_version} for SKU {sku}")
            return model, metadata
            
        except Exception as e:
            logging.error(f"Error loading model for SKU {sku}: {e}")
            return None, None

    def get_model_metadata(self, sku: str) -> Optional[Dict]:
        """
        Get metadata for latest model version
        """
        try:
            # Check cache first
            if sku in self.model_cache:
                return self.model_cache[sku]['metadata']
            
            # Get latest version
            response = self.s3.get_object(
                Bucket=self.bucket_name,
                Key=f'models/{sku}/latest_version.txt'
            )
            latest_version = response['Body'].read().decode('utf-8')
            
            # Get metadata
            response = self.s3.get_object(
                Bucket=self.bucket_name,
                Key=f'models/{sku}/{latest_version}/metadata.json'
            )
            metadata = json.loads(response['Body'].read().decode('utf-8'))
            
            return metadata
            
        except Exception as e:
            logging.error(f"Error getting model metadata for SKU {sku}: {e}")
            return None

    def list_model_versions(self, sku: str) -> list:
        """
        List all available versions for a SKU
        """
        try:
            versions = []
            paginator = self.s3.get_paginator('list_objects_v2')
            prefix = f'models/{sku}/'
            
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if obj['Key'].endswith('/metadata.json'):
                            version = obj['Key'].split('/')[-2]
                            versions.append(version)
            
            return sorted(versions)
            
        except Exception as e:
            logging.error(f"Error listing model versions for SKU {sku}: {e}")
            return []

    def cleanup_old_versions(self, sku: str, keep_versions: int = 5):
        """
        Clean up old model versions, keeping the specified number of recent versions
        """
        try:
            versions = self.list_model_versions(sku)
            if len(versions) <= keep_versions:
                return
            
            versions_to_delete = versions[:-keep_versions]
            for version in versions_to_delete:
                prefix = f'models/{sku}/{version}/'
                
                # List all objects with this prefix
                response = self.s3.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix
                )
                
                # Delete objects
                if 'Contents' in response:
                    for obj in response['Contents']:
                        self.s3.delete_object(
                            Bucket=self.bucket_name,
                            Key=obj['Key']
                        )
            
            logging.info(f"Cleaned up {len(versions_to_delete)} old versions for SKU {sku}")
            
        except Exception as e:
            logging.error(f"Error cleaning up old versions for SKU {sku}: {e}")
            
    
