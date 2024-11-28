# app/main.py
from typing import List
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import logging
import boto3
import os
import pandas as pd  # Add this import
from .ingestion import DataIngestion
from .training import ModelTrainer, train_from_s3
from .schemas import (
    PredictionRequest,
    MultiDayPredictionRequest,
    MultiDayPredictionResponse,
    TrainingResponse,
    ModelStatusResponse,
    DataIngestionResponse,
    RetrainingRequest, 
    RetrainingResponse,
    ErrorResponse,
    ModelMetadata
)
from .utils import ModelRegistry, get_skus_from_s3, setup_logging

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Sales Prediction API",
    description="API for sales prediction using Prophet models with S3 storage and Elasticsearch data",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
data_ingestion = DataIngestion()
model_registry = ModelRegistry()
model_trainer = ModelTrainer()
setup_logging()

# Scheduler for periodic tasks
scheduler = BackgroundScheduler()

async def update_data_and_retrain():
    """Weekly task to update data and retrain models"""
    skus = os.getenv('PRODUCT_SKUS', '').split(',')
    
    for sku in skus:
        try:
            # Update data from Elasticsearch
            if data_ingestion.fetch_es_data_and_upload(sku):
                # Train new model using updated data
                success = await train_from_s3(sku)
                if success:
                    logging.info(f"Successfully retrained model for SKU: {sku}")
                else:
                    logging.error(f"Failed to retrain model for SKU: {sku}")
            else:
                logging.warning(f"No new data found for SKU: {sku}, skipping retraining")
        except Exception as e:
            logging.error(f"Error in weekly update for SKU {sku}: {e}")

# Schedule weekly updates
scheduler.add_job(
    update_data_and_retrain,
    'cron',
    day_of_week='mon',
    hour=0,
    minute=0
)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        scheduler.start()
        logging.info("Scheduler started successfully")
    except Exception as e:
        logging.error(f"Error starting scheduler: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    scheduler.shutdown()
    logging.info("Scheduler shut down successfully")

async def verify_sku(sku: str) -> str:
    """Dependency to verify SKU exists in configuration"""
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        bucket_name = os.getenv('S3_BUCKET_NAME')
        
        # Get SKUs from S3
        skus = get_skus_from_s3(s3_client, bucket_name)
        
        if not skus:
            logging.warning("No SKUs found in S3")
            raise HTTPException(
                status_code=404,
                detail="No SKUs found in S3 bucket"
            )
            
        if sku not in skus:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid SKU: {sku}. Available SKUs: {skus}"
            )
            
        return sku
        
    except Exception as e:
        logging.error(f"Error verifying SKU: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error verifying SKU: {str(e)}"
        )

@app.post("/predict_multi_day", response_model=MultiDayPredictionResponse, responses={
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    500: {"model": ErrorResponse}
})
async def predict_multi_day(
    request: MultiDayPredictionRequest,
    sku: str = Depends(verify_sku)
):
    """
    Get predictions for multiple days (5-7 days ahead)
    
    - **product_sku**: Product SKU to predict for
    - **start_date**: Start date for predictions
    - **num_days**: Number of days to predict (5-7)
    """
    try:
        # Load latest model
        model, metadata = model_registry.get_latest_model(request.product_sku)
        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"No model found for SKU: {request.product_sku}"
            )
        
        # Make predictions
        predictions = model_trainer.make_multi_day_prediction(
            model,
            request.product_sku,
            request.start_date,
            request.num_days
        )
        
        if not predictions:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate predictions"
            )
        
        return MultiDayPredictionResponse(predictions=predictions)
        
    except HTTPException as e:
        raise
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making predictions: {str(e)}"
        )

@app.post("/upload_historical_data", response_model=DataIngestionResponse)
async def upload_historical_data(background_tasks: BackgroundTasks):
    """Upload historical CSV data to S3"""
    try:
        data_dir = os.getenv('DATA_DIRECTORY')
        if not data_dir:
            raise HTTPException(
                status_code=400,
                detail="DATA_DIRECTORY environment variable not set"
            )
        
        # Get all CSV files
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            raise HTTPException(
                status_code=404,
                detail="No CSV files found in data directory"
            )
            
        processed_files = []
        total_records = 0
        
        for file in csv_files:
            file_path = os.path.join(data_dir, file)
            sku = data_ingestion.get_sku_from_filename(file)
            
            # Read file to count records
            df = pd.read_csv(file_path)
            total_records += len(df)
            
            # Schedule upload
            background_tasks.add_task(
                data_ingestion.upload_csv_to_s3,
                file_path,
                sku
            )
            processed_files.append(file)
        
        return DataIngestionResponse(
            status="initiated",
            files_processed=processed_files,
            records_processed=total_records,
            success=True,
            sku="all"
        )
        
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        return DataIngestionResponse(
            status="failed",
            files_processed=[],
            records_processed=0,
            success=False,
            sku="all",
            error_message=str(e)
        )
        
@app.get("/skus", response_model=List[str])
async def list_skus():
    """List all available SKUs from S3"""
    skus = model_registry.get_available_skus()
    if not skus:
        raise HTTPException(
            status_code=404,
            detail="No SKUs found in S3 bucket"
        )
    return skus


@app.post("/train_all", response_model=RetrainingResponse)
async def train_all_models(
    request: RetrainingRequest = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Trigger training for all available SKUs
    
    - **force**: Force retraining even if no new data (optional, default: False)
    - **parameters**: Optional override parameters for training
    """
    try:
        skus = model_registry.get_available_skus()
        if not skus:
            raise HTTPException(
                status_code=404,
                detail="No SKUs found in S3 bucket"
            )
            
        training_id = f"train_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Use request parameters if provided, otherwise use defaults
        parameters = None
        if request:
            parameters = request.parameters if request.force else None
        
        # Start training for each SKU
        for sku in skus:
            background_tasks.add_task(train_from_s3, sku, parameters)
            logging.info(f"Queued training for SKU: {sku}")
            
        return RetrainingResponse(
            sku="all",
            status="initiated",
            training_id=training_id,
            start_time=datetime.now(),
            estimated_completion=datetime.now() + timedelta(minutes=30 * len(skus)),
            skus_queued=skus
        )
        
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error initiating training: {str(e)}"
        )

@app.post("/train/{sku}", response_model=RetrainingResponse)
async def train_model_endpoint(
    sku: str,
    request: RetrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger model training for a specific SKU
    
    - **sku**: Product SKU to train (from path)
    - **force**: Force retraining even if no new data
    - **parameters**: Optional training parameters
    """
    try:
        # Verify SKU exists
        skus = model_registry.get_available_skus()
        if sku not in skus:
            raise HTTPException(
                status_code=404,
                detail=f"SKU {sku} not found in S3 bucket"
            )
        
        training_id = f"train_{sku}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get previous version if exists
        previous_version = None
        try:
            metadata = model_registry.get_model_metadata(sku)
            previous_version = metadata.get('version')
        except:
            pass
            
        # Start training
        parameters = request.parameters if request.force else None
        background_tasks.add_task(train_from_s3, sku, parameters)
        
        return RetrainingResponse(
            sku=sku,
            status="initiated",
            training_id=training_id,
            start_time=datetime.now(),
            estimated_completion=datetime.now() + timedelta(minutes=30),
            previous_version=previous_version
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Training error for SKU {sku}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error initiating training: {str(e)}"
        )

@app.get("/model_status/{sku}", response_model=ModelStatusResponse)
async def get_model_status(sku: str = Depends(verify_sku)):
    """
    Get the status and metadata of the latest model for a SKU
    
    - **sku**: Product SKU to check
    """
    try:
        model = model_registry.get_latest_model(sku)
        if model is None:
            return ModelStatusResponse(
                sku=sku,
                status="not_found",
                error_message=f"No model found for SKU: {sku}"
            )
            
        metadata = model_registry.get_model_metadata(sku)
        return ModelStatusResponse(
            sku=sku,
            status="available",
            metadata=ModelMetadata(**metadata),
            last_updated=datetime.now()
        )
        
    except Exception as e:
        logging.error(f"Error getting model status for SKU {sku}: {str(e)}")
        return ModelStatusResponse(
            sku=sku,
            status="error",
            error_message=str(e)
        )