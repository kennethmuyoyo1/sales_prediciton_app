# app/schemas.py

from pydantic import BaseModel, Field
from datetime import date, datetime
from typing import List, Optional, Dict, Any

class Prediction(BaseModel):
    """Single prediction value with confidence intervals"""
    high: float = Field(..., description="Upper bound of prediction interval")
    low: float = Field(..., description="Lower bound of prediction interval")
    median: float = Field(..., description="Median prediction value")

class SimplePrediction(BaseModel):
    """Response model for a single prediction"""
    product_sku: str = Field(..., description="Product SKU")
    prediction_date: date = Field(..., description="Prediction date")
    prediction_value: Prediction = Field(..., description="Prediction values")

class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    product_sku: str = Field(..., description="Product SKU to predict for")
    target_date: date = Field(..., description="Date to predict for")

class MultiDayPredictionRequest(BaseModel):
    """Request model for multiple day predictions"""
    product_sku: str = Field(..., description="Product SKU to predict for")
    start_date: date = Field(..., description="Start date for predictions")
    num_days: int = Field(
        ..., 
        description="Number of days to predict",
        ge=5,
        le=7
    )

class MultiDayPredictionResponse(BaseModel):
    """Response model for multiple day predictions"""
    predictions: List[SimplePrediction] = Field(
        ...,
        description="List of daily predictions"
    )

class TrainingResponse(BaseModel):
    """Response model for training operations"""
    sku: str = Field(..., description="SKU being trained")
    status: str = Field(..., description="Training status")
    message: str = Field(..., description="Status message")
    training_id: Optional[str] = Field(None, description="Training job identifier")
    start_time: datetime = Field(
        default_factory=datetime.now,
        description="Training start time"
    )

class ModelMetadata(BaseModel):
    """Model metadata information"""
    version: str = Field(..., description="Model version")
    training_date: datetime = Field(..., description="Date model was trained")
    metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    parameters: Dict[str, Any] = Field(..., description="Model parameters")

class ModelStatusResponse(BaseModel):
    """Response model for model status endpoint"""
    sku: str = Field(..., description="Product SKU")
    status: str = Field(..., description="Model status")
    metadata: Optional[ModelMetadata] = Field(None, description="Model metadata if available")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")
    error_message: Optional[str] = Field(None, description="Error message if any")

class DataIngestionResponse(BaseModel):
    """Response model for data ingestion operations"""
    sku: str = Field(..., description="Product SKU")
    status: str = Field(..., description="Ingestion status")
    files_processed: List[str] = Field(..., description="List of processed files")
    records_processed: int = Field(..., description="Number of records processed")
    success: bool = Field(..., description="Overall success status")
    error_message: Optional[str] = Field(None, description="Error message if any")

class RetrainingRequest(BaseModel):
    """Request model for model retraining"""
    force: bool = Field(
        False,
        description="Force retraining even if no new data"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional override parameters"
    ) 
    
class RetrainingResponse(BaseModel):
    """Response model for retraining operation"""
    sku: str = Field(..., description="Product SKU or 'all'")
    status: str = Field(..., description="Retraining status")
    training_id: str = Field(..., description="Training job identifier")
    start_time: datetime = Field(..., description="Retraining start time")
    estimated_completion: Optional[datetime] = Field(
        None,
        description="Estimated completion time"
    )
    previous_version: Optional[str] = Field(
        None,
        description="Previous model version"
    )
    skus_queued: Optional[List[str]] = Field(
        None,
        description="List of SKUs queued for training (only for train_all)"
    )

class ErrorResponse(BaseModel):
    """Standard error response"""
    detail: str = Field(..., description="Error detail message")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Error timestamp"
    )